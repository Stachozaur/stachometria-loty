# -*- coding: utf-8 -*-
"""
Opcjonalny helper do wysyłania setpointów MAVLink do PX4 (tryb offboard).
Używany do realizacji trajektorii scenariuszy (np. lot 200 m prosto).
Połączenie do PX4 na UDP 14550 (port GCS) — symulator używa 4560.
"""
from __future__ import annotations

import time
from typing import Optional

try:
    from pymavlink import mavutil
    from pymavlink.dialects.v20 import common as mavlink2
    HAS_PYMAVLINK = True
except ImportError:
    HAS_PYMAVLINK = False


# NED: x North, y East, z Down (w dół dodatnie)
# type_mask: 0 = use position, 3576 = use velocity (ignoruj pozycję)
# https://mavlink.io/en/messages/common.html#SET_POSITION_TARGET_LOCAL_NED
POSITION_TARGET_TYPEMASK_POSITION = 0
POSITION_TARGET_TYPEMASK_VELOCITY = 1 << 3 | 1 << 4 | 1 << 5  # xyz velocity
POSITION_TARGET_TYPEMASK_ACCEL = 1 << 6 | 1 << 7 | 1 << 8


class MavlinkOffboard:
    """Połączenie MAVLink do PX4 w celu wysyłania setpointów (offboard)."""

    def __init__(self, connection_string: str = "udpin:0.0.0.0:14550"):
        if not HAS_PYMAVLINK:
            raise RuntimeError("pymavlink nie jest zainstalowany")
        # udpin:0.0.0.0:14550 = nasłuchuj na 14550, PX4 SITL łączy się tam jako GCS
        self.connection_string = connection_string
        self._conn: Optional[mavutil.mavlink_connection] = None
        self._system_id = 255
        self._component_id = 1
        self._target_system = 1
        self._target_component = 1

    def bind(self) -> bool:
        """Otwiera port (bind) bez czekania na heartbeat. Wywołaj przed pętlą symulacji."""
        try:
            self._conn = mavutil.mavlink_connection(
                self.connection_string,
                source_system=self._system_id,
                source_component=self._component_id,
            )
            return True
        except Exception:
            return False

    def try_recv_heartbeat(self) -> bool:
        """Krótka próba odbioru HEARTBEAT (timeout 5 ms), żeby nie blokować symulacji. Zwraca True jeśli odebrano."""
        if self._conn is None:
            return False
        # blocking=False przy recv_msg() i tak blokuje na socket.recv() — używamy krótkiego timeoutu
        msg = self._conn.recv_match(type="HEARTBEAT", blocking=True, timeout=0.005)
        if msg is not None:
            self._target_system = msg.get_srcSystem()
            self._target_component = msg.get_srcComponent()
            return True
        return False

    def connect(self, timeout_s: float = 15.0) -> bool:
        """Nawiązuje połączenie (bind + czekaj na heartbeat). Blokuje — nie używać gdy symulacja musi iść."""
        if self.bind():
            t0 = time.time()
            while time.time() - t0 < timeout_s:
                msg = self._conn.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
                if msg is not None:
                    self._target_system = msg.get_srcSystem()
                    self._target_component = msg.get_srcComponent()
                    return True
        return False

    def is_connected(self) -> bool:
        return self._conn is not None

    # PX4 custom_mode: union { reserved(uint16), main_mode(uint8), sub_mode(uint8) } → (main<<16)|(sub<<24)
    # MAIN_MODE_AUTO=4, SUB_MODE_AUTO_TAKEOFF=2 → Takeoff
    PX4_CUSTOM_MODE_TAKEOFF = (4 << 16) | (2 << 24)

    def set_mode_takeoff_px4(self) -> bool:
        """Ustawia tryb Takeoff (PX4 AUTO_TAKEOFF). Po ARM dron sam wznosi do MIS_TAKEOFF_ALT."""
        if self._conn is None:
            return False
        # SET_MODE: base_mode=MAV_MODE_FLAG_CUSTOM_MODE_ENABLED (1), custom_mode=Takeoff
        self._conn.mav.set_mode_send(
            self._target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            self.PX4_CUSTOM_MODE_TAKEOFF,
        )
        return True

    def arm_and_takeoff_no_heartbeat(self, altitude_m: float = 10.0) -> bool:
        """Wysyła ARM i TAKEOFF bez czekania na heartbeat (np. na port 14580 — adres PX4 Onboard). target_system=1, target_component=1."""
        if self._conn is None:
            return False
        self._target_system = 1
        self._target_component = 1
        self._conn.mav.command_long_send(
            self._target_system, self._target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0,
        )
        self._conn.mav.command_long_send(
            self._target_system, self._target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, float(altitude_m),
        )
        return True

    def arm(self) -> bool:
        """Wysyła komendę ARM."""
        if self._conn is None:
            return False
        self._conn.mav.command_long_send(
            self._target_system, self._target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0,
        )
        return True

    def takeoff(self, altitude_m: float = 10.0) -> bool:
        """Wysyła komendę TAKEOFF (MAV_CMD_NAV_TAKEOFF). altitude_m w metrach (AMSL lub względem startu w zależności od ramy)."""
        if self._conn is None:
            return False
        # MAV_CMD_NAV_TAKEOFF: param7 = altitude (meters)
        self._conn.mav.command_long_send(
            self._target_system, self._target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, float(altitude_m),
        )
        return True

    # PX4 MAIN_MODE_OFFBOARD = 6 (px4_custom_mode.h)
    PX4_CUSTOM_MODE_OFFBOARD = (6 << 16) | (0 << 24)
    # MAIN_MODE_AUTO = 4, SUB_MODE_AUTO_LAND = 6 — lądowanie w miejscu (po doleceniu nad cel)
    PX4_CUSTOM_MODE_AUTO_LAND = (4 << 16) | (6 << 24)

    def set_mode_offboard(self) -> bool:
        """Przełącza w tryb OFFBOARD (PX4). Wymaga ciągłego wysyłania setpointów (np. velocity NED)."""
        if self._conn is None:
            return False
        self._conn.mav.set_mode_send(
            self._target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            self.PX4_CUSTOM_MODE_OFFBOARD,
        )
        return True

    def set_mode_auto_land_px4(self) -> bool:
        """Przełącza w tryb AUTO LAND (PX4). Ląduje w bieżącym punkcie poziomym; wyłącza OFFBOARD."""
        if self._conn is None:
            return False
        self._conn.mav.set_mode_send(
            self._target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            self.PX4_CUSTOM_MODE_AUTO_LAND,
        )
        return True

    def send_position_target_local_ned(
        self,
        time_boot_ms: int,
        x: float, y: float, z: float,
        vx: float = 0.0, vy: float = 0.0, vz: float = 0.0,
        afx: float = 0.0, afy: float = 0.0, afz: float = 0.0,
        yaw: float = 0.0, yaw_rate: float = 0.0,
        type_mask: int = 0,
    ) -> bool:
        """Wysyła SET_POSITION_TARGET_LOCAL_NED. Pozycja w NED (m), prędkość w NED (m/s)."""
        if self._conn is None:
            return False
        self._conn.mav.set_position_target_local_ned_send(
            time_boot_ms,
            self._target_system, self._target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            x, y, z,
            vx, vy, vz,
            afx, afy, afz,
            yaw, yaw_rate,
        )
        return True

    def send_velocity_target_ned(self, time_boot_ms: int, vx: float, vy: float, vz: float) -> bool:
        """Wysyła setpoint prędkości (NED). type_mask: ignoruj pozycję i przyspieszenie."""
        # Bits 0,1,2 = ignore x,y,z position; 6,7,8 = ignore ax,ay,az
        vel_only_mask = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 6) | (1 << 7) | (1 << 8)
        return self.send_position_target_local_ned(
            time_boot_ms, 0, 0, 0, vx, vy, vz, type_mask=vel_only_mask,
        )

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
