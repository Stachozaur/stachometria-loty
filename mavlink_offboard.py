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

    def force_px4_sitl_target(self) -> bool:
        """Wymusza target MAVLink na PX4 SITL autopilot (sys=1, comp=1)."""
        if self._conn is None:
            return False
        self._target_system = 1
        self._target_component = 1
        return True

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

    def disarm(self) -> bool:
        """Wysyła komendę DISARM (MAV_CMD_COMPONENT_ARM_DISARM, param1=0)."""
        if self._conn is None:
            return False
        self._conn.mav.command_long_send(
            self._target_system,
            self._target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        return True

    def disarm_px4_sitl_default(self) -> bool:
        """DISARM z target_system=1 (SITL), gdy nie było heartbeat / znanych adresów."""
        if self._conn is None:
            return False
        self._target_system = 1
        self._target_component = 1
        return self.disarm()

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

    def send_nav_land_in_place(self) -> bool:
        """
        Wysyła MAV_CMD_NAV_LAND — PX4 ląduje w bieżącej pozycji (parametry globalne jako NaN).
        Użyteczne po zatrzymaniu w powietrzu, gdy sam AUTO LAND nie wystartuje zejścia.
        """
        if self._conn is None:
            return False
        nan = float("nan")
        self._conn.mav.command_long_send(
            self._target_system,
            self._target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0,
            0.0,
            0.0,
            0.0,
            nan,
            nan,
            nan,
            nan,
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

    def send_velocity_target_ned(
        self,
        time_boot_ms: int,
        vx: float,
        vy: float,
        vz: float,
        yaw: Optional[float] = None,
    ) -> bool:
        """Wysyła setpoint prędkości (NED). Opcjonalny yaw (rad, NED) — dron skierowany dziobem w kierunku lotu."""
        # Bits 0,1,2 = ignore x,y,z position; 6,7,8 = ignore ax,ay,az
        vel_only_mask = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 6) | (1 << 7) | (1 << 8)
        if yaw is None:
            # ignoruj yaw — PX4 sam trzyma ostatni heading
            vel_only_mask |= (1 << 10)
        return self.send_position_target_local_ned(
            time_boot_ms, 0, 0, 0, vx, vy, vz,
            yaw=float(yaw) if yaw is not None else 0.0,
            type_mask=vel_only_mask,
        )

    def set_px4_parameters(self, params: dict[str, float], repeats: int = 3) -> bool:
        """
        Ustawia parametry PX4 przez MAVLink PARAM_SET.

        Uwaga: PX4 może być „wrażliwe” na zmiany parametrów w trakcie lotu/trybów,
        dlatego domyślnie wysyłamy je kilka razy.
        """
        if self._conn is None:
            return False

        # Pymavlink: MAV_PARAM_TYPE_REAL32 = float
        try:
            param_type = mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        except Exception:
            param_type = 9  # fallback: najczęściej REAL32

        ok = True
        # Param id ma maks długość 16 znaków w MAVLink (bez null-terminacji).
        items = [(str(k)[:16], float(v)) for k, v in params.items()]
        for _ in range(max(1, int(repeats))):
            for param_id, param_value in items:
                try:
                    self._conn.mav.param_set_send(
                        self._target_system,
                        self._target_component,
                        param_id,
                        param_value,
                        param_type,
                    )
                except Exception:
                    ok = False
            time.sleep(0.02)
        return ok

    def read_px4_param(self, param_id: str, timeout_s: float = 1.0) -> Optional[float]:
        """
        Odczytuje pojedynczy parametr PX4 przez MAVLink PARAM_REQUEST_READ.

        PX4 SITL offboard port (14580) wysyła odpowiedzi (PARAM_VALUE) na stały adres
        zdefiniowany przez -o w px4-rc.mavlink (domyślnie 14540), nie z powrotem do nadawcy.
        Dlatego nasłuchujemy na dodatkowym gnieździe udpin:0.0.0.0:14540.

        Zwraca float (param_value) lub None, jeśli timeout / brak odpowiedzi.
        """
        if self._conn is None:
            return None

        pid = str(param_id)[:16]

        # Osobne gniazdo odbiorcze na 14540 — tam PX4 wysyła PARAM_VALUE (flaga -o w px4-rc.mavlink).
        rx_conn = None
        try:
            rx_conn = mavutil.mavlink_connection(
                "udpin:0.0.0.0:14540",
                source_system=self._system_id,
                source_component=self._component_id,
            )
        except Exception:
            rx_conn = None

        try:
            self._conn.mav.param_request_read_send(
                self._target_system,
                self._target_component,
                pid,
                -1,  # param index: -1 = szukaj po nazwie
            )
        except Exception:
            if rx_conn is not None:
                try:
                    rx_conn.close()
                except Exception:
                    pass
            return None

        def _match_param_value(conn) -> Optional[float]:
            t0 = time.time()
            while time.time() - t0 < float(timeout_s):
                try:
                    msg = conn.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.1)
                except Exception:
                    msg = None
                if msg is None:
                    continue
                try:
                    got_id = getattr(msg, "param_id", None)
                    if isinstance(got_id, bytes):
                        got_id = got_id.decode("utf-8", errors="ignore")
                    got_id = str(got_id).replace("\x00", "")
                    if got_id[:16] == pid:
                        return float(getattr(msg, "param_value", None))
                except Exception:
                    continue
            return None

        result = None
        # Najpierw odbiornik 14540 (gdzie PX4 faktycznie wysyła odpowiedź).
        if rx_conn is not None:
            result = _match_param_value(rx_conn)
            try:
                rx_conn.close()
            except Exception:
                pass
        # Fallback: spróbuj też na głównym gnieździe (np. gdy PX4 odpowiada do nadawcy).
        if result is None:
            result = _match_param_value(self._conn)
        return result

    def read_px4_params(self, param_ids: list[str], timeout_s: float = 1.0) -> dict[str, float]:
        """
        Odczytuje listę parametrów (kolejno) i zwraca tylko te, które udało się odczytać.
        """
        out: dict[str, float] = {}
        for pid in param_ids:
            v = self.read_px4_param(pid, timeout_s=timeout_s)
            if v is not None:
                out[pid] = v
        return out

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
