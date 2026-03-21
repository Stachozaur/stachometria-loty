# -*- coding: utf-8 -*-
"""
Runner Stachometr: uruchamia Pegasus IRIS + PX4 w Isaac Sim, wykonuje scenariusze
z docs/py/scenarios.md, zapisuje parametry (JSON) i logi (CSV) dla każdego runu.

Użycie:
  Preview (z GUI; scenariusz 1 = losowe parametry, **zawsze z wiatrem**):
    ./python.sh stachometr/run_stachometr.py --preview --scenario 1
    # Opcjonalnie: --preview-pair (alias, ten sam jeden lot z wiatrem — zostawione dla starych skryptów)

  Scenariusz 1 z ustalonymi parametrami — osobno z wiatrem lub bez (jawne flagi):
    ./python.sh stachometr/run_stachometr.py --preview --scenario_1_no_wind
    ./python.sh stachometr/run_stachometr.py --preview --scenario_1_wind

  Headless (50 lotów per scenariusz, bez GUI):
    ./python.sh stachometr/run_stachometr.py --headless --scenario 1 --runs 50 --output-dir ./stachometr_output

  Scenariusz 1: 10 losowań (każdy lot **z wiatrem**), długi dystans — zwiększ --duration-s:
    ./python.sh stachometr/run_stachometr.py --headless --scenario 1 --runs 10 --duration-s 600 --output-dir ./stachometr_output

  Podgląd losowego scenariusza 1 z wiatrem:
    ./python.sh stachometr/run_stachometr.py --preview --scenario 1 --seed $RANDOM

  Scenariusz 2 (lot do celu 500–1000 m + lądowanie PX4 AUTO LAND), z wiatrem:
    ./python.sh stachometr/run_stachometr.py --preview --scenario 2 --seed $RANDOM
  Większe podłoże / "kratka" w viewport (~2 km bok, domyślnie włączone):
    # opcjonalnie: --ground-extent-m 0 wyłącza; --ground-extent-m 3000 → 3 km

  Dla wszystkich scenariuszy po 50 lotów (batch):
    for s in $(seq 1 10); do ./python.sh stachometr/run_stachometr.py --headless --scenario $s --runs 50; done
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Dodaj ścieżkę do stachometr (import scenario_params)
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Uruchomienie Isaac Sim musi być na początku (przed importami omni/pegasus)
def _parse_args_before_isaac():
    parser = argparse.ArgumentParser(description="Stachometr: scenariusze lotów Pegasus IRIS + PX4")
    parser.add_argument("--preview", action="store_true", help="Jedno uruchomienie z GUI (na żywo)")
    parser.add_argument(
        "--preview-pair",
        action="store_true",
        help="Alias (scenariusz 1): ten sam podgląd co --preview --scenario 1 — jeden lot, z wiatrem",
    )
    parser.add_argument("--headless", action="store_true", help="Bez GUI (do batch)")
    parser.add_argument("--scenario", type=int, default=1, choices=range(1, 11), metavar="1..10", help="Numer scenariusza")
    parser.add_argument("--scenario_1_no_wind", action="store_true", help="Scenariusz 1 z ustalonymi parametrami, bez wiatru (jeden run)")
    parser.add_argument("--scenario_1_wind", action="store_true", help="Scenariusz 1 z ustalonymi parametrami, z wiatrem (jeden run)")
    parser.add_argument("--runs", type=int, default=1, help="Liczba lotów (runów) w jednej sesji (headless)")
    parser.add_argument("--output-dir", type=str, default=None, help="Katalog na JSON+CSV (domyślnie: stachometr_output)")
    parser.add_argument("--seed", type=int, default=None, help="Seed dla powtarzalności (opcjonalnie)")
    parser.add_argument("--duration-s", type=float, default=120.0, help="Czas jednego runu w sekundach (symulacja)")
    parser.add_argument(
        "--ground-extent-m",
        type=float,
        default=8000.0,
        help="Docelowy bok plaskeigo podloza (m) po przeskalowaniu — wieksza 'kratka'/ziemia na dlugie loty; 0=wylacz",
    )
    return parser.parse_known_args()[0]

_ARGS_EARLY = _parse_args_before_isaac()

# Teraz start Isaac Sim
import carb
from isaacsim import SimulationApp

_HEADLESS = _ARGS_EARLY.headless if _ARGS_EARLY.headless else (not _ARGS_EARLY.preview)
simulation_app = SimulationApp({"headless": _HEADLESS})

# --- Follow camera + Attitude cam (tylko w trybie GUI) ---
_HAS_FOLLOW_CAM = False
_set_camera_view_fn = None
_np_cam = None
_attitude_cam_vp = None     # ViewportWindow dla attitude cam (Viewport 2)

# Debug-draw trail — zero USD / zero PhysX (pure viewport overlay)
_HAS_DEBUG_DRAW = False
_debug_draw_iface = None
_trail_deque = None          # collections.deque of (x,y,z) tuples

if not _HEADLESS:
    try:
        from isaacsim.core.utils.viewports import set_camera_view as _set_camera_view_fn
        import numpy as _np_cam
        _HAS_FOLLOW_CAM = True
    except Exception:
        pass
    try:
        from isaacsim.util.debug_draw import _debug_draw as _dd_mod
        _debug_draw_iface = _dd_mod.acquire_debug_draw_interface()
        _HAS_DEBUG_DRAW = True
    except Exception:
        try:
            import omni.isaac.debug_draw._debug_draw as _dd_mod
            _debug_draw_iface = _dd_mod.acquire_debug_draw_interface()
            _HAS_DEBUG_DRAW = True
        except Exception:
            pass

# --- Reszta importów po uruchomieniu Sim ---
import collections
import json
import omni.physx
import omni.timeline
import omni.usd
from omni.isaac.core.world import World
from pxr import Gf, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

from scenario_params import (
    K_WIND_N_PER_MS,
    draw_scenario,
    get_scenario_1_fixed_params,
    get_scenario_description,
    merge_wind_defaults,
    PHASE_LABELS_DOC,
)
from wind_generator import make_wind_generator

# Odczyty na żywo przez HTTP (localhost) — tylko w preview
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

LIVE_DISPLAY_PORT = 8765


def _wind_force_legacy_ned(w_ned: tuple[float, float, float], k_wind: float) -> tuple[float, float, float]:
    """Legacy: F = k * ||w_h|| * (w_h / ||w_h||) — skalowanie jak K * prędkość, kierunek jak bieżący wektor wiatru."""
    wn, we, _wz = w_ned
    h = math.hypot(wn, we)
    if h < 1e-9:
        return (0.0, 0.0, 0.0)
    s = k_wind * h
    return (s * wn / h, s * we / h, 0.0)


def _px4_velocity_mission_limits(params: dict[str, Any], v_cmd_ms: float) -> dict[str, float]:
    """
    Limity prędkości PX4 (PARAM_SET) przed lotem OFFBOARD.

    **Skąd "blokada" ~12–20 m/s:** autopilot stosuje m.in. **MPC_XY_VEL_MAX** jako sufit na |v_xy|.
    Stałe 20 m/s w kodzie obcinały setpoint przy ``v_cmd_ms`` do 30 m/s — w CSV (GT) widać było
    niższą prędkość mimo większego MAVLinka. To nie jest limit Pegasus/Isaac z tego skryptu,
    tylko **parametr PX4** (do "ominiecia" trzeba go podnieść, nie ma magicznego bypassu).

    Opcjonalne klucze w ``params`` / JSON runu:
      - ``px4_mpc_xy_vel_max`` — jawny sufit XY (m/s)
      - ``px4_mpc_z_vel_max_dn`` / ``px4_mpc_z_vel_max_up`` — Z NED (m/s)
      - ``px4_mpc_acc_hor`` → **MPC_ACC_HOR** (m/s²), tylko jeśli ustawione — szybsze dochodzenie do dużego v_cmd
    """
    v = max(0.1, float(v_cmd_ms))
    xy_raw = params.get("px4_mpc_xy_vel_max")
    if xy_raw is None:
        # Zapas na śledzenie setpointu + wiatr; must-have: v_cmd do 30 m/s.
        # W poprzedniej wersji ~20 m/s bywało widoczne w GT (PX4 sufit).
        # Dajemy wyraźnie wyższy limit, żeby nie klamrować setpointu.
        mpc_xy = max(v * 2.0 + 10.0, 45.0)
    else:
        mpc_xy = float(xy_raw)
    mpc_xy = max(mpc_xy, v + 5.0)

    zdn_raw = params.get("px4_mpc_z_vel_max_dn")
    if zdn_raw is None:
        mpc_z_dn = 6.0 + 0.20 * min(v, 30.0)
    else:
        mpc_z_dn = float(zdn_raw)

    zup_raw = params.get("px4_mpc_z_vel_max_up")
    if zup_raw is None:
        mpc_z_up = 7.0 + 0.20 * min(v, 30.0)
    else:
        mpc_z_up = float(zup_raw)

    out: dict[str, float] = {
        "MPC_XY_VEL_MAX": mpc_xy,
        # Dodatkowe prędkości horyzontalne (część ścieżek kontrolera odnosi się do tych limitów).
        # Trzymamy je <= MPC_XY_VEL_MAX, ale wystarczająco wysoko dla v_cmd.
        "MPC_XY_CRUISE": min(mpc_xy, max(v, 8.0)),
        "MPC_VEL_MANUAL": min(mpc_xy, max(v, 8.0)),
        "MPC_Z_VEL_MAX_DN": mpc_z_dn,
        "MPC_Z_VEL_MAX_UP": mpc_z_up,
    }
    acc_raw = params.get("px4_mpc_acc_hor")
    if acc_raw is not None:
        out["MPC_ACC_HOR"] = float(acc_raw)
    return out


def _meteo_wind_from_deg(wn: float, we: float) -> float:
    """Kierunek meteo "z którego wieje" [°], 0=N, 90=E — z wektora prędkości powietrza w NED (poziomo)."""
    if abs(wn) < 1e-9 and abs(we) < 1e-9:
        return 0.0
    rad = math.atan2(-we, -wn)
    return math.degrees(rad) % 360.0


def _wind_force_drag_ned(
    v_ned: tuple[float, float, float],
    w_ned: tuple[float, float, float],
    rho: float,
    cd_a: float,
) -> tuple[float, float, float]:
    """Opór względem powietrza: v_air = v_drona - w_wiatru (NED); tylko płaszczyzna pozioma. F = -0.5 ρ CdA |v_h| v_h."""
    vn, ve, _vz = float(v_ned[0]), float(v_ned[1]), float(v_ned[2])
    wn, we, _wz = float(w_ned[0]), float(w_ned[1]), float(w_ned[2])
    an = vn - wn
    ae = ve - we
    mag = math.hypot(an, ae)
    if mag < 1e-9:
        return (0.0, 0.0, 0.0)
    c = -0.5 * rho * cd_a * mag
    return (c * an, c * ae, 0.0)


def _make_live_display_handler(state_ref):
    """Handler zwracający state_ref z zewnątrz (closure)."""
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/data":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(state_ref.get("data", {})).encode("utf-8"))
                return
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                html = _LIVE_HTML
                self.wfile.write(html.encode("utf-8"))
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):
            pass
    return _Handler


_LIVE_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Stachometr — odczyty</title>
<style>
body { font-family: monospace; background: #1e1e1e; color: #0f0; padding: 12px; margin: 0; }
h1 { font-size: 14px; }
h2 { font-size: 13px; color: #8f8; margin: 0 0 6px 0; }
h3 { font-size: 11px; color: #aa8; margin: 8px 0 2px 0; font-weight: normal; }
.wrap { display: flex; gap: 18px; flex-wrap: wrap; align-items: flex-start; }
.col-readings { flex: 1.25; min-width: 300px; }
.col-params { flex: 0.9; min-width: 200px; }
.readings-2col { display: flex; gap: 6px; align-items: flex-start; }
.readings-col { flex: 1; min-width: 0; }
/* Jedna szerokość etykiety wszędzie → wartości w jednej pionowej linii; wartości do prawej */
table.kv-table {
  width: 100%;
  table-layout: fixed;
  border-collapse: collapse;
  margin-bottom: 4px;
  font-size: 12px;
}
table.kv-table td { padding: 2px 0; vertical-align: baseline; }
table.kv-table td.kv-k {
  box-sizing: border-box;
  width: 24ch;
  min-width: 24ch;
  max-width: 24ch;
  color: #888;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 8px;
}
table.kv-table td.kv-v {
  text-align: right;
  color: #0f0;
  font-variant-numeric: tabular-nums;
}
.aktywny-manewr {
  background: #1a2200;
  border: 1px solid #aaff00;
  border-radius: 3px;
  padding: 5px 10px;
  margin-bottom: 10px;
  font-size: 15px;
  letter-spacing: 0.03em;
}
.manewr-label { color: #888; font-size: 11px; }
.manewr-val   { color: #aaff00; font-weight: bold; font-size: 15px; }
</style></head>
<body>
<h1>Stachometr — odczyty na żywo</h1>
<div class="wrap">
  <div class="col-readings" id="readings"></div>
  <div class="col-params" id="params"></div>
</div>
<script>
function refresh() {
  fetch('/data').then(r => r.json()).then(d => {
    const paramKeys = Object.entries(d).filter(([k]) => k.startsWith('param_'));
    const row = (k, v) =>
      '<tr><td class="kv-k">' +
      String(k).replace(/^param_/, '') +
      '</td><td class="kv-v">' +
      v +
      '</td></tr>';
    const WIND_KEYS = ['wiatr', 'param_wind_speed_ms', 'param_wind_dir_deg', 'param_wind_dynamic_enabled', 'param_wind_force_model'];
    const WIND_PARAM_LABELS = {
      'param_wind_speed_ms': 'wind_speed_ms — średnia bazy (nie to co |w| w locie)',
      'param_wind_dir_deg': 'wind_dir_deg — średni kierunek (meteo)',
      'param_wind_dynamic_enabled': 'wind_dynamic_enabled',
      'param_wind_force_model': 'wind_force_model',
      'wiatr': 'wiatr (run z plikiem *_wind)',
    };
    let manewrHtml = '';
    if ('aktywny_manewr' in d) {
      manewrHtml = '<div class="aktywny-manewr">'
        + '<span class="manewr-label">MANEWR:</span> '
        + '<span class="manewr-val">' + d['aktywny_manewr'] + '</span>'
        + '</div>';
    }
    const LEFT = [
      { lab: 'Czas i dystans', keys: ['czas_symulacji_s', 'czas_rzeczywisty_s', 'przeleciono_m', 'cel_m'] },
      { lab: 'Wiatr (symulacja, bieżąco)', keys: ['wiatr_bieżący_m/s', 'wiatr_z_którego_°', 'wiatr_podmuch_aktywny'] },
      { lab: 'Prędkość (GT)', keys: ['vel_x', 'vel_y', 'vel_z'] },
      { lab: 'Barometr', keys: ['baro_pressure_hPa', 'baro_temp_C', 'baro_alt_m'] },
      { lab: 'Żyroskop (body)', keys: ['gyro_x', 'gyro_y', 'gyro_z'] },
    ];
    const RIGHT = [
      { lab: 'Pozycja (GT)', keys: ['pos_x', 'pos_y', 'pos_z'] },
      { lab: 'Orientacja (GT)', keys: ['roll_deg', 'pitch_deg', 'yaw_deg', 'kwat_w', 'kwat_x', 'kwat_y', 'kwat_z'] },
      { lab: 'Akcelerometr (body)', keys: ['acc_x', 'acc_y', 'acc_z'] },
      { lab: 'Silniki', keys: ['motor_1', 'motor_2', 'motor_3', 'motor_4', 'throttle'] },
    ];
    function renderCols(sections) {
      let h = '';
      for (const sec of sections) {
        const rows = sec.keys.filter(k => k in d).map(k => row(k, d[k])).join('');
        if (rows) h += '<h3>' + sec.lab + '</h3><table class="kv-table">' + rows + '</table>';
      }
      return h;
    }
    const seen = new Set([
      ...LEFT.flatMap(s => s.keys),
      ...RIGHT.flatMap(s => s.keys),
      ...WIND_KEYS,
      'aktywny_manewr',
    ]);
    let readingsHtml = '<h2>Odczyty na żywo</h2>' + manewrHtml + '<div class="readings-2col">';
    readingsHtml += '<div class="readings-col">' + renderCols(LEFT) + '</div>';
    readingsHtml += '<div class="readings-col">' + renderCols(RIGHT) + '</div></div>';
    const rest = Object.entries(d).filter(([k]) => !k.startsWith('param_') && !seen.has(k));
    if (rest.length) readingsHtml += '<h3>Inne</h3><table class="kv-table">' + rest.map(([k, v]) => row(k, v)).join('') + '</table>';
    document.getElementById('readings').innerHTML = readingsHtml;
    const windRow = (k, v) => {
      const lab = WIND_PARAM_LABELS[k] || String(k).replace(/^param_/, '');
      return '<tr><td class="kv-k">' + lab + '</td><td class="kv-v">' + v + '</td></tr>';
    };
    const windRows = WIND_KEYS.filter(k => k in d).map(k => windRow(k, d[k])).join('');
    let paramsHtml = '<h2>Parametry scenariusza</h2>';
    if (windRows) paramsHtml += '<h3>Wiatr</h3><table class="kv-table">' + windRows + '</table>';
    if (paramKeys.length) paramsHtml += '<h3>Ustawienia</h3><table class="kv-table">' + paramKeys.map(([k, v]) => row(k, v)).join('') + '</table>';
    document.getElementById('params').innerHTML = paramsHtml;
  }).catch(() => {});
}
setInterval(refresh, 100);
refresh();
</script>
</body></html>
"""


def _params_to_display(params: dict | None) -> dict:
    """Z formatowanych parametrów scenariusza buduje słownik do wyświetlenia (tylko proste wartości)."""
    out = {}
    if not params:
        return out
    skip = {"phase_times", "phase_times_description", "scenario_1_runs"}
    for k, v in params.items():
        if k in skip or isinstance(v, (list, dict)):
            continue
        if isinstance(v, float):
            out[f"param_{k}"] = f"{v:.4f}" if abs(v) < 1e6 else f"{v:.2f}"
        else:
            out[f"param_{k}"] = str(v)
    return out


class _LiveDisplay:
    """Serwer HTTP na localhost — odczyty w przeglądarce pod http://127.0.0.1:8765/ ."""

    def __init__(self, wind_suffix: str | None = None, params: dict | None = None):
        self._state = {"data": {}}
        self._server = None
        self._thread = None
        self._wind_enabled = wind_suffix == "wind"
        self._wind_speed_ms = params.get("wind_speed_ms", "") if params else ""
        self._wind_dir_deg = params.get("wind_dir_deg", "") if params else ""
        self._params_display = _params_to_display(params)
        self._start_wall_s = time.time()
        try:
            handler = _make_live_display_handler(self._state)
            self._server = HTTPServer(("127.0.0.1", LIVE_DISPLAY_PORT), handler)
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            carb.log_info(
                f"Stachometr: odczyty na żywo → otwórz w przeglądarce: http://127.0.0.1:{LIVE_DISPLAY_PORT}/ "
                + (f"({wind_suffix})" if wind_suffix else "")
            )
        except OSError as e:
            carb.log_warn(f"Stachometr: nie udało się uruchomić serwera odczytów (port {LIVE_DISPLAY_PORT}): {e}")
            self._server = None

    def update(
        self,
        vehicle,
        time_s: float,
        flown_m: float | None = None,
        distance_m_cel: float | None = None,
        wind_snapshot: dict | None = None,
        current_phase: str | None = None,
    ) -> None:
        if self._server is None:
            return
        data = {
            "czas_symulacji_s": f"{time_s:.2f}",
            "czas_rzeczywisty_s": f"{time.time() - self._start_wall_s:.2f}",
        }
        if current_phase:
            data["aktywny_manewr"] = current_phase
        if flown_m is not None:
            data["przeleciono_m"] = f"{flown_m:.1f}"
            if distance_m_cel is not None:
                data["cel_m"] = f"{distance_m_cel:.0f}"
        else:
            data["przeleciono_m"] = "—"
            data["cel_m"] = "—"
        try:
            s = getattr(vehicle, "_state", None)
            if s is not None:
                p = getattr(s, "position", [0, 0, 0])
                for i, ax in enumerate("xyz"):
                    data[f"pos_{ax}"] = f"{p[i]:.4f}"
                v = getattr(s, "linear_velocity", [0, 0, 0])
                for i, ax in enumerate("xyz"):
                    data[f"vel_{ax}"] = f"{v[i]:.4f}"
                # Kwaternion jak w CSV: attitude[0..2]=qx,qy,qz, attitude[3]=qw (Body→NED, docs/scenarios.md)
                att = getattr(s, "attitude", None)
                if att is not None and len(att) >= 4:
                    qx, qy, qz, qw = (float(att[0]), float(att[1]), float(att[2]), float(att[3]))
                    # Te same wartości co qw,qx,qy,qz w CSV; krótsze etykiety w panelu: kwat_* = kwaternion (w,x,y,z), w = składowa „skalarna”.
                    data["kwat_w"] = f"{qw:.5f}"
                    data["kwat_x"] = f"{qx:.5f}"
                    data["kwat_y"] = f"{qy:.5f}"
                    data["kwat_z"] = f"{qz:.5f}"
                    try:
                        rot = Rotation.from_quat([qx, qy, qz, qw])
                        # Euler ZYX [°]: yaw (o Z NED / "heading"), pitch (o Y), roll (o X) — scipy, jednoznacznie z kwaternionu
                        yaw_deg, pitch_deg, roll_deg = rot.as_euler("ZYX", degrees=True)
                        data["roll_deg"] = f"{roll_deg:.2f}"
                        data["pitch_deg"] = f"{pitch_deg:.2f}"
                        data["yaw_deg"] = f"{yaw_deg:.2f}"
                    except Exception:
                        data["roll_deg"] = "—"
                        data["pitch_deg"] = "—"
                        data["yaw_deg"] = "—"
            try:
                if getattr(vehicle, "_sensors", None) and len(vehicle._sensors) > 0:
                    baro = vehicle._sensors[0].state
                    bp = baro.get("absolute_pressure", "")
                    bt = baro.get("temperature", "")
                    alt = baro.get("pressure_altitude", "")
                    data["baro_pressure_hPa"] = f"{bp:.2f}" if bp != "" and bp is not None else "—"
                    data["baro_temp_C"] = f"{bt:.2f}" if bt != "" and bt is not None else "—"
                    data["baro_alt_m"] = f"{alt:.2f}" if alt != "" and alt is not None else "—"
                if getattr(vehicle, "_sensors", None) and len(vehicle._sensors) > 1:
                    imu = vehicle._sensors[1].state
                    acc = imu.get("linear_acceleration", [0, 0, 0])
                    gyro = imu.get("angular_velocity", [0, 0, 0])
                    for i, ax in enumerate("xyz"):
                        data[f"acc_{ax}"] = f"{acc[i]:.4f}"
                    for i, ax in enumerate("xyz"):
                        data[f"gyro_{ax}"] = f"{gyro[i]:.4f}"
                if getattr(vehicle, "_sensors", None) and len(vehicle._sensors) > 3:
                    gps = vehicle._sensors[3].state
                    glat = gps.get("latitude", "")
                    glon = gps.get("longitude", "")
                    galt = gps.get("altitude", "")
                    data["gps_lat"] = f"{glat:.8f}" if glat != "" and glat is not None else "—"
                    data["gps_lon"] = f"{glon:.8f}" if glon != "" and glon is not None else "—"
                    data["gps_alt"] = f"{galt:.2f}" if galt != "" and galt is not None else "—"
            except Exception:
                pass
            try:
                if getattr(vehicle, "_thrusters", None):
                    ref = getattr(vehicle._thrusters, "_input_reference", [0, 0, 0, 0])
                    mins = getattr(vehicle._thrusters, "min_rotor_velocity", [0, 0, 0, 0])
                    maxs = getattr(vehicle._thrusters, "max_rotor_velocity", [1100] * 4)
                    t = 0.0
                    for i in range(4):
                        r = ref[i] if i < len(ref) else 0
                        mn = mins[i] if i < len(mins) else 0
                        mx = maxs[i] if i < len(maxs) else 1100
                        span = max(mx - mn, 1e-9)
                        n = max(0, min(1, (r - mn) / span))
                        data[f"motor_{i+1}"] = f"{n:.3f}"
                        t += n
                    data["throttle"] = f"{(t/4):.3f}"
            except Exception:
                pass
        except Exception:
            pass
        data["wiatr"] = "tak" if self._wind_enabled else "nie"
        # Bieżący wektor wiatru z PhysX (ten sam co w CSV); kierunek = meteo "z którego wieje"
        if self._wind_enabled and wind_snapshot is not None:
            try:
                wn = float(wind_snapshot.get("wind_vel_n", 0.0))
                we = float(wind_snapshot.get("wind_vel_e", 0.0))
                spd = math.hypot(wn, we)
                data["wiatr_bieżący_m/s"] = f"{spd:.3f}"
                data["wiatr_z_którego_°"] = f"{_meteo_wind_from_deg(wn, we):.2f}"
                ig = int(wind_snapshot.get("wind_is_gust", 0) or 0)
                data["wiatr_podmuch_aktywny"] = "tak" if ig else "nie"
            except Exception:
                data["wiatr_bieżący_m/s"] = "—"
                data["wiatr_z_którego_°"] = "—"
                data["wiatr_podmuch_aktywny"] = "—"
        else:
            data["wiatr_bieżący_m/s"] = "—"
            data["wiatr_z_którego_°"] = "—"
            data["wiatr_podmuch_aktywny"] = "—"
        self._state["data"] = {**self._params_display, **data}

    def close(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:
                pass
            self._server = None


# Opcjonalnie: MAVLink do arm + takeoff (port GCS 14550)
try:
    from mavlink_offboard import MavlinkOffboard
    HAS_MAVLINK_OFFBOARD = True
except Exception:
    HAS_MAVLINK_OFFBOARD = False

# Stałe faz wspólnych dla wszystkich scenariuszy (etykiety do klasyfikacji)
# Common phases shared by all scenarios:
# - warmup: PX4/EKF window while the vehicle is starting
# - climb: Takeoff wait before the actual scenario begins
#
# User request (earlier): warmup 3x krótszy i climb 4x krótszy.
WARMUP_S = 8.0
TAKEOFF_WAIT_S = 8.0
MISSION_START_OFFSET_S = WARMUP_S + TAKEOFF_WAIT_S  # start of the actual scenario


def _ensure_output_dir(output_dir: str | None) -> Path:
    if output_dir is None:
        # Domyślnie w katalogu głównym Isaac Sim (obok stachometr/)
        output_dir = SCRIPT_DIR.parent / "stachometr_output"
    p = Path(output_dir).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _build_full_phase_times(params: dict) -> list:
    """Buduje pełną listę phase_times: rozgrzewka, wznoszenie, potem fazy scenariusza (misja) z offsetem czasowym."""
    common = [
        {"phase": "rozgrzewka", "t_start_s": 0.0, "t_end_s": round(WARMUP_S, 2)},
        {"phase": "wznoszenie", "t_start_s": round(WARMUP_S, 2), "t_end_s": round(MISSION_START_OFFSET_S, 2)},
    ]
    scenario_phases = params.get("phase_times", [])
    mission_phases = [
        {
            "phase": p["phase"],
            "t_start_s": round(p["t_start_s"] + MISSION_START_OFFSET_S, 2),
            "t_end_s": round(p["t_end_s"] + MISSION_START_OFFSET_S, 2),
        }
        for p in scenario_phases
    ]
    return common + mission_phases


def _json_serializable(obj):
    """Konwersja numpy/niestandardowych typów do formatu JSON."""
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _save_params_json(params: dict, scenario_id: int, run_id: int, output_dir: Path, run_start_time: str) -> Path:
    """Zapisuje parametry runu do JSON z opisem scenariusza, parametrów losowych i etykiet faz (phase_times)."""
    desc = get_scenario_description(scenario_id)
    full_phase_times = _build_full_phase_times(params)
    params_copy = {k: v for k, v in params.items() if k != "phase_times"}
    out = {
        "run_start_time": run_start_time,
        "scenario_id": scenario_id,
        "scenario_description": desc,
        "run_id": run_id,
        "params_description": (
            "Wszystkie wartości poniżej (oprócz scenario_id/name) zostały wylosowane "
            "z zakresów podanych w docs/py/scenarios.md dla tego scenariusza."
        ),
        "phase_labels_description": PHASE_LABELS_DOC,
        "mission_start_offset_s": MISSION_START_OFFSET_S,
        "phase_times": full_phase_times,
        "phase_times_disclaimer": (
            "PRE-FLIGHT ESTIMATE ONLY: built from scenario formulas (distance/v_cmd, etc.), not measured during the run. "
            "Real transition times are in the companion file *_flight_timeline.json (English marks, t_sim_s = CSV time_s)."
        ),
        "phase_times_note": (
            "Czas 0–{:.0f} s: rozgrzewka (PX4/EKF). {:.0f}–{:.0f} s: wznoszenie. Od {:.0f} s: prawdziwa misja (fazy scenariusza)."
            .format(WARMUP_S, WARMUP_S, MISSION_START_OFFSET_S, MISSION_START_OFFSET_S)
        ),
        "log_columns_note": (
            "CSV: jeden wiersz na krok fizyki świata (physics_dt = World.get_physics_dt(), typowo 1/60 s, nie 1/250). "
            "time_s = krok × physics_dt — musi być zgodne z integracją PhysX. "
            "Opis grup kolumn poniżej (log_columns)."
        ),
        "log_columns": {
            "ground_truth_inertial": [
                "time_s (s)",
                "pos_x, pos_y, pos_z (m, NED)",
                "vel_x, vel_y, vel_z (m/s, NED)",
                "qw, qx, qy, qz (kwaternion orientacji Body→NED)",
            ],
            "imu": [
                "acc_x, acc_y, acc_z (Body FRD, m/s²)",
                "gyro_x, gyro_y, gyro_z (Body FRD, rad/s)",
            ],
            "barometer": [
                "baro_pressure (hPa)",
                "baro_temp (degC)",
                "baro_pressure_altitude (m, wysokość z baro)",
            ],
            "magnetometer": [
                "mag_x, mag_y, mag_z (Body FRD, pole magnetyczne – do yaw/orientacji)",
            ],
            "gps": [
                "gps_lat, gps_lon (deg), gps_alt (m)",
                "gps_velocity_north, gps_velocity_east, gps_velocity_down (m/s, NED)",
                "gps_speed (m/s)",
            ],
            "ground_truth_body": [
                "body_vel_x, body_vel_y, body_vel_z (u,v,w w układzie body FLU, m/s)",
                "body_rates_p, body_rates_q, body_rates_r (p,q,r w body, rad/s) – do porównań z IMU",
            ],
            "motors": [
                "motor_1..4, throttle (znormalizowany setpoint 0–1)",
                "motor_velocity_1..4 (rad/s, faktyczna prędkość wirnika)",
                "motor_force_1..4 (N, siła na rotor)",
            ],
            "setpoints": [
                "setpoint_vel_x, setpoint_vel_y, setpoint_vel_z (m/s, OFFBOARD)",
            ],
        },
        "acc_unit": "m/s2",
        "gyro_unit": "rad/s",
        "time_unit": "s",
        "position_unit": "m",
        "velocity_unit": "m/s",
        "baro_pressure_unit": "hPa",
        "baro_temp_unit": "degC",
        "baro_pressure_altitude_unit": "m",
        "mag_unit": "Gauss",
        "gps_lat_lon_unit": "deg",
        "gps_alt_unit": "m",
        "gps_velocity_unit": "m/s",
        "gps_speed_unit": "m/s",
        "body_velocity_unit": "m/s",
        "body_rates_unit": "rad/s",
        "motor_unit": "1",
        "motor_velocity_unit": "rad/s",
        "motor_force_unit": "N",
        **params_copy,
    }
    fname = f"run_s{scenario_id}_{run_start_time}_r{run_id}_params.json"
    fpath = output_dir / fname
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=_json_serializable)
    return fpath


def _log_state_csv_path(
    output_dir: Path, scenario_id: int, run_id: int, run_start_time: str, wind_suffix: str | None = None
) -> Path:
    base = f"run_s{scenario_id}_{run_start_time}_r{run_id}"
    if wind_suffix:
        return output_dir / f"{base}_{wind_suffix}_state.csv"
    return output_dir / f"{base}_state.csv"


class _FlightTimeline:
    """Rejestracja znaczników lotu.

    `t_sim_s` musi być spójne z tym co trafia do CSV (`time_s`), żeby porównania dystans/czas miały sens.
    Gdy podamy `time_getter`, używamy prawdziwego czasu symulacji z Isaaca (world.current_time), a nie
    prostej zależności `step*physics_dt` (która bywa niezgodna, jeśli jedno `world.step()` wykonuje więcej
    pod-kroków fizyki).
    """

    def __init__(self, physics_dt: float, time_getter=None):
        self.physics_dt = float(physics_dt)
        self._time_getter = time_getter
        self._t0_wall = time.perf_counter()
        self.marks: list[dict[str, Any]] = []
        self._once: set[str] = set()

    def t_sim(self, step: int) -> float:
        if self._time_getter is not None:
            return float(self._time_getter())
        return float(step) * self.physics_dt

    def mark(self, name: str, step: int, **extra: Any) -> None:
        entry: dict[str, Any] = {
            "mark": name,
            "t_sim_s": round(self.t_sim(step), 4),
            "wall_elapsed_s": round(time.perf_counter() - self._t0_wall, 4),
        }
        for k, v in extra.items():
            if v is None:
                continue
            if isinstance(v, float):
                entry[k] = round(v, 5)
            elif isinstance(v, (bool, str, int)):
                entry[k] = v
            else:
                entry[k] = v
        self.marks.append(entry)

    def mark_once(self, name: str, step: int, **extra: Any) -> None:
        if name in self._once:
            return
        self._once.add(name)
        self.mark(name, step, **extra)


def _save_flight_timeline_json(
    timeline: _FlightTimeline,
    csv_path: Path,
    scenario_id: int,
    run_id: int,
    run_start_time: str,
    meta: dict[str, Any],
) -> Path | None:
    """Zapis ``*_flight_timeline.json`` obok CSV (to samo ``run_s*`` + ``_flight_timeline``)."""
    if not timeline.marks:
        return None
    stem = csv_path.stem
    if stem.endswith("_state"):
        stem = stem[: -len("_state")]
    fname = f"{stem}_flight_timeline.json"
    fpath = csv_path.with_name(fname)
    out = {
        "schema": "stachometr_flight_timeline_v1",
        "run_start_time": run_start_time,
        "scenario_id": scenario_id,
        "run_id": run_id,
        "physics_dt_s": timeline.physics_dt,
        "time_base": "t_sim_s matches CSV column time_s (from world.current_time base)",
        "wall_clock_note": "wall_elapsed_s = seconds since timeline object creation (start of CSV logging loop)",
        "marks": timeline.marks,
        **meta,
    }
    try:
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=_json_serializable)
        carb.log_info(f"Stachometr: zapisano oś czasu lotu: {fpath}")
    except Exception as e:
        carb.log_warn(f"Stachometr: zapis flight_timeline: {e}")
        return None
    return fpath


def _pick_xform_scale_target(mesh_or_gprim: Usd.Prim) -> Usd.Prim:
    """Pierwszy przodek Xform pod /World, inaczej sam prim."""
    p = mesh_or_gprim.GetParent()
    while p and p.IsValid():
        ps = str(p.GetPath())
        if ps == "/World":
            break
        if p.IsA(UsdGeom.Xform):
            return p
        p = p.GetParent()
    return mesh_or_gprim


def _apply_uniform_xy_scale_on_prim(scale_prim: Usd.Prim, factor_xy: float) -> bool:
    """Mnoży istniejący Scale Xform o (factor_xy, factor_xy, 1) lub dodaje Scale op."""
    if factor_xy <= 1.001:
        return False
    xf = UsdGeom.Xformable(scale_prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            try:
                v = op.Get()
                op.Set(
                    Gf.Vec3d(float(v[0]) * factor_xy, float(v[1]) * factor_xy, float(v[2]))
                )
                return True
            except Exception:
                break
    xf.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(factor_xy, factor_xy, 1.0)
    )
    return True


def _scale_flat_ground_extent_xy(stage, extent_xy_m: float) -> None:
    """
    Powiększa **wszystkie** duże płaskie fragmenty podłoża (Flat Plane bywa złożony z kilku meshów
    ~500×500 m — wcześniej skalowaliśmy tylko jeden). Skaluje osobno każdy pasujący Gprim (jego Xform).
    """
    if extent_xy_m < 100.0:
        return
    try:
        world = stage.GetPrimAtPath("/World")
        if not world or not world.IsValid():
            return
        skip_sub = ("quadrotor", "iris", "camera", "light", "dome", "px4", "sun", "shadow")
        try:
            purposes = [
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy,
            ]
        except Exception:
            purposes = ["default", "render", "proxy"]
        for _tok in ("physics", "guide"):
            try:
                purposes.append(getattr(UsdGeom.Tokens, _tok))
            except Exception:
                if _tok == "physics":
                    purposes.append("physics")
                elif _tok == "guide":
                    purposes.append("guide")
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=purposes, useExtentsHint=True)
        candidates: list[tuple[Usd.Prim, float, float, float]] = []
        for prim in Usd.PrimRange(world):
            pstr = str(prim.GetPath()).lower()
            if any(s in pstr for s in skip_sub):
                continue
            if not prim.IsActive():
                continue
            if not prim.IsA(UsdGeom.Gprim):
                continue
            try:
                wb = cache.ComputeWorldBound(prim)
                rng = wb.ComputeAlignedRange()
            except Exception:
                continue
            mn, mx = rng.GetMin(), rng.GetMax()
            sx = float(mx[0] - mn[0])
            sy = float(mx[1] - mn[1])
            sz = float(mx[2] - mn[2])
            # Płaskie "płyty"; nieco luźniejsze progi (kolizja / kilka meshów)
            if sz > 45.0 or sx < 12.0 or sy < 12.0:
                continue
            if sx * sy < 2000.0:
                continue
            candidates.append((prim, sx, sy, sz))
        if not candidates:
            carb.log_warn(
                "Stachometr: nie znaleziono płaskiego podłoża pod /World — pomijam --ground-extent-m."
            )
            return
        scaled_xforms: set[str] = set()
        n_done = 0
        for prim, sx, sy, _sz in candidates:
            cur = max(sx, sy)
            if cur < 1.0:
                continue
            factor = float(extent_xy_m) / cur
            if factor <= 1.01:
                continue
            target = _pick_xform_scale_target(prim)
            tpath = str(target.GetPath())
            if tpath in scaled_xforms:
                continue
            if _apply_uniform_xy_scale_on_prim(target, factor):
                scaled_xforms.add(tpath)
                n_done += 1
                carb.log_info(
                    f"Stachometr: podłoże ×{factor:.3f} (≈{cur:.0f}→{extent_xy_m:.0f} m XY) "
                    f"`{tpath}` (z mesha `{prim.GetPath()}`)."
                )
        if n_done == 0:
            carb.log_info(
                f"Stachometr: wszystkie wykryte płyty już ≥ {extent_xy_m:.0f} m — bez skalowania."
            )
    except Exception as e:
        carb.log_warn(f"Stachometr: skalowanie podłoża (--ground-extent-m): {e}")


def _add_stachometr_ground_collision_cube(stage: Usd.Stage, extent_xy_m: float) -> None:
    """
    Duży sześcian z samą kolizją (niewidoczny): górna powierzchnia przy z≈0.
    Zabezpiecza lądowanie / fizykę, gdy skalowanie istniejącego mesha nic nie złapie lub bbox był wcześniej pusty.
    """
    if extent_xy_m < 100.0:
        return
    try:
        path = "/World/StachometrGroundCollision"
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            stage.RemovePrim(path)
        side = float(extent_xy_m)
        cz = -0.5 * side  # środek: wierzch sześcianu przy z=0 (Pegasus z↑)
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(side)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, cz))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        try:
            UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
        except Exception:
            pass
        carb.log_info(
            f"Stachometr: niewidzialna płyta kolizji ~{side:.0f}×{side:.0f} m (góra z≈0) `{path}` "
            "(fizyka nawet bez skalowania widocznego terenu)."
        )
    except Exception as e:
        carb.log_warn(f"Stachometr: nie udało się dodać płyty kolizji (--ground-extent-m): {e}")


def _apply_ground_extent_after_world_reset(ground_extent_xy_m: float) -> None:
    """Wywołaj po `world.reset()`: bbox podłoża jest wtedy wiarygodny; skalowanie + backup kolizji."""
    if ground_extent_xy_m < 100.0:
        return
    st = omni.usd.get_context().get_stage()
    if st is None:
        return
    _scale_flat_ground_extent_xy(st, ground_extent_xy_m)
    _add_stachometr_ground_collision_cube(st, ground_extent_xy_m)


def _add_bright_lighting() -> None:
    """Dodaje umiarkowane oświetlenie — scena czytelna (szarości), dron ciemny na jaśniejszym tle (kontrast)."""
    try:
        from pxr import Sdf, UsdLux

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        # DomeLight — rozproszone, bez przepalenia (niska intensywność i exposure)
        path = "/World/DomeLightStachometr"
        if stage.GetPrimAtPath(path):
            return  # już dodane
        dome = UsdLux.DomeLight.Define(stage, path)
        dome.CreateIntensityAttr(400.0)
        dome.CreateExposureAttr(0.5)
        if hasattr(dome, "CreateEnableColorTemperatureAttr"):
            dome.CreateEnableColorTemperatureAttr(True)
            dome.CreateColorTemperatureAttr(6500.0)
        # DistantLight — kierunkowe, umiarkowane
        path_dist = "/World/DistantLightStachometr"
        if not stage.GetPrimAtPath(path_dist):
            distant = UsdLux.DistantLight.Define(stage, path_dist)
            distant.CreateIntensityAttr(800.0)
            distant.CreateExposureAttr(0.0)
            if hasattr(distant, "CreateEnableColorTemperatureAttr"):
                distant.CreateEnableColorTemperatureAttr(True)
                distant.CreateColorTemperatureAttr(5500.0)
            rot = distant.AddRotateXYZOp()
            if rot:
                rot.Set((-35.0, 45.0, 0.0))
        carb.log_info("Dodano oświetlenie (DomeLight + DistantLight, umiarkowane — kontrast dron/tło)")
    except Exception as e:
        carb.log_warn(f"Oświetlenie: {e}")


_cam_eye: list = []         # smoothed chase-cam eye   [x, y, z] ENU
_cam_tgt: list = []         # smoothed chase-cam target [x, y, z] ENU
_att_cam_eye: list = []     # smoothed attitude-cam eye   [x, y, z] ENU
_att_cam_tgt: list = []     # smoothed attitude-cam target [x, y, z] ENU






_TRAIL_MAXLEN = 300          # ~5 s przy 60 Hz
_TRAIL_LINE_W = 3.0          # szerokość linii w pikselach
_TRAIL_COLOR  = (1.0, 0.5, 0.05)   # RGB pomarańczowy


def _init_trail_draw() -> None:
    """Inicjalizuje deque; debug_draw_iface jest już gotowy na poziomie modułu."""
    global _trail_deque
    if not _HAS_DEBUG_DRAW:
        return
    _trail_deque = collections.deque(maxlen=_TRAIL_MAXLEN)


def _update_trail(vehicle) -> None:
    """Dodaje pozycję za dronem do deque i przerysowuje smugę przez debug_draw (zero fizyki)."""
    if not _HAS_DEBUG_DRAW or _trail_deque is None or _debug_draw_iface is None:
        return
    if vehicle is None or not getattr(vehicle, "_state", None):
        return
    try:
        p = vehicle.state.position
        vel = vehicle.state.linear_velocity
        px, py, pz = float(p[0]), float(p[1]), float(p[2])
        vx, vy = float(vel[0]), float(vel[1])
        spd_h = math.sqrt(vx * vx + vy * vy)
        if spd_h > 0.3:
            ox, oy = -vx / spd_h * 0.4, -vy / spd_h * 0.4
        else:
            ox, oy = 0.0, 0.0
        _trail_deque.append((px + ox, py + oy, pz))

        n = len(_trail_deque)
        if n < 2:
            return

        pts = list(_trail_deque)
        starts = pts[:-1]
        ends   = pts[1:]
        n_seg  = len(starts)
        r, g, b = _TRAIL_COLOR
        colors = [(r, g, b, i / (n_seg - 1)) for i in range(n_seg)]
        widths = [_TRAIL_LINE_W] * n_seg

        _debug_draw_iface.clear_lines()
        _debug_draw_iface.draw_lines(starts, ends, colors, widths)
    except Exception:
        pass


def _clear_trail() -> None:
    if _HAS_DEBUG_DRAW and _debug_draw_iface is not None:
        try:
            _debug_draw_iface.clear_lines()
        except Exception:
            pass
    if _trail_deque is not None:
        _trail_deque.clear()


def _update_follow_camera(vehicle, heading_rad: float, phase_label: str) -> None:
    """Przesuwa kamerę viewport za dronem co krok symulacji (60 Hz).

    Dwie stałe czasowe:
    - alpha_tgt = 0.40: target (= pozycja drona) goni bardzo szybko → dron zawsze w centrum kadru
    - alpha_eye = 0.10: eye (pozycja kamery) goni wolniej → płynny ruch bez teleportów

    Tryby kamer (przełączane przez phase_label):
    - slalom_*        → kamera boczna  (widać wahnięcia L/P)
    - zawrot / zakret_gwaltowny → kamera z góry  (widać łuk U-turnu)
    - hamowanie / ladowanie_px4 → kamera z przodu (dron jedzie w obiektyw)
    - pozostałe       → chase cam 9 m z tyłu, 5 m z góry
    """
    global _cam_eye, _cam_tgt
    if not _HAS_FOLLOW_CAM or vehicle is None or not getattr(vehicle, "_state", None):
        return
    try:
        p = vehicle.state.position          # ENU: x=North, y=East, z=Up
        px, py, pz = float(p[0]), float(p[1]), float(p[2])
        fwd_x = math.cos(heading_rad)
        fwd_y = math.sin(heading_rad)
        rgt_x =  math.sin(heading_rad)
        rgt_y = -math.cos(heading_rad)

        # target zawsze = dron (lekki offset w górę → dron na dolnej 1/3 kadru)
        tgt_d = [px, py, pz + 0.8]

        _is_topdown = phase_label in (
            "slalom_lagodny", "slalom_sredni", "slalom_ostry",
            "zakret_lagodny", "zakret_sredni", "zakret_ostry",
            "zakret_gwaltowny", "zawrot",
        )
        if _is_topdown:
            # Top-down: prosto z góry, bez heading/rotacji, wolne śledzenie → dron wylatuje z kadru
            eye_d = [px, py, pz + 50.0]
            tgt_d = [px, py, pz]
            alpha_eye = 0.04
            alpha_tgt = 0.04
        elif phase_label in ("hamowanie", "ladowanie_px4"):
            # Kamera z przodu — dron leci w stronę kamery
            eye_d = [px + 10.0 * fwd_x, py + 10.0 * fwd_y, pz + 5.0]
            tgt_d = [px, py, pz + 0.8]
            alpha_eye = 0.10
            alpha_tgt = 0.40
        else:
            # Chase cam: 9 m z tyłu, 5 m z góry
            eye_d = [px - 9.0 * fwd_x, py - 9.0 * fwd_y, pz + 5.0]
            tgt_d = [px, py, pz + 0.8]
            alpha_eye = 0.10
            alpha_tgt = 0.40

        if not _cam_eye:
            _cam_eye[:] = list(eye_d)
            _cam_tgt[:] = list(tgt_d)
        else:
            for i in range(3):
                _cam_eye[i] += alpha_eye * (eye_d[i] - _cam_eye[i])
                _cam_tgt[i] += alpha_tgt * (tgt_d[i] - _cam_tgt[i])

        _set_camera_view_fn(
            eye=_np_cam.array(_cam_eye),
            target=_np_cam.array(_cam_tgt),
        )
    except Exception:
        pass


def _init_attitude_cam(stage) -> None:
    """Tworzy Camera prim /World/attitude_cam i otwiera Viewport 2 z widokiem od południa.
    Wywołaj raz po world.reset(), gdy stage jest gotowy i nie jesteśmy w trybie headless.
    """
    global _attitude_cam_vp
    if not _HAS_FOLLOW_CAM:
        return
    try:
        import omni.usd
        from omni.kit.viewport.utility import create_viewport_window
        from pxr import Sdf

        st = stage if stage is not None else omni.usd.get_context().get_stage()

        # Utwórz prim kamery (jeśli nie istnieje)
        cam_path = "/World/attitude_cam"
        if not st.GetPrimAtPath(cam_path).IsValid():
            st.DefinePrim(cam_path, "Camera")

        # Otwórz Viewport 2 z tą kamerą
        _attitude_cam_vp = create_viewport_window(
            name="Attitude Cam",
            width=640,
            height=480,
            camera_path=Sdf.Path(cam_path),
        )
        carb.log_info("Stachometr: Attitude Cam (Viewport 2) gotowy — /World/attitude_cam")
    except Exception as e:
        carb.log_warn(f"Stachometr: attitude_cam init: {e}")


def _update_attitude_cam(vehicle) -> None:
    """Kamera 3.5 m przed nosem drona, zawsze patrzy na jego środek — oko w oko."""
    if not _HAS_FOLLOW_CAM or _attitude_cam_vp is None:
        return
    if vehicle is None or not getattr(vehicle, "_state", None):
        return
    try:
        p = vehicle.state.position       # ENU: x=North, y=East, z=Up
        px, py, pz = float(p[0]), float(p[1]), float(p[2])

        # Yaw z kwaternionu orientacji [w, x, y, z] w ENU
        q = vehicle.state.attitude       # [w, x, y, z]
        qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                         1.0 - 2.0 * (qy * qy + qz * qz))

        # Forward drona w ENU (dziób wskazuje kierunek)
        fwd_x = math.cos(yaw)
        fwd_y = math.sin(yaw)

        # Kamera 3.5 m przed dziobem, ta sama wysokość → czyste "oko w oko"
        dist = 3.5
        _set_camera_view_fn(
            eye=_np_cam.array([px + fwd_x * dist, py + fwd_y * dist, pz]),
            target=_np_cam.array([px, py, pz]),
            camera_prim_path="/World/attitude_cam",
            viewport_api=_attitude_cam_vp.viewport_api,
        )
    except Exception:
        pass


def _vz_ned_from_altitude_zup(
    pz_up: float,
    vz_up: float,
    z_target_up: float,
    kp: float,
    kd: float,
    cap: float,
) -> float:
    """
    Pegasus/Isaac: position[2] i linear_velocity[2] to oś **w górę** (z rośnie z wysokością).
    MAVLink OFFBOARD: **vz w NED** — dodatnie = w dół. v_down = −dh/dt przy h = z_up.
    """
    dh_dt_cmd = kp * (z_target_up - pz_up) - kd * vz_up
    vz_ned = -dh_dt_cmd
    return max(-cap, min(cap, vz_ned))


def run_single(
    scenario_id: int,
    run_id: int,
    params: dict,
    output_dir: Path,
    duration_s: float,
    simulation_app,
    run_start_time: str,
    wind_suffix: str | None = None,
    show_live_display: bool = False,
    random_seed: int | None = None,
    ground_extent_xy_m: float = 0.0,
) -> None:
    """Wykonuje jeden run: świat + IRIS + PX4, logowanie stanu do CSV. wind_suffix: np. 'nowind'/'wind' dla scenariusza 1."""
    run_params = merge_wind_defaults(dict(params))
    timeline = omni.timeline.get_timeline_interface()
    pg = PegasusInterface()

    # Reset singletona świata jeśli wcześniej był używany (np. poprzedni run)
    if pg._world is not None:
        try:
            pg.clear_scene()
        except Exception as e:
            carb.log_warn(f"clear_scene: {e}")

    pg._world = World(**pg._world_settings)
    world = pg.world

    # Nieskończona płaska powierzchnia (terrain), żeby dron mógł lecieć 200 m+ bez ścian.
    pg.load_environment(SIMULATION_ENVIRONMENTS["Flat Plane"])
    # Skalowanie podłoża — dopiero po world.reset() (patrz niżej); tu bbox bywa jeszcze pusty.
    _add_bright_lighting()

    # Konfiguracja IRIS + PX4 (jak w 1_px4_single_vehicle.py)
    config_multirotor = MultirotorConfig()
    mavlink_config = PX4MavlinkBackendConfig({
        "vehicle_id": 0,
        "px4_autolaunch": True,
        "px4_dir": pg.px4_path,
        "px4_vehicle_model": pg.px4_default_airframe,
    })
    config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]

    # Start na ziemi (jak w przykładzie Pegasus). Wysokość docelowa z parametrów — osiągniemy przez ARM + TAKEOFF.
    alt_target = float(params.get("altitude_m", params.get("altitude_start_m", 20.0)))
    init_z = 0.07  # nad podłożem, żeby nie kolidować z ziemią
    # Punkt "startu" na ziemi (XY) — musi być zgodny z pierwszym argumentem pozycji Multirotor poniżej.
    # Scenariusz 1: koniec misji, gdy odległość pozioma środka drona od tego punktu >= distance_m (dowolny kierunek).
    spawn_xy = (0.0, 0.0)

    Multirotor(
        "/World/quadrotor",
        ROBOTS["Iris"],
        0,
        [spawn_xy[0], spawn_xy[1], init_z],
        Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
        config=config_multirotor,
    )

    world.reset()

    _apply_ground_extent_after_world_reset(ground_extent_xy_m)
    _stage_ref = omni.usd.get_context().get_stage()
    _init_attitude_cam(_stage_ref)
    _init_trail_draw()
    _clear_trail()

    csv_path = _log_state_csv_path(output_dir, scenario_id, run_id, run_start_time, wind_suffix=wind_suffix)

    live_display = _LiveDisplay(wind_suffix, run_params) if show_live_display else None

    # Plik CSV: nagłówek — GT, IMU, baro (+ pressure_altitude), mag, GPS, GT body (u,v,w + p,q,r), silniki (norm + velocity + force), setpointy
    csv_header = [
        "time_s", "pos_x", "pos_y", "pos_z",
        "vel_x", "vel_y", "vel_z",
        "qw", "qx", "qy", "qz",
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z",
        "baro_pressure", "baro_temp", "baro_pressure_altitude",
        "mag_x", "mag_y", "mag_z",
        "gps_lat", "gps_lon", "gps_alt",
        "gps_velocity_north", "gps_velocity_east", "gps_velocity_down", "gps_speed",
        "body_vel_x", "body_vel_y", "body_vel_z",
        "body_rates_p", "body_rates_q", "body_rates_r",
        "motor_1", "motor_2", "motor_3", "motor_4", "throttle",
        "motor_velocity_1", "motor_velocity_2", "motor_velocity_3", "motor_velocity_4",
        "motor_force_1", "motor_force_2", "motor_force_3", "motor_force_4",
        "wind_vel_n",
        "wind_vel_e",
        "wind_vel_d",
        "wind_force_n",
        "wind_force_e",
        "wind_force_d",
        "v_air_n",
        "v_air_e",
        "v_air_d",
        "wind_is_gust",
        "wind_gust_phase",
        "wind_gust_rising",
        "wind_gust_from_schedule",
        "wind_gust_is_lull",
        "setpoint_vel_x", "setpoint_vel_y", "setpoint_vel_z",
        "seg_phase",
    ]
    log_file = open(csv_path, "w", encoding="utf-8", newline="")
    writer = csv.writer(log_file)
    writer.writerow(csv_header)

    # Mutable: ostatni setpoint prędkości (OFFBOARD) — ustawiane w pętli misji, odczytywane w log_state
    last_setpoint = [0.0, 0.0, 0.0]
    # Mutable: etykieta aktywnej fazy/manewru (scenariusz 3) — ustawiane w pętli misji, odczytywane w log_state
    current_seg_label: list[str] = [""]
    # Mutable: stan podmuchu z poprzedniego kroku — do detekcji startu/końca gustu w flight_timeline
    _gust_active_prev: list[bool] = [False]
    # Kontekst misji: start_xy = spawn; distance_m = promień / zadany dystans; scenariusz 2: target_xy = punkt lądowania w XY
    mission_ctx: dict = {"start_xy": None, "distance_m": None, "target_xy": None}
    if scenario_id == 1:
        mission_ctx["start_xy"] = spawn_xy
        mission_ctx["distance_m"] = float(params.get("distance_m", 200.0))
    elif scenario_id == 2:
        mission_ctx["start_xy"] = spawn_xy
        mission_ctx["distance_m"] = float(params.get("distance_m", 200.0))
        dm = mission_ctx["distance_m"]
        yd = math.radians(float(params.get("yaw_deg", 0.0)))
        mission_ctx["target_xy"] = (
            spawn_xy[0] + dm * math.cos(yd),
            spawn_xy[1] + dm * math.sin(yd),
        )

    timeline.play()

    vehicle = pg.get_vehicle("/World/quadrotor")
    # Pegasus: świeży GT (pozycja, CSV, panel) jest na tym obiekcie. Kolejne pg.get_vehicle() zwraca czasem
    # uchwyt bez aktualnego state — wtedy dystans misji wychodził ~0 i cel 200 m nie kończył runu przy ~900 m na panelu.
    spawn_goal_reached = False
    # physics_dt  — wewnętrzny krok fizyki PhysX (1/250 s dla PX4); używany w windgen i MAVLink.
    # step_dt     — czas symulacji na jedno wywołanie world.step() = rendering_dt (1/60 s).
    #               Każde _do_step() przesuwa world.current_time o step_dt, NIE o physics_dt.
    #               Pętle segmentów, mission_time_s, step-counting muszą używać step_dt.
    physics_dt = 1.0 / 250.0
    try:
        physics_dt = float(world.get_physics_dt())
    except Exception as e:
        carb.log_warn(f"Stachometr: world.get_physics_dt() — {e}; fallback physics_dt={physics_dt}")
    if physics_dt <= 0.0:
        physics_dt = 1.0 / 250.0
        carb.log_warn(f"Stachometr: nieprawidłowy physics_dt, używam {physics_dt}")

    step_dt = 1.0 / 60.0
    try:
        step_dt = float(world.get_rendering_dt())
    except Exception as e:
        carb.log_warn(f"Stachometr: world.get_rendering_dt() — {e}; fallback step_dt={step_dt}")
    if step_dt <= 0.0:
        step_dt = 1.0 / 60.0

    carb.log_info(
        f"Stachometr: physics_dt={physics_dt:.6f} s ({1.0/physics_dt:.0f} Hz), "
        f"step_dt={step_dt:.6f} s ({1.0/step_dt:.0f} Hz) — "
        f"step_dt/physics_dt={step_dt/physics_dt:.2f} sub-steps/frame"
    )
    _steps_per_sim_second = max(1, int(round(1.0 / step_dt)))

    csv_sim_time_base_s = float(getattr(world, "current_time", 0.0) or 0.0)
    flight_timeline = _FlightTimeline(physics_dt, time_getter=lambda: float(world.current_time) - csv_sim_time_base_s)
    flight_timeline.mark("simulation_csv_logging_started", 0)

    # Wiatr: generator prędkości (stały / dynamiczny) + model siły legacy albo drag; apply_force_at_pos w NED.
    _wind_stage_id = None
    _wind_body_id = None
    _wind_gen = None
    _wind_apply_pos_warned = False
    last_wind_log = {
        "wind_vel_n": 0.0,
        "wind_vel_e": 0.0,
        "wind_vel_d": 0.0,
        "wind_force_n": 0.0,
        "wind_force_e": 0.0,
        "wind_force_d": 0.0,
        "v_air_n": 0.0,
        "v_air_e": 0.0,
        "v_air_d": 0.0,
        "wind_is_gust": 0,
        "wind_gust_phase": "",
        "wind_gust_rising": 0,
        "wind_gust_from_schedule": 0,
        "wind_gust_is_lull": 0,
    }
    if wind_suffix == "wind":
        _wind_gen = make_wind_generator(run_params, random_seed, run_id)
        if _wind_gen is not None:
            try:
                stage = omni.usd.get_context().get_stage()
                if stage is not None:
                    _wind_stage_id = omni.usd.get_context().get_stage_id()
                    for drone_prim_path in ("/World/quadrotor/body", "/World/quadrotor"):
                        prim = stage.GetPrimAtPath(drone_prim_path)
                        if prim and prim.IsValid():
                            _wind_body_id = PhysicsSchemaTools.sdfPathToInt(prim.GetPath())
                            wm = str(run_params.get("wind_force_model", "legacy"))
                            dyn = bool(run_params.get("wind_dynamic_enabled", True))
                            carb.log_info(
                                f"Stachometr: wiatr — model={wm}, dynamic={dyn}, prim={drone_prim_path}; "
                                "siła co krok z generatora + apply_force_at_pos (pozycja drona)."
                            )
                            break
                    else:
                        carb.log_warn("Stachometr: wiatr: nie znaleziono prima /World/quadrotor/body ani /World/quadrotor.")
            except Exception as e:
                carb.log_warn(f"Stachometr: wiatr PhysX init: {e}")

    def _apply_wind_force():
        """Siła wiatru w PhysX; **zawsze** aktualizuje last_wind_log z krokiem generatora (OU+podmuchy), nawet gdy brak prima — wtedy tylko brak apply_force."""
        nonlocal _wind_apply_pos_warned, last_wind_log
        if _wind_gen is None:
            last_wind_log = {
                "wind_vel_n": 0.0,
                "wind_vel_e": 0.0,
                "wind_vel_d": 0.0,
                "wind_force_n": 0.0,
                "wind_force_e": 0.0,
                "wind_force_d": 0.0,
                "v_air_n": 0.0,
                "v_air_e": 0.0,
                "v_air_d": 0.0,
                "wind_is_gust": 0,
                "wind_gust_phase": "",
                "wind_gust_rising": 0,
                "wind_gust_from_schedule": 0,
                "wind_gust_is_lull": 0,
            }
            return
        try:
            apply_at = None
            v_src = vehicle
            if v_src is None or not getattr(v_src, "_state", None):
                v_src = pg.get_vehicle("/World/quadrotor")
            v_ned = (0.0, 0.0, 0.0)
            if v_src is not None and getattr(v_src, "_state", None) is not None:
                st = v_src.state
                p = st.position
                lv = getattr(st, "linear_velocity", None)
                if p is not None and len(p) >= 3:
                    apply_at = Gf.Vec3f(float(p[0]), float(p[1]), float(p[2]))
                if lv is not None and len(lv) >= 3:
                    v_ned = (float(lv[0]), float(lv[1]), float(lv[2]))
            t_sim = step * physics_dt
            w_ned, gust_ph = _wind_gen.step(physics_dt, t_sim)
            ex = getattr(_wind_gen, "step_extras", {})
            wn, we, wd = w_ned
            van = v_ned[0] - wn
            vae = v_ned[1] - we
            vad = v_ned[2] - wd
            model = str(run_params.get("wind_force_model", "legacy")).lower()
            if model == "drag":
                rho = float(run_params.get("wind_rho_kg_m3", 1.225))
                cd_a = float(run_params.get("wind_cd_times_area_m2", 1.5))
                fn, fe, fd = _wind_force_drag_ned(v_ned, w_ned, rho, cd_a)
            else:
                fn, fe, fd = _wind_force_legacy_ned(w_ned, K_WIND_N_PER_MS)
            gust_on = bool(gust_ph and str(gust_ph).strip())
            last_wind_log = {
                "wind_vel_n": wn,
                "wind_vel_e": we,
                "wind_vel_d": wd,
                "wind_force_n": fn,
                "wind_force_e": fe,
                "wind_force_d": fd,
                "v_air_n": van,
                "v_air_e": vae,
                "v_air_d": vad,
                "wind_is_gust": 1 if gust_on else 0,
                "wind_gust_phase": gust_ph if gust_ph else "",
                "wind_gust_rising": int(ex.get("wind_gust_rising", 0) or 0),
                "wind_gust_from_schedule": int(ex.get("wind_gust_from_schedule", 0) or 0),
                "wind_gust_is_lull": int(ex.get("wind_gust_is_lull", 0) or 0),
            }
            if _wind_body_id is None or _wind_stage_id is None:
                if not _wind_apply_pos_warned:
                    carb.log_warn(
                        "Stachometr: wiatr — brak prima PhysX; generator nadal krokowany (CSV/panel), bez apply_force."
                    )
                    _wind_apply_pos_warned = True
                return
            if apply_at is None:
                if not _wind_apply_pos_warned:
                    carb.log_warn(
                        "Stachometr: wiatr — brak vehicle.state.position; pomijam apply_force (bez (0,0,0))."
                    )
                    _wind_apply_pos_warned = True
                return
            psi = omni.physx.get_physx_simulation_interface()
            psi.apply_force_at_pos(
                _wind_stage_id,
                _wind_body_id,
                Gf.Vec3f(float(fn), float(fe), float(fd)),
                apply_at,
                "Force",
            )
        except Exception as e:
            if not _wind_apply_pos_warned:
                carb.log_warn(f"Stachometr: wiatr apply_force: {e}")
                _wind_apply_pos_warned = True

    def _do_step():
        _apply_wind_force()
        world.step(render=not _HEADLESS)

    def _maybe_mark_spawn_goal_done() -> None:
        """Scenariusz 1: jeśli promień poziomy od spawn_xy ≥ distance_m — kończymy cały run (także w rozgrzewce / wznoszeniu)."""
        nonlocal spawn_goal_reached
        if spawn_goal_reached or scenario_id != 1:
            return
        dm = mission_ctx.get("distance_m")
        if dm is None or vehicle is None or not getattr(vehicle, "_state", None):
            return
        p = vehicle.state.position
        d = math.hypot(float(p[0]) - spawn_xy[0], float(p[1]) - spawn_xy[1])
        if d >= dm:
            spawn_goal_reached = True
            carb.log_info(
                f"Scenariusz 1: osiągnięto cel dystansu od spawn ({d:.1f} m ≥ {dm:.0f} m); kończę run."
            )

    def _get_extra_columns(v):
        """Pobiera wszystkie dodatkowe kolumny CSV w kolejności nagłówka: IMU, baro (+ pressure_altitude), mag, GPS, body vel/rates, silniki (norm + velocity + force)."""
        out = []
        # IMU (sensors[1]: acc body FRD m/s², gyro rad/s)
        try:
            if getattr(v, "_sensors", None) and len(v._sensors) > 1:
                imu = v._sensors[1].state
                acc = imu.get("linear_acceleration", [0, 0, 0])
                gyro = imu.get("angular_velocity", [0, 0, 0])
                out.extend([acc[0], acc[1], acc[2], gyro[0], gyro[1], gyro[2]])
            else:
                out.extend([""] * 6)
        except Exception:
            out.extend([""] * 6)
        # Barometer (sensors[0]: pressure hPa, temp °C, pressure_altitude m)
        try:
            if getattr(v, "_sensors", None) and len(v._sensors) > 0:
                baro = v._sensors[0].state
                out.append(baro.get("absolute_pressure", ""))
                out.append(baro.get("temperature", ""))
                out.append(baro.get("pressure_altitude", ""))
            else:
                out.extend(["", "", ""])
        except Exception:
            out.extend(["", "", ""])
        # Magnetometr (sensors[2]: magnetic_field body FRD, np. µT / Gauss)
        try:
            if getattr(v, "_sensors", None) and len(v._sensors) > 2:
                mag = v._sensors[2].state
                m = mag.get("magnetic_field", [0, 0, 0])
                out.extend([m[0] if len(m) > 0 else "", m[1] if len(m) > 1 else "", m[2] if len(m) > 2 else ""])
            else:
                out.extend([""] * 3)
        except Exception:
            out.extend([""] * 3)
        # GPS (sensors[3]: lat, lon, alt, velocity NED, speed — bez cog, w Pegasusie zawsze 0)
        try:
            if getattr(v, "_sensors", None) and len(v._sensors) > 3:
                gps = v._sensors[3].state
                out.append(gps.get("latitude", ""))
                out.append(gps.get("longitude", ""))
                out.append(gps.get("altitude", ""))
                out.append(gps.get("velocity_north", ""))
                out.append(gps.get("velocity_east", ""))
                out.append(gps.get("velocity_down", ""))
                out.append(gps.get("speed", ""))
            else:
                out.extend([""] * 7)
        except Exception:
            out.extend([""] * 7)
        # GT body: linear_body_velocity (u,v,w), angular_velocity (p,q,r) — state w układzie body (FLU)
        try:
            s = getattr(v, "_state", None)
            if s is not None:
                lb = getattr(s, "linear_body_velocity", [0, 0, 0])
                av = getattr(s, "angular_velocity", [0, 0, 0])
                out.extend([lb[0], lb[1], lb[2], av[0], av[1], av[2]])
            else:
                out.extend([""] * 6)
        except Exception:
            out.extend([""] * 6)
        # Silniki: znormalizowany setpoint [0,1] + throttle
        try:
            if getattr(v, "_thrusters", None):
                ref = getattr(v._thrusters, "_input_reference", [0, 0, 0, 0])
                mins = getattr(v._thrusters, "min_rotor_velocity", [0, 0, 0, 0])
                maxs = getattr(v._thrusters, "max_rotor_velocity", [1100, 1100, 1100, 1100])
                motors = []
                for i in range(4):
                    r = ref[i] if i < len(ref) else 0
                    mn = mins[i] if i < len(mins) else 0
                    mx = maxs[i] if i < len(maxs) else 1100
                    span = max(mx - mn, 1e-9)
                    motors.append(round(max(0.0, min(1.0, (r - mn) / span)), 6))
                thr = sum(motors) / 4.0 if motors else ""
                out.extend(motors + [thr])
            else:
                out.extend([""] * 5)
        except Exception:
            out.extend([""] * 5)
        # Thrusters: faktyczna prędkość wirnika (rad/s) i siła (N)
        try:
            if getattr(v, "_thrusters", None):
                vel = getattr(v._thrusters, "velocity", getattr(v._thrusters, "_velocity", [0, 0, 0, 0]))
                force = getattr(v._thrusters, "force", getattr(v._thrusters, "_force", [0, 0, 0, 0]))
                for i in range(4):
                    out.append(vel[i] if i < len(vel) else "")
                for i in range(4):
                    out.append(force[i] if i < len(force) else "")
            else:
                out.extend([""] * 8)
        except Exception:
            out.extend([""] * 8)
        return out

    def log_state(step_count: int) -> None:
        t = float(world.current_time) - csv_sim_time_base_s
        flown_m = None
        if live_display and mission_ctx["start_xy"] is not None and vehicle is not None and getattr(vehicle, "state", None) is not None:
            p = vehicle.state.position
            dx = float(p[0]) - mission_ctx["start_xy"][0]
            dy = float(p[1]) - mission_ctx["start_xy"][1]
            flown_m = math.sqrt(dx * dx + dy * dy)
        if vehicle is not None and vehicle._state is not None:
            s = vehicle.state
            extra = _get_extra_columns(vehicle)
            row = [
                f"{t:.4f}",
                s.position[0], s.position[1], s.position[2],
                s.linear_velocity[0], s.linear_velocity[1], s.linear_velocity[2],
                s.attitude[3], s.attitude[0], s.attitude[1], s.attitude[2],
            ]
            row.extend(extra)
            row.extend(
                [
                    last_wind_log["wind_vel_n"],
                    last_wind_log["wind_vel_e"],
                    last_wind_log["wind_vel_d"],
                    last_wind_log["wind_force_n"],
                    last_wind_log["wind_force_e"],
                    last_wind_log["wind_force_d"],
                    last_wind_log["v_air_n"],
                    last_wind_log["v_air_e"],
                    last_wind_log["v_air_d"],
                    last_wind_log["wind_is_gust"],
                    last_wind_log["wind_gust_phase"],
                    last_wind_log["wind_gust_rising"],
                    last_wind_log["wind_gust_from_schedule"],
                    last_wind_log["wind_gust_is_lull"],
                ]
            )
            row.extend([last_setpoint[0], last_setpoint[1], last_setpoint[2]])
            row.append(current_seg_label[0])
            writer.writerow(row)
            # Attitude cam + trail update (każdy krok, wszystkie scenariusze)
            _update_attitude_cam(vehicle)
            _update_trail(vehicle)
            # Gust transition detection → flight_timeline event
            gust_now = bool(last_wind_log.get("wind_is_gust", 0))
            if gust_now and not _gust_active_prev[0]:
                flight_timeline.mark(
                    "gust_start", step_count,
                    wind_speed_n_ms=round(float(last_wind_log["wind_vel_n"]), 3),
                    wind_speed_e_ms=round(float(last_wind_log["wind_vel_e"]), 3),
                    gust_from_schedule=bool(last_wind_log.get("wind_gust_from_schedule", 0)),
                    gust_is_lull=bool(last_wind_log.get("wind_gust_is_lull", 0)),
                    seg_phase=current_seg_label[0],
                )
            elif not gust_now and _gust_active_prev[0]:
                flight_timeline.mark(
                    "gust_end", step_count,
                    wind_speed_n_ms=round(float(last_wind_log["wind_vel_n"]), 3),
                    wind_speed_e_ms=round(float(last_wind_log["wind_vel_e"]), 3),
                    seg_phase=current_seg_label[0],
                )
            _gust_active_prev[0] = gust_now
            if live_display:
                live_display.update(
                    vehicle,
                    t,
                    flown_m=flown_m,
                    distance_m_cel=mission_ctx.get("distance_m"),
                    wind_snapshot=last_wind_log if wind_suffix == "wind" else None,
                    current_phase=current_seg_label[0] if current_seg_label[0] else None,
                )

    step = 0

    # 1) Rozgrzewka (etykieta: rozgrzewka) — logowanie od t=0; dajemy PX4 czas na start
    warmup_steps = int(WARMUP_S / step_dt)
    for _ in range(warmup_steps):
        if not simulation_app.is_running():
            break
        _do_step()
        log_state(step)
        _maybe_mark_spawn_goal_done()
        step += 1
        if spawn_goal_reached:
            break
    flight_timeline.mark_once("warmup_complete_px4_ekf_window", step)

    # 2) Tryb Takeoff + ARM (port 14580 Onboard PX4)
    # PX4: najpierw SET_MODE Takeoff (AUTO_TAKEOFF), potem ARM — wtedy po uzbrojeniu dron sam wznosi (do MIS_TAKEOFF_ALT).
    # MAV_CMD_NAV_TAKEOFF przy samym ARM powodowało "Disarmed by auto preflight disarming".
    fallback_ok = False
    if not spawn_goal_reached and HAS_MAVLINK_OFFBOARD:
        try:
            mav_out = MavlinkOffboard("udpout:127.0.0.1:14580")
            if mav_out.bind():
                mav_out.force_px4_sitl_target()
                if not mav_out.try_recv_heartbeat():
                    for _ in range(20):
                        if mav_out.try_recv_heartbeat():
                            break
                mav_out.force_px4_sitl_target()

                # Kluczowe: ustaw limity prędkości PX4 PRZED ARM/takeoff.
                # Wiele parametrów bywa odrzucanych/ignorowanych po uzbrojeniu.
                if scenario_id in (1, 2, 3):
                    if scenario_id == 3:
                        _seg_v_max = max(
                            (float(s.get("v_end_ms", 0)) for s in params.get("segments", [])),
                            default=20.0,
                        )
                        _prearm_v_cmd = max(float(params.get("v_initial_ms", 20.0)), _seg_v_max)
                    else:
                        _prearm_v_cmd = float(params.get("v_cmd_ms", 10.0) or 10.0)
                    _prearm_px4_lim = _px4_velocity_mission_limits(params, _prearm_v_cmd)
                    # Preferuj dwukierunkowy link parametryczny:
                    # - udpin:14550 (GCS) najczęściej dostaje broadcast PARAM_VALUE
                    # - potem udpin:14540
                    # - udpout:14580 zostaw jako ostatni fallback.
                    _prearm_rb: dict[str, float] = {}
                    _prearm_keys = ["MPC_XY_VEL_MAX", "MPC_XY_CRUISE", "MPC_VEL_MANUAL"]
                    _prearm_param_link_used = "udpout:14580"
                    for _conn_label, _conn_str in [
                        ("udpin:14550", "udpin:0.0.0.0:14550"),
                        ("udpin:14540", "udpin:0.0.0.0:14540"),
                    ]:
                        if len(_prearm_rb) == len(_prearm_keys):
                            break
                        _param_link = None
                        try:
                            _param_link = MavlinkOffboard(_conn_str)
                            if _param_link.bind():
                                _param_link.force_px4_sitl_target()
                                _prearm_param_link_used = _conn_label
                                if not _param_link.try_recv_heartbeat():
                                    for _ in range(50):
                                        if _param_link.try_recv_heartbeat():
                                            break
                                _param_link.force_px4_sitl_target()
                                _param_link.set_px4_parameters(_prearm_px4_lim, repeats=12)
                                for _ in range(5):
                                    for _k in _prearm_keys:
                                        if _k in _prearm_rb:
                                            continue
                                        _v = _param_link.read_px4_param(_k, timeout_s=1.0)
                                        if _v is not None:
                                            _prearm_rb[_k] = _v
                                    if len(_prearm_rb) == len(_prearm_keys):
                                        break
                        except Exception as _e:
                            carb.log_warn(f"PX4 prearm param link error ({_conn_label}): {_e}")
                        finally:
                            if _param_link is not None:
                                try:
                                    _param_link.close()
                                except Exception:
                                    pass

                    # Fallback: jeżeli udpin nie dał pełnego readbacku, spróbuj na istniejącym linku 14580.
                    if len(_prearm_rb) < len(_prearm_keys):
                        mav_out.force_px4_sitl_target()
                        mav_out.set_px4_parameters(_prearm_px4_lim, repeats=8)
                        for _ in range(3):
                            for _k in _prearm_keys:
                                if _k in _prearm_rb:
                                    continue
                                _v = mav_out.read_px4_param(_k, timeout_s=0.8)
                                if _v is not None:
                                    _prearm_rb[_k] = _v
                            if len(_prearm_rb) == len(_prearm_keys):
                                break

                    _prearm_xy_rb = _prearm_rb.get("MPC_XY_VEL_MAX")
                    # Ważne diagnostycznie: warning, aby było widoczne nawet przy filtrowaniu INFO.
                    carb.log_warn(
                        "PX4 prearm velocity readback: "
                        f"MPC_XY_VEL_MAX={_prearm_rb.get('MPC_XY_VEL_MAX', None)} "
                        f"(set={_prearm_px4_lim['MPC_XY_VEL_MAX']:.1f}), "
                        f"MPC_XY_CRUISE={_prearm_rb.get('MPC_XY_CRUISE', None)} "
                        f"(set={_prearm_px4_lim['MPC_XY_CRUISE']:.1f}), "
                        f"MPC_VEL_MANUAL={_prearm_rb.get('MPC_VEL_MANUAL', None)} "
                        f"(set={_prearm_px4_lim['MPC_VEL_MANUAL']:.1f}), link={_prearm_param_link_used}, "
                        f"v_cmd={_prearm_v_cmd:.1f}."
                    )
                    flight_timeline.mark_once(
                        "px4_mpc_xy_vel_max_prearm_readback",
                        step,
                        mpc_xy_vel_max_prearm_set=_prearm_px4_lim["MPC_XY_VEL_MAX"],
                        mpc_xy_vel_max_prearm_readback=_prearm_xy_rb,
                        v_command_ms=_prearm_v_cmd,
                    )

                mav_out.set_mode_takeoff_px4()
                flight_timeline.mark_once("px4_auto_takeoff_mode_command_sent", step)
                for _ in range(int(0.5 / step_dt)):
                    if not simulation_app.is_running():
                        break
                    _do_step()
                    log_state(step)
                    _maybe_mark_spawn_goal_done()
                    step += 1
                    if spawn_goal_reached:
                        break
                if not spawn_goal_reached:
                    mav_out.arm()
                    flight_timeline.mark_once("px4_arm_command_sent", step)
                mav_out.close()
                if not spawn_goal_reached:
                    carb.log_info("Wysłano SET_MODE Takeoff + ARM (port 14580); wznoszenie do MIS_TAKEOFF_ALT")
                fallback_ok = True
        except Exception as e:
            carb.log_warn(f"MAVLink 14580: {e}")
    if not spawn_goal_reached and not fallback_ok and HAS_MAVLINK_OFFBOARD:
        carb.log_warn("MAVLink: nie udało się wysłać Tryb Takeoff + ARM na 14580 — uzbrój/takeoff ręcznie (QGC).")

    # 3) Wznoszenie (etykieta: wznoszenie) — logowanie
    if not spawn_goal_reached:
        takeoff_steps = int(TAKEOFF_WAIT_S / step_dt)
        takeoff_end_reason = "takeoff_wait_full_duration"
        for _ in range(takeoff_steps):
            if not simulation_app.is_running():
                break
            _do_step()
            log_state(step)
            _maybe_mark_spawn_goal_done()
            step += 1
            if spawn_goal_reached:
                break
        flight_timeline.mark_once(
            "takeoff_climb_phase_end",
            step,
            reason=takeoff_end_reason,
            altitude_target_m=round(float(alt_target), 3),
        )

    # 4) Prawdziwa misja — fazy scenariusza (logowanie)
    if not spawn_goal_reached and scenario_id == 1 and HAS_MAVLINK_OFFBOARD:
        # Scenariusz 1: lot prosto (setpoint prędkości NED). Koniec gdy odległość pozioma od spawn_xy ≥ distance_m.
        v_cmd = params.get("v_cmd_ms", 10.0)
        yaw_deg = params.get("yaw_deg", 0.0)
        distance_m = params.get("distance_m", 200.0)
        horiz_s = float(distance_m) / max(0.1, float(v_cmd))
        _need_mission_s = horiz_s + 60.0
        mission_duration_s = float(duration_s)
        if mission_duration_s < _need_mission_s:
            carb.log_warn(
                f"Scenariusz 1: --duration-s={mission_duration_s:.0f}s jest za małe na dystans {distance_m:.0f} m "
                f"przy v={float(v_cmd):.1f} m/s — **wydłużam fazę misji** do ~{_need_mission_s:.0f}s (ustaw wyższe --duration-s dla pełnej kontroli)."
            )
            mission_duration_s = _need_mission_s
        steps_total = int(mission_duration_s / step_dt)
        yaw_rad = math.radians(yaw_deg)
        vx = v_cmd * math.cos(yaw_rad)
        vy = v_cmd * math.sin(yaw_rad)
        _alt_ref = float(params.get("altitude_m", alt_target))
        _var_frac = min(0.05, max(0.0, float(params.get("altitude_variation_frac", 0.0) or 0.0)))
        _wobble_period = max(3.0, float(params.get("altitude_variation_period_s", 40.0) or 40.0))

        def _mission_vz_altitude_wave(t_mission_s: float) -> float:
            """h(t)=alt_ref·(1+var·sin(ωt)); vz NED = −dh/dt (wzrost wysokości → ujemne vz). var ≤ 5 %."""
            if _var_frac <= 0.0:
                return 0.0
            om = 2.0 * math.pi / _wobble_period
            return -_alt_ref * _var_frac * om * math.cos(om * t_mission_s)

        # Dystans kończący misję: odległość pozioma od spawn_xy (nie od pozycji po wzniesieniu).
        # sqrt(dx²+dy²) ≥ distance_m — dowolny kierunek (wiatr + lot liczą się tak samo).
        carb.log_info(
            f"Scenariusz 1: punkt odniesienia dystansu = spawn xy=({spawn_xy[0]:.2f}, {spawn_xy[1]:.2f}), "
            f"cel: ≥{distance_m:.0f} m w poziomie od spawnu. "
            'Komunikaty "przeleciono ... m" i "koniec misji" w tym samym logu (terminal / Isaac Sim Console).'
        )
        if _var_frac > 0.0:
            carb.log_info(
                f"Scenariusz 1: modulacja wysokości w locie ±{_var_frac*100:.0f}% względem altitude_m={_alt_ref:.1f} m, "
                f"okres {_wobble_period:.1f} s (setpoint vz w NED)."
            )

        def _flown_m() -> float:
            """Odległość pozioma od spawn_xy — zawsze z `vehicle` (ten sam GT co CSV / panel). Nie używać pg.get_vehicle(): zwraca obiekt bez świeżego state."""
            if vehicle is None or getattr(vehicle, "state", None) is None:
                return 0.0
            p = vehicle.state.position
            dx = float(p[0]) - spawn_xy[0]
            dy = float(p[1]) - spawn_xy[1]
            return math.sqrt(dx * dx + dy * dy)

        try:
            mav_mission = MavlinkOffboard("udpout:127.0.0.1:14580")
            if mav_mission.bind():
                mav_mission.force_px4_sitl_target()
                # Upewnij się, że mamy poprawne target_system/target_component z HEARTBEAT.
                # Bez tego PARAM_SET może nie trafić w PX4 (stąd np. brak zmiany limitów i efekt ~12 m/s).
                if not mav_mission.try_recv_heartbeat():
                    for _ in range(20):
                        if mav_mission.try_recv_heartbeat():
                            break
                mav_mission.force_px4_sitl_target()
                # PX4: MPC_XY_VEL_MAX itd. — bez tego OFFBOARD jest obcinany (wcześniej sztywne 20 m/s).
                _px4_lim = _px4_velocity_mission_limits(params, float(v_cmd))
                mav_mission.set_px4_parameters(_px4_lim, repeats=3)
                # Readback: sprawdź czy PX4 faktycznie przyjął MPC_XY_VEL_MAX.
                _mpc_xy_after = mav_mission.read_px4_param("MPC_XY_VEL_MAX", timeout_s=0.3)
                carb.log_info(
                    f"Scenariusz 1: readback PX4 MPC_XY_VEL_MAX={_mpc_xy_after if _mpc_xy_after is not None else 'None'} m/s (ustawiane={_px4_lim['MPC_XY_VEL_MAX']:.1f})."
                )
                flight_timeline.mark_once(
                    "px4_mpc_xy_vel_max_readback",
                    step,
                    mpc_xy_vel_max_readback=_mpc_xy_after,
                    mpc_xy_vel_max_set=_px4_lim["MPC_XY_VEL_MAX"],
                )
                if (
                    _mpc_xy_after is not None
                    and _mpc_xy_after < _px4_lim["MPC_XY_VEL_MAX"] * 0.9
                ):
                    carb.log_warn(
                        "Scenariusz 1: MPC_XY_VEL_MAX nie zmienił się jak oczekiwano — ponawiam PARAM_SET."
                    )
                    mav_mission.set_px4_parameters(_px4_lim, repeats=8)
                mav_mission.set_mode_offboard()
                carb.log_info(
                    f"Scenariusz 1: OFFBOARD, lot prosto v={float(v_cmd):.1f} m/s, yaw={yaw_deg:.0f}°, "
                    f"max {distance_m:.0f} m lub {mission_duration_s:.0f} s; "
                    f"PX4 MPC_XY_VEL_MAX={_px4_lim['MPC_XY_VEL_MAX']:.1f} m/s (sufit na |v_xy|)."
                )
                mission_time_s = 0.0
                s1_first_cmd = True
                s1_cruise_fast_since: float | None = None
                _s1_v_frac = 0.88
                _s1_v_sustain_s = 0.35
                for _ in range(steps_total):
                    if not simulation_app.is_running():
                        break
                    flown = _flown_m()
                    if step > 0 and step % _steps_per_sim_second == 0:
                        carb.log_info(f"Scenariusz 1: przeleciono {flown:.1f} m (cel {distance_m:.0f} m)")
                    if flown >= distance_m:
                        carb.log_info(f"Scenariusz 1: przeleciano {flown:.1f} m, koniec misji.")
                        flight_timeline.mark_once(
                            "cruise_distance_goal_reached",
                            step,
                            flown_horizontal_m=flown,
                            distance_m=distance_m,
                        )
                        break
                    vz_sp = _mission_vz_altitude_wave(mission_time_s)
                    last_setpoint[0], last_setpoint[1], last_setpoint[2] = vx, vy, vz_sp
                    time_boot_ms = int(step * physics_dt * 1000)
                    if s1_first_cmd:
                        flight_timeline.mark_once(
                            "offboard_velocity_stream_started",
                            step,
                            scenario_id=1,
                            v_command_ms=float(v_cmd),
                        )
                        s1_first_cmd = False
                    if vehicle is not None and getattr(vehicle, "state", None) is not None:
                        lv = vehicle.state.linear_velocity
                        vh1 = math.hypot(float(lv[0]), float(lv[1]))
                        if vh1 >= float(v_cmd) * _s1_v_frac:
                            if s1_cruise_fast_since is None:
                                s1_cruise_fast_since = mission_time_s
                            elif mission_time_s - s1_cruise_fast_since >= _s1_v_sustain_s:
                                flight_timeline.mark_once(
                                    "cruise_horizontal_speed_sustained_near_command",
                                    step,
                                    v_horizontal_ground_truth_ms=vh1,
                                    v_command_ms=float(v_cmd),
                                    sustained_threshold_frac=_s1_v_frac,
                                    sustained_min_s=_s1_v_sustain_s,
                                )
                        else:
                            s1_cruise_fast_since = None
                    mav_mission.send_velocity_target_ned(time_boot_ms, vx, vy, vz_sp)
                    _do_step()
                    log_state(step)
                    _maybe_mark_spawn_goal_done()
                    step += 1
                    mission_time_s += step_dt
                    if spawn_goal_reached:
                        break
                mav_mission.close()
            else:
                mission_time_s = 0.0
                for _ in range(steps_total):
                    if not simulation_app.is_running():
                        break
                    if _flown_m() >= distance_m:
                        break
                    vz_sp = _mission_vz_altitude_wave(mission_time_s)
                    last_setpoint[0], last_setpoint[1], last_setpoint[2] = vx, vy, vz_sp
                    _do_step()
                    log_state(step)
                    _maybe_mark_spawn_goal_done()
                    step += 1
                    mission_time_s += step_dt
                    if spawn_goal_reached:
                        break
        except Exception as e:
            carb.log_warn(f"Scenariusz 1 OFFBOARD: {e}; tylko logowanie.")
            mission_time_s = 0.0
            for _ in range(steps_total):
                if not simulation_app.is_running():
                    break
                if _flown_m() >= distance_m:
                    break
                vz_sp = _mission_vz_altitude_wave(mission_time_s)
                last_setpoint[0], last_setpoint[1], last_setpoint[2] = vx, vy, vz_sp
                _do_step()
                log_state(step)
                _maybe_mark_spawn_goal_done()
                step += 1
                mission_time_s += step_dt
                if spawn_goal_reached:
                    break

    elif not spawn_goal_reached and scenario_id == 2 and HAS_MAVLINK_OFFBOARD:
        # Scenariusz 2: cruise (PD wysokości z↑ Pegasus) → hamowanie (snapshot z↑) + |v_xy|→0 → PX4 AUTO LAND + NAV_LAND.
        v_cmd = float(params.get("v_cmd_ms", 10.0))
        yaw_deg = float(params.get("yaw_deg", 0.0))
        distance_m = float(params.get("distance_m", 200.0))
        tgt = mission_ctx.get("target_xy")
        if tgt is None:
            carb.log_warn("Scenariusz 2: brak target_xy w mission_ctx — pomijam misję.")
        else:
            tx, ty = float(tgt[0]), float(tgt[1])
            horiz_s = distance_m / max(0.1, v_cmd)
            land_to = float(params.get("land_phase_timeout_s", 180.0))
            brake_to = float(params.get("land_brake_timeout_s", 70.0))
            # Nie pozwalamy, by hamowanie przeciągało się dłużej niż 5 s.
            brake_to = min(brake_to, 5.0)
            _need_mission_s = horiz_s + brake_to + 100.0 + land_to
            mission_duration_s = float(duration_s)
            if mission_duration_s < _need_mission_s:
                carb.log_warn(
                    f"Scenariusz 2: --duration-s={mission_duration_s:.0f}s za małe na lot+lądowanie — wydłużam do ~{_need_mission_s:.0f}s."
                )
                mission_duration_s = _need_mission_s
            steps_total = int(mission_duration_s / step_dt)
            yaw_rad = math.radians(yaw_deg)
            vx_cruise = v_cmd * math.cos(yaw_rad)
            vy_cruise = v_cmd * math.sin(yaw_rad)
            _alt_ref = float(params.get("altitude_m", alt_target))
            _wobble_period = max(3.0, float(params.get("altitude_variation_period_s", 40.0) or 40.0))
            _wobble_frac = params.get("land_cruise_z_wobble_frac")
            if _wobble_frac is None:
                _wobble_frac = min(0.05, max(0.0, float(params.get("altitude_variation_frac", 0.0) or 0.0)))
            else:
                _wobble_frac = max(0.0, float(_wobble_frac))
            cruise_z_kp = float(params.get("land_cruise_z_hold_kp", 1.35))
            cruise_z_kd = float(params.get("land_cruise_z_hold_kd", 0.7))
            cruise_z_cap = float(params.get("land_cruise_z_vz_cap_ms", 3.5))
            cruise_kp_boost = float(params.get("land_cruise_z_kp_boost", 1.9))
            cruise_acquire_s = max(0.0, float(params.get("land_cruise_acquire_boost_s", 14.0)))
            brake_v_stop = float(params.get("land_brake_v_stop_ms", 0.12))
            brake_settle_s = max(0.0, float(params.get("land_brake_settle_s", 1.25)))
            z_hold_kp = float(params.get("land_brake_z_hold_kp", 1.45))
            z_hold_kd = float(params.get("land_brake_z_hold_kd", 0.65))
            z_vz_cap = float(params.get("land_brake_z_vz_cap_ms", 2.2))
            z_land = float(params.get("landed_pos_z_max_m", 0.4))
            v_land = float(params.get("landed_total_speed_max_ms", 0.6))
            land_timeout_steps = int(land_to / step_dt)
            land_stream_n = 15
            s2_first_cmd = True
            s2_cruise_fast_since: float | None = None
            _s2_v_frac = 0.88
            _s2_v_sustain_s = 0.35
            s2_land_cmd_z: float | None = None
            s2_descent_gt_marked = False
            land_after_touchdown_s = float(params.get("land_after_touchdown_s", 2.0))
            land_after_disarm_s = float(params.get("land_after_disarm_s", 2.0))
            touchdown_sim_time_s: float | None = None
            post_touchdown_disarm_sent = False
            disarm_sent_sim_time_s: float | None = None

            def _flown_spawn_s2() -> float:
                if vehicle is None or getattr(vehicle, "state", None) is None:
                    return 0.0
                p = vehicle.state.position
                dx = float(p[0]) - spawn_xy[0]
                dy = float(p[1]) - spawn_xy[1]
                return math.sqrt(dx * dx + dy * dy)

            carb.log_info(
                f"Scenariusz 2: referencja XY celu ({tx:.1f}, {ty:.1f}) m (spawn + {distance_m:.0f} m, yaw {yaw_deg:.0f}°); "
                f"po przeleceniu {distance_m:.0f} m — hamowanie na **tej samej wysokości**, potem **PX4 AUTO LAND + NAV_LAND**."
            )

            phase = "cruise"
            mission_t0_sim = float(getattr(world, "current_time", 0.0) or 0.0)
            mission_time_s = 0.0
            brake_z_hold_ned: float | None = None
            brake_enter_s: float | None = None
            brake_vel_ok_since: float | None = None
            land_commanded = False
            land_steps = 0
            mav_mission = None

            # Acceleration phase recognition (cruise start):
            # "ruszanie" od małych v_xy do blisko v_cmd (Vmax w praktyce).
            s2_acc_started = False
            s2_acc_start_v_xy_ms: float | None = None
            s2_acc_peak_accel_xy_m_s2: float | None = None
            s2_acc_peak_dv_m_s: float | None = None
            s2_acc_peak_v_before: float | None = None
            s2_acc_peak_v_after: float | None = None
            s2_acc_peak_step: int | None = None
            s2_acc_prev_v_xy: float | None = None
            s2_acc_prev_t_sim: float | None = None
            # Real (measured) vmax estimate while accelerating (GT).
            s2_vxy_peak_real: float | None = None
            s2_vxy_peak_real_last_update_s: float | None = None
            s2_near_v_reached = False

            # Start threshold for acceleration phase.
            s2_acc_start_v_thresh = 0.6
            # End-of-acceleration criteria:
            # 1) v_xy within +/-6% of the measured peak
            s2_acc_vmax_tol_frac = 0.06
            # 2) peak must stay stable (no meaningful increase) for a bit
            s2_acc_vmax_stable_s = 0.25
            # How much a new peak must beat the current one (avoid noise spikes).
            s2_acc_vxy_peak_update_abs_eps = 0.05
            # Keep derivative tracking for vx/vy deltas (not just |v_xy|).
            s2_acc_prev_vx_xy_ms: float | None = None
            s2_acc_prev_vy_xy_ms: float | None = None
            # How long we've already stayed within +/- tolerance (for "stable plateau" labeling).
            s2_acc_vxy_tol_ok_since_s: float | None = None

            try:
                mav_mission = MavlinkOffboard("udpout:127.0.0.1:14580")
                if mav_mission.bind():
                    mav_mission.force_px4_sitl_target()
                    # Upewnij się, że mamy poprawne target_system/target_component z HEARTBEAT.
                    if not mav_mission.try_recv_heartbeat():
                        for _ in range(20):
                            if mav_mission.try_recv_heartbeat():
                                break
                    mav_mission.force_px4_sitl_target()
                    # PX4: MPC_XY_VEL_MAX — sufitem na prędkość poziomą (must-have v_cmd 4–30 m/s).
                    _px4_lim = _px4_velocity_mission_limits(params, float(v_cmd))
                    mav_mission.set_px4_parameters(_px4_lim, repeats=3)
                    # Readback: sprawdź czy PX4 faktycznie przyjął MPC_XY_VEL_MAX.
                    _mpc_xy_after = mav_mission.read_px4_param("MPC_XY_VEL_MAX", timeout_s=0.3)
                    carb.log_info(
                        f"Scenariusz 2: readback PX4 MPC_XY_VEL_MAX={_mpc_xy_after if _mpc_xy_after is not None else 'None'} m/s (ustawiane={_px4_lim['MPC_XY_VEL_MAX']:.1f})."
                    )
                    flight_timeline.mark_once(
                        "px4_mpc_xy_vel_max_readback",
                        step,
                        mpc_xy_vel_max_readback=_mpc_xy_after,
                        mpc_xy_vel_max_set=_px4_lim["MPC_XY_VEL_MAX"],
                    )
                    if (
                        _mpc_xy_after is not None
                        and _mpc_xy_after < _px4_lim["MPC_XY_VEL_MAX"] * 0.9
                    ):
                        carb.log_warn(
                            "Scenariusz 2: MPC_XY_VEL_MAX nie zmienił się jak oczekiwano — ponawiam PARAM_SET."
                        )
                        mav_mission.set_px4_parameters(_px4_lim, repeats=8)
                    mav_mission.set_mode_offboard()
                    carb.log_info(
                        f"Scenariusz 2: OFFBOARD cruise v={v_cmd:.1f} m/s, wysokość PD→{_alt_ref:.1f} m (z↑), "
                        f"wobble celu ≤{_wobble_frac*100:.1f} %; koniec cruise przy ≥{distance_m:.0f} m od spawnu; "
                        f"PX4 MPC_XY_VEL_MAX={_px4_lim['MPC_XY_VEL_MAX']:.1f} m/s."
                    )
                    for _ in range(steps_total):
                        if not simulation_app.is_running():
                            break
                        mission_time_s = float(getattr(world, "current_time", 0.0) or 0.0) - mission_t0_sim
                        time_boot_ms = int(step * physics_dt * 1000)

                        if not land_commanded:
                            if phase == "cruise":
                                flown = _flown_spawn_s2()
                                if flown >= distance_m or mission_time_s > horiz_s * 5.0:
                                    phase = "brake"
                                    brake_enter_s = mission_time_s
                                    brake_vel_ok_since = None
                                    if vehicle is not None and getattr(vehicle, "state", None) is not None:
                                        brake_z_hold_ned = float(vehicle.state.position[2])
                                    flight_timeline.mark_once(
                                        "brake_phase_started",
                                        step,
                                        flown_horizontal_m=flown,
                                        distance_m_commanded=distance_m,
                                        mission_timeout_s=bool(mission_time_s > horiz_s * 5.0),
                                    )
                                    carb.log_info(
                                        f"Scenariusz 2: hamowanie — przeleciano z spawnu ~{flown:.1f} m "
                                        f"(próg {distance_m:.0f} m); hold wys. z↑="
                                        f"{brake_z_hold_ned if brake_z_hold_ned is not None else float('nan'):.2f} m; "
                                        f"OFFBOARD **vx=vy=0** (pełne zatrzymanie w poziomie), potem lądowanie."
                                    )
                                else:
                                    z_tgt = _alt_ref
                                    if _wobble_frac > 0.0 and _wobble_period > 0.0:
                                        om = 2.0 * math.pi / _wobble_period
                                        z_tgt += _alt_ref * _wobble_frac * math.sin(
                                            om * mission_time_s
                                        )
                                    kp_use = (
                                        cruise_kp_boost
                                        if mission_time_s < cruise_acquire_s
                                        else cruise_z_kp
                                    )
                                    if vehicle is None or getattr(vehicle, "state", None) is None:
                                        vz_sp = 0.0
                                    else:
                                        p = vehicle.state.position
                                        lv = vehicle.state.linear_velocity
                                        vz_sp = _vz_ned_from_altitude_zup(
                                            float(p[2]),
                                            float(lv[2]),
                                            z_tgt,
                                            kp_use,
                                            cruise_z_kd,
                                            cruise_z_cap,
                                        )
                                    last_setpoint[0] = vx_cruise
                                    last_setpoint[1] = vy_cruise
                                    last_setpoint[2] = vz_sp
                                    if s2_first_cmd:
                                        flight_timeline.mark_once(
                                            "offboard_velocity_stream_started",
                                            step,
                                            scenario_id=2,
                                            v_command_ms=float(v_cmd),
                                        )
                                        s2_first_cmd = False
                                    if vehicle is not None and getattr(vehicle, "state", None) is not None:
                                        lv2 = vehicle.state.linear_velocity
                                        vh2 = math.hypot(float(lv2[0]), float(lv2[1]))
                                        vx2 = float(lv2[0])
                                        vy2 = float(lv2[1])

                                        # --- Acceleration phase recognition (cruise start) ---
                                        if not s2_acc_started and vh2 >= s2_acc_start_v_thresh:
                                            s2_acc_started = True
                                            s2_acc_start_v_xy_ms = vh2
                                            # Initialize derivative tracking right at the moment it starts.
                                            s2_acc_prev_v_xy = vh2
                                            s2_acc_prev_t_sim = mission_time_s
                                            s2_acc_prev_vx_xy_ms = vx2
                                            s2_acc_prev_vy_xy_ms = vy2
                                            # Initialize measured vmax with the current value.
                                            s2_vxy_peak_real = vh2
                                            s2_vxy_peak_real_last_update_s = mission_time_s
                                            flight_timeline.mark_once(
                                                "acceleration_phase_started",
                                                step,
                                                v_xy_start_m_s=vh2,
                                                v_cmd_ms=float(v_cmd),
                                                threshold_m_s=s2_acc_start_v_thresh,
                                            )

                                        if s2_acc_started:
                                            if s2_acc_prev_v_xy is not None and s2_acc_prev_t_sim is not None:
                                                dt = mission_time_s - s2_acc_prev_t_sim
                                                if dt > 1e-4:
                                                    dv = vh2 - float(s2_acc_prev_v_xy)
                                                    accel_xy = dv / dt
                                                    # Component-wise deltas: helps to attribute the measured "acceleration" to vx/vy.
                                                    dvx = vx2 - float(s2_acc_prev_vx_xy_ms) if s2_acc_prev_vx_xy_ms is not None else 0.0
                                                    dvy = vy2 - float(s2_acc_prev_vy_xy_ms) if s2_acc_prev_vy_xy_ms is not None else 0.0
                                                    accel_x = dvx / dt
                                                    accel_y = dvy / dt
                                                    if (
                                                        s2_acc_peak_accel_xy_m_s2 is None
                                                        or accel_xy > float(s2_acc_peak_accel_xy_m_s2)
                                                    ):
                                                        s2_acc_peak_accel_xy_m_s2 = accel_xy
                                                        s2_acc_peak_dv_m_s = dv
                                                        s2_acc_peak_v_before = float(s2_acc_prev_v_xy)
                                                        s2_acc_peak_v_after = vh2
                                                        s2_acc_peak_step = step
                                                        # Znacznik dla "peak" nie jest potrzebny do stabilnego
                                                        # etykietowania końca przyspieszenia (wystarczy start + near_vmax).

                                            # Update measured vmax peak while accelerating.
                                            if s2_vxy_peak_real is None:
                                                s2_vxy_peak_real = vh2
                                                s2_vxy_peak_real_last_update_s = mission_time_s
                                            elif vh2 > float(s2_vxy_peak_real) + s2_acc_vxy_peak_update_abs_eps:
                                                s2_vxy_peak_real = vh2
                                                s2_vxy_peak_real_last_update_s = mission_time_s

                                            s2_acc_prev_v_xy = vh2
                                            s2_acc_prev_t_sim = mission_time_s
                                            s2_acc_prev_vx_xy_ms = vx2
                                            s2_acc_prev_vy_xy_ms = vy2

                                        # End-of-acceleration: reached measured vmax plateau
                                        if not s2_near_v_reached and s2_vxy_peak_real is not None:
                                            vmax_real = float(s2_vxy_peak_real)
                                            if vmax_real > 1e-6:
                                                rel_err = abs(vh2 - vmax_real) / vmax_real
                                                last_peak_update = (
                                                    float(s2_vxy_peak_real_last_update_s)
                                                    if s2_vxy_peak_real_last_update_s is not None
                                                    else mission_time_s
                                                )
                                                stable_for_s = mission_time_s - last_peak_update
                                                if rel_err <= s2_acc_vmax_tol_frac:
                                                    if s2_acc_vxy_tol_ok_since_s is None:
                                                        s2_acc_vxy_tol_ok_since_s = mission_time_s
                                                else:
                                                    # We left the +/- tolerance band -> reset.
                                                    s2_acc_vxy_tol_ok_since_s = None

                                                tol_hold_s = (
                                                    mission_time_s - s2_acc_vxy_tol_ok_since_s
                                                    if s2_acc_vxy_tol_ok_since_s is not None
                                                    else 0.0
                                                )
                                                if (
                                                    tol_hold_s >= s2_acc_vmax_stable_s
                                                    and stable_for_s >= s2_acc_vmax_stable_s
                                                ):
                                                    s2_near_v_reached = True
                                                    dv_from_start = None
                                                    if s2_acc_start_v_xy_ms is not None:
                                                        dv_from_start = vh2 - s2_acc_start_v_xy_ms
                                                    flight_timeline.mark_once(
                                                        "acceleration_near_vmax_reached",
                                                        step,
                                                        v_xy_near_vmax_m_s=vh2,
                                                        v_cmd_ms=float(v_cmd),
                                                        v_max_real_m_s=vmax_real,
                                                        rel_err_to_vmax_real=rel_err,
                                                        stable_since_vmax_real_s=stable_for_s,
                                                        tol_hold_s=tol_hold_s,
                                                        dv_from_acc_start_m_s=dv_from_start,
                                                        peak_accel_xy_m_s2=float(s2_acc_peak_accel_xy_m_s2)
                                                        if s2_acc_peak_accel_xy_m_s2 is not None
                                                        else None,
                                                        peak_dv_m_s=float(s2_acc_peak_dv_m_s)
                                                        if s2_acc_peak_dv_m_s is not None
                                                        else None,
                                                    )

                                        if vh2 >= float(v_cmd) * _s2_v_frac:
                                            if s2_cruise_fast_since is None:
                                                s2_cruise_fast_since = mission_time_s
                                            elif mission_time_s - s2_cruise_fast_since >= _s2_v_sustain_s:
                                                flight_timeline.mark_once(
                                                    "cruise_horizontal_speed_sustained_near_command",
                                                    step,
                                                    v_horizontal_ground_truth_ms=vh2,
                                                    v_command_ms=float(v_cmd),
                                                    sustained_threshold_frac=_s2_v_frac,
                                                    sustained_min_s=_s2_v_sustain_s,
                                                )
                                        else:
                                            s2_cruise_fast_since = None
                                    mav_mission.send_velocity_target_ned(
                                        time_boot_ms, vx_cruise, vy_cruise, vz_sp
                                    )
                            # Osobno od cruise: po przejściu cruise→brake w tej samej iteracji wyślij hamowanie (nie elif).
                            if phase == "brake":
                                if vehicle is None or getattr(vehicle, "state", None) is None:
                                    break
                                lv = vehicle.state.linear_velocity
                                vx = float(lv[0])
                                vy = float(lv[1])
                                vz_meas = float(lv[2])
                                pz = float(vehicle.state.position[2])
                                if brake_z_hold_ned is None:
                                    brake_z_hold_ned = pz
                                vz_sp = _vz_ned_from_altitude_zup(
                                    pz, vz_meas, brake_z_hold_ned, z_hold_kp, z_hold_kd, z_vz_cap
                                )
                                v_h = math.hypot(vx, vy)
                                b_el = (
                                    (mission_time_s - brake_enter_s)
                                    if brake_enter_s is not None
                                    else 0.0
                                )
                                if v_h <= brake_v_stop:
                                    if brake_vel_ok_since is None:
                                        brake_vel_ok_since = mission_time_s
                                else:
                                    brake_vel_ok_since = None
                                settled = (
                                    brake_vel_ok_since is not None
                                    and (mission_time_s - brake_vel_ok_since) >= brake_settle_s
                                )
                                if settled or b_el > brake_to:
                                    if settled:
                                        flight_timeline.mark_once(
                                            "brake_horizontal_stabilized",
                                            step,
                                            v_xy_horizontal_ms=v_h,
                                            brake_settle_s=brake_settle_s,
                                        )
                                    if not settled and b_el > brake_to:
                                        carb.log_warn(
                                            f"Scenariusz 2: timeout hamowania ({b_el:.0f}s), |v_xy|={v_h:.2f} m/s — wymuszam lądowanie PX4."
                                        )
                                        flight_timeline.mark_once(
                                            "brake_timeout_forcing_land",
                                            step,
                                            v_xy_horizontal_ms=v_h,
                                            brake_elapsed_s=b_el,
                                        )
                                    if (
                                        vehicle is not None
                                        and getattr(vehicle, "state", None) is not None
                                    ):
                                        s2_land_cmd_z = float(vehicle.state.position[2])
                                    ok_land = mav_mission.set_mode_auto_land_px4()
                                    ok_cmd = mav_mission.send_nav_land_in_place()
                                    flight_timeline.mark_once(
                                        "px4_auto_land_mode_command_sent",
                                        step,
                                        mavlink_ok=bool(ok_land),
                                    )
                                    flight_timeline.mark_once(
                                        "px4_nav_land_command_sent",
                                        step,
                                        mavlink_ok=bool(ok_cmd),
                                    )
                                    land_commanded = True
                                    land_steps = 0
                                    last_setpoint[:] = [0.0, 0.0, 0.0]
                                    carb.log_info(
                                        "Scenariusz 2: po zatrzymaniu — "
                                        f"AUTO LAND={'OK' if ok_land else 'fail'}, NAV_LAND={'OK' if ok_cmd else 'fail'}; "
                                        + (
                                            f"|v_xy|≤{brake_v_stop:.2f} m/s przez ≥{brake_settle_s:.1f} s."
                                            if settled
                                            else "timeout hamowania."
                                        )
                                    )
                                else:
                                    # Pełne zatrzymanie w poziomie (nie −k·v — to powodowało "skręt" i dalszy lot).
                                    vx_sp = 0.0
                                    vy_sp = 0.0
                                    last_setpoint[0] = vx_sp
                                    last_setpoint[1] = vy_sp
                                    last_setpoint[2] = vz_sp
                                    mav_mission.send_velocity_target_ned(
                                        time_boot_ms, vx_sp, vy_sp, vz_sp
                                    )
                        elif mav_mission is not None and land_stream_n > 0:
                            last_setpoint[:] = [0.0, 0.0, 0.0]
                            mav_mission.send_velocity_target_ned(time_boot_ms, 0.0, 0.0, 0.0)
                            land_stream_n -= 1
                            # Nie zamykaj tutaj — potrzebne jest połączenie do DISARM po touchdown.

                        _do_step()
                        log_state(step)
                        step += 1
                        if land_commanded:
                            land_steps += 1
                            if vehicle is not None and getattr(vehicle, "state", None) is not None:
                                p = vehicle.state.position
                                lv = vehicle.state.linear_velocity
                                spd = math.sqrt(
                                    float(lv[0]) ** 2 + float(lv[1]) ** 2 + float(lv[2]) ** 2
                                )
                                if (
                                    s2_land_cmd_z is not None
                                    and not s2_descent_gt_marked
                                    and float(p[2]) < float(s2_land_cmd_z) - 0.35
                                ):
                                    flight_timeline.mark_once(
                                        "vertical_descent_ground_truth_started",
                                        step,
                                        z_at_land_command_m=float(s2_land_cmd_z),
                                        z_now_m=float(p[2]),
                                    )
                                    s2_descent_gt_marked = True
                                now_sim_t = float(getattr(world, "current_time", 0.0) or 0.0)
                                if touchdown_sim_time_s is None and float(p[2]) <= z_land and spd < v_land:
                                    flight_timeline.mark_once(
                                        "touchdown_stable_ground_truth",
                                        step,
                                        pos_z_m=float(p[2]),
                                        total_speed_ms=spd,
                                    )
                                    carb.log_info(
                                        f"Scenariusz 2: zakończono po lądowaniu (z≤{z_land:.2f} m, |v|<{v_land:.2f} m/s)."
                                    )
                                    touchdown_sim_time_s = now_sim_t
                                # Po touchdown: N s → DISARM → M s → koniec run (CSV dalej do końca).
                                if touchdown_sim_time_s is not None:
                                    dt_td = now_sim_t - touchdown_sim_time_s
                                    if dt_td >= land_after_touchdown_s and not post_touchdown_disarm_sent:
                                        ok_d = False
                                        if mav_mission is not None:
                                            ok_d = bool(mav_mission.disarm())
                                        else:
                                            m_dis = MavlinkOffboard("udpout:127.0.0.1:14580")
                                            if m_dis.bind():
                                                ok_d = bool(m_dis.disarm_px4_sitl_default())
                                                m_dis.close()
                                        post_touchdown_disarm_sent = True
                                        disarm_sent_sim_time_s = now_sim_t
                                        flight_timeline.mark_once(
                                            "px4_disarm_command_sent",
                                            step,
                                            mavlink_ok=bool(ok_d),
                                        )
                                        carb.log_info(
                                            f"Scenariusz 2: DISARM po {land_after_touchdown_s:.2f}s od touchdown "
                                            f"(MAVLink ok={ok_d})."
                                        )
                                    if (
                                        post_touchdown_disarm_sent
                                        and disarm_sent_sim_time_s is not None
                                        and (now_sim_t - disarm_sent_sim_time_s) >= land_after_disarm_s
                                    ):
                                        carb.log_info(
                                            f"Scenariusz 2: koniec run po {land_after_disarm_s:.2f}s od DISARM."
                                        )
                                        break
                            if land_steps >= land_timeout_steps:
                                carb.log_warn("Scenariusz 2: timeout fazy lądowania — kończę run.")
                                break

                    if mav_mission is not None:
                        mav_mission.close()
                        mav_mission = None
                else:
                    carb.log_warn("Scenariusz 2: MAVLink bind nieudany — brak OFFBOARD/lądowania.")
            except Exception as e:
                carb.log_warn(f"Scenariusz 2: {e}")
                if mav_mission is not None:
                    try:
                        mav_mission.close()
                    except Exception:
                        pass

    elif not spawn_goal_reached and scenario_id == 3 and HAS_MAVLINK_OFFBOARD:
        # Scenariusz 3: sekwencja segmentów (sprint/hamowanie/wznoszenie/zakręty/slalomy).
        # Każdy segment — liniowa interpolacja v, PD wysokości, obrót nagłówku o omega*dt.
        segments = params.get("segments", [])
        z_kp = float(params.get("s3_z_kp", 1.35))
        z_kd = float(params.get("s3_z_kd", 0.7))
        z_cap = float(params.get("s3_z_vz_cap_ms", 4.0))
        z_kp_b = float(params.get("s3_z_kp_boost", 1.9))
        z_boost_s = float(params.get("s3_z_kp_boost_s", 8.0))
        heading = math.radians(float(params.get("yaw_start_deg", 0.0)))

        def _interp_v(v0: float, v1: float, t: float, ramp_s: float) -> float:
            return v0 + (v1 - v0) * min(1.0, t / max(ramp_s, step_dt))

        def _z_ramp(z0: float, z1: float, t: float, dur: float) -> float:
            return z0 + (z1 - z0) * min(1.0, t / max(dur, step_dt))

        def _get_vz(veh, z_tgt: float, kp: float, kd: float, cap: float) -> float:
            if veh is None or not getattr(veh, "_state", None):
                return 0.0
            p = veh.state.position
            lv = veh.state.linear_velocity
            return _vz_ned_from_altitude_zup(float(p[2]), float(lv[2]), z_tgt, kp, kd, cap)

        try:
            mav3 = MavlinkOffboard("udpout:127.0.0.1:14580")
            if mav3.bind():
                mav3.force_px4_sitl_target()
                if not mav3.try_recv_heartbeat():
                    for _ in range(20):
                        if mav3.try_recv_heartbeat():
                            break
                mav3.force_px4_sitl_target()
                mav3.set_mode_offboard()
                mission_time_s = 0.0
                flight_timeline.mark_once("offboard_velocity_stream_started", step, scenario_id=3)

                for seg_idx, seg in enumerate(segments):
                    if not simulation_app.is_running():
                        break
                    seg_type = seg.get("type", "sprint")
                    v_start = float(seg["v_start_ms"])
                    v_end = float(seg["v_end_ms"])
                    dur = float(seg["duration_s"])
                    z_start = float(seg.get("z_start_m", 15.0))
                    z_end = float(seg.get("z_end_m", z_start))
                    ramp_s = float(seg.get("accel_ramp_s", dur))
                    radius_m = float(seg.get("radius_m", 0.0))
                    turn_dir = float(seg.get("turn_dir", 0.0))
                    n_ht = max(1, int(seg.get("n_halfturns", 1)))
                    phase_lbl = seg.get("phase_label", "lot")
                    seg_steps = max(1, int(dur / step_dt))

                    current_seg_label[0] = phase_lbl
                    carb.log_info(
                        f"Scen3 seg {seg_idx + 1}/{len(segments)} [{phase_lbl}] "
                        f"v={v_start:.1f}→{v_end:.1f} m/s z={z_start:.1f}→{z_end:.1f} m "
                        f"R={radius_m:.0f} m dur={dur:.1f} s"
                    )
                    flight_timeline.mark(
                        "segment_start", step,
                        seg_idx=seg_idx,
                        seg_type=seg_type,
                        phase_label=phase_lbl,
                        v_start_ms=round(v_start, 2),
                        v_end_ms=round(v_end, 2),
                        z_start_m=round(z_start, 1),
                        z_end_m=round(z_end, 1),
                        radius_m=round(radius_m, 1),
                        angle_deg=round(float(seg.get("angle_deg", 0.0)), 1),
                        turn_dir=int(turn_dir),
                        duration_s=round(dur, 2),
                        n_halfturns=n_ht if seg_type.startswith("slalom") else 1,
                    )

                    if seg_type.startswith("slalom"):
                        ht_steps = max(1, seg_steps // n_ht)
                        total_done = 0
                        for ht in range(n_ht):
                            if not simulation_app.is_running():
                                break
                            d_now = turn_dir * ((-1.0) ** ht)
                            for i in range(ht_steps):
                                if not simulation_app.is_running():
                                    break
                                t_in = (total_done + i) * step_dt
                                v = _interp_v(v_start, v_end, t_in, ramp_s)
                                om = (v / radius_m) * d_now if radius_m > 0.0 else 0.0
                                heading += om * physics_dt
                                z_tgt = _z_ramp(z_start, z_end, t_in, dur)
                                kp_use = z_kp_b if mission_time_s < z_boost_s else z_kp
                                vz = _get_vz(vehicle, z_tgt, kp_use, z_kd, z_cap)
                                vx = v * math.cos(heading)
                                vy = v * math.sin(heading)
                                last_setpoint[:] = [vx, vy, vz]
                                mav3.send_velocity_target_ned(int(step * physics_dt * 1000), vx, vy, vz)
                                _do_step()
                                log_state(step)
                                _update_follow_camera(vehicle, heading, current_seg_label[0])
                                step += 1
                                mission_time_s += step_dt
                            total_done += ht_steps
                    else:
                        for i in range(seg_steps):
                            if not simulation_app.is_running():
                                break
                            t_in = i * step_dt
                            v = _interp_v(v_start, v_end, t_in, ramp_s)
                            om = (v / radius_m) * turn_dir if radius_m > 0.0 else 0.0
                            heading += om * physics_dt
                            z_tgt = _z_ramp(z_start, z_end, t_in, dur)
                            kp_use = z_kp_b if mission_time_s < z_boost_s else z_kp
                            vz = _get_vz(vehicle, z_tgt, kp_use, z_kd, z_cap)
                            vx = v * math.cos(heading)
                            vy = v * math.sin(heading)
                            last_setpoint[:] = [vx, vy, vz]
                            mav3.send_velocity_target_ned(int(step * physics_dt * 1000), vx, vy, vz)
                            _do_step()
                            log_state(step)
                            _update_follow_camera(vehicle, heading, current_seg_label[0])
                            step += 1
                            mission_time_s += step_dt

                    flight_timeline.mark(
                        "segment_end", step,
                        seg_idx=seg_idx,
                        seg_type=seg_type,
                        phase_label=phase_lbl,
                    )

                current_seg_label[0] = ""
                flight_timeline.mark_once("scenario_3_segments_complete", step, segments_count=len(segments))

                # Lądowanie: hamowanie (vx=vy=0, hold z) → PX4 AUTO LAND + NAV_LAND → touchdown → DISARM
                current_seg_label[0] = "hamowanie"
                brake_s = float(params.get("s3_land_brake_s", 6.0))
                bz_kp = float(params.get("s3_land_brake_z_kp", 1.45))
                bz_kd = float(params.get("s3_land_brake_z_kd", 0.65))
                bz_cap = float(params.get("s3_land_brake_z_vz_cap_ms", 2.5))
                brake_z_hold = None
                if vehicle is not None and getattr(vehicle, "_state", None) is not None:
                    brake_z_hold = float(vehicle.state.position[2])
                for _ in range(max(1, int(brake_s / step_dt))):
                    if not simulation_app.is_running():
                        break
                    vz = _get_vz(vehicle, brake_z_hold, bz_kp, bz_kd, bz_cap) if brake_z_hold else 0.0
                    last_setpoint[:] = [0.0, 0.0, vz]
                    mav3.send_velocity_target_ned(int(step * physics_dt * 1000), 0.0, 0.0, vz)
                    _do_step()
                    log_state(step)
                    _update_follow_camera(vehicle, heading, current_seg_label[0])
                    step += 1
                    mission_time_s += step_dt

                current_seg_label[0] = "ladowanie_px4"
                mav3.set_mode_auto_land_px4()
                mav3.send_nav_land_in_place()
                flight_timeline.mark_once("scenario_3_auto_land_commanded", step)
                carb.log_info("Scenariusz 3: AUTO LAND + NAV_LAND — czekam na touchdown.")

                z_land_max = float(params.get("s3_land_z_max_m", 0.4))
                v_land_max = float(params.get("s3_land_v_max_ms", 0.6))
                land_timeout_steps = int(float(params.get("s3_land_timeout_s", 120.0)) / step_dt)
                after_td_s = float(params.get("s3_land_after_touchdown_s", 2.0))
                after_dis_s = float(params.get("s3_land_after_disarm_s", 2.0))
                touchdown_sim_t: float | None = None
                disarm_sent_s3 = False
                disarm_t_s3: float | None = None

                for _ in range(land_timeout_steps):
                    if not simulation_app.is_running():
                        break
                    last_setpoint[:] = [0.0, 0.0, 0.0]
                    _do_step()
                    log_state(step)
                    _update_follow_camera(vehicle, heading, current_seg_label[0])
                    step += 1
                    mission_time_s += step_dt
                    if vehicle is not None and getattr(vehicle, "_state", None) is not None:
                        p3 = vehicle.state.position
                        lv3 = vehicle.state.linear_velocity
                        spd3 = math.sqrt(sum(float(x) ** 2 for x in lv3))
                        now3 = float(getattr(world, "current_time", 0.0) or 0.0)
                        if touchdown_sim_t is None and float(p3[2]) <= z_land_max and spd3 < v_land_max:
                            touchdown_sim_t = now3
                            flight_timeline.mark_once(
                                "touchdown_stable_ground_truth", step,
                                pos_z_m=float(p3[2]), total_speed_ms=spd3,
                            )
                            carb.log_info(f"Scenariusz 3: touchdown (z={float(p3[2]):.2f} m, v={spd3:.2f} m/s).")
                        if touchdown_sim_t is not None:
                            if not disarm_sent_s3 and now3 - touchdown_sim_t >= after_td_s:
                                mav3.disarm()
                                disarm_sent_s3 = True
                                disarm_t_s3 = now3
                                flight_timeline.mark_once("px4_disarm_command_sent", step)
                            if disarm_sent_s3 and disarm_t_s3 is not None and now3 - disarm_t_s3 >= after_dis_s:
                                carb.log_info("Scenariusz 3: koniec run po DISARM.")
                                break

                current_seg_label[0] = ""
                mav3.close()
            else:
                carb.log_warn("Scenariusz 3: MAVLink bind nieudany — brak OFFBOARD.")
        except Exception as e:
            carb.log_warn(f"Scenariusz 3: {e}")
            current_seg_label[0] = ""

    elif not spawn_goal_reached:
        steps_total = int(duration_s / step_dt)
        for _ in range(steps_total):
            if not simulation_app.is_running():
                break
            _do_step()
            log_state(step)
            step += 1

    try:
        _tl_meta: dict[str, Any] = {
            "commanded_cruise_speed_ms": float(params.get("v_cmd_ms", 0.0) or 0.0),
            "speed_measurement_note": (
                "v_horizontal_ground_truth_ms is from Pegasus/Isaac linear_velocity XY (NED). "
                "PX4 MPC_XY_VEL_MAX is raised from params/v_cmd before flight so OFFBOARD is not clamped to ~12–20 m/s; "
                "measured speed can still lag v_cmd_ms (tracking, wind, altitude loop). "
                "Override: px4_mpc_xy_vel_max, px4_mpc_z_vel_max_dn/up, px4_mpc_acc_hor in run params JSON."
            ),
        }
        _save_flight_timeline_json(
            flight_timeline,
            csv_path,
            scenario_id,
            run_id,
            run_start_time,
            _tl_meta,
        )
    except Exception as e:
        carb.log_warn(f"Stachometr: flight_timeline zapis: {e}")

    log_file.close()
    if live_display:
        live_display.close()
    timeline.stop()
    carb.log_info(f"Run s{scenario_id} r{run_id} zakończony. Zapis: {csv_path}")


def main():
    args = _ARGS_EARLY
    output_dir = _ensure_output_dir(args.output_dir)
    show_live = not _HEADLESS  # okno z odczytami tylko w trybie z GUI (preview)

    if getattr(args, "preview_pair", False) and not args.preview:
        carb.log_warn("Stachometr: --preview-pair wymaga --preview — uruchamiam bez pary.")
    if getattr(args, "preview_pair", False) and args.preview and args.scenario != 1:
        carb.log_warn("Stachometr: --preview-pair ma sens tylko z --scenario 1 — ignoruję.")

    # Tryb z ustalonymi parametrami: jeden run scenariusza 1, z wiatrem lub bez (osobne uruchomienia)
    if args.scenario_1_no_wind or args.scenario_1_wind:
        wind_enabled = args.scenario_1_wind
        scenario_id = 1
        params = get_scenario_1_fixed_params(wind_enabled)
        params["scenario_1_runs"] = [
            {"wind_suffix": "wind" if wind_enabled else "nowind", "wind_enabled": wind_enabled},
        ]
        run_start_time = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        _save_params_json(params, scenario_id, 0, output_dir, run_start_time)
        run_single(
            scenario_id=scenario_id,
            run_id=0,
            params=params,
            output_dir=output_dir,
            duration_s=args.duration_s,
            simulation_app=simulation_app,
            run_start_time=run_start_time,
            wind_suffix="wind" if wind_enabled else "nowind",
            show_live_display=show_live,
            random_seed=args.seed,
            ground_extent_xy_m=float(getattr(args, "ground_extent_m", 2000.0)),
        )
        carb.log_info("Stachometr: zakończono (scenariusz 1, parametry ustalone).")
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        simulation_app.close()
        return

    scenario_id = args.scenario
    runs = args.runs if args.headless else 1
    if args.preview:
        runs = 1

    for run_id in range(runs):
        seed = (args.seed + run_id) if args.seed is not None else None
        params = draw_scenario(scenario_id, seed=seed)
        if scenario_id in (1, 2, 3):
            params["scenario_1_runs"] = [
                {"wind_suffix": "wind", "wind_enabled": True},
            ]
        run_start_time = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        _save_params_json(params, scenario_id, run_id, output_dir, run_start_time)

        if scenario_id in (1, 2, 3):
            # Scenariusze 1, 2, 3 (losowane): przelot **z wiatrem** (CSV *_wind_state.csv)
            run_single(
                scenario_id=scenario_id,
                run_id=run_id,
                params=params,
                output_dir=output_dir,
                duration_s=args.duration_s,
                simulation_app=simulation_app,
                run_start_time=run_start_time,
                wind_suffix="wind",
                show_live_display=show_live,
                random_seed=seed,
                ground_extent_xy_m=float(getattr(args, "ground_extent_m", 2000.0)),
            )
        else:
            run_single(
                scenario_id=scenario_id,
                run_id=run_id,
                params=params,
                output_dir=output_dir,
                duration_s=args.duration_s,
                simulation_app=simulation_app,
                run_start_time=run_start_time,
                wind_suffix=None,
                show_live_display=show_live,
                random_seed=seed,
                ground_extent_xy_m=float(getattr(args, "ground_extent_m", 2000.0)),
            )

        # Po pierwszym runie w trybie wielokrotnym nie zamykamy app — robimy clear_scene i kolejny run
        if run_id < runs - 1:
            pg = PegasusInterface()
            try:
                pg.clear_scene()
            except Exception as e:
                carb.log_warn(f"clear_scene między runami: {e}")

    carb.log_info("Stachometr: zakończono.")
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
