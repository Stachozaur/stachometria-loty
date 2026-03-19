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
    return parser.parse_known_args()[0]

_ARGS_EARLY = _parse_args_before_isaac()

# Teraz start Isaac Sim
import carb
from isaacsim import SimulationApp

_HEADLESS = _ARGS_EARLY.headless if _ARGS_EARLY.headless else (not _ARGS_EARLY.preview)
simulation_app = SimulationApp({"headless": _HEADLESS})

# --- Reszta importów po uruchomieniu Sim ---
import json
import omni.physx
import omni.timeline
import omni.usd
from omni.isaac.core.world import World
from pxr import Gf, PhysicsSchemaTools
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


def _meteo_wind_from_deg(wn: float, we: float) -> float:
    """Kierunek meteo „z którego wieje” [°], 0=N, 90=E — z wektora prędkości powietrza w NED (poziomo)."""
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
    const LEFT = [
      { lab: 'Czas i dystans', keys: ['czas_symulacji_s', 'czas_rzeczywisty_s', 'przeleciono_m', 'cel_m'] },
      { lab: 'Wiatr (symulacja, bieżąco)', keys: ['wiatr_bieżący_m/s', 'wiatr_z_którego_°', 'wiatr_podmuch_aktywny', 'wiatr_faza_podmuchu', 'wiatr_podmuch_narasta', 'wiatr_podmuch_z_planu', 'wiatr_zelżenie'] },
      { lab: 'Prędkość (GT)', keys: ['vel_x', 'vel_y', 'vel_z'] },
      { lab: 'Barometr', keys: ['baro_pressure_hPa', 'baro_temp_C', 'baro_alt_m'] },
      { lab: 'Żyroskop (body)', keys: ['gyro_x', 'gyro_y', 'gyro_z'] },
    ];
    const RIGHT = [
      { lab: 'Pozycja (GT)', keys: ['pos_x', 'pos_y', 'pos_z'] },
      { lab: 'Orientacja (GT)', keys: ['roll_deg', 'pitch_deg', 'yaw_deg', 'qw', 'qx', 'qy', 'qz'] },
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
    ]);
    let readingsHtml = '<h2>Odczyty na żywo</h2><div class="readings-2col">';
    readingsHtml += '<div class="readings-col">' + renderCols(LEFT) + '</div>';
    readingsHtml += '<div class="readings-col">' + renderCols(RIGHT) + '</div></div>';
    const rest = Object.entries(d).filter(([k]) => !k.startsWith('param_') && !seen.has(k));
    if (rest.length) readingsHtml += '<h3>Inne</h3><table class="kv-table">' + rest.map(([k, v]) => row(k, v)).join('') + '</table>';
    document.getElementById('readings').innerHTML = readingsHtml;
    const windRows = WIND_KEYS.filter(k => k in d).map(k => row(k, d[k])).join('');
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
    ) -> None:
        if self._server is None:
            return
        data = {
            "czas_symulacji_s": f"{time_s:.2f}",
            "czas_rzeczywisty_s": f"{time.time() - self._start_wall_s:.2f}",
        }
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
                    data["qw"] = f"{qw:.5f}"
                    data["qx"] = f"{qx:.5f}"
                    data["qy"] = f"{qy:.5f}"
                    data["qz"] = f"{qz:.5f}"
                    try:
                        rot = Rotation.from_quat([qx, qy, qz, qw])
                        # Euler ZYX [°]: yaw (o Z NED / „heading”), pitch (o Y), roll (o X) — scipy, jednoznacznie z kwaternionu
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
        # Bieżący wektor wiatru z PhysX (ten sam co w CSV); kierunek = meteo „z którego wieje”
        if self._wind_enabled and wind_snapshot is not None:
            try:
                wn = float(wind_snapshot.get("wind_vel_n", 0.0))
                we = float(wind_snapshot.get("wind_vel_e", 0.0))
                spd = math.hypot(wn, we)
                data["wiatr_bieżący_m/s"] = f"{spd:.2f}"
                data["wiatr_z_którego_°"] = f"{_meteo_wind_from_deg(wn, we):.1f}"
                gp = wind_snapshot.get("wind_gust_phase", "") or ""
                ig = int(wind_snapshot.get("wind_is_gust", 0) or 0)
                data["wiatr_podmuch_aktywny"] = "tak" if ig else "nie"
                data["wiatr_faza_podmuchu"] = gp if str(gp).strip() else "—"
                rs = int(wind_snapshot.get("wind_gust_rising", 0) or 0)
                data["wiatr_podmuch_narasta"] = "tak" if (ig and rs) else ("nie" if ig else "—")
                fs = int(wind_snapshot.get("wind_gust_from_schedule", 0) or 0)
                data["wiatr_podmuch_z_planu"] = "tak" if fs else ("nie" if ig else "—")
                ll = int(wind_snapshot.get("wind_gust_is_lull", 0) or 0)
                data["wiatr_zelżenie"] = "tak" if ll else ("nie" if ig else "—")
            except Exception:
                data["wiatr_bieżący_m/s"] = "—"
                data["wiatr_z_którego_°"] = "—"
                data["wiatr_podmuch_aktywny"] = "—"
                data["wiatr_faza_podmuchu"] = "—"
                data["wiatr_podmuch_narasta"] = "—"
                data["wiatr_podmuch_z_planu"] = "—"
                data["wiatr_zelżenie"] = "—"
        else:
            data["wiatr_bieżący_m/s"] = "—"
            data["wiatr_z_którego_°"] = "—"
            data["wiatr_podmuch_aktywny"] = "—"
            data["wiatr_faza_podmuchu"] = "—"
            data["wiatr_podmuch_narasta"] = "—"
            data["wiatr_podmuch_z_planu"] = "—"
            data["wiatr_zelżenie"] = "—"
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
WARMUP_S = 3.0
TAKEOFF_WAIT_S = 5.0  # 3 + 5 = 8 s do startu misji
MISSION_START_OFFSET_S = WARMUP_S + TAKEOFF_WAIT_S  # 8.0 s — od tego momentu zaczyna się "prawdziwa misja"


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
    alt_target = params.get("altitude_m", params.get("altitude_start_m", 20.0))
    if scenario_id == 10:
        alt_target = 5.0
    init_z = 0.07  # nad podłożem, żeby nie kolidować z ziemią
    # Punkt „startu” na ziemi (XY) — musi być zgodny z pierwszym argumentem pozycji Multirotor poniżej.
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
    ]
    log_file = open(csv_path, "w", encoding="utf-8", newline="")
    writer = csv.writer(log_file)
    writer.writerow(csv_header)

    # Mutable: ostatni setpoint prędkości (OFFBOARD) — ustawiane w pętli misji, odczytywane w log_state
    last_setpoint = [0.0, 0.0, 0.0]
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
    # Krok czasu **fizyki** musi pochodzić ze świata Isaac (PhysX). Błąd: założenie 1/250 przy domyślnym 1/60
    # skraca `time_s` w CSV ~4× względem prawdziwego postępu (dystans/prędkość przestają się zgadzać z time_s).
    physics_dt = 1.0 / 60.0
    try:
        physics_dt = float(world.get_physics_dt())
    except Exception as e:
        carb.log_warn(f"Stachometr: world.get_physics_dt() — {e}; fallback physics_dt={physics_dt}")
    if physics_dt <= 0.0:
        physics_dt = 1.0 / 60.0
        carb.log_warn(f"Stachometr: nieprawidłowy physics_dt, używam {physics_dt}")
    carb.log_info(f"Stachometr: physics_dt={physics_dt:.6f} s/krok (~{1.0/physics_dt:.1f} Hz) — time_s, wiatr, MAVLink")
    _steps_per_sim_second = max(1, int(round(1.0 / physics_dt)))

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
        """Siła wiatru w świecie PhysX; aktualizuje last_wind_log do CSV."""
        nonlocal _wind_apply_pos_warned, last_wind_log
        if _wind_body_id is None or _wind_stage_id is None or _wind_gen is None:
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
        t = step_count * physics_dt
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
            writer.writerow(row)
            if live_display:
                live_display.update(
                    vehicle,
                    t,
                    flown_m=flown_m,
                    distance_m_cel=mission_ctx.get("distance_m"),
                    wind_snapshot=last_wind_log if wind_suffix == "wind" else None,
                )

    step = 0

    # 1) Rozgrzewka (etykieta: rozgrzewka) — logowanie od t=0; dajemy PX4 czas na start
    warmup_steps = int(WARMUP_S / physics_dt)
    for _ in range(warmup_steps):
        if not simulation_app.is_running():
            break
        _do_step()
        log_state(step)
        _maybe_mark_spawn_goal_done()
        step += 1
        if spawn_goal_reached:
            break

    # 2) Tryb Takeoff + ARM (port 14580 Onboard PX4)
    # PX4: najpierw SET_MODE Takeoff (AUTO_TAKEOFF), potem ARM — wtedy po uzbrojeniu dron sam wznosi (do MIS_TAKEOFF_ALT).
    # MAV_CMD_NAV_TAKEOFF przy samym ARM powodowało "Disarmed by auto preflight disarming".
    fallback_ok = False
    if not spawn_goal_reached and HAS_MAVLINK_OFFBOARD:
        try:
            mav_out = MavlinkOffboard("udpout:127.0.0.1:14580")
            if mav_out.bind():
                mav_out.set_mode_takeoff_px4()
                for _ in range(int(0.5 / physics_dt)):
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
        takeoff_steps = int(TAKEOFF_WAIT_S / physics_dt)
        for _ in range(takeoff_steps):
            if not simulation_app.is_running():
                break
            _do_step()
            log_state(step)
            _maybe_mark_spawn_goal_done()
            step += 1
            if spawn_goal_reached:
                break
            if vehicle is not None and vehicle._state is not None:
                if vehicle.state.position[2] >= alt_target - 1.5:
                    break

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
        steps_total = int(mission_duration_s / physics_dt)
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
            "Komunikaty „przeleciono … m” i „koniec misji” w tym samym logu (terminal / Isaac Sim Console)."
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
                mav_mission.set_mode_offboard()
                carb.log_info(f"Scenariusz 1: OFFBOARD, lot prosto v={v_cmd:.1f} m/s, yaw={yaw_deg:.0f}°, max {distance_m:.0f} m lub {mission_duration_s:.0f} s")
                mission_time_s = 0.0
                for _ in range(steps_total):
                    if not simulation_app.is_running():
                        break
                    flown = _flown_m()
                    if step > 0 and step % _steps_per_sim_second == 0:
                        carb.log_info(f"Scenariusz 1: przeleciono {flown:.1f} m (cel {distance_m:.0f} m)")
                    if flown >= distance_m:
                        carb.log_info(f"Scenariusz 1: przeleciano {flown:.1f} m, koniec misji.")
                        break
                    vz_sp = _mission_vz_altitude_wave(mission_time_s)
                    last_setpoint[0], last_setpoint[1], last_setpoint[2] = vx, vy, vz_sp
                    time_boot_ms = int(step * physics_dt * 1000)
                    mav_mission.send_velocity_target_ned(time_boot_ms, vx, vy, vz_sp)
                    _do_step()
                    log_state(step)
                    _maybe_mark_spawn_goal_done()
                    step += 1
                    mission_time_s += physics_dt
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
                    mission_time_s += physics_dt
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
                mission_time_s += physics_dt
                if spawn_goal_reached:
                    break

    elif not spawn_goal_reached and scenario_id == 2 and HAS_MAVLINK_OFFBOARD:
        # Scenariusz 2: cruise (±5 % wys.) → wyrownanie_xy (vz=0) → schodzenie pionowe → AUTO LAND przy z ≤ progu.
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
            _need_mission_s = horiz_s + 120.0 + land_to
            mission_duration_s = float(duration_s)
            if mission_duration_s < _need_mission_s:
                carb.log_warn(
                    f"Scenariusz 2: --duration-s={mission_duration_s:.0f}s za małe na lot+lądowanie — wydłużam do ~{_need_mission_s:.0f}s."
                )
                mission_duration_s = _need_mission_s
            steps_total = int(mission_duration_s / physics_dt)
            yaw_rad = math.radians(yaw_deg)
            vx_cruise = v_cmd * math.cos(yaw_rad)
            vy_cruise = v_cmd * math.sin(yaw_rad)
            _alt_ref = float(params.get("altitude_m", alt_target))
            _var_frac = min(0.05, max(0.0, float(params.get("altitude_variation_frac", 0.0) or 0.0)))
            _wobble_period = max(3.0, float(params.get("altitude_variation_period_s", 40.0) or 40.0))
            appr_start = float(params.get("land_approach_start_m", 35.0))
            appr_v_max = float(params.get("land_approach_v_max_ms", 4.0))
            xy_align = float(params.get("land_xy_align_m", 4.0))
            descend_vz = float(params.get("land_descend_vz_ms", 0.85))
            z_auto_land = float(params.get("land_auto_land_alt_z_m", 3.0))
            xy_drift_max = float(params.get("land_descend_xy_drift_max_m", 7.0))
            z_land = float(params.get("landed_pos_z_max_m", 0.4))
            v_land = float(params.get("landed_total_speed_max_ms", 0.6))
            land_timeout_steps = int(land_to / physics_dt)
            land_stream_n = 15
            land_align_timeout_s = float(params.get("land_align_timeout_s", 95.0))

            def _mission_vz_altitude_wave_s2(t_mission_s: float) -> float:
                if _var_frac <= 0.0:
                    return 0.0
                om = 2.0 * math.pi / _wobble_period
                return -_alt_ref * _var_frac * om * math.cos(om * t_mission_s)

            def _dist_to_target_s2() -> float:
                if vehicle is None or getattr(vehicle, "state", None) is None:
                    return 1.0e9
                p = vehicle.state.position
                return math.hypot(float(p[0]) - tx, float(p[1]) - ty)

            def _flown_spawn_s2() -> float:
                if vehicle is None or getattr(vehicle, "state", None) is None:
                    return 0.0
                p = vehicle.state.position
                dx = float(p[0]) - spawn_xy[0]
                dy = float(p[1]) - spawn_xy[1]
                return math.sqrt(dx * dx + dy * dy)

            carb.log_info(
                f"Scenariusz 2: cel XY = ({tx:.1f}, {ty:.1f}) m (spawn+{distance_m:.0f} m, yaw {yaw_deg:.0f}°); "
                f"wyrownanie do ≤{xy_align:.0f} m lub timeout {land_align_timeout_s:.0f}s; potem zejście + AUTO LAND."
            )

            phase = "cruise"
            mission_time_s = 0.0
            align_enter_s: float | None = None
            land_commanded = False
            land_steps = 0
            mav_mission = None

            try:
                mav_mission = MavlinkOffboard("udpout:127.0.0.1:14580")
                if mav_mission.bind():
                    mav_mission.set_mode_offboard()
                    carb.log_info(
                        f"Scenariusz 2: OFFBOARD cruise v={v_cmd:.1f} m/s, modulacja wys. ≤5 %; "
                        f"wyrownanie_xy gdy d_do_celu ≤ {appr_start:.0f} m"
                    )
                    for _ in range(steps_total):
                        if not simulation_app.is_running():
                            break
                        d_tgt = _dist_to_target_s2()
                        time_boot_ms = int(step * physics_dt * 1000)

                        if not land_commanded:
                            if phase == "cruise":
                                flown = _flown_spawn_s2()
                                if d_tgt <= appr_start or mission_time_s > horiz_s * 5.0:
                                    phase = "align_xy"
                                    align_enter_s = mission_time_s
                                    carb.log_info(
                                        f"Scenariusz 2: wyrownanie_xy — d_do_celu={d_tgt:.1f} m, z spawn ~{flown:.1f} m."
                                    )
                                vz_sp = _mission_vz_altitude_wave_s2(mission_time_s)
                                last_setpoint[0] = vx_cruise
                                last_setpoint[1] = vy_cruise
                                last_setpoint[2] = vz_sp
                                mav_mission.send_velocity_target_ned(
                                    time_boot_ms, vx_cruise, vy_cruise, vz_sp
                                )
                            elif phase == "align_xy":
                                if vehicle is None or getattr(vehicle, "state", None) is None:
                                    break
                                p = vehicle.state.position
                                px, py = float(p[0]), float(p[1])
                                dx = tx - px
                                dy = ty - py
                                dist = math.hypot(dx, dy)
                                _align_elapsed = (
                                    (mission_time_s - align_enter_s)
                                    if align_enter_s is not None
                                    else 0.0
                                )
                                if dist <= xy_align or _align_elapsed > land_align_timeout_s:
                                    if dist > xy_align:
                                        carb.log_warn(
                                            f"Scenariusz 2: timeout wyrownania XY ({_align_elapsed:.0f}s), "
                                            f"d_xy={dist:.1f} m — start schodzenia."
                                        )
                                    phase = "descend"
                                    carb.log_info(f"Scenariusz 2: schodzenie (d_xy={dist:.2f} m).")
                                else:
                                    v_cap = min(appr_v_max, max(0.5, 0.42 * dist))
                                    vx_ap = v_cap * dx / dist if dist > 0.05 else 0.0
                                    vy_ap = v_cap * dy / dist if dist > 0.05 else 0.0
                                    last_setpoint[0] = vx_ap
                                    last_setpoint[1] = vy_ap
                                    last_setpoint[2] = 0.0
                                    mav_mission.send_velocity_target_ned(
                                        time_boot_ms, vx_ap, vy_ap, 0.0
                                    )
                            else:
                                if vehicle is None or getattr(vehicle, "state", None) is None:
                                    break
                                p = vehicle.state.position
                                px, py = float(p[0]), float(p[1])
                                pz = float(p[2])
                                if pz <= z_auto_land:
                                    carb.log_info(
                                        f"Scenariusz 2: z={pz:.2f} m ≤ {z_auto_land:.1f} m — AUTO LAND (PX4)."
                                    )
                                    mav_mission.set_mode_auto_land_px4()
                                    land_commanded = True
                                    land_steps = 0
                                    last_setpoint[:] = [0.0, 0.0, 0.0]
                                else:
                                    dx = tx - px
                                    dy = ty - py
                                    dh = math.hypot(dx, dy)
                                    vx_d = 0.0
                                    vy_d = 0.0
                                    if dh > xy_drift_max and dh > 0.05:
                                        vn = min(1.2, 0.2 * (dh - xy_drift_max))
                                        vx_d = vn * dx / dh
                                        vy_d = vn * dy / dh
                                    vz_sp = descend_vz
                                    last_setpoint[0] = vx_d
                                    last_setpoint[1] = vy_d
                                    last_setpoint[2] = vz_sp
                                    mav_mission.send_velocity_target_ned(
                                        time_boot_ms, vx_d, vy_d, vz_sp
                                    )
                        elif mav_mission is not None and land_stream_n > 0:
                            last_setpoint[:] = [0.0, 0.0, 0.0]
                            mav_mission.send_velocity_target_ned(time_boot_ms, 0.0, 0.0, 0.0)
                            land_stream_n -= 1
                            if land_stream_n == 0:
                                mav_mission.close()
                                mav_mission = None

                        _do_step()
                        log_state(step)
                        step += 1
                        mission_time_s += physics_dt
                        if land_commanded:
                            land_steps += 1
                            if vehicle is not None and getattr(vehicle, "state", None) is not None:
                                p = vehicle.state.position
                                lv = vehicle.state.linear_velocity
                                spd = math.sqrt(
                                    float(lv[0]) ** 2 + float(lv[1]) ** 2 + float(lv[2]) ** 2
                                )
                                if float(p[2]) <= z_land and spd < v_land:
                                    carb.log_info(
                                        f"Scenariusz 2: zakończono po lądowaniu (z≤{z_land:.2f} m, |v|<{v_land:.2f} m/s)."
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

    elif not spawn_goal_reached:
        steps_total = int(duration_s / physics_dt)
        for _ in range(steps_total):
            if not simulation_app.is_running():
                break
            _do_step()
            log_state(step)
            step += 1

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
        if scenario_id in (1, 2):
            params["scenario_1_runs"] = [
                {"wind_suffix": "wind", "wind_enabled": True},
            ]
        run_start_time = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        _save_params_json(params, scenario_id, run_id, output_dir, run_start_time)

        if scenario_id in (1, 2):
            # Scenariusze 1 i 2 (losowane): przelot **z wiatrem** (CSV *_wind_state.csv)
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
