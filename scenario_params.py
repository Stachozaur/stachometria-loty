# -*- coding: utf-8 -*-
"""
Parametry scenariuszy Stachometr (docs/py/scenarios.md).
Dla każdego scenariusza: losowanie parametrów z podanych zakresów i zapis do JSON.
"""
from __future__ import annotations

import math
import os
import random
from typing import Any

# --- Etykiety faz do klasyfikacji (zgodnie z sekcją "Etykiety faz z definicji scenariusza" w scenarios.md) ---
# Każdy run musi zapisywać phase_times: lista przedziałów [t_start_s, t_end_s] z przypisaną klasą fazy.
# Użycie: wejście = okno czasowe (np. 100–200 ms) z IMU/FC; etykieta = klasa fazy (samoetykietowanie z planu misji).

PHASE_LABELS_DOC = (
    "phase_times: wektor faz z przedziałami [t_start_s, t_end_s] i klasą fazy. "
    "W każdym runie na początku są fazy wspólne: rozgrzewka (0–1 s), wznoszenie (1–2.25 s); od 2.25 s zaczyna się misja (fazy scenariusza). "
    "Etykiety do klasyfikacji: rozgrzewka, wznoszenie, hamowanie, przyspieszanie, zawis, lot_prosty, zakręt, wiatr_step, start, podejście, itd. "
    "Zgodnie z sekcją 'Etykiety faz z definicji scenariusza (samoetykietowanie)' w docs/py/scenarios.md."
)

# --- Scenariusze (id, opis, funkcja losująca zwracająca dict do zapisu) ---

SCENARIO_DESCRIPTIONS = {
    1: "Lot prosto 500–1000 m (losowy dystans), wysokość modulowana w locie (±5%); wiatr ~75% U(2,4) m/s, ~25% U(0.01,9) (ogon). Każdy run z wiatrem.",
    2: "Jak scenariusz 1 (±5% wys.): lot do celu 500–1000 m (yaw), wycentrowanie XY, zejście pionowe, PX4 AUTO LAND. Wiatr jak w scenariuszu 1.",
    3: (
        "Zakręty (łagodny/średni/ostry/gwałtowny), zawroty (>150°), slalomy L-R, lot jednostajny (2×), "
        "zmiany prędkości i wysokości. Prędkości 5–30 m/s. "
        "Sekwencja 14 segmentów (12 typów obowiązkowych + 2× lot_jednostajny). Wiatr jak scen. 1/2. Wysokość 10–40 m."
    ),
    4: "Zawis (hover) z zakłóceniami wiatru. Utrzymanie pozycji, wiatr zmienia się w czasie.",
    5: "Ósemki / slalomy. Lot po torze: ósemka lub slalom lewo-prawo.",
    6: "Wznoszenie / opadanie pod kątem (200 m w poziomie). Lot 200 m z jednoczesnym wznoszeniem/opadaniem.",
    7: "Staccato: krótkie impulsy przyspieszenia i hamowania.",
    8: "Lot z jednym silnym podmuchem wiatru (step). Wiatr skacze z niskiego na wysoki w ustalonym momencie.",
    9: "Zmienna wysokość (fale). Lot 200 m z sinusoidalną zmianą wysokości.",
    10: "Mix faz: start → przyspieszanie → prosty → zakręt → zawis → podejście do lądowania (stop na 2 m).",
}


def _r(a: float, b: float) -> float:
    """R(min, max) - jednorazowy draw z zakresu."""
    return random.uniform(a, b)


def _r_discrete(*options: Any) -> Any:
    """R_discrete(opcje) - jednorazowy wybór z listy."""
    return random.choice(options)


# --- Wiatr (PhysX): siła pozioma ∝ K_WIND_N_PER_MS * wind_speed_ms (run_stachometr.py) ---
#
# TY EDYTUJESZ TYLKO `wind_speed_ms_when_enabled` w SCENARIO_1_FIXED — to są **m/s w terenie** (w logu / JSON / panelu).
# K_WIND_N_PER_MS liczy się sam z kalibracji legacy (nie musisz go dotykać przy zmianie 3.7 → 5 itd.).
#
# Kalibracja (stała, „pod spodem”): kiedyś było k=0.5 i w parametrze 33; uznaliśmy, że to ~3.7 m/s w polu.
_WIND_LEGACY_K_N_PER_MS = 0.5
_WIND_LEGACY_PARAM_MS = 33.0
_WIND_LEGACY_MEANT_REAL_MS = 3.7
K_WIND_N_PER_MS = _WIND_LEGACY_K_N_PER_MS * (_WIND_LEGACY_PARAM_MS / _WIND_LEGACY_MEANT_REAL_MS)

# --- Wiatr: model siły + dynamika (domyślnie: generator OU + podmuchy; stały wektor tylko po jawnej nadpisie wind_dynamic_enabled=False) ---
# wind_cd_times_area_m2: dobrane ~ tak, by przy |v_air|≈ wind_speed siła drag była rzędu legacy (K_WIND * w).
DEFAULT_WIND_PHYSICS_PARAMS: dict[str, Any] = {
    "wind_force_model": "legacy",  # "legacy" | "drag"
    "wind_dynamic_enabled": True,
    # OU (turbulencja): losowe odchylenia składowych wiatru w płaszczyźnie NED **N (North) i E (East)** —
    # wektor wiatru poziomo „pływa” w czasie wokół średniej (to jest ten „dryf” składowych n/e, nie przesuwanie całej pogody).
    # Większe tau → wolniejsza zmiana tych składowych; sigma → rząd wielkości odchyłek [m/s].
    "wind_tau_s": 0.55,
    "wind_sigma_ms": 0.35,
    "wind_gust_prob_per_s": 0.018,
    # Całkowity czas podmuchu T = t_narastania + t_opadania; clamp do [T_min, T_max] (przy domyślnych rise/fall min 1,6 s).
    "wind_gust_T_min_s": 1.6,
    "wind_gust_T_max_s": 15.0,
    # Podmuch 1−cos: dodatek wzdłuż średniego wiatru w szczycie — losowo U(A_min, A_max) [m/s].
    "wind_gust_A_rel_min_ms": 1.5,
    "wind_gust_A_rel_max_ms": 3.0,
    # Zachowanie wsteczne: gdy w JSON jest tylko ten klucz (bez min/max), generator i tak bierze min=max z A_rel_ms.
    "wind_gust_A_rel_ms": 2.25,
    # Czas narastania do szczytu i czas opadania (s symulacji); ustawione oba → losowe podmuchy **nie** używają peak_frac×T.
    "wind_gust_rise_time_min_s": 0.8,
    "wind_gust_rise_time_max_s": 2.0,
    "wind_gust_fall_time_min_s": 0.8,
    "wind_gust_fall_time_max_s": 2.0,
    # Legacy (gdy **usuniesz** z params oba klucze wind_gust_rise_time_*): peak_frac * T, peak_frac ∈ [min, max].
    "wind_gust_peak_frac_min": 0.1,
    "wind_gust_peak_frac_max": 0.9,
    "wind_gust_lull_enabled": True,
    "wind_gust_lull_prob": 0.35,
    "wind_rho_kg_m3": 1.225,
    "wind_cd_times_area_m2": 1.5,
}


def merge_wind_defaults(params: dict[str, Any]) -> dict[str, Any]:
    """Uzupełnia brakujące klucze wiatru domyślnymi (nie nadpisuje już ustawionych)."""
    out = dict(params)
    for k, v in DEFAULT_WIND_PHYSICS_PARAMS.items():
        out.setdefault(k, v)
    return out


# Stałe parametry dla trybu --scenario_1_no_wind / --scenario_1_wind (ten sam zestaw, różnica tylko wiatr)
# wind_dir_deg: kierunek, Z KTÓREGO wieje wiatr (konwencja meteo). 0° = z północy → wieje na południe = headwind (w twarz, utrudnia lot na północ przy yaw=0)
# wind_speed_ms_when_enabled: **średnia prędkość wiatru [m/s]** (baza dla generatora; przy wind_dynamic_enabled=True dochodzi szum + podmuchy).
# wind_dynamic_enabled: True (domyślnie) = OU + podmuchy; False = stały wektor = wind_speed_ms / wind_dir_deg.
# Inne klucze wiatru (opcjonalnie): wind_force_model, wind_tau_s, wind_sigma_ms, wind_gust_* — patrz DEFAULT_WIND_PHYSICS_PARAMS.
SCENARIO_1_FIXED = {
    "v_cmd_ms": 12.0,
    "yaw_deg": 0.0,
    "altitude_m": 15.0,
    "distance_m": 1000.0,
    "wind_speed_ms_when_enabled": 3,
    "wind_dir_deg": 65.0,
    "wind_dynamic_enabled": True,
    # Turbulencja OU: większe tau / mniejsze sigma → wolniejsza zmiana kierunku „z którego wieje” na panelu (nadal chwilowy wektor).
    # Przy bardzo słabej średniej (np. 1 m/s) i dużym sigma domyślne składowa wzdłużna często wpada w obcięcie (along≥0) → skoki kierunku ~90°.
    "wind_tau_s": 1.4,
    "wind_sigma_ms": 0.35,
    # Harmonogram: opcjonalnie "peak_fraction" (0–1) = część duration do szczytu; brak → szczyt w połowie (symetrycznie).
    "wind_gust_schedule": [
        {"t_start_s": 11.67, "duration_s": 9.5, "A_rel_ms": 3, "peak_fraction": 0.35},
    ],
    "altitude_variation_frac": 0.05,
    "altitude_variation_period_s": 60.0,
}


def get_scenario_1_fixed_params(wind_enabled: bool) -> dict:
    """Scenariusz 1 z ustalonymi (nie losowymi) parametrami. wind_enabled=True → wiatr włączony, False → bez wiatru."""
    v_cmd = SCENARIO_1_FIXED["v_cmd_ms"]
    distance_m = SCENARIO_1_FIXED["distance_m"]
    t_end = distance_m / v_cmd if v_cmd > 0 else 60.0
    phase_times = [
        {"phase": "lot_prosty", "t_start_s": 0.0, "t_end_s": round(t_end, 2)},
    ]
    base = {
        "scenario_id": 1,
        "scenario_name": "lot_prosto_200m",
        "v_cmd_ms": v_cmd,
        "yaw_deg": SCENARIO_1_FIXED["yaw_deg"],
        "altitude_m": SCENARIO_1_FIXED["altitude_m"],
        "wind_speed_ms": SCENARIO_1_FIXED["wind_speed_ms_when_enabled"] if wind_enabled else 0.0,
        "wind_dir_deg": SCENARIO_1_FIXED["wind_dir_deg"] if wind_enabled else 0.0,
        "distance_m": distance_m,
        "phase_times": phase_times,
        "phase_times_description": "Jedna faza: lot_prosty na całym dystansie 200 m. Etykiety do klasyfikacji faz.",
        "fixed_params_note": "Parametry ustalone (--scenario_1_no_wind / --scenario_1_wind), nie losowe.",
    }
    if "wind_gust_schedule" in SCENARIO_1_FIXED:
        base["wind_gust_schedule"] = [dict(ev) for ev in SCENARIO_1_FIXED["wind_gust_schedule"]]
    # Opcjonalne nadpisania z SCENARIO_1_FIXED (generator / model siły) — bez duplikatu speed/dir
    _skip = frozenset({"wind_speed_ms_when_enabled", "wind_dir_deg"})
    for k, v in SCENARIO_1_FIXED.items():
        if k in {"altitude_variation_frac", "altitude_variation_period_s"}:
            base[k] = v
        elif k.startswith("wind_") and k not in _skip and k in {
            "wind_dynamic_enabled",
            "wind_force_model",
            "wind_tau_s",
            "wind_sigma_ms",
            "wind_gust_prob_per_s",
            "wind_gust_T_min_s",
            "wind_gust_T_max_s",
            "wind_gust_A_rel_ms",
            "wind_gust_A_rel_min_ms",
            "wind_gust_A_rel_max_ms",
            "wind_gust_rise_time_min_s",
            "wind_gust_rise_time_max_s",
            "wind_gust_fall_time_min_s",
            "wind_gust_fall_time_max_s",
            "wind_gust_peak_frac_min",
            "wind_gust_peak_frac_max",
            "wind_gust_lull_enabled",
            "wind_gust_lull_prob",
            "wind_rho_kg_m3",
            "wind_cd_times_area_m2",
        }:
            base[k] = v
    return merge_wind_defaults(base)


def _draw_wind_speed_scenario_1() -> float:
    """
    Mieszanina dwóch rozkładów jednostajnych (symulacja „najczęściej umiarkowanie, czasem skrajnie”):

    - **~75%** losowań: **U(2, 4)** m/s — typowy, najczęstszy pas (średnia 3 m/s).
    - **~25%** losowań: **U(0.01, 9)** m/s — cały szeroki zakres; tu pojawiają się **słaby** wiatr (<2),
      **umiarkowany** (część 2–4 nakłada się z górnego rozkładu), i **mocny** aż do **9 m/s**.

    Łącznie P(wiatr w [2,4]) ≈ 75% + 25%×(2/8.99) ≈ **81%**; reszta to skrajne wartości z „ogona”.
    """
    if random.random() < 0.75:
        return random.uniform(2.0, 4.0)
    return random.uniform(0.01, 9.0)


def draw_scenario_1() -> dict:
    """Scenariusz 1: lot prosto na losowy dystans 500–1000 m, lekka modulacja wysokości (sinus ±5 %), wiatr jak wyżej."""
    v_cmd = _r(4.0, 30.0)
    distance_m = _r(500.0, 1000.0)
    altitude_m = _r(10.0, 20.0)
    t_end = distance_m / v_cmd if v_cmd > 0 else 60.0
    # Kilka pełnych „fal” wysokości w czasie szacowanego lotu poziomego
    period_s = max(18.0, min(100.0, t_end / 2.2))
    phase_times = [
        {"phase": "lot_prosty", "t_start_s": 0.0, "t_end_s": round(t_end, 2)},
    ]
    base = {
        "scenario_id": 1,
        "scenario_name": "lot_prosto_500_1000m",
        "v_cmd_ms": v_cmd,
        "yaw_deg": _r(0.0, 360.0),
        "altitude_m": altitude_m,
        "wind_speed_ms": _draw_wind_speed_scenario_1(),
        "wind_dir_deg": _r(0.0, 360.0),
        "distance_m": distance_m,
        # sinus: amplituda ±5 % altitude_m (setpoint vz w NED); okres [s]
        "altitude_variation_frac": 0.05,
        "altitude_variation_period_s": round(period_s, 2),
        "phase_times": phase_times,
        "phase_times_description": "Jedna faza: lot_prosty; dystans poziomy 500–1000 m; wysokość modulowana w locie (±5 %).",
        # Jawne dla czytelności JSON: domyślnie i tak True z merge_wind_defaults
        "wind_dynamic_enabled": True,
    }
    return merge_wind_defaults(base)


def _stub_scenario(scenario_id: int, name: str, snippet: str) -> dict:
    """Stub: zwraca minimalny dict; scenariusz będzie kodowany osobno po dopracowaniu poprzednich."""
    return {
        "scenario_id": scenario_id,
        "scenario_name": name,
        "phase_times": [{"phase": "todo", "t_start_s": 0.0, "t_end_s": 60.0}],
        "phase_times_description": f"TODO — scenariusz {scenario_id} będzie kodowany osobno. Snippet: {snippet}",
    }


def draw_scenario_2() -> dict:
    """
    Scenariusz 2: jak scenariusz 1 (v, yaw, dystans poziomy od spawnu, wysokość ±5 %, wiatr),
    potem przy osiągnięciu distance_m — hamowanie do ~0 prędkości poziomej, potem PX4 AUTO LAND + NAV_LAND w miejscu.
    Punkt referencyjny XY: spawn + distance_m w azymucie yaw (target_xy / etykiety).
    """
    v_cmd = _r(4.0, 30.0)
    distance_m = _r(500.0, 1000.0)
    altitude_m = _r(10.0, 20.0)
    yaw_deg = _r(0.0, 360.0)
    t_cruise = distance_m / v_cmd if v_cmd > 0 else 60.0
    period_s = max(18.0, min(100.0, t_cruise / 2.2))
    # Szacunek faz (etykiety; faktyczny czas = PX4 / wiatr)
    t_brake_s = 45.0
    t_land_s = 120.0
    phase_times = [
        {"phase": "lot_prosty", "t_start_s": 0.0, "t_end_s": round(t_cruise, 2)},
        {
            "phase": "hamowanie",
            "t_start_s": round(t_cruise, 2),
            "t_end_s": round(t_cruise + t_brake_s, 2),
        },
        {
            "phase": "ladowanie_px4",
            "t_start_s": round(t_cruise + t_brake_s, 2),
            "t_end_s": round(t_cruise + t_brake_s + t_land_s, 2),
        },
    ]
    yrad = math.radians(yaw_deg)
    base = {
        "scenario_id": 2,
        "scenario_name": "lot_do_celu_ladowanie_500_1000m",
        "v_cmd_ms": v_cmd,
        "yaw_deg": yaw_deg,
        "altitude_m": altitude_m,
        "wind_speed_ms": _draw_wind_speed_scenario_1(),
        "wind_dir_deg": _r(0.0, 360.0),
        "distance_m": distance_m,
        # Scen. 2: w locie wysokość = PD względem altitude_m (patrz run_stachometr), nie sinus vz jak w scen. 1.
        "altitude_variation_frac": 0.0,
        "altitude_variation_period_s": round(period_s, 2),
        "land_cruise_z_hold_kp": 1.35,
        "land_cruise_z_hold_kd": 0.7,
        "land_cruise_z_vz_cap_ms": 3.5,
        "land_cruise_z_wobble_frac": 0.012,
        "land_cruise_z_kp_boost": 1.9,
        "land_cruise_acquire_boost_s": 14.0,
        # Cel w płaszczyźnie XY względem spawnu (0,0) — zgodnie z run_stachometr (spawn_xy)
        "target_xy_offset_m": [round(distance_m * math.cos(yrad), 3), round(distance_m * math.sin(yrad), 3)],
        # Hamowanie po distance_m: |v_xy|→0 + hold wys. (z↑); potem PX4 AUTO LAND + MAV_CMD_NAV_LAND (bez długiego offboard zejścia).
        # Hamowanie: OFFBOARD vx=vy=0 + hold wys.; przejście do lądowania gdy |v_xy| ≤ stop przez settle_s (lub timeout).
        # Skrócone hamowanie: nie ma wisieć długo w powietrzu.
        # Uwaga: runtime używa też kryterium timeout; poniższe wartości są domyślne.
        "land_brake_v_stop_ms": 0.35,
        "land_brake_settle_s": 0.5,
        "land_brake_timeout_s": 5.0,
        # Hamowanie: ta sama konwencja z↑ co Pegasus (nie „NED z” z pozycji).
        "land_brake_z_hold_kp": 1.45,
        "land_brake_z_hold_kd": 0.65,
        "land_brake_z_vz_cap_ms": 2.2,
        "land_descend_vz_ms": 0.85,
        "land_auto_land_alt_z_m": 3.0,
        "land_phase_timeout_s": 180.0,
        "landed_pos_z_max_m": 0.4,
        "landed_total_speed_max_ms": 0.6,
        # Po stabilnym touchdown: czekaj → DISARM → czekaj → koniec run (symulacja).
        "land_after_touchdown_s": 2.0,
        "land_after_disarm_s": 2.0,
        "phase_times": phase_times,
        "phase_times_description": (
            "cruise: PD w z↑ Pegasus + małe wahanie celu (wobble_frac); "
            "hamowanie: snapshot wysokości, |v_xy|→0; potem PX4 AUTO LAND + NAV_LAND w miejscu. phase_times: szacunek."
        ),
        "wind_dynamic_enabled": True,
    }
    return merge_wind_defaults(base)


def draw_scenario_3() -> dict:
    """
    Scenariusz 3: wszystkie 11 typów manewrów (każdy co najmniej raz), krótkie segmenty 10–20 s
    (arc_lagodny do 25 s), łącznie ~2–2.5 min misji + lądowanie jak scen. 2.
    Lot w obrębie ~2 km×2 km: prędkości 5–18 m/s, skrócone łuki.
    """
    _ALL_TYPES = [
        "sprint", "brake", "climb_sprint", "descent_sprint",
        "arc_lagodny", "arc_sredni", "arc_ostry", "sharp_reversal",
        "slalom_lagodny", "slalom_sredni", "slalom_ostry",
        "zawrot",
    ]

    def _cv(v: float) -> float:
        return max(5.0, min(30.0, v))

    def _cz(z: float) -> float:
        return max(10.0, min(40.0, z))

    def _arc_dur(angle_deg: float, radius_m: float, v: float, max_s: float) -> tuple[float, float]:
        """Zwraca (duration_s, angle_deg) — jeśli byłoby za długo, skraca kąt."""
        dur = math.radians(angle_deg) * radius_m / max(v, 1.0)
        if dur > max_s:
            angle_deg = math.degrees(max_s * max(v, 1.0) / radius_m)
            dur = max_s
        return max(8.0, dur), max(30.0, angle_deg)

    v_current = _r(10.0, 22.0)
    z_current = _r(15.0, 28.0)
    heading_start = _r(0.0, 360.0)

    # Buduj listę typów: wszystkie 12 obowiązkowo + 2× lot_jednostajny, w losowej kolejności.
    # Sprint zawsze pierwszy; reszta przetasowana.
    mandatory = list(_ALL_TYPES) + ["lot_jednostajny", "lot_jednostajny"]
    mandatory.remove("sprint")
    random.shuffle(mandatory)
    type_sequence = ["sprint"] + mandatory

    # Wstaw wymuszone sprinty (recovery) po sharp_reversal i arc_ostry.
    # Nie możemy jeszcze wiedzieć które to będą, więc zrobimy post-processing.
    # Buduj segmenty iteracyjnie.
    segments: list[dict] = []
    remaining = list(type_sequence)  # kolejka typów do wykonania

    while remaining:
        seg_type = remaining.pop(0)

        # Jeśli poprzedni był sharp_reversal, arc_ostry lub zawrot, a obecny nie jest sprint —
        # wstaw sprint przed nim.
        if segments and segments[-1]["type"] in ("sharp_reversal", "arc_ostry", "zawrot") and seg_type != "sprint":
            remaining.insert(0, seg_type)
            seg_type = "sprint"

        v_start = v_current
        radius_m = 0.0
        turn_dir = 0
        n_halfturns = 1
        angle_deg = 0.0
        a_c = 0.0

        if seg_type == "sprint":
            v_end = _cv(v_start + _r(3.0, 10.0))
            z_end = _cz(z_current + _r(0.0, 5.0))
            duration_s = _r(2.0, 10.0)
            accel_ramp_s = duration_s * 0.65
            phase_label = "przyspieszanie"

        elif seg_type == "brake":
            # hamowanie blisko zera — v_end z szerokiego zakresu, ale docelowo niska prędkość
            v_end = _cv(v_start - _r(v_start * 0.5, v_start * 0.95))
            z_end = _cz(z_current - _r(0.0, 4.0))
            duration_s = _r(3.0, 6.0)
            accel_ramp_s = duration_s * 0.85
            phase_label = "hamowanie"

        elif seg_type == "climb_sprint":
            v_end = _cv(v_start * _r(0.9, 1.1))
            dz = _r(8.0, 20.0)
            z_end = _cz(z_current + dz)
            duration_s = _r(2.0, 8.0)
            accel_ramp_s = duration_s
            phase_label = "wznoszenie"

        elif seg_type == "descent_sprint":
            v_end = _cv(v_start * _r(0.9, 1.1))
            # Gwałtowne opadanie: duża różnica wysokości w krótkim czasie
            dz = _r(8.0, 20.0)
            z_end = _cz(z_current - dz)
            duration_s = _r(2.0, 8.0)
            accel_ramp_s = duration_s
            phase_label = "opadanie"

        elif seg_type == "arc_lagodny":
            radius_m = _r(60.0, 150.0)
            v_end = _cv(v_start * _r(0.95, 1.05))
            angle_deg = _r(60.0, 180.0)
            turn_dir = random.choice([-1, 1])
            duration_s, angle_deg = _arc_dur(angle_deg, radius_m, v_start, 25.0)
            accel_ramp_s = duration_s
            z_end = _cz(z_current + _r(2.0, 4.0) * random.choice([-1, 1]))
            a_c = v_start ** 2 / radius_m
            phase_label = "zakret_lagodny"

        elif seg_type == "arc_sredni":
            radius_m = _r(20.0, 60.0)
            v_entry = _cv(v_start * 0.85)
            v_end = v_entry
            angle_deg = _r(60.0, 150.0)
            turn_dir = random.choice([-1, 1])
            duration_s, angle_deg = _arc_dur(angle_deg, radius_m, v_entry, 20.0)
            accel_ramp_s = duration_s * 0.3
            z_end = _cz(z_current + _r(2.0, 6.0))
            a_c = v_entry ** 2 / radius_m
            phase_label = "zakret_sredni" if a_c < 5.0 else "zakret_ostry"

        elif seg_type == "arc_ostry":
            radius_m = _r(8.0, 22.0)
            v_entry = _cv(v_start * 0.75)
            v_end = v_entry
            angle_deg = _r(60.0, 120.0)
            turn_dir = random.choice([-1, 1])
            duration_s, angle_deg = _arc_dur(angle_deg, radius_m, v_entry, 18.0)
            accel_ramp_s = duration_s * 0.2
            z_end = _cz(z_current + _r(3.0, 8.0))
            a_c = v_entry ** 2 / radius_m
            phase_label = "zakret_ostry" if a_c < 15.0 else "zakret_gwaltowny"

        elif seg_type == "sharp_reversal":
            radius_m = _r(3.0, 10.0)
            v_entry = _cv(v_start * 0.5)
            v_end = v_entry
            angle_deg = _r(120.0, 180.0)
            turn_dir = random.choice([-1, 1])
            duration_s, angle_deg = _arc_dur(angle_deg, radius_m, v_entry, 16.0)
            accel_ramp_s = duration_s * 0.5
            z_end = _cz(z_current + _r(4.0, 10.0))
            a_c = v_entry ** 2 / radius_m
            phase_label = "zakret_gwaltowny"

        elif seg_type == "zawrot":
            # Zawrót (U-turn) >150°: zwinny, szybki, mały promień.
            # Promień 4-8m + wysoka prędkość → czas trwania 2-4 s symulacyjne.
            radius_m = _r(4.0, 8.0)
            v_entry = _cv(v_start * _r(0.6, 0.85))
            v_end = v_entry
            angle_deg = _r(160.0, 200.0)
            turn_dir = random.choice([-1, 1])
            duration_s = max(2.0, min(4.0, math.radians(angle_deg) * radius_m / max(v_entry, 1.0)))
            accel_ramp_s = duration_s * 0.3
            z_end = _cz(z_current + _r(2.0, 6.0))
            a_c = v_entry ** 2 / radius_m
            phase_label = "zawrot"

        elif seg_type == "slalom_lagodny":
            # Łagodne wahnięcia L-R; budżet 10-18s, min 10 half-turnów (5 cykli)
            # Kąt max 25° na wahnięcie — widoczne ale nie skręca z trasy
            radius_m = _r(25.0, 60.0)
            v_end = _cv(v_start)
            omega = v_start / max(radius_m, 0.1)
            max_ht_s = math.radians(30.0) / max(omega, 0.01)
            half_turn_s = min(_r(0.8, 2.0), max_ht_s)
            half_turn_s = max(0.4, half_turn_s)
            budget_s = _r(10.0, 18.0)
            n_halfturns = max(10, int(budget_s / half_turn_s))
            if n_halfturns % 2 != 0:
                n_halfturns += 1
            duration_s = n_halfturns * half_turn_s
            angle_deg = math.degrees(omega * half_turn_s)
            turn_dir = random.choice([-1, 1])
            accel_ramp_s = duration_s
            z_end = _cz(z_current + _r(1.0, 3.0) * random.choice([-1, 1]))
            a_c = v_start ** 2 / radius_m
            phase_label = "slalom_lagodny"

        elif seg_type == "slalom_sredni":
            # Wyraźne wahnięcia; budżet 8-16s, min 10 half-turnów; kąt max 55°
            radius_m = _r(10.0, 28.0)
            v_end = _cv(v_start * _r(0.88, 1.05))
            omega = v_start / max(radius_m, 0.1)
            max_ht_s = math.radians(66.0) / max(omega, 0.01)
            half_turn_s = min(_r(0.5, 1.5), max_ht_s)
            half_turn_s = max(0.3, half_turn_s)
            budget_s = _r(8.0, 16.0)
            n_halfturns = max(10, int(budget_s / half_turn_s))
            if n_halfturns % 2 != 0:
                n_halfturns += 1
            duration_s = n_halfturns * half_turn_s
            angle_deg = math.degrees(omega * half_turn_s)
            turn_dir = random.choice([-1, 1])
            accel_ramp_s = duration_s * 0.5
            z_end = _cz(z_current + _r(2.0, 5.0) * random.choice([-1, 1]))
            a_c = v_start ** 2 / radius_m
            phase_label = "slalom_sredni"

        elif seg_type == "slalom_ostry":
            # Agresywne wahnięcia; budżet 6-14s, min 10 half-turnów; kąt 56–80° na wahnięcie
            # Losuję kąt i czas wahnięcia → R = v / omega (gwarantuje zakres kąta)
            v_end = _cv(v_start * 0.8)
            angle_ht_rad = _r(math.radians(67.2), math.radians(96.0))
            half_turn_s = _r(0.4, 1.2)
            omega = angle_ht_rad / half_turn_s
            radius_m = max(2.5, v_end / max(omega, 0.1))
            budget_s = _r(6.0, 14.0)
            n_halfturns = max(10, int(budget_s / half_turn_s))
            if n_halfturns % 2 != 0:
                n_halfturns += 1
            duration_s = n_halfturns * half_turn_s
            angle_deg = math.degrees(omega * half_turn_s)
            turn_dir = random.choice([-1, 1])
            accel_ramp_s = duration_s * 0.3
            z_end = _cz(z_current + _r(2.0, 6.0) * random.choice([-1, 1]))
            a_c = v_end ** 2 / radius_m
            phase_label = "slalom_ostry"

        elif seg_type == "lot_jednostajny":
            # Lot jednostajny ~15 s: stała prędkość (±5%), prosta trasa, lekka zmiana wysokości.
            v_end = _cv(v_start * _r(0.95, 1.05))
            z_end = _cz(z_current + _r(-3.0, 3.0))
            duration_s = _r(12.0, 18.0)
            accel_ramp_s = 2.0
            phase_label = "lot_jednostajny"

        else:
            v_end = v_start
            z_end = z_current
            duration_s = 10.0
            accel_ramp_s = 10.0
            phase_label = "lot_prosty"

        seg: dict[str, Any] = {
            "type": seg_type,
            "v_start_ms": round(v_start, 3),
            "v_end_ms": round(v_end, 3),
            "accel_ramp_s": round(accel_ramp_s, 3),
            "duration_s": round(duration_s, 3),
            "z_start_m": round(z_current, 3),
            "z_end_m": round(z_end, 3),
            "radius_m": round(radius_m, 3),
            "angle_deg": round(angle_deg, 3),
            "turn_dir": int(turn_dir),
            "n_halfturns": int(n_halfturns),
            "phase_label": phase_label,
            "a_c_ms2": round(a_c, 4),
        }
        segments.append(seg)
        v_current = v_end
        z_current = z_end

    t_acc = 0.0
    phase_times: list[dict] = []
    for seg in segments:
        phase_times.append({
            "phase": seg["phase_label"],
            "t_start_s": round(t_acc, 3),
            "t_end_s": round(t_acc + seg["duration_s"], 3),
        })
        t_acc += seg["duration_s"]

    base: dict[str, Any] = {
        "scenario_id": 3,
        "scenario_name": "zakręty_slalomy_zmiana_v_i_h",
        "yaw_start_deg": round(heading_start, 3),
        "altitude_initial_m": round(segments[0]["z_start_m"], 3),
        "v_initial_ms": round(segments[0]["v_start_ms"], 3),
        "wind_speed_ms": _draw_wind_speed_scenario_1(),
        "wind_dir_deg": _r(0.0, 360.0),
        "wind_dynamic_enabled": True,
        "wind_gust_prob_per_s": 0.030,   # wyraźnie wyższy niż default 0.018 → ~3-4 podmuchy/2 min
        "s3_z_kp": 1.35,
        "s3_z_kd": 0.7,
        "s3_z_vz_cap_ms": 4.0,
        "s3_z_kp_boost": 1.9,
        "s3_z_kp_boost_s": 8.0,
        # Lądowanie po segmentach (jak scen. 2)
        "s3_land_brake_s": 2.0,
        "s3_land_brake_z_kp": 1.45,
        "s3_land_brake_z_kd": 0.65,
        "s3_land_brake_z_vz_cap_ms": 2.5,
        "s3_land_z_max_m": 0.4,
        "s3_land_v_max_ms": 0.6,
        "s3_land_timeout_s": 120.0,
        "s3_land_after_touchdown_s": 2.0,
        "s3_land_after_disarm_s": 2.0,
        "segments": segments,
        "phase_times": phase_times,
        "phase_times_description": (
            "Jeden wpis na segment + faza ladowanie_px4 na końcu. "
            "t_start/t_end szacowane. Wszystkie 12 typów manewrów wystąpią przynajmniej raz."
        ),
    }
    return merge_wind_defaults(base)


def draw_scenario_4() -> dict:
    # Snippet: Zawis (hover) z zakłóceniami wiatru. Utrzymanie pozycji, wiatr zmienia się w czasie.
    return _stub_scenario(4, "zawis_zaklocenia_wiatru", "zawis przy wietrze początkowym, potem skok wiatru")


def draw_scenario_5() -> dict:
    # Snippet: Ósemki / slalomy. Lot po torze: ósemka lub slalom lewo-prawo.
    return _stub_scenario(5, "osemki_slalomy", "ósemka lub slalom, fazy zmiana_kierunku")


def draw_scenario_6() -> dict:
    # Snippet: Wznoszenie/opadanie pod kątem (200 m w poziomie). Lot 200 m z jednoczesnym wznoszeniem/opadaniem.
    return _stub_scenario(6, "wznoszenie_opadanie_200m", "lot po ślizgu (path_angle), 200 m")


def draw_scenario_7() -> dict:
    # Snippet: Staccato — krótkie impulsy przyspieszenia i hamowania.
    return _stub_scenario(7, "staccato_impulsy", "naprzemienne impuls_gaz / impuls_hamuj")


def draw_scenario_8() -> dict:
    # Snippet: Lot z jednym silnym podmuchem wiatru (step). Wiatr skacze z niskiego na wysoki w ustalonym momencie.
    return _stub_scenario(8, "wiatr_step", "lot prosty → wiatr_step (skok wiatru)")


def draw_scenario_9() -> dict:
    # Snippet: Zmienna wysokość (fale). Lot 200 m z sinusoidalną zmianą wysokości.
    return _stub_scenario(9, "fale_wysokosc_200m", "lot z sinusoidalną zmianą wysokości")


def draw_scenario_10() -> dict:
    # Snippet: Mix faz — start → przyspieszanie → prosty → zakręt → zawis → podejście do lądowania (stop na 2 m).
    return _stub_scenario(10, "mix_faz", "start, przyspieszanie, lot_prosty, zakręt, zawis, podejście")


DRAW_FUNCTIONS = {
    1: draw_scenario_1,
    2: draw_scenario_2,
    3: draw_scenario_3,
    4: draw_scenario_4,
    5: draw_scenario_5,
    6: draw_scenario_6,
    7: draw_scenario_7,
    8: draw_scenario_8,
    9: draw_scenario_9,
    10: draw_scenario_10,
}


def draw_scenario(scenario_id: int, seed: int | None = None) -> dict:
    """Losuje parametry dla danego scenariusza. Opcjonalnie seed dla powtarzalności.

    Przy ``seed is None`` ustawiamy entropię z ``os.urandom`` i **przywracamy** stan globalnego
    ``random`` po losowaniu — inaczej Isaac/Omniverse mógł wcześniej ustawić stały seed i każdy
    run bez ``--seed`` dawałby te same parametry (np. zawsze to samo ``v_cmd_ms``).
    """
    if scenario_id not in DRAW_FUNCTIONS:
        raise ValueError(f"Nieznany scenario_id={scenario_id}. Dostępne: 1..10")
    rng_state = random.getstate()
    try:
        if seed is not None:
            random.seed(seed)
        else:
            random.seed(int.from_bytes(os.urandom(8), "little"))
        return merge_wind_defaults(DRAW_FUNCTIONS[scenario_id]())
    finally:
        random.setstate(rng_state)


def get_scenario_description(scenario_id: int) -> str:
    """Zwraca opis scenariusza do zapisu w pliku metadanych."""
    return SCENARIO_DESCRIPTIONS.get(scenario_id, f"Scenariusz {scenario_id}")
