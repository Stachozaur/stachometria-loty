# -*- coding: utf-8 -*-
"""
Parametry scenariuszy Stachometr (docs/py/scenarios.md).
Dla każdego scenariusza: losowanie parametrów z podanych zakresów i zapis do JSON.
"""
from __future__ import annotations

import math
import random
from typing import Any

# --- Etykiety faz do klasyfikacji (zgodnie z sekcją "Etykiety faz z definicji scenariusza" w scenarios.md) ---
# Każdy run musi zapisywać phase_times: lista przedziałów [t_start_s, t_end_s] z przypisaną klasą fazy.
# Użycie: wejście = okno czasowe (np. 100–200 ms) z IMU/FC; etykieta = klasa fazy (samoetykietowanie z planu misji).

PHASE_LABELS_DOC = (
    "phase_times: wektor faz z przedziałami [t_start_s, t_end_s] i klasą fazy. "
    "W każdym runie na początku są fazy wspólne: rozgrzewka (0–3 s), wznoszenie (3–8 s); od 8 s zaczyna się misja (fazy scenariusza). "
    "Etykiety do klasyfikacji: rozgrzewka, wznoszenie, hamowanie, przyspieszanie, zawis, lot_prosty, zakręt, wiatr_step, start, podejście, itd. "
    "Zgodnie z sekcją 'Etykiety faz z definicji scenariusza (samoetykietowanie)' w docs/py/scenarios.md."
)

# --- Scenariusze (id, opis, funkcja losująca zwracająca dict do zapisu) ---

SCENARIO_DESCRIPTIONS = {
    1: "Lot prosto 500–1000 m (losowy dystans), wysokość modulowana w locie (±5%); wiatr ~75% U(2,4) m/s, ~25% U(0.01,9) (ogon). Każdy run z wiatrem.",
    2: "Jak scenariusz 1 (±5% wys.): lot do celu 500–1000 m (yaw), wycentrowanie XY, zejście pionowe, PX4 AUTO LAND. Wiatr jak w scenariuszu 1.",
    3: "Jedno przyspieszanie 0 → V_max na 200 m. Start z zatrzymania, przyspieszanie do V_max.",
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
    # OU: umiarkowana zmienność (bez „szalonego” skoku); większe tau = wolniejszy dryf kierunku
    "wind_tau_s": 0.55,
    "wind_sigma_ms": 0.28,
    "wind_gust_prob_per_s": 0.018,
    "wind_gust_T_min_s": 2.0,
    "wind_gust_T_max_s": 15.0,
    # Amplituda podmuchu wzdłuż średniego — zmniejszona vs 2.0, żeby nie skakało o kilka m/s naraz
    "wind_gust_A_rel_ms": 1.15,
    # Losowy moment szczytu podmuchu: peak_frac * T, peak_frac ∈ [min, max] (losowe Poisson).
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
    "wind_sigma_ms": 0.26,
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
    v_cmd = _r(5.0, 25.0)
    distance_m = _r(500.0, 1000.0)
    altitude_m = _r(10.0, 50.0)
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
    Scenariusz 2: bliźniak scenariusza 1 — ten sam lot (v, yaw, dystans, wysokość, wiatr),
    ale z fazą podejścia do wylosowanego punktu docelowego i zakończeniem przez PX4 AUTO LAND.
    Punkt celu: spawn (0,0) + distance_m w azymucie yaw (płaszczyzna XY).
    """
    v_cmd = _r(5.0, 25.0)
    distance_m = _r(500.0, 1000.0)
    altitude_m = _r(10.0, 50.0)
    yaw_deg = _r(0.0, 360.0)
    t_cruise = distance_m / v_cmd if v_cmd > 0 else 60.0
    period_s = max(18.0, min(100.0, t_cruise / 2.2))
    # Szacunek faz misji (etykiety; faktyczny czas zależy od PX4 / wiatru)
    t_align_s = 55.0
    t_descend_s = 90.0
    t_land_s = 120.0
    phase_times = [
        {"phase": "lot_prosty", "t_start_s": 0.0, "t_end_s": round(t_cruise, 2)},
        {
            "phase": "wyrownanie_xy",
            "t_start_s": round(t_cruise, 2),
            "t_end_s": round(t_cruise + t_align_s, 2),
        },
        {
            "phase": "schodzenie_pionowe",
            "t_start_s": round(t_cruise + t_align_s, 2),
            "t_end_s": round(t_cruise + t_align_s + t_descend_s, 2),
        },
        {
            "phase": "ladowanie",
            "t_start_s": round(t_cruise + t_align_s + t_descend_s, 2),
            "t_end_s": round(t_cruise + t_align_s + t_descend_s + t_land_s, 2),
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
        "altitude_variation_frac": 0.05,
        "altitude_variation_period_s": round(period_s, 2),
        # Cel w płaszczyźnie XY względem spawnu (0,0) — zgodnie z run_stachometr (spawn_xy)
        "target_xy_offset_m": [round(distance_m * math.cos(yrad), 3), round(distance_m * math.sin(yrad), 3)],
        "land_approach_start_m": 35.0,
        "land_approach_v_max_ms": 4.0,
        "land_xy_align_m": 4.0,
        "land_descend_vz_ms": 0.85,
        "land_auto_land_alt_z_m": 3.0,
        "land_descend_xy_drift_max_m": 7.0,
        "land_align_timeout_s": 95.0,
        "land_phase_timeout_s": 180.0,
        "landed_pos_z_max_m": 0.4,
        "landed_total_speed_max_ms": 0.6,
        "phase_times": phase_times,
        "phase_times_description": (
            "lot_prosty (±5 % wys.) → wyrownanie_xy nad punktem → schodzenie_pionowe (vz w dół NED) → "
            "ladowanie (AUTO LAND gdy z ≤ land_auto_land_alt_z_m). Czas faz w phase_times: szacunek."
        ),
        "wind_dynamic_enabled": True,
    }
    return merge_wind_defaults(base)


def draw_scenario_3() -> dict:
    # Snippet: Jedno przyspieszanie 0 → V_max na 200 m. Start z zatrzymania, przyspieszanie do V_max.
    return _stub_scenario(3, "przyspieszanie_0_Vmax_200m", "przyspieszanie od 0 do V_max na 200 m")


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
    """Losuje parametry dla danego scenariusza. Opcjonalnie seed dla powtarzalności."""
    if seed is not None:
        random.seed(seed)
    if scenario_id not in DRAW_FUNCTIONS:
        raise ValueError(f"Nieznany scenario_id={scenario_id}. Dostępne: 1..10")
    return merge_wind_defaults(DRAW_FUNCTIONS[scenario_id]())


def get_scenario_description(scenario_id: int) -> str:
    """Zwraca opis scenariusza do zapisu w pliku metadanych."""
    return SCENARIO_DESCRIPTIONS.get(scenario_id, f"Scenariusz {scenario_id}")
