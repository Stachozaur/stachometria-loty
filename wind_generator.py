# -*- coding: utf-8 -*-
"""
Generator prędkości wiatru w NED dla Stachometra: średnia (meteo) + turbulencja OU + podmuchy 1−cos (+ lulls).

Obwiednia podmuchu: dwa odcinki cosinusowe (narastanie do szczytu w t_peak, potem opadanie do zera).
Losowe podmuchy: albo jawne czasy narastania/opadania + zakres amplitudy A, albo (legacy) losowe T ∈ [T_min, T_max]
oraz losowy moment szczytu (peak_frac ∈ [peak_frac_min, peak_frac_max]).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any


def wind_mean_ned_from_meteo(wind_speed_ms: float, wind_dir_deg: float) -> tuple[float, float, float]:
    """Kierunek meteo: z którego wieje (0° = N). Wektor prędkości powietrza w NED (poziomo)."""
    rad = math.radians(wind_dir_deg)
    ux = -math.cos(rad)
    uy = -math.sin(rad)
    return (wind_speed_ms * ux, wind_speed_ms * uy, 0.0)


def horizontal_unit(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    h = math.hypot(vec[0], vec[1])
    if h < 1e-9:
        return (1.0, 0.0, 0.0)
    return (vec[0] / h, vec[1] / h, 0.0)


def _clamp_t_peak_s(T: float, t_peak_s: float) -> float:
    """Szczyt obwiedni w (0, T), żeby uniknąć dzielenia przez zero w cosinusach."""
    T = max(T, 1e-6)
    eps = min(0.05 * T, 0.25)  # co najmniej ~5% lub 250 ms od końców
    eps = max(eps, 1e-4)
    if T <= 2.0 * eps:
        return 0.5 * T
    return max(eps, min(T - eps, t_peak_s))


def gust_envelope(t_loc: float, T: float, t_peak_s: float) -> float:
    """
    Obwiednia 0…1: 0 w t=0, 1 w t=t_peak_s, 0 w t=T; gładkie połączenie w szczycie.
    """
    if t_loc <= 0.0 or t_loc >= T:
        return 0.0
    tp = _clamp_t_peak_s(T, t_peak_s)
    if t_loc <= tp:
        return 0.5 * (1.0 - math.cos(math.pi * t_loc / tp))
    tf = T - tp
    if tf <= 1e-12:
        return 0.0
    s = (t_loc - tp) / tf
    return 0.5 * (1.0 + math.cos(math.pi * s))


@dataclass
class _ActiveGust:
    t_start: float
    duration_s: float
    """Czas od t_start, w którym obwiednia osiąga 1 (szczyt |dodatku|)."""
    t_peak_s: float
    """Dodatkowa prędkość w m/s w szczycie podmuchu (wektor, zwykle równoległy do średniego wiatru)."""
    peak_vec_ned: tuple[float, float, float]
    from_schedule: bool = False
    is_lull: bool = False


class ConstantWindGenerator:
    """Stały wektor wiatru (jak wcześniej w Stachometrze, ale jako prędkość NED)."""

    def __init__(self, wind_speed_ms: float, wind_dir_deg: float) -> None:
        self._w = wind_mean_ned_from_meteo(wind_speed_ms, wind_dir_deg)

    def step(self, dt: float, sim_time_s: float) -> tuple[tuple[float, float, float], str]:
        del dt, sim_time_s
        self.step_extras = {
            "wind_gust_from_schedule": 0,
            "wind_gust_rising": 0,
            "wind_gust_is_lull": 0,
        }
        return (self._w, "")


class DynamicWindGenerator:
    """
    Średnia + OU(NE) + losowe podmuchy 1−cos (opcjonalnie lulls: ujemna amplituda wzdłuż średniego).
    """

    def __init__(self, params: dict[str, Any], rng: random.Random) -> None:
        self._rng = rng
        speed = float(params.get("wind_speed_ms", 0.0) or 0.0)
        direction = float(params.get("wind_dir_deg", 0.0) or 0.0)
        self._mean = wind_mean_ned_from_meteo(speed, direction)
        self._u_mean = horizontal_unit(self._mean)

        self._tau_s = max(1e-6, float(params.get("wind_tau_s", 0.3)))
        self._sigma_ms = max(0.0, float(params.get("wind_sigma_ms", 0.4)))

        self._gust_prob_per_s = max(0.0, float(params.get("wind_gust_prob_per_s", 0.02)))
        self._gust_T_min = max(1e-3, float(params.get("wind_gust_T_min_s", 2.0)))
        self._gust_T_max = max(self._gust_T_min, float(params.get("wind_gust_T_max_s", 15.0)))
        # Amplituda wzdłuż średniego: U(A_min, A_max) lub pojedyncze wind_gust_A_rel_ms (min=max).
        a_single = float(params.get("wind_gust_A_rel_ms", 2.0))
        a_min_p = params.get("wind_gust_A_rel_min_ms")
        a_max_p = params.get("wind_gust_A_rel_max_ms")
        if a_min_p is not None or a_max_p is not None:
            self._gust_A_min = max(0.0, float(a_min_p if a_min_p is not None else a_single))
            self._gust_A_max = max(self._gust_A_min, float(a_max_p if a_max_p is not None else a_single))
        else:
            self._gust_A_min = self._gust_A_max = max(0.0, a_single)
        self._peak_frac_min = float(params.get("wind_gust_peak_frac_min", 0.1))
        self._peak_frac_max = float(params.get("wind_gust_peak_frac_max", 0.9))
        if self._peak_frac_max < self._peak_frac_min:
            self._peak_frac_min, self._peak_frac_max = self._peak_frac_max, self._peak_frac_min
        self._peak_frac_min = max(0.02, min(0.98, self._peak_frac_min))
        self._peak_frac_max = max(self._peak_frac_min, min(0.98, self._peak_frac_max))
        # Jawny czas narastania do szczytu (sekundy symulacji); jeśli oba skonfigurowane → losowe podmuchy ignorują peak_frac×T.
        gr0 = params.get("wind_gust_rise_time_min_s")
        gr1 = params.get("wind_gust_rise_time_max_s")
        self._gust_rise_mode = gr0 is not None and gr1 is not None
        if self._gust_rise_mode:
            self._gust_rise_min = max(1e-3, float(gr0))
            self._gust_rise_max = max(self._gust_rise_min, float(gr1))
            gf0 = params.get("wind_gust_fall_time_min_s", gr0)
            gf1 = params.get("wind_gust_fall_time_max_s", gr1)
            self._gust_fall_min = max(1e-3, float(gf0))
            self._gust_fall_max = max(self._gust_fall_min, float(gf1))
        else:
            self._gust_rise_min = self._gust_rise_max = 0.0
            self._gust_fall_min = self._gust_fall_max = 0.0
        self._lull_enabled = bool(params.get("wind_gust_lull_enabled", True))
        self._lull_prob = float(params.get("wind_gust_lull_prob", 0.35))  # część podmuchów to lull

        self._ou_n = 0.0
        self._ou_e = 0.0
        self._active: _ActiveGust | None = None
        # Harmonogram: lista {"t_start_s", "duration_s", "A_rel_ms"} — jeśli niepusta, **tylko** te podmuchy (bez losowych).
        raw_sched = params.get("wind_gust_schedule")
        self._schedule: list[dict[str, Any]] = []
        if isinstance(raw_sched, list) and len(raw_sched) > 0:
            for ev in raw_sched:
                if not isinstance(ev, dict):
                    continue
                row: dict[str, Any] = {
                    "t_start_s": float(ev.get("t_start_s", 0.0)),
                    "duration_s": max(1e-3, float(ev.get("duration_s", 2.0))),
                    "A_rel_ms": float(ev.get("A_rel_ms", 2.0)),
                }
                if "peak_fraction" in ev and ev["peak_fraction"] is not None:
                    row["peak_fraction"] = float(ev["peak_fraction"])
                self._schedule.append(row)
            self._schedule.sort(key=lambda x: x["t_start_s"])
        self._sched_queue: list[dict[str, Any]] = list(self._schedule)
        self._random_gusts_enabled = len(self._schedule) == 0

    def _ou_step(self, dt: float) -> None:
        if self._sigma_ms <= 0.0 or dt <= 0.0:
            return
        tau = self._tau_s
        phi = math.exp(-dt / tau)
        # Stacjonarna wariancja sigma^2 dla OU dx = -x/tau dt + sigma*sqrt(2/tau) dW: Var = sigma^2
        scale = self._sigma_ms * math.sqrt(max(0.0, 1.0 - phi * phi))
        self._ou_n = phi * self._ou_n + scale * self._rng.gauss(0.0, 1.0)
        self._ou_e = phi * self._ou_e + scale * self._rng.gauss(0.0, 1.0)

    def _try_spawn_scheduled(self, sim_time_s: float) -> None:
        """Kolejka: start dopiero gdy brak aktywnego podmuchu; zdarzenia z t_start w przeszłości wystartują po kolei."""
        if self._active is not None or not self._sched_queue:
            return
        ev = self._sched_queue[0]
        if sim_time_s + 1e-12 < ev["t_start_s"]:
            return
        self._sched_queue.pop(0)
        A = ev["A_rel_ms"]
        T = ev["duration_s"]
        t_nom = float(ev["t_start_s"])
        # Jeśli czekaliśmy w kolejce po wcześniejszym podmuchu, pełne okno 1−cos od teraz (nie od t_nom — inaczej t_loc > T od razu).
        t_start_eff = sim_time_s if sim_time_s > t_nom + 1e-9 else t_nom
        peak_vec = (A * self._u_mean[0], A * self._u_mean[1], 0.0)
        if "peak_fraction" in ev:
            t_peak = _clamp_t_peak_s(T, float(ev["peak_fraction"]) * T)
        else:
            t_peak = _clamp_t_peak_s(T, 0.5 * T)
        self._active = _ActiveGust(
            t_start=t_start_eff,
            duration_s=T,
            t_peak_s=t_peak,
            peak_vec_ned=peak_vec,
            from_schedule=True,
            is_lull=A < 0.0,
        )

    def _try_spawn_gust(self, dt: float) -> None:
        if self._active is not None or not self._random_gusts_enabled:
            return
        if self._gust_prob_per_s <= 0.0 or self._gust_A_max <= 0.0:
            return
        p = 1.0 - math.exp(-self._gust_prob_per_s * max(dt, 0.0))
        if self._rng.random() >= p:
            return
        if self._gust_rise_mode:
            t_peak = self._rng.uniform(self._gust_rise_min, self._gust_rise_max)
            t_fall = self._rng.uniform(self._gust_fall_min, self._gust_fall_max)
            T = t_peak + t_fall
            T = max(self._gust_T_min, min(self._gust_T_max, T))
            # Zachowaj t_peak ≤ T−eps (clamp z _clamp_t_peak_s).
            t_peak = _clamp_t_peak_s(T, t_peak)
        else:
            T = self._rng.uniform(self._gust_T_min, self._gust_T_max)
            pf = self._rng.uniform(self._peak_frac_min, self._peak_frac_max)
            t_peak = _clamp_t_peak_s(T, pf * T)
        A = self._rng.uniform(self._gust_A_min, self._gust_A_max)
        if self._lull_enabled and self._rng.random() < self._lull_prob:
            A = -abs(A)
        peak_vec = (A * self._u_mean[0], A * self._u_mean[1], 0.0)
        self._active = _ActiveGust(
            t_start=0.0,
            duration_s=T,
            t_peak_s=t_peak,
            peak_vec_ned=peak_vec,
            from_schedule=False,
            is_lull=A < 0.0,
        )  # t_start ustawimy w step

    def step(self, dt: float, sim_time_s: float) -> tuple[tuple[float, float, float], str]:
        self._ou_step(dt)

        gust_add = (0.0, 0.0, 0.0)
        phase_str = ""
        self.step_extras = {
            "wind_gust_from_schedule": 0,
            "wind_gust_rising": 0,
            "wind_gust_is_lull": 0,
        }

        if self._active is None:
            self._try_spawn_scheduled(sim_time_s)
        if self._active is None:
            self._try_spawn_gust(dt)
            if self._active is not None and not self._active.from_schedule:
                self._active.t_start = sim_time_s

        if self._active is not None:
            g = self._active
            t_loc = sim_time_s - g.t_start
            if t_loc >= g.duration_s or t_loc < 0.0:
                self._active = None
            else:
                phase = t_loc / g.duration_s
                env = gust_envelope(t_loc, g.duration_s, g.t_peak_s)
                gust_add = (
                    g.peak_vec_ned[0] * env,
                    g.peak_vec_ned[1] * env,
                    0.0,
                )
                phase_str = f"{phase:.6f}"
                self.step_extras["wind_gust_from_schedule"] = 1 if g.from_schedule else 0
                self.step_extras["wind_gust_rising"] = 1 if t_loc + 1e-12 < g.t_peak_s else 0
                self.step_extras["wind_gust_is_lull"] = 1 if g.is_lull else 0

        wn = self._mean[0] + self._ou_n + gust_add[0]
        we = self._mean[1] + self._ou_e + gust_add[1]
        # Po lull: nie pozwól, żeby składowa wzdłuż średniego wiatru była ujemna (brak „wiatru wstecz”)
        along = wn * self._u_mean[0] + we * self._u_mean[1]
        if along < 0.0:
            wn -= along * self._u_mean[0]
            we -= along * self._u_mean[1]

        return ((wn, we, 0.0), phase_str)


def make_wind_generator(params: dict[str, Any], seed: int | None, run_id: int = 0):
    """
    Zwraca ConstantWindGenerator lub DynamicWindGenerator wg params.
    seed: z CLI --seed; łączony z run_id dla powtarzalności batchy.
    """
    wind_on = float(params.get("wind_speed_ms", 0) or 0) > 0.0
    if not wind_on:
        return None

    s = (int(seed) if seed is not None else 0) * 1000003 + int(run_id)
    rng = random.Random(s)

    if bool(params.get("wind_dynamic_enabled", True)):
        return DynamicWindGenerator(params, rng)

    speed = float(params.get("wind_speed_ms", 0.0))
    direction = float(params.get("wind_dir_deg", 0.0))
    return ConstantWindGenerator(speed, direction)
