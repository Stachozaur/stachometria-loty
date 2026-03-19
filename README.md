# Stachometr – scenariusze lotów (Pegasus IRIS + PX4 + Isaac Sim)

Pipeline do zbierania danych zgodnie z **docs/py/scenarios.md**: uruchamianie scenariuszy w Isaac Sim z dronem Pegasus IRIS i PX4, zapis parametrów losowych i logów.

## Wymagania

- Isaac Sim 5.1 (standalone)
- Pegasus Simulator (rozszerzenie / pakiet `pegasus.simulator`) z konfiguracją PX4 (`px4_dir`, `px4_default_airframe` w configu Pegasusa)
- PX4-Autopilot (SITL)

## Co jest zapisywane dla każdego runu

1. **Plik parametrów (JSON)**  
   `run_s<scenario_id>_<timestamp>_r<run_id>_params.json`  
   Zawiera:
   - `run_start_time` – znacznik czasu startu runu
   - `scenario_id`, `scenario_description` – numer i opis scenariusza
   - `params_description` – informacja, że wartości są wylosowane z zakresów z docs/py/scenarios.md
   - **`phase_labels_description`** – opis etykiet faz do klasyfikacji (zgodnie z sekcją „Etykiety faz z definicji scenariusza” w scenarios.md)
   - **`phase_times`** – **obowiązkowo** wektor faz: lista przedziałów `{ phase: "<klasa_fazy>", t_start_s: float, t_end_s: float }`. **W każdym scenariuszu** na początku są dwie fazy wspólne: **rozgrzewka** (0–3 s), **wznoszenie** (3–8 s); dopiero potem zaczyna się „prawdziwa misja” (fazy scenariusza: lot_prosty, przyspieszanie, hamowanie itd.). Czas w CSV (`time_s`) jest od 0 i pokrywa cały run, więc przedziały z `phase_times` można mapować 1:1 na okna do klasyfikacji.
   - `mission_start_offset_s` – czas (s), od którego zaczyna się misja właściwa (domyślnie 8).
   - `phase_times_note` – krótki opis podziału czasu (rozgrzewka / wznoszenie / misja).
   - `phase_times_description` – krótki opis faz scenariusza (per-scenario)
   - Wszystkie **parametry wylosowane** dla tego scenariusza (np. `v_cmd_ms`, `yaw_deg`, `wind_speed_ms`, …) – zgodnie z listami w scenarios.md

2. **Log stanu (CSV)**  
   `run_s<scenario_id>_<timestamp>_r<run_id>_state.csv`  
   Zgodnie z **docs/py/scenarios.md** „Co logujemy” — jeden strumień zsynchronizowany po `time_s`:
   - **GT:** `time_s`, `pos_x`, `pos_y`, `pos_z`, `vel_x`, `vel_y`, `vel_z`, `qw`, `qx`, `qy`, `qz` (pozycja/prędkość ENU, kwaternion).
   - **IMU:** `acc_x`, `acc_y`, `acc_z` (Body, m/s²), `gyro_x`, `gyro_y`, `gyro_z` (rad/s) — z sensorów Pegasus.
   - **Barometr:** `baro_pressure` (hPa), `baro_temp` (°C).
   - **Moc/silniki:** `motor_1`…`motor_4` (znormalizowane 0–1), `throttle` (średnia).
   - **Setpointy:** `setpoint_vel_x`, `setpoint_vel_y`, `setpoint_vel_z` (m/s, gdy używany OFFBOARD).
   Częstotliwość ~250 Hz (krok fizyki PX4). Szczegóły zmian w scenariuszu: **stachometr/SCENARIOS_CHANGELOG.md**.

Synchronizacja: ten sam `<timestamp>` w nazwie pliku JSON i CSV odpowiada temu samemu runowi.

## Uruchomienie

Z katalogu głównego Isaac Sim (np. `isaac-sim-standalone-5.1.0-linux-x86_64/`):

### Podgląd na żywo (1 lot, z GUI)

- Jeden scenariusz (np. 1):
  ```bash
  ./python.sh stachometr/run_stachometr.py --preview --scenario 1
  ```
- Dla każdego scenariusza po jednym podglądzie (scenariusze 1–10):
  ```bash
  for s in $(seq 1 10); do ./python.sh stachometr/run_stachometr.py --preview --scenario $s; done
  ```

### Headless (wiele lotów, bez GUI)

- 50 lotów dla scenariusza 1, wynik w `./stachometr_output`:
  ```bash
  ./python.sh stachometr/run_stachometr.py --headless --scenario 1 --runs 50 --output-dir ./stachometr_output
  ```
- Wszystkie scenariusze 1–10, po 50 lotów każdy:
  ```bash
  for s in $(seq 1 10); do ./python.sh stachometr/run_stachometr.py --headless --scenario $s --runs 50 --output-dir ./stachometr_output; done
  ```

### Opcje

| Opcja | Znaczenie |
|--------|-----------|
| `--preview` | Jedno uruchomienie z GUI (na żywo). |
| `--headless` | Bez GUI (do batch). |
| `--scenario N` | Numer scenariusza 1–10. |
| `--runs K` | W jednej sesji wykona K runów (sensowne z `--headless`). |
| `--output-dir DIR` | Katalog na JSON i CSV (domyślnie: `stachometr_output/` w katalogu głównym). |
| `--seed N` | Seed dla generatora losowego (powtarzalność). |
| `--duration-s T` | Czas trwania jednego runu w sekundach symulacji (domyślnie 120). |

## Opis parametrów losowych (scenarios.md)

Dla każdego scenariusza w JSON zapisywane są dokładnie te zmienne, które są wylosowane według **docs/py/scenarios.md**, np.:

- **Scenariusz 1:** `v_cmd_ms`, `yaw_deg`, `altitude_m`, `wind_speed_ms`, `wind_dir_deg`, `scenario_id`, `distance_m`
- **Scenariusz 2:** `n_phases`, `phase_lengths_m`, `v_accel_ms`, `v_decel_ms`, `transition_time_s`, `wind_*`, …
- **Scenariusz 10:** `v_cruise_ms`, `straight_leg_m`, `turn_angle_deg`, `turn_direction`, `hover_duration_s`, `wind_*`, …

Pełna lista i zakresy są w **docs/py/scenarios.md**; plik JSON zawiera zarówno `scenario_description`, jak i wszystkie wylosowane wartości.

## Etykiety faz do klasyfikacji (phase_times)

Zgodnie z **docs/py/scenarios.md** (sekcja „Etykiety faz z definicji scenariusza (samoetykietowanie)”): dla każdego runu w metadanych zapisywany jest wektor faz **phase_times** – lista przedziałów czasowych z przypisaną klasą fazy. **Każdy run** ma na początku te same dwie fazy: **rozgrzewka** (0–3 s), **wznoszenie** (3–8 s); od 8 s zaczyna się misja właściwa (fazy zależne od scenariusza). Dzięki temu bez ręcznej anotacji można budować zbiór do klasyfikacji: wejście = okno czasowe (np. 100–200 ms) z IMU/FC; etykieta = klasa fazy (rozgrzewka, wznoszenie, hamowanie, przyspieszanie, zawis, lot_prosty, zakręt, wiatr_step, start, podejście, impuls_gaz, impuls_hamuj, lot_fale, lot_po_slizgu, zmiana_kierunku, zawis_wiatr_niski, zawis_wiatr_wysoki itd.). Czas `time_s` w CSV jest od zera i obejmuje cały run, więc przedziały z **phase_times** w JSON można stosować bezpośrednio do etykietowania okien.

## Start symulacji (PX4, ARM, takeoff)

- Dron jest **spawnowany na ziemi** (z ≈ 0.07 m). Aby uniknąć „spadania” przed gotowością PX4:
  1. **Rozgrzewka** (~3 s): symulacja kręci się, skrypt loguje stan; PX4 ma czas na start.
  2. **ARM + TAKEOFF** przez MAVLink na port **14580** (kanał Onboard PX4): skrypt wysyła uzbrojenie i wznoszenie do wysokości z parametrów (bez czekania na heartbeat).
  3. **Wznoszenie** (~5 s lub do osiągnięcia wysokości).
  4. Potem **główna pętla** z logowaniem stanu do CSV.

- Jeśli wysłanie na 14580 się nie uda (np. inna konfiguracja PX4), w trybie **preview** możesz uzbroić i wykonać takeoff ręcznie z QGroundControl (QGC na 14550).

## Uwagi

- **Wiatr:** Parametry wiatru są losowane i zapisywane w JSON. W symulacji Isaac Sim/Pegasus wiatr może nie być jeszcze zaimplementowany (TODO w backendzie) – mimo to zapis parametrów wiatru jest zgodny ze specyfikacją.
- **Trajektoria:** Obecnie symulacja działa z PX4 (pozycja startowa, czas runu); pełna realizacja trajektorii scenariuszy (np. dokładnie 200 m prosto, ósemki) może wymagać trybu offboard i wysyłania setpointów (moduł `mavlink_offboard.py` jest przygotowany do ewentualnego rozszerzenia).
- **Logowanie IMU/baro/FC:** Skrypt zapisuje w CSV (oprócz GT) także IMU (acc, gyro), baro (pressure, temp), znormalizowane wyjścia silników i setpointy prędkości — zgodnie z listą zmiennych z **docs/py/scenarios.md** (patrz **SCENARIOS_CHANGELOG.md**).
