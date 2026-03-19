# Wiatr w Stachometrze – opis i implementacja

## Czas symulacji vs czas na zegarku (ważne)

- W logu CSV kolumna **`time_s`** i argument **`sim_time_s`** w generatorze wiatru to **czas w świecie symulacji**: `krok × physics_dt`, gdzie **`physics_dt` = `World.get_physics_dt()`** (PhysX — często **1/60 s**, a nie 1/250). Wcześniejsze błędne założenie 250 Hz **skracało `time_s` ~4×** względem faktycznej fizyki (dystans/prędkość nie zgadzały się z czasem w CSV).
- **`duration_s` podmuchów (`wind_gust_T_*`, harmonogram `duration_s`)** jest liczony w **tych samych sekundach symulacji** — czyli „fizycznie” w modelu podmuch trwa dokładnie tyle, ile wynika z `time_s` w CSV (np. od początku do końca okna gustu).
- **Zegarek na ścianie** może iść **znacznie wolniej**: jeden krok PhysX + rendering + Pegasus/PX4 często trwa **dłużej** niż `physics_dt` (np. 4 ms) w rzeczywistości. Wtedy **1 s w `time_s` zajmuje np. 3–10 s rzeczywistych** (współczynnik czasu rzeczywistego RTF ≪ 1). To normalne przy GUI i ciężkiej scenie; w headless bywa bliżej 1:1, ale bez gwarancji.
- Jeśli potrzebujesz, żeby **15 s podmuchu = 15 s na stoperze**, trzeba by **sztucznie synchronizować** symulację z czasem rzeczywistym (osobny temat — Stachometr tego nie wymusza).

## Panel na żywo (8765): średnia vs bieżący wiatr

- **`wind_speed_ms` w kolumnie „Parametry”** to **średnia z losowania** (baza OU + podmuchów) — **nie zmienia się** w trakcie runu.
- **Zmienny wiatr w symulacji** jest w sekcji **„Wiatr (symulacja, bieżąco)”**: **`wiatr_bieżący_m/s`**, **`wind_vel_n`**, **`wind_vel_e`** — te same wartości co kolumny `wind_vel_*` w CSV.
- Gdy **`wind_dynamic_enabled: true`**, generator **zawsze** jest krokowany co `physics_dt`; wcześniej przy braku prima PhysX log/panel mogły pokazywać zera — to poprawione: bieżący wektor i tak pochodzi z generatora.

## Stan do tej pory (przed implementacją fizyki wiatru)

**Wiatr nie był stosowany do symulacji fizyki.** W kodzie:

- `wind_speed_ms` i `wind_dir_deg` są **tylko**:
  - zapisywane w pliku JSON (parametry runu),
  - wyświetlane na stronie odczytów na żywo (127.0.0.1:8765),
  - używane do nazewnictwa plików (np. `*_wind_state.csv` vs `*_nowind_state.csv`).

- **Żadna siła wiatru nie jest przekazywana do silnika fizyki** (PhysX / Isaac Sim). Dron w scenariuszu „z wiatrem” leci tak samo jak „bez wiatru” – różnica jest tylko w metadanych i w nazwach plików.

Dlatego porównanie `--scenario_1_wind` i `--scenario_1_no_wind` nie pokazywało różnicy w zachowaniu drona (np. w throttle, prędkości nad ziemią, przeleconym dystansie).

---

## Skąd „dokumentacja” (ważne)

- **Nie ma w Isaac Sim osobnego poradnika „jak zrobić wiatr”.** W kodzie jest:
  - **konwencja meteo** (kierunek, z którego wieje) + **model z dupy**: stała siła ∝ `wind_speed` (dopasowanie `k_wind` pod scenariusz);
  - **API PhysX w Omniverse:** `omni.physx.get_physx_simulation_interface().apply_force_at_pos(...)` — opis w `extscache/omni.physx-.../omni/physx/bindings/_physx.pyi` (argument `pos`: *„World position where the force is applied”*).
- **Oficjalny wzorzec użycia** (NVIDIA): np. `isaacsim/replicator/examples/tests/test_sdg_randomizer_snippets.py` — przy `apply_force_at_pos` przekazują **`box_position` = `ExtractTranslation()` z world transformu obiektu**, a nie `(0,0,0)`.

### Błąd, który powodował crash przy każdej niezerowej sile wiatru

- Użycie **`pos = (0,0,0)`** oznacza: siła działa w **punkcie początku układu świata**, a nie w środku masy drona.
- Gdy dron jest np. 50–200 m dalej, powstaje **duże ramię** `r × F` → **olbrzymi moment** → zwichrowanie i „wypierdal” **niezależnie** od tego, czy `wind_speed_ms` to 2, 5 czy 7.
- Poprawnie: **`pos` = aktualna pozycja drona w świecie** (w Stachometrze: `vehicle.state.position` z Pegasusa, spójna z logiem CSV).

---

## Jak wiatr powinien działać (research / uproszczenie)

- **Konwencja kierunku (meteo):** `wind_dir_deg` = kierunek, **z którego** wieje wiatr (0° = z północy, 90° = ze wschodu itd.). Wektor prędkości wiatru (kierunek ruchu powietrza) w NED:  
  `wind_vec = wind_speed * (-cos(θ), -sin(θ), 0)`, θ w radianach.
- **Headwind względem lotu (scenariusz 1):** setpoint prędkości to `(vx, vy) = v_cmd * (cos(yaw_deg), sin(yaw_deg))` w NED. Żeby siła wiatru była **przeciwna do tego wektora**, ustaw **`wind_dir_deg = yaw_deg`** (wtedy „wiatr w twarz” temu kierunkowi). Jeśli `yaw` i `wind_dir` się nie zgadzają, dostajesz głównie **wiatr skośny** — mniejsza różnica na throttle przy tym samym `|v|` z kontrolera.

- **Efekt na drona:**  
  Wiatr działa na drona jak **siła zewnętrzna** (opór/parcie w kierunku wiatru). Przy zadanym setpoincie prędkości (np. 12 m/s w przód):
  - **Headwind (wiatr w twarz):** dron musi dawać więcej ciągu, żeby utrzymać prędkość nad ziemią → wyższy throttle, mniejsza prędkość nad ziemią przy tym samym setpoincie.
  - **Tailwind (wiatr w plecy):** mniejszy opór / „pomaganie” wiatru → niższy throttle przy tej samej prędkości nad ziemią lub wyższa prędkość przy tym samym throttle.

- **Modele siły (przełącznik `wind_force_model` w parametrach runu):**
  - **`legacy`:**  
    `F = K_WIND_N_PER_MS * ‖w_h‖ * (w_h / ‖w_h‖)` w płaszczyźnie poziomej NED,  
    gdzie `w_h` to **bieżący** wektor prędkości wiatru (średnia ± turbulencja ± podmuch). Przy stałym wiatrze jest to równoważne dawnemu `K * wind_speed * kierunek`.
  - **`drag`:** uproszczony opór względem powietrza, tylko **XY**:  
    `v_air = v_drona - w_wiatru` (NED),  
    `F = −0.5 · ρ · (Cd·A) · ‖v_air,xy‖ · v_air,xy`.  
    Parametry: `wind_rho_kg_m3` (domyślnie 1.225), `wind_cd_times_area_m2` (efektywne Cd·A, domyślnie ~1.5 — szacunek pod rząd wielkości jak legacy przy typowym locie).

---

## Generator prędkości wiatru (`wind_generator.py`)

### Co to jest „dryf” składowych **N / E**

W układzie **NED** wiatr ma składowe **`wind_vel_n`** (North) i **`wind_vel_e`** (East) w płaszczyźnie poziomej. Model **Ornstein–Uhlenbeck** co krok dodaje **losowe, skorelowane w czasie** odchylenia do **obu** składowych — wektor wiatru w poziomie **powoli zmienia się wokół średniej** (płynne „pływanie”), a nie stoi idealnie na jednej wartości. To właśnie ten **dryf składowych n/e**: **nie** oznacza przesuwania całej masy powietrza w świecie, tylko **fluktuacje modelu turbulencji** wokół zadanej średniej prędkości i kierunku.

### Amplituda i czas narastania losowych podmuchów

- **Dodatek w szczycie [m/s]** wzdłuż średniego wiatru: domyślnie losowo **`U(wind_gust_A_rel_min_ms, wind_gust_A_rel_max_ms)`** (np. **1,5…3,0**). Pojedyncze **`wind_gust_A_rel_ms`** służy jako wartość domyślna przy braku min/max w surowym dict (albo jako dokumentacja środka); po `merge_wind_defaults` używane są min/max.
- To **nie** jest zastąpienie średniej — średnia z `wind_speed_ms` / `wind_dir_deg` zostaje; podmuch dokłada składową równoległą. **Lull:** część losowań ma ujemną amplitudę (zelżenie).
- **Czas narastania do szczytu:** jeśli w parametrach są **oba** `wind_gust_rise_time_min_s` i `wind_gust_rise_time_max_s`, to dla losowych Poisson losuje się **`t_peak ~ U(min, max)`** (np. **0,8…2,0 s** symulacji). **Czas opadania** po szczycie: **`U(wind_gust_fall_time_min_s, wind_gust_fall_time_max_s)`** (domyślnie także **0,8…2,0 s**). Całkowite **`T = t_peak + t_opadania`**, potem **clamp** do **`[wind_gust_T_min_s, wind_gust_T_max_s]`**.
- **Tryb legacy (peak_frac):** gdy **obu** kluczy `wind_gust_rise_time_*` **nie ma** w dict przekazanym do generatora, losowy podmuch bierze **`T ~ U(T_min, T_max)`** i **`t_peak = peak_frac × T`**, `peak_frac ~ U(peak_frac_min, peak_frac_max)` — jak wcześniej. Harmonogram `wind_gust_schedule` nadal używa `duration_s` i opcjonalnie `peak_fraction`.

- **`wind_dynamic_enabled: true` (domyślnie w `merge_wind_defaults`):** średnia z `wind_speed_ms` / `wind_dir_deg` (meteo) **+** turbulencja **Ornstein–Uhlenbeck** na N/E (`wind_tau_s`, `wind_sigma_ms`) **+** podmuchy **1−cos** w jednym z dwóch trybów:
    - **Harmonogram `wind_gust_schedule`:** lista zdarzeń `{"t_start_s", "duration_s", "A_rel_ms"}` oraz opcjonalnie **`peak_fraction`** ∈ (0,1) — ułamek `duration_s`, po którym jest **szczyt** obwiedni (brak klucza = **0,5**, symetrycznie). **`A_rel_ms > 0`** — podmuch wzdłuż średniego wiatru; **`< 0`** — **zelżenie (lull)**. Czas **`t_start_s`** liczony od pierwszego kroku symulacji. Jeśli lista jest **niepusta**, działają **wyłącznie** te zdarzenia (bez losowych Poisson). Jeśli **pusta** / brak klucza — **losowe** podmuchy Poisson: amplituda **`U(A_min, A_max)`**, czas narastania/opadania jak wyżej (**rise/fall**), lub tryb legacy (**`T`** + **`peak_frac`**); **`wind_gust_prob_per_s`**, lull (`wind_gust_lull_enabled`, `wind_gust_lull_prob`). Zdarzenia z kolejki startują **po kolei**; jeśli nominalny `t_start_s` minął podczas czekania, okno liczy się od **faktycznego startu**.
    - Składowa wiatru wzdłuż średniego kierunku jest **ograniczana z dołu** (brak „odwrotnego” wiatru po tej osi), z wyjątkiem jawnego **lulla** w harmonogramie. Gdy turbulencja **przeważa** nad słabą średnią, ta korekta bywa **gwałtowna** — wektor na chwilę staje się głównie **prostopadły** do średniego kierunku, więc kąt meteo na panelu może „skoczyć” o ~90°. To **nie** jest powolna zmiana pogody, tylko **chwilowy** wynik modelu (średnia + OU + podmuch); żeby kierunek wizualnie płynął wolniej: **zwiększ `wind_tau_s`**, **zmniejsz `wind_sigma_ms`**, trzymaj **średni wiatr** znacznie większy od σ.
    - **Obwiednia:** dwa odcinki cosinusowe (narastanie do **szczytu** w `t_peak`, opadanie do końca okna). Dla losowych podmuchów `t_peak` jest **losowy**; `wind_gust_phase` w CSV to nadal **postęp czasu** `t_loc / T` (0…1), a **`wind_gust_rising`** = 1 **przed** szczytem (`t_loc < t_peak`), 0 po szczycie.
- **`wind_dynamic_enabled: false` (jawne nadpisanie w JSON / scenariuszu):** **stały** wektor wiatru — tylko gdy świadomie chcesz wyłączyć OU i podmuchy.
- **Seed:** `random_seed` przekazywany do `run_single` (CLI `--seed` + `run_id`) — powtarzalność batchy.

---

## Implementacja w kodzie (`run_stachometr.py`)

- Gdy `wind_suffix == "wind"` i `make_wind_generator` zwraca generator (`wind_speed_ms > 0`):
  - co krok: `w_ned = generator.step(dt, t)`, potem siła wg `wind_force_model`, potem  
    `apply_force_at_pos(..., world_pos = pozycja drona)`.
- **Kolumny CSV (etykiety / kontekst dla AI):**  
  `wind_vel_n,e,d`, `wind_force_n,e,d`, `v_air_n,e,d` (`v_air = v_drona − w_wiatru`),  
  `wind_is_gust` (1 w trakcie aktywnego okna 1−cos),  
  `wind_gust_phase` (poza podmuchem puste; w podmuchu: **0…1** = `t_loc / duration`, postęp w oknie),  
  `wind_gust_rising` (1 gdy `wind_is_gust` i **przed szczytem** obwiedni — patrz opis generatora),  
  `wind_gust_from_schedule` (1 gdy aktywny podmuch pochodzi z `wind_gust_schedule`),  
  `wind_gust_is_lull` (1 przy zaplanowanym lub losowym **lullu**).
- **Panel na żywo (8765):** m.in. `wiatr_faza_podmuchu` (= faza), `wiatr_podmuch_narasta`, `wiatr_podmuch_z_planu`, `wiatr_zelżenie` — te same informacje co w CSV.
- **Ścieżka rigid body:** `/World/quadrotor/body` lub `/World/quadrotor`.
- **`K_WIND_N_PER_MS`:** [`scenario_params.py`](scenario_params.py) — kalibracja legacy; przy `legacy` i stałym wiatrze zachowanie jak wcześniej.

---

## Parametry domyślne (JSON)

Wszystkie klucze z `DEFAULT_WIND_PHYSICS_PARAMS` w [`scenario_params.py`](scenario_params.py) są **doklejane** przez `merge_wind_defaults` do dict scenariusza (nie nadpisują już ustawionych wartości). Edytuj w `SCENARIO_1_FIXED` / losowaniu albo w zapisanym JSON przed runem.
