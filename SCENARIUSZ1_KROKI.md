# Scenariusz 1 — krok po kroku (co się dzieje w runie)

## Kolejność zdarzeń

### 1. Inicjalizacja (przed `timeline.play()`)
- Wczytywanie środowiska **Flat Plane** (płaska powierzchnia bez ścian), spawn drona **na ziemi** (pozycja z ≈ 0.07 m).
- Konfiguracja PX4 backendu: Pegasus uruchamia PX4 SITL w tle; łączenie symulatora z PX4 po **TCP 4560** (sensory + sterowanie).
- Otwarcie pliku CSV do logowania stanu.

### 2. `timeline.play()` — start fizyki
- Oś czasu Isaac Sim startuje.
- PX4 SITL startuje; po połączeniu z 4560 otwiera kanały MAVLink (Onboard — port **14580**).
- W logach PX4 pojawia się m.in. „Ready for takeoff!”.

### 3. Faza „rozgrzewka” (0–3 s symulacji)
- Pętla: **750 kroków** (3 s × 250 Hz).
- W każdym kroku: **`world.step(render=…)`** i **`log_state(step)`** — fizyka + zapis do CSV. Dajemy PX4 czas na pełny start.

### 4. ARM + TAKEOFF (zawsze na port 14580)
- Skrypt tworzy **udpout:127.0.0.1:14580** (adres PX4 kanału Onboard).
- **`arm_and_takeoff_no_heartbeat(alt_target)`** — wysłanie ARM i MAV_CMD_NAV_TAKEOFF (target 1/1).
- W logu: *„Wysłano ARM + TAKEOFF … (port 14580 Onboard PX4)”*. Jeśli się nie uda: *„nie udało się wysłać ARM+TAKEOFF na 14580”*.

### 5. Faza „wznoszenie” (3–8 s)
- Do **1250 kroków** (5 s) lub do osiągnięcia wysokości `alt_target - 1.5` m.
- W każdym kroku: `world.step()`, `log_state(step)`.
- Dron powinien wznosić się, jeśli w kroku 4 wykonano arm + takeoff.

### 6. Faza „misja” (od 8 s)
- Pętla przez **duration_s** sekund (domyślnie 120 s): `world.step()`, `log_state(step)`.
- Dla scenariusza 1 nie ma jeszcze komend offboard (lot 200 m prosto) — dron wisi w miejscu (position hold po takeoff).

### 7. Zakończenie
- Zamknięcie pliku CSV, `timeline.stop()`.

---

## Co logujemy w JSON (phase_times)

- **rozgrzewka:** 0–3 s  
- **wznoszenie:** 3–8 s  
- **lot_prosty** (scenariusz 1): od 8 s (przedział liczony z parametrów scenariusza + offset 8 s).

---

## Gdy „nic się nie dzieje” (brak uzbrojenia / takeoff)

1. **Port 14580**  
   - Skrypt wysyła komendy na **127.0.0.1:14580**. PX4 musi mieć uruchomiony kanał Onboard (domyślnie tak; w logu PX4: „Onboard, udp port 14580 remote port 14540”).

2. **ARM + TAKEOFF na 14580**  
   - Skrypt wysyła ARM+TAKEOFF na **udpout:127.0.0.1:14580** (port źródłowy PX4 dla kanału Onboard; w logu PX4: „Onboard, udp port 14580 remote port 14540”).

3. **Uzbrojenie ręczne (preview)**  
   - Jeśli 14580 nie zadziała: w trybie **preview** możesz w QGC (podłączonym do 14550) uzbroić i wykonać takeoff ręcznie; skrypt i tak loguje stan od 0 s.
