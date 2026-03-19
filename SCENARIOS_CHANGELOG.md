# Różnice w docs/py/scenarios.md (więcej zapisów)

Porównanie z poprzednią wersją: w dokumencie dodano **szczegółową listę zmiennych do logowania** i wymóg jednego zsynchronizowanego strumienia czasowego.

---

## 1. Nowa sekcja „Co logujemy (każdy run)”

- **Wymóg:** Jeden strumień (np. jeden CSV) z **wszystkimi** zmiennymi zsynchronizowanymi po **jednym timestampie** (`time_s`).
- **Było:** Zapis pozycji, prędkości, kwaternionu (Ground Truth).
- **Jest:** GT + **IMU (surowe)** + **barometr** + **moc/silniki/FC** (+ opcjonalnie setpointy).

---

## 2. Minimalna lista zmiennych w logu (strumień czasowy)

| Źródło | Zmienne | Jednostki / uwagi |
|--------|---------|-------------------|
| **Czas** | `time_s` | Sekundy od startu runu. |
| **Pozycja / GPS (GT)** | `pos_x`, `pos_y`, `pos_z` | NED, m. |
| | `vel_x`, `vel_y`, `vel_z` | Prędkość NED, m/s. |
| | `qw`, `qx`, `qy`, `qz` | Kwaternion (Body→NED). |
| **IMU (surowe)** | `acc_x`, `acc_y`, `acc_z` | Akcelerometr Body, m/s² (lub g – w metadanych). |
| | `gyro_x`, `gyro_y`, `gyro_z` | Żyroskop, rad/s (lub °/s – w metadanych). |
| **Barometr** | `baro_pressure` | Ciśnienie (Pa lub hPa – w metadanych). |
| | `baro_temp` | Temperatura (°C), jeśli dostępna. |
| **Moc / silniki / FC** | `motor_1` … `motor_4` (lub `actuator_0` … `actuator_3`) | PWM, throttle [0–1] lub normalized thrust. |
| | `throttle` | Łączny throttle / thrust setpoint, jeśli dostępny. |
| | `setpoint_vel_x`, `setpoint_vel_y`, `setpoint_vel_z` | (Opcjonalnie) setpointy z kontrolera. |
| | `setpoint_yaw` | (Opcjonalnie) setpoint yaw. |

- **Częstotliwość:** IMU i moc najlepiej z tą samą częstością (np. 200 Hz); pozycja/GT może być rzadsza; **każdy wiersz** ma ten sam `time_s` i wszystkie kolumny (brakujące: NaN lub interpolacja).
- **Parametry runu:** Osobny plik JSON na run (scenario_id, parametry, `phase_times`) — bez zmian.

---

## 3. „Co masz teraz vs co potrzeba”

- **Masz:** `time_s`, `pos_*`, `vel_*`, `qw,qx,qy,qz` → tylko GT.
- **Brakuje:** `acc_*`, `gyro_*` (IMU), `baro_*`, **moc na silniki** (`motor_1`…`motor_4` lub `actuator_*`, `throttle`). Bez IMU i mocy model nie zobaczy „jak wygląda hamowanie / przyspieszanie / wiatr”; bez baro — wysokość i kontekst gęstości/dryfu.

---

## 4. Scenariusze 1–10

Opisy scenariuszy i listy parametrów do zapisania w logu (JSON) **bez merytorycznych zmian** — nadal te same listy (`v_cmd_ms`, `yaw_deg`, itd.) i wymóg **phase_times** (wektor czasów początku/końca faz) w metadanych.

---

## 5. Wdrożenie w Stachometrze

- **CSV:** Rozszerzony o kolumny: `acc_x`, `acc_y`, `acc_z` (Body, m/s²), `gyro_x`, `gyro_y`, `gyro_z` (rad/s), `baro_pressure` (hPa), `baro_temp` (°C), `motor_1`…`motor_4` (znormalizowane [0–1]), `throttle`, `setpoint_vel_x`, `setpoint_vel_y`, `setpoint_vel_z`.
- **Źródła danych:** IMU i barometr z sensorów Pegasus (`vehicle._sensors`), moc z `vehicle._thrusters._input_reference` (normalizacja do [0,1]), setpointy z wartości wysyłanych w OFFBOARD (scenariusz 1).
- **Metadane JSON:** Opcjonalna notatka o jednostkach kolumn CSV (np. `log_columns_note`: IMU m/s², rad/s; baro hPa, °C; motor normalized 0–1).
