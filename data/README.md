# Data / Données

| File | Source | Rows |
|---|---|---|
| `n24/n24_2026_gt3_laps.csv` | ADAC 24h Nürburgring 2026, `41_24h_Race_Lap_by_Lap.pdf` + `40_24h_Race_Result_Class_Final.pdf` (timing: wige SOLUTIONS), GT3 cars only (class SP 9 + GT3 cars of AT1) | one row per lap |
| `rx_france/rx_france_2026_laps.csv` | Championnat de France de Rallycross 2026, ITS Chrono results (its-results.com), Supercar + Super 1600, heats, semi-finals, finals | one row per driver, race and lap |

Columns of the endurance lap table: `car, lap, lap_time_s, driver_idx, driver, in_pit, cls, team,
car_model, laps_completed, status, class_position, event, flag`. `in_pit` = the lap ended in the pit
lane.

Columns of the rallycross table: `event, date, category, kind (heat/semi/final), race, position,
car_no, driver, car_model, nat, heat_time_s, gap, points, penalty, lap, lap_time_s, is_joker,
jkr1_s, jkr2_s` (`jkr1/jkr2` = the two split times of the joker lap).

Rebuild with `mlab data n24 --year 2026` and `mlab data rx --year 2026`. The official documents
remain the source of truth. / Les documents officiels restent la référence.
