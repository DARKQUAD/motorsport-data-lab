# Data / Données

| File | Source | Rows |
|---|---|---|
| `n24/n24_{2017,2018,2020,2022,2023,2026}_gt3_laps.csv` | ADAC 24h Nürburgring race *Lap by Lap* + *Result by Class* PDFs (timing: wige SOLUTIONS; 24h-rennen.de, livetiming.wige.de, 24h-information.de), GT3 cars only (class SP 9 + GT3 cars of AT classes). 2019, 2021, 2024 and 2025 have no public lap-by-lap file. 2017–2018 files give no driver per lap; the 2020 file gives no time for lap 1. | one row per lap: 3,538 / 3,081 / 2,007 / 3,552 / 3,692 / 4,445 |
| `rx_france/rx_france_2026_laps.csv` | Championnat de France de Rallycross 2026, ITS Chrono results (its-results.com), Supercar + Super 1600, heats, semi-finals, finals. Development set. | one row per driver, race and lap |
| `rx_france/rx_france_2026_other_categories_laps.csv` | Same source, Division 3, Division 4, Division 5 and Féminines. **Blind test set**: never used to build or tune the models. ITS only publishes 2026, earlier seasons are not public lap by lap. | one row per driver, race and lap |

Columns of the endurance lap table: `car, lap, lap_time_s, driver_idx, driver, in_pit, cls, team,
car_model, laps_completed, status, class_position, event, flag`. `in_pit` = the lap ended in the pit
lane.

Columns of the rallycross table: `event, date, category, kind (heat/semi/final), race, position,
car_no, driver, car_model, nat, heat_time_s, gap, points, penalty, lap, lap_time_s, is_joker,
jkr1_s, jkr2_s` (`jkr1/jkr2` = the two split times of the joker lap).

Rebuild with `mlab data n24 --year 2017` (… 2018, 2020, 2022, 2023, 2026) and
`mlab data rx --year 2026`. Every parsed edition is checked against the official lap count of
each car (tests). The official documents remain the source of truth. / Les documents officiels
restent la référence.
