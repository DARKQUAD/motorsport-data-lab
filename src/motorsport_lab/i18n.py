"""English / French strings for the Streamlit app."""

from __future__ import annotations

TEXT: dict[str, dict[str, str]] = {
    "title": {
        "en": "Motorsport Data Lab",
        "fr": "Motorsport Data Lab",
    },
    "subtitle": {
        "en": (
            "Race-engineering data science on **real** timing data: GT3 at the Nürburgring "
            "24h, Le Mans (bring your own file) and the French Rallycross Championship."
        ),
        "fr": (
            "De la data science d'ingénieur de course sur des **vraies** données de "
            "chronométrage : les GT3 aux 24h du Nürburgring, Le Mans (avec ton propre "
            "fichier) et le Championnat de France de Rallycross."
        ),
    },
    "nav": {
        "en": "Section",
        "fr": "Rubrique",
    },
    "home": {
        "en": "Start here",
        "fr": "Commencer ici",
    },
    "n24": {
        "en": "Nürburgring 24h — GT3",
        "fr": "24h du Nürburgring — GT3",
    },
    "lemans": {
        "en": "Le Mans 24h — GT3 (your file)",
        "fr": "24h du Mans — GT3 (ton fichier)",
    },
    "rx": {
        "en": "French Rallycross",
        "fr": "Rallycross France",
    },
    "ml": {
        "en": "Machine learning & statistics",
        "fr": "Machine learning et statistiques",
    },
    "glossary": {
        "en": "Glossary in 30 seconds",
        "fr": "Lexique en 30 secondes",
    },
    "glossary_body": {
        "en": (
            "- **Lap time**: time for one lap. The Nordschleife + GP circuit is 25.4 km, so "
            "~8 min.\n"
            "- **Stint**: the laps between two pit stops (~8 laps / 70 min at the "
            "Nürburgring).\n"
            "- **In-lap / out-lap**: the lap into the pits and the lap out (includes the "
            "stop).\n"
            "- **Code 60 / FCY**: zones or whole track limited to 60 km/h after an incident.\n"
            "- **Track index**: how slow the whole field is right now (1.00 = normal, 1.20 = "
            "20 % slower).\n"
            "- **Joker lap** (rallycross): a longer alternative route every driver must take "
            "once per race.\n"
            "- **MAE**: mean absolute error, the average size of a forecast miss, in seconds."
        ),
        "fr": (
            "- **Temps au tour** : le temps d'un tour. Nordschleife + circuit GP = 25,4 km, "
            "soit ~8 min.\n"
            "- **Relais (stint)** : les tours entre deux arrêts au stand (~8 tours / 70 min "
            "au Nürburgring).\n"
            "- **In-lap / out-lap** : le tour qui rentre au stand et celui qui en sort (arrêt"
            " compris).\n"
            "- **Code 60 / FCY** : zones ou piste entière limitées à 60 km/h après un "
            "incident.\n"
            "- **Indice de piste** : à quel point tout le plateau est lent (1,00 = normal, "
            "1,20 = 20 % plus lent).\n"
            "- **Joker lap** (rallycross) : un tracé alternatif plus long, obligatoire une "
            "fois par course.\n"
            "- **MAE** : erreur absolue moyenne, la taille moyenne d'une erreur de prévision,"
            " en secondes."
        ),
    },
    "sources": {
        "en": (
            "Data: official timing documents — ADAC / wige (24h-rennen.de), ITS Chrono / FFSA"
            " (its-results.com), Al Kamel Systems / FIA WEC (your download)."
        ),
        "fr": (
            "Données : documents officiels de chronométrage — ADAC / wige (24h-rennen.de), "
            "ITS Chrono / FFSA (its-results.com), Al Kamel Systems / FIA WEC (ton "
            "téléchargement)."
        ),
    },
    "track_index_title": {
        "en": "How slow was the whole GT3 field?",
        "fr": "À quel point le plateau GT3 était-il lent ?",
    },
    "track_index_help": {
        "en": (
            "Above the dashed line the race was neutralised (Code 60 / FCY) or wet: those "
            "laps are excluded from pace comparisons."
        ),
        "fr": (
            "Au-dessus des pointillés, la course était neutralisée (Code 60 / FCY) ou "
            "mouillée : ces tours sont exclus des comparaisons de rythme."
        ),
    },
    "car_pick": {
        "en": "Cars to compare",
        "fr": "Voitures à comparer",
    },
    "laps_title": {
        "en": "Every lap of the selected cars",
        "fr": "Tous les tours des voitures choisies",
    },
    "pit_title": {
        "en": "Pit stops: time lost",
        "fr": "Arrêts au stand : temps perdu",
    },
    "bop_title": {
        "en": "Car models: who is quick? (best 20 % of clean laps)",
        "fr": "Modèles : qui est rapide ? (20 % meilleurs tours propres)",
    },
    "drivers_title": {
        "en": "Drivers vs team-mates (same car, conditions corrected)",
        "fr": "Pilotes vs coéquipiers (même voiture, conditions corrigées)",
    },
    "evolution_title": {
        "en": "Lap-time change within a stint",
        "fr": "Évolution du temps au tour dans un relais",
    },
    "evolution_help": {
        "en": (
            "Negative = the car gets faster as fuel burns. Intervals crossing zero mean "
            "traffic noise is larger than tyre wear: timing data alone cannot isolate "
            "degradation at the Nürburgring."
        ),
        "fr": (
            "Négatif = la voiture accélère quand le carburant s'allège. Un intervalle qui "
            "traverse zéro veut dire que le trafic brouille plus que l'usure : les chronos "
            "seuls ne suffisent pas à isoler la dégradation au Nürburgring."
        ),
    },
    "strategy_title": {
        "en": "Pit-strategy simulator (Monte-Carlo, 24 h)",
        "fr": "Simulateur de stratégie (Monte-Carlo, 24 h)",
    },
    "strategy_help": {
        "en": (
            "Parameters are calibrated on the real 2026 race; move the sliders to test what-ifs."
        ),
        "fr": (
            "Les paramètres sont calibrés sur la vraie course 2026 ; bouge les curseurs pour "
            "tester des scénarios."
        ),
    },
    "deg": {
        "en": "Tyre wear (s per lap of tyre age)",
        "fr": "Usure pneus (s par tour d'âge du pneu)",
    },
    "tyre_extra": {
        "en": "Extra time to change tyres (s)",
        "fr": "Temps en plus pour changer les pneus (s)",
    },
    "neutral_share": {
        "en": "Share of the race neutralised",
        "fr": "Part de la course neutralisée",
    },
    "sims": {
        "en": "Simulated races",
        "fr": "Courses simulées",
    },
    "upload": {
        "en": "Drop an Al Kamel `23_Analysis_Race*.CSV` file",
        "fr": "Dépose un fichier Al Kamel `23_Analysis_Race*.CSV`",
    },
    "upload_help": {
        "en": (
            "Download it from fiawec.alkamelsystems.com (season > LE MANS > FIA WEC > Race > "
            "Analysis). Only GT3 classes (LMGT3) are kept."
        ),
        "fr": (
            "Télécharge-le sur fiawec.alkamelsystems.com (saison > LE MANS > FIA WEC > Race >"
            " Analysis). Seules les classes GT3 (LMGT3) sont gardées."
        ),
    },
    "category": {
        "en": "Category",
        "fr": "Catégorie",
    },
    "event": {
        "en": "Event",
        "fr": "Épreuve",
    },
    "joker_cost_title": {
        "en": "What does the joker cost on each track?",
        "fr": "Combien coûte le joker sur chaque circuit ?",
    },
    "joker_when_title": {
        "en": "When do drivers take the joker?",
        "fr": "Quand les pilotes prennent-ils le joker ?",
    },
    "heatmap_title": {
        "en": "Simulator: best joker lap for each grid slot",
        "fr": "Simulateur : meilleur tour de joker selon la place sur la grille",
    },
    "heatmap_help": {
        "en": (
            "Expected finishing position (lower is better) of an average-pace driver. Track "
            "numbers (lap time, joker cost, lap-1 cost) come from the real event; traffic "
            "rules are assumptions."
        ),
        "fr": (
            "Position finale attendue (plus bas = mieux) d'un pilote au rythme moyen. Les "
            "chiffres du circuit (temps au tour, coût du joker, coût du 1er tour) viennent de"
            " l'épreuve réelle ; les règles de trafic sont des hypothèses."
        ),
    },
    "pass": {
        "en": "Pace advantage needed to overtake (s/lap)",
        "fr": "Avance de rythme nécessaire pour doubler (s/tour)",
    },
    "ml_intro": {
        "en": (
            "Every model is compared with a simple race-engineer rule on data it never saw. "
            "Grey bars = simple rules, blue = models. An interval that does not cross zero "
            "means the difference is real, not luck."
        ),
        "fr": (
            "Chaque modèle est comparé à une règle simple d'ingénieur, sur des données qu'il "
            "n'a jamais vues. Barres grises = règles simples, bleues = modèles. Un intervalle "
            "qui ne touche pas zéro veut dire que l'écart est réel, pas de la chance."
        ),
    },
    "no_ml": {
        "en": "Run `mlab ml forecast` (and `classic`, `ratings`, `rx-win`) to create results.",
        "fr": "Lance `mlab ml forecast` (puis `classic`, `ratings`, `rx-win`) pour les créer.",
    },
    "ml_forecast": {
        "en": "1. Forecast the next stint (8 laps)",
        "fr": "1. Prévoir le prochain relais (8 tours)",
    },
    "ml_forecast_help": {
        "en": (
            "Chronos-2 and AutoGluon (Amazon) against classical statistics (ETS, Theta, "
            "ARIMA from statsforecast) and the 'same as the last clean lap' rule. "
            "Rolling backtest: 5 cut-offs in the 2026 race."
        ),
        "fr": (
            "Chronos-2 et AutoGluon (Amazon) face aux statistiques classiques (ETS, Theta, "
            "ARIMA de statsforecast) et à la règle « comme le dernier tour propre ». "
            "Backtest glissant : 5 instants de coupure dans la course 2026."
        ),
    },
    "ml_next_lap": {
        "en": "2. Predict the next lap: the model zoo",
        "fr": "2. Prédire le tour suivant : le zoo de modèles",
    },
    "ml_next_lap_help": {
        "en": (
            "Trained on hours 0-16, tested on hours 16-24. Ridge, Huber, random forest, "
            "LightGBM, XGBoost, CatBoost, LightGBM tuned with Optuna, AutoGluon. SHAP "
            "explains the prediction; conformal intervals give an 80 % band whose coverage "
            "is checked on the unseen hours."
        ),
        "fr": (
            "Entraîné sur les heures 0-16, testé sur les heures 16-24. Ridge, Huber, forêt "
            "aléatoire, LightGBM, XGBoost, CatBoost, LightGBM réglé avec Optuna, AutoGluon. "
            "SHAP explique la prédiction ; les intervalles conformes donnent une fourchette "
            "à 80 % dont la couverture est vérifiée sur les heures jamais vues."
        ),
    },
    "ml_drivers": {
        "en": "3. Driver ratings (mixed-effects model)",
        "fr": "3. Note des pilotes (modèle à effets mixtes)",
    },
    "ml_drivers_help": {
        "en": (
            "Seconds per lap faster (<0) or slower (>0) than expected in their car, with a "
            "90 % interval. Drivers with few laps are pulled towards 0 (partial pooling), so "
            "6 lucky laps cannot top the table."
        ),
        "fr": (
            "Secondes par tour plus rapide (<0) ou plus lent (>0) qu'attendu dans sa voiture, "
            "avec un intervalle à 90 %. Un pilote avec peu de tours est ramené vers 0 "
            "(regroupement partiel) : 6 tours chanceux ne suffisent pas pour être en tête."
        ),
    },
    "ml_seasons": {
        "en": "5. Blind test on other years (2017-2026)",
        "fr": "5. Test à l'aveugle sur d'autres années (2017-2026)",
    },
    "ml_seasons_help": {
        "en": (
            "Six Nürburgring 24h races. Each model only learns from the races before the one "
            "it is scored on, like a team preparing next year. Red-flag laps of several hours "
            "are left out: no lap history can foresee them."
        ),
        "fr": (
            "Six éditions des 24h du Nürburgring. Chaque modèle n'apprend que sur les courses "
            "d'avant celle sur laquelle il est noté, comme une équipe qui prépare l'année "
            "suivante. Les tours de drapeau rouge de plusieurs heures sont exclus : aucun "
            "historique ne peut les prévoir."
        ),
    },
    "ml_lineup": {
        "en": "6. Driver style and who drives the next stint",
        "fr": "6. Style des pilotes et qui prend le prochain relais",
    },
    "ml_lineup_help": {
        "en": (
            "First table: does a driver's style (pace, consistency, stint fade, night delta, "
            "cold-tyre lap) in one race predict it in the next? Second: does it hold within a "
            "race? Then a simulator: at each stop, choose the next driver by learning every "
            "driver's pace during the race (Bayesian update, Thompson sampling) and compare "
            "with a fixed rotation. Change the rules below."
        ),
        "fr": (
            "Premier tableau : le style d'un pilote (rythme, régularité, baisse en fin de "
            "relais, écart de nuit, tour pneus froids) sur une course prédit-il la suivante ? "
            "Deuxième : tient-il au sein d'une même course ? Puis un simulateur : à chaque "
            "arrêt, choisir le prochain pilote en apprenant le rythme de chacun pendant la "
            "course (mise à jour bayésienne, Thompson sampling), comparé à une rotation fixe. "
            "Change les règles ci-dessous."
        ),
    },
    "lineup_drivers": {"en": "Drivers per car", "fr": "Pilotes par voiture"},
    "lineup_min": {"en": "Minimum stints per driver", "fr": "Relais minimum par pilote"},
    "lineup_corr": {
        "en": "How well last race predicts this one",
        "fr": "Fiabilité du profil de la course précédente",
    },
    "ml_rx": {
        "en": "4. Rallycross: what wins a race?",
        "fr": "4. Rallycross : qu'est-ce qui fait gagner ?",
    },
    "ml_rx_help": {
        "en": (
            "Logistic regression: odds of winning by joker timing at equal pace (odds ratio "
            "> 1 = better than a mid-race joker). Then three classifiers, each tested on a "
            "whole meeting it never saw, then on 4 unseen categories (Division 3, 4, 5, "
            "Féminines). This is a link, not proof of cause: leaders can choose to joker late."
        ),
        "fr": (
            "Régression logistique : chances de gagner selon le moment du joker, à rythme "
            "égal (rapport de cotes > 1 = mieux qu'un joker en milieu de course). Puis trois "
            "classifieurs, chacun testé sur un meeting entier jamais vu, puis sur 4 catégories "
            "jamais vues (Division 3, 4, 5, Féminines). C'est un lien, pas une preuve de "
            "cause : les leaders peuvent choisir de jokerer tard."
        ),
    },
}


def t(key: str, lang: str) -> str:
    """Text for ``key`` in ``lang`` ("en" or "fr"), English if missing."""
    entry = TEXT[key]
    return entry.get(lang, entry["en"])
