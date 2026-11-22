"""Machine learning on top of the lap tables.

* ``forecasting``: next-stint lap-time forecasting at the Nürburgring 24h: Amazon Chronos-2
  (zero-shot foundation model) and AutoGluon-TimeSeries (extra ``[ml]``), ETS / Theta /
  ARIMA from statsforecast (extra ``[models]``), against race-engineer baselines on a
  rolling-origin backtest;
* ``classic``: next-lap model zoo (ridge, Huber, random forest, LightGBM, XGBoost, CatBoost,
  Optuna tuning, SHAP, conformal intervals), extra ``[models]``;
* ``tabular``: an AutoGluon-Tabular lap model with the same time-based split.
"""
