# Field dataset integration

PULSE synthetic dataset v1.4 (case `baghewala`, seed 42, noise 0.5): 300 wells, 2,739 CSS cycles,
599,522 well-days and 613 rod/pump failures. The file guide is in `docs/dataset/PULSE_Dataset_File_Guide.pdf`.

## Files (`data/field`)

| File | Notes |
|---|---|
| `production_history.parquet`, `srp_vfd_data.parquet`, `css_cycle_log.parquet` | The delivered CSVs converted to zstd Parquet. The conversion is checked to be lossless and the source SHA-256 is recorded in `MANIFEST.json`. |
| `rod_failure_log.csv`, `generation_config.json`, `validation_report.csv`, `early_warning_report.json`, `validation_plots.jpg` | As delivered. |

The loader also accepts `.csv` or `.csv.gz` for any table, so a new version can simply be dropped in
and imported with `python -m scripts.import_dataset --src <folder>`.

## Pipeline (`backend/app/field`)

`python -m scripts.build_field_store` (or automatically on API start when the store is missing or the
data changed) builds `backend/field_store/` in about 1.5-3 minutes:

1. Joins production and pump tables one-to-one on `well_id, cycle, day`, and builds one row per cycle (controls, oil, SOR, failures) and one row per well.
2. Trains and evaluates every model with **5-fold cross-validation grouped by well**. The guide's three rules hold by construction: splits by well; `condition_label` and the failure log are labels only; latent `reservoir_temp_c` / `oil_viscosity_cp` never feed a rate model.
3. Writes daily columns as memory-mapped `.npy` files (the API uses about 60 MB for them), plus Parquet tables, LightGBM models and `meta.json` with every metric.

## Models

| Model | Inputs | Target | Out-of-fold result |
|---|---|---|---|
| Thermal residual (nowcast) | physics prior, cycle controls, production observed up to 7 days earlier | log(observed / prior oil) | MAE 1.26 bbl/d vs. 4.10 prior-only, 1.36 persistence |
| CSS cycle response | steam rate, injection days, quality, temperature, soak, production days, cycle number, previous-cycle results | log(oil per cycle-day) | R² 0.77 (monotone constraints: more steam never lowers oil, later cycles never raise it); log-linear cross-check R² 0.76, soak optimum about 10.5 days |
| Failure early warning | rolling 3/7/14/30-day pump signals, trends, float/pound exposure since workover, days since restart | failure within 14 days | PR-AUC 0.54, ROC-AUC 0.98, 92% of failures flagged at least 3 days ahead at 2% false flags (dataset baseline report: 0.51 on its own held-out wells) |
| SPM advisor | physics only | | rods floating on 0.02% of days vs. 12.9% logged; avoidable fluid pound 23.8% to 0% of days; pump energy down 15%; oil essentially unchanged (conservative scoring) |

Temperature and viscosity residuals vs. the prior are not predictable from controls (held-out R² below 0),
so the physics prior is used for them unchanged.

## Physics recovered from the data

The rod-floating index in `srp_vfd_data.csv` is reproduced to a 0.4% median error (r = 0.9998) by:

    mu_mix = exp((1 - wc) ln mu_oil + wc ln 0.31 cP)           # produced-fluid mixture
    v_fall = min(1.2, 0.2 * (10000 / mu_mix) ** 0.25)  m/s     # config: rod_float_*
    RFI    = (2 * stroke * SPM / 60) / v_fall

Pump capacity is exactly `0.1166 x 2² x 96 x SPM x 0.8 = 35.8 x SPM bbl/d`. The viscosity table in
the config is the Walther law through 11,500 cP at 50 °C and 20 cP at 200 °C. These live in `app/field/case.py`.

## Two calibrations

The dataset pages use the field case above. The live twin (12 simulated wells) keeps its original
demonstration calibration: 1,500 cP at 50 °C and a Couette rod-drag model, which the dynamometer-card
generator and CNN were built around. Moving the twin to the field case means re-tuning its drag model
and fault scenarios, which is left as an extension.
