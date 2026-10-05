# Baghewala field dataset v1.4

Already here: `css_cycle_log.parquet`, `rod_failure_log.csv`, `generation_config.json`,
`validation_report.csv`, `early_warning_report.json`, `validation_plots.jpg`, `MANIFEST.json`.

Add the two large tables: copy `production_history.csv` and `srp_vfd_data.csv` into this folder, then
run `python -m scripts.import_dataset --src ../data/field` from `backend/`. That writes lossless
`production_history.parquet` and `srp_vfd_data.parquet` (the files to commit), so you can delete the
CSVs afterwards. See docs/DATASET.md.
