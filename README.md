# PULSE — Predictive Unified Lift & Steam Engine

Digital twin and **advisory** optimiser for a Cyclic Steam Stimulation (CSS) heavy-oil field
(Baghewala-style, SIH25120 / SIH26120). It couples a thermal-viscosity model of the near-wellbore
reservoir with a sucker-rod-pump (SRP) dynamometer diagnostic engine, then recommends steam
volumes and pump speeds that stay inside hard physical safety limits.

> **Data disclosure.** No field data ships with this repository. The 12-well fleet, telemetry,
> dynamometer cards, training data and every reported metric are **synthetic**, generated from
> the physics models. Replace them with your historian/SCADA feed before drawing any operational
> conclusion. See [Limitations](#limitations).

## What is in the box

The dashboard has two sections.

**Field history: the 300-well Baghewala dataset (v1.4)** in `data/field`

| View | What it shows |
|---|---|
| **Field Overview** | As-of-day replay of the whole field: RAG map of 300 wells, total oil, field SOR, pump power, failures, ranked failure risk, sortable well table |
| **Well History** | Synchronised charts per well: observed oil vs. physics prior vs. digital-twin nowcast, temperature and viscosity, SPM and rod-floating index, fillage and load variability, 14-day failure risk with failure markers, cycle table, risk drivers (TreeSHAP) |
| **CSS Cycle Optimizer** | What-if sliders (steam rate, injection days, quality, temperature, soak, production days), recommended next-cycle settings within a steam budget with the reasons, sensitivities, expected recovery vs. time, oil-vs-steam frontier |
| **SPM Advisor** | "Reduce SPM to 5.0"-style advice from the dataset's own rod-float physics, operating envelope, 14-day outlook, replay over the well's history, field back-test |
| **Failure Early Warning** | Model vs. rule baselines (and the dataset's baseline report), PR curves, feature importance, ranked wells on any day with explanations |
| **Dataset & Validation** | Provenance, file checksums, the 62 automated checks, calibration vs. literature, field-case parameters, model cards, rebuild |

**Live twin: 12 simulated wells in real time** (WebSocket stream, dynamometer cards + CNN, MPC optimizer with advisory approvals). Its fluid and rod-drag calibration is the original demo calibration, not the dataset's (see docs/DATASET.md).

Models trained on the dataset (5-fold cross-validation grouped by well; every well is scored by a model that never saw it):

| Model | Held-out result | Baseline |
|---|---|---|
| Thermal residual (oil-rate nowcast, 7 days ahead) | MAE 1.26 bbl/d | physics prior 4.10, persistence 1.36 |
| CSS cycle response (oil per cycle-day) | R² 0.77 | field mean by cycle 0.51, repeat previous cycle 0.23 |
| Failure early warning (14 days) | PR-AUC 0.54, 92% of failures flagged ≥ 3 days ahead at a 2% false-flag rate | best rule 0.18; dataset baseline report 0.51 |
| SPM advisor (physics, no ML) | rods floating on 0.02% of days | logged operation 12.9% |

## Quick start (local)

Requirements: Python 3.11+, Node 20+.

**Step 0: the two large dataset tables.** Copy `production_history.csv` and `srp_vfd_data.csv`
(from the dataset delivery) into `data/field/`, then convert them once to Parquet (lossless, about 20 s):

```bash
cd backend
pip install pandas pyarrow            # or the full install below first
python -m scripts.import_dataset --src ../data/field
```

The CSVs are git-ignored (they exceed GitHub's 100 MB limit); the Parquet files (42 MB and 33 MB) are
what you commit. The other dataset files are already in `data/field`.

```bash
# 1. backend
cd backend
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.txt
python -m scripts.train_models          # card CNN + live-twin thermal model (~20 s)
python -m scripts.build_field_store     # 300-well dataset store + models (~2 min)
uvicorn app.main:app --port 8000        # API: http://localhost:8000/docs

# 2. dashboard (second terminal)
cd frontend
npm ci
npm run dev                             # http://localhost:5173
```

The two `scripts.*` steps are optional: on first start the API trains the models and builds the
dataset store by itself (the dashboard shows a progress bar meanwhile). On macOS/Linux,
`make install`, `make setup`, `make dev-api` and `make dev-ui` do the same.

Single process: `cd frontend && npm run build`, then
`cd backend && PULSE_FRONTEND_DIST=../frontend/dist uvicorn app.main:app` and open http://localhost:8000.

Docker: `cp .env.example .env && docker compose up --build`, then http://localhost:8080.

## Push to GitHub

```bash
cd pulse-digital-twin
git init -b main
git add .
git commit -m "PULSE digital twin with Baghewala field dataset"
git remote add origin https://github.com/<you>/pulse-digital-twin.git
git push -u origin main
```

Every file is under GitHub's 100 MB limit (the two large tables are lossless Parquet, 42 MB and
33 MB). Trained models, the derived store, `node_modules` and builds are git-ignored and are
regenerated on first run. CI (`.github/workflows/ci.yml`) runs lint, tests and the production build.

## Updating the data

```bash
cd backend
python -m scripts.import_dataset --src /path/to/folder/with/csvs   # schema check, lossless Parquet, SHA-256 manifest
python -m scripts.build_field_store                                 # retrain all dataset models
```

Real field data in the same schema can be imported the same way. See docs/DATASET.md.

## Repository layout

```
backend/
  app/
    api/            REST routers + WebSocket (/api/v1, /ws/stream)
    engines/
      thermal/      steam properties, radial conduction, cycle chaining, viscosity, LightGBM residual
      srp/          rods, wave equation, loads/fatigue, card generator/analysis, CNN classifier
      optimizer/    safety limits, MPC, cycle planner, XAI
    (services/optimizer_service.py = advisory lifecycle; services/scada.py = SCADA adapter)
    services/       ETL, storage (SQLAlchemy Core), simulator, jobs, streaming hub, twin
    domain/         well fleet definition (synthetic)
    config.py       PULSE_* settings         security.py   API key, rate limits, headers
    field/          dataset pipeline: loading, features, models, SPM advisor, cycle planner, store
  scripts/          train_models.py, import_dataset.py, build_field_store.py, generate_samples.py
  tests/            physics, SRP, optimiser, ETL and API tests
frontend/           React 19 + TypeScript + Tailwind 4 + ECharts 6 (Vite)
data/field/         Baghewala dataset v1.4 (Parquet tables, config, validation report, failure log, plot)
data/samples/       demo CSVs for the live twin's upload pages
docs/               ARCHITECTURE.md, API.md, PHYSICS.md, DATASET.md, dataset/ (file guide PDF)
```

## Safety model

* **Advisory by default.** Nothing is written to the plant without an operator approval. `PULSE_CONTROL_MODE=closed_loop` lets policy-approved set-points flow to the SCADA adapter, still through the same safety layer.
* **Hard limits are physics, not ML.** Rod stress (Goodman), float index, unit structure load, minimum SPM and a +1.5 SPM/day rate limit are checked on recommend, on approve and on manual set-point (HTTP 409 if unsafe). ML can only shift the thermal prediction within ×0.5–×2 of the physical estimate.
* **API hardening.** Optional `X-API-Key` on all mutating routes, in-memory rate limits, upload size/row/zip-bomb checks, strict Pydantic schemas (`extra=forbid`), CSP + security headers, WebSocket origin check, validation errors that never echo input.
* Set `PULSE_API_KEY` in any deployment reachable by others; the demo mode is open.

## Testing

```bash
make test     # pytest (backend) + tsc + vitest (frontend)
make lint     # ruff
```

CI (`.github/workflows/ci.yml`) runs ruff, pytest, typecheck, unit tests, build and `npm audit`.

## Limitations

* **All data is synthetic.** The 300-well dataset comes from a physics generator calibrated to published Baghewala figures (no Oil India well data), so models learn the generator's assumptions. The live twin's 12 wells are simulated too.
* **No dynamometer cards in the dataset**, so the card CNN is trained on simulated cards only; dataset failure warning uses daily pump signals.
* Early-warning scores are an optimistic ceiling: in the generator, failures are caused by the same exposures used as features.
* The live twin keeps its own demo calibration (1,500 cP at 50 °C, Couette rod drag); the dataset pages use the field case (11,500 cP at 50 °C, rod-fall-speed law).
* Docker and the TimescaleDB/PostgreSQL path are untested here; SQLite is the tested store. Background jobs use a thread pool (Celery + Redis is the documented swap-in).
* Not a certified control system: advisory use only until validated against real wells.

## License

MIT — see [LICENSE](LICENSE).
