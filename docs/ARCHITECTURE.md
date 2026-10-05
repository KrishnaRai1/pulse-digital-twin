# Architecture

```
 Browser ── React SPA ──► nginx (:8080) ──/api──► FastAPI (:8000) ──► SQLite / TimescaleDB
                              └────/ws────────►  WebSocket hub
                                                  ├─ Twin (thermal engine + LightGBM residual)
                                                  ├─ Diagnostics (wave equation + CNN)
                                                  ├─ Optimizer (safety layer + MPC + XAI)
                                                  ├─ ETL + JobManager (thread pool)
                                                  └─ Simulator / SCADA adapter
```

The UI and API are separate containers and communicate only through REST and one WebSocket
(`/ws/stream`). No shared code or database access.

## Backend modules

| Module | Responsibility |
|---|---|
| `engines/thermal` | `steam` (Antoine, Watson latent heat, Marx–Langenheim), `radial` (log-grid implicit conduction), `cycle` (injection → soak → production, residual heat chaining), `viscosity` (Walther + emulsion), `ml_correction` (LightGBM on ln(q_obs/q_phys), clipped ×0.5–×2, SHAP) |
| `engines/srp` | `rods` (taper design), `wave` (2×2 transfer matrices), `loads` (polished-rod loads, float index, Goodman fatigue, power), `cards` (fault-card generator, alignment, featurisation), `cnn` (1-D CNN, ~16.7k parameters, `weights_only` load) |
| `engines/optimizer` | `limits` (hard SPM limits by bisection), `mpc` (receding-horizon DP, 0.5 SPM grid, 14-day horizon), `cycle_planner` (steam-volume sweep), `xai` (drivers, contributions, narrative); the advisory lifecycle lives in `services/optimizer_service.py` |
| `services` | `etl` (CSV/Excel cleaning), `storage` (SQLAlchemy Core), `simulator`, `jobs`, `stream`, `twin`, `scada` |

## Advisory lifecycle

`pending → applied | rejected | expired`. The safety layer re-checks at approval time; an unsafe
or stale advisory is refused. In `closed_loop` mode the same check gates the SCADA write.
Every decision lands in the audit log (`GET /api/v1/audit`).

## Data flow

1. Simulator (or a real adapter) produces telemetry every tick (3 sim-minutes / 2 s by default).
2. Faults emerge from physics (cooling → viscosity ↑ → rod float; gas interference; fluid pound; pump-unsetting risk). Cards are classified by the CNN; alerts are debounced over 2 ticks.
3. Ticks, cards, alerts, advisories and job updates are pushed over the WebSocket.
4. The optimizer evaluates candidate SPM plans against the thermal model and safety limits.

## Storage

SQLite by default (tested). Set `PULSE_DATABASE_URL=postgresql+psycopg://...` for
PostgreSQL/TimescaleDB; the schema creation attempts a hypertable on telemetry. **This path is untested.**

## Extension points

* **Field dataset:** see docs/DATASET.md (pipeline in `app/field`, routes in `app/api/dataset.py`).
* **Real data feed:** implement a loader replacing `domain/wells.build_fleet` and a `ScadaAdapter`; set `PULSE_SIM_ENABLED=false` and push telemetry through `/api/v1/ingest/production`.
* **Retrain the CNN with field cards:** `python -m scripts.train_models --only cnn --cards labelled.csv`. The CSV holds one card per row with columns `well_id, spm, label, position, load`; `position` (metres) and `load` (kN) are JSON arrays, `well_id` must exist in the fleet and `label` is one of NORMAL, ROD_FLOATING, PUMP_UNSETTING_RISK, FLUID_POUND, GAS_INTERFERENCE. Labelled cards are up-weighted ×5 in training.
* **Thermal retraining from DB:** not implemented; `engines/thermal/ml_correction.fit_residual_model` fits from observed-vs-physics cycle pairs and is the entry point.
* **Celery + Redis:** replace `services/jobs.JobManager.submit/get` with a Celery task and result backend; router code only depends on `submit` and `get`.
* **Auth:** the API key is a minimal guard; put an identity-aware gateway in front for production.
