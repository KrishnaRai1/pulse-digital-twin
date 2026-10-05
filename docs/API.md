# REST & WebSocket API

Interactive docs: `http://localhost:8000/docs` (OpenAPI). Base path `/api/v1`.
Mutating routes need `X-API-Key` when `PULSE_API_KEY` is set. Validation errors return `422` with `loc/msg/type` only.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health`, `/system/info` | liveness; mode, limits, model status |
| GET | `/samples`, `/samples/{name}` | bundled demo CSVs |
| GET | `/field/overview`, `/field/trend?hours=` | KPIs, RAG wells; totals over time |
| GET | `/alerts` | active alerts |
| GET | `/wells`, `/wells/{id}` | fleet and single-well state |
| GET | `/wells/{id}/telemetry`, `/profile`, `/viscosity` | sensor history; wellbore + radial profiles |
| GET/POST | `/wells/{id}/cycles` | cycle history / add cycle |
| PUT/DELETE | `/wells/{id}/cycles/{n}` | replace (full body) / remove cycle |
| POST | `/wells/{id}/setpoint` | manual SPM override; `409` if it violates safety limits |
| POST | `/thermal/predict`, `/thermal/whatif` | prediction; up to 6 scenarios |
| POST | `/diagnostics/classify` | classify one card (`points` or `position`+`load`, unit fields) |
| GET | `/wells/{id}/card`, `/cards` | latest card + baseline; history |
| GET | `/wells/{id}/recommendation` | optimiser output with XAI |
| POST | `/wells/{id}/advisories?force=` | create advisory |
| GET | `/advisories`, `/audit` | list; audit log |
| POST | `/advisories/{id}/approve`, `/reject` | decision (`actor`, `note`) |
| POST | `/wells/{id}/cycle-plan` | steam-volume sweep, returns `202` + job |
| POST | `/ingest/production` | CSV/Excel/zip with cleaning report; `align_to_now` form field |
| POST | `/ingest/dynamometer` | multipart cards CSV; `202` + job |
| GET | `/jobs/{id}` | job state/result |

Uploads over `PULSE_MAX_UPLOAD_MB` return `413`; rate limits return `429`.

## WebSocket `/ws/stream`

Same-origin or allowed CORS origin only. Client messages: `{"action":"subscribe","well_id":"BGW-03"}`, `{"action":"ping"}`.
Server message types: `hello`, `subscribed`, `tick`, `dyno`, `alert`, `alert_cleared`, `advisory`, `job`, `ingest`.

## Field dataset (`/api/v1/dataset`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/dataset/status`, `/summary` | store state (503 with progress while building); dataset totals and headline metrics |
| GET | `/dataset/field?day=`, `/field/trend?step=` | all wells on one day (RAG, rates, risk, KPIs); field totals per day |
| GET | `/dataset/wells`, `/wells/{id}`, `/wells/{id}/daily?cycle=` | well table; cycles and failures; daily history |
| GET | `/dataset/wells/{id}/risk?day=` | 14-day failure risk with TreeSHAP drivers (out-of-fold model) |
| GET | `/dataset/wells/{id}/spm?day=`, `/spm-history` | SPM advice with reasons, envelope and outlook; advisor replay |
| GET / POST | `/dataset/wells/{id}/cycle-plan`, `/cycle-whatif` | recommended CSS settings within a steam budget; what-if for up to 6 scenarios |
| GET | `/dataset/models`, `/early-warning`, `/early-warning/alerts?day=` | model cards; warning metrics vs. rules and the baseline report; ranked wells |
| GET | `/dataset/validation`, `/files/{name}` | validation checks, manifest, field-case parameters; whitelisted downloads |
| POST | `/dataset/rebuild` | rebuild the store and retrain (API key, background) |
