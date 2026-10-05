# PULSE — Predictive Unified Lift & Steam Engine
### Digital Twin & Advisory Optimizer for Cyclic Steam Stimulation (CSS) Heavy-Oil Fields
**Smart India Hackathon (SIH) | Problem Statement: SIH25120 / SIH26120**

<div align="center">

[![Team Name](https://img.shields.io/badge/Team-FLUXORA-orange?style=for-the-badge&logo=target)](https://github.com)
[![Team Code](https://img.shields.io/badge/Team%20Code-178187-blue?style=for-the-badge&logo=hashnode)](https://github.com)
[![Problem Statement](https://img.shields.io/badge/Problem%20Statement-SIH26120%20%2F%20SIH25120-green?style=for-the-badge)](https://github.com)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%20%7C%20Python%203.11+-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React%2019%20%7C%20TypeScript%20%7C%20Vite-61DAFB?style=for-the-badge&logo=react)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch%20CNN-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Deployment-Docker%20%26%20Compose-2496ED?style=for-the-badge&logo=docker)](https://www.docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Executive Summary

**PULSE (Predictive Unified Lift & Steam Engine)** is an end-to-end physics-informed digital twin and advisory optimization platform built specifically for **Cyclic Steam Stimulation (CSS)** heavy-oil reservoirs operating with **Sucker Rod Pumps (SRP)** (modeled on the Baghewala heavy-oil field, Rajasthan).

PULSE integrates **first-principles reservoir thermodynamics**, **downhole rod-string wave mechanics**, and **deep learning computer vision** to solve the four most critical operational hurdles in thermal heavy-oil extraction:
1. **Excessive Steam-to-Oil Ratio (SOR)** and sub-optimal thermal soaking.
2. **Downhole Rod Floating** caused by viscous oil damping on the pump downstroke.
3. **Catastrophic Equipment Failures** (parted rods, pump sticking, gearbox fatigue).
4. **Lack of Unified Advisory Automation**: Closing the loop between reservoir stimulation cycles and surface pumping set-points with hard physical safety boundaries.

---

## 🏆 Smart India Hackathon Details

| Parameter | Details |
|---|---|
| **Event** | Smart India Hackathon (SIH) |
| **Problem Statement ID** | `SIH25120` / `SIH26120` |
| **Domain** | Energy & Natural Resources / Heavy Oil Exploitation / Digital Twin & AI |
| **Team Name** | **Fluxora** |
| **Team Code** | **178187** |
| **Core Innovation** | Hybrid physics-informed thermal reservoir PDE coupled with 1D wave-equation dynamometer card CNN diagnostics & safety-constrained advisory MPC |

---

## 🎯 Industrial Problem Statement & Key Challenges

In ultra-heavy oil fields like **Baghewala** (~11,500 cP at 50 °C, 15° API gravity), cyclic steam injection is required to heat the near-wellbore rock and lower oil viscosity enough for sucker-rod pumps to produce. Operators face extreme operational trade-offs:

1. **The Thermal-Viscosity Trade-off:** Steam generation is energy-intensive and carbon-heavy. Over-steaming yields diminishing returns, while under-steaming leaves oil too cold and viscous to pump efficiently.
2. **Rod Floating & Buckling:** On the pump downstroke, heavy oil in the tubing creates immense viscous drag opposing gravity. If the rod string cannot fall fast enough, the polished rod goes slack, the bridle jumps off the horsehead, and rods buckle or fatigue fail.
3. **Unscheduled Downtime:** Rod fatigue, unmitigated fluid pounding, and mechanical overload account for over 70% of field downtime, costing millions in deferred production and workover rigs.
4. **Operator Trust & Safety:** Autonomous AI systems cannot be deployed to oilfields without strict, non-negotiable physical safety guardrails and operator-in-the-loop approvals.

---

## 💡 The PULSE Solution Architecture

```mermaid
flowchart TD
    subgraph Data Sources & Ingestion
        A[300-Well Field Dataset\nBaghewala v1.4] --> E[ETL & Parquet Ingest\nSHA-256 Verified]
        B[Real-Time SCADA Telemetry\n12-Well Live Stream] --> S[Stream Hub\nWebSocket Server]
    end

    subgraph Physics Engines
        E & S --> P1[Radial Thermal Conduction PDE\nNear-Wellbore T(r,t)]
        P1 --> P2[Gibbs-Vogel Viscosity Model\nμ(T, Cycle)]
        S --> P3[Everitt-Jennings 1D Wave Equation\nSurface to Pump Card Transformation]
    end

    subgraph AI & ML Layer
        P1 & P2 --> M1[Thermal Residual Nowcaster\nLightGBM 7-Day Ahead]
        P3 --> M2[Dynamometer Card Classifier\n2D CNN Inverted Cards]
        E --> M3[Failure Early Warning Engine\n14-Day Advance PR-AUC 0.54]
        M1 & M3 --> XAI[Explainability Engine\nTreeSHAP Risk Drivers]
    end

    subgraph Decision & Optimization
        P2 & M1 --> OPT1[CSS Cycle Optimizer\nSteam Budget & Soaking Frontier]
        P3 & M2 --> OPT2[SPM Advisor & Rod-Float Envelope\nHard Goodman Stress Limits]
        OPT1 & OPT2 --> MPC[Advisory Engine\nOperator Approval / Safe Closed-Loop]
    end

    subgraph Operator Dashboard
        MPC & XAI --> UI[React 19 + TypeScript + ECharts\nResponsive Modern UI]
    end
```

---

## ⚡ Key Modules & Capabilities

### 1. 🌐 Field Overview (300-Well Dataset Replay)
* Interactive **RAG (Red/Amber/Green) Risk Map** of 300 Baghewala wells as of any chosen day.
* Macro KPIs: Total field oil rate (bbl/d), field Steam-Oil Ratio (SOR), total pump power (kW), and active failure count.
* Real-time ranked failure risk list with instant jump-to-well analysis.

### 2. 📈 Well History & Physics-Informed Nowcast
* Synchronized time-series telemetry:
  * **Observed Oil Rate vs. Physics Prior vs. Digital-Twin Nowcast** (7-day forecast).
  * **Near-Wellbore Temperature & Dynamic Viscosity**.
  * **Pumping Speed (SPM) and Rod-Floating Risk Index**.
  * **Pump Fillage & Peak Polished Rod Load (PPRL) Variability**.
  * **14-Day Failure Probability with Failure Markers & TreeSHAP Drivers**.

### 3. ♨️ CSS Cycle Optimizer
* Interactive **What-If simulation sliders**:
  * Steam Injection Rate (t/d)
  * Injection Duration (days)
  * Steam Quality (%)
  * Steam Temperature (°C)
  * Soak Time (days)
  * Target Production Period (days)
* Evaluates next-cycle recommendations against an operator-defined steam budget.
* Computes the **Oil-vs-Steam Pareto Frontier** to maximize Net Energy Gain.

### 4. ⚙️ SPM Advisor & Rod-Floating Physics
* Recommends safe pumping speeds based on fluid viscosity and rod-fall terminal velocity:
  $$\text{Terminal Fall Speed } v_{\text{fall}} = \frac{(\rho_{\text{steel}} - \rho_{\text{fluid}}) \cdot g \cdot d_{\text{rod}}^2}{32 \cdot \mu_{\text{eff}}}$$
* Slashes rod-floating occurrences from **12.9%** down to **0.02%**.
* Dynamically computes the allowable operating speed window for every well.

### 5. 🚨 Failure Early Warning System
* Predicts rod-parting, pump-sticking, and gearbox overload **up to 14 days in advance**.
* **PR-AUC 0.54** (vs. 0.18 for traditional rule-based threshold baselines).
* Flags **92% of failures $\ge$ 3 days before catastrophic breakdown** at a low 2% false-positive rate.
* Provides feature contribution breakdowns (TreeSHAP) so maintenance crews know *why* an alert fired.

### 6. 🎛️ Real-Time Live Twin & Dynamometer Card CNN
* Live streaming of 12 active wells via high-speed WebSockets.
* 2D PyTorch CNN classifies surface and downhole dynamometer cards into 5 operational regimes:
  * Normal Pumping
  * Fluid Pound
  * Gas Interference / Gas Lock
  * Rod Floating / Delayed Downstroke
  * Parted Rod / Deep Mechanical Jam

---

## 🔬 Empirical Model Validation & Benchmarks

All models are trained with **5-Fold Cross-Validation grouped strictly by well** (no data leakage; every well was scored by a model that never saw any of its past cycles):

| Model Component | PULSE Metric (Held-out) | Industry Baseline | Performance Lift |
|---|---|---|---|
| **Thermal Residual Nowcast** (7-day ahead oil rate) | **MAE 1.26 bbl/d** | Physics prior: 4.10<br>Persistence: 1.36 | **+69.3% vs Prior** |
| **CSS Cycle Response** (Oil per cycle day) | **$R^2 = 0.77$** | Field mean: 0.51<br>Repeat previous: 0.23 | **+51.0% vs Baseline** |
| **Failure Early Warning** (14-day horizon) | **PR-AUC 0.54** (92% caught $\ge 3\text{d}$) | Rule-based baseline: 0.18 | **+200% Detection Gain** |
| **Rod-Float Prevention** (SPM Advisor) | **0.02% of days floating** | Historic operation: 12.9% | **99.8% Floating Reduction** |

---

## 🛡️ Industrial Safety & Human-in-the-Loop Guardrails

Safety in heavy-oil operations is non-negotiable. PULSE enforces a multi-tier defense:

* **Advisory by Default:** All optimizations generate an explicit recommendation with human-readable rationale. Set-points are never dispatched to SCADA without operator sign-off.
* **Closed-Loop Safety Envelope:** If configured in autonomous closed-loop mode (`PULSE_CONTROL_MODE=closed_loop`), set-points are passed through strict physical safety interceptors.
* **Physics Guardrails Override ML:**
  * **Goodman Stress Limits:** Peak and minimum rod stresses are calculated; over-stressed set-points are automatically rejected.
  * **Ramp-Rate Limiting:** Pumping speed changes are capped at **+1.5 SPM/day** to prevent thermal and mechanical shock.
  * **Structure Limits:** Rejects set-points that exceed beam unit gearbox torque or structure ratings (HTTP 409 Conflict).
* **Enterprise API Security:** Optional `X-API-Key` authentication, strict Pydantic `extra="forbid"` models, CSP headers, rate-limiting, and sanitized validation error responses.

---

## 💻 Tech Stack

### Backend & AI Engine
* **Framework:** FastAPI, Python 3.11+, Uvicorn, Pydantic v2
* **Physics Modeling:** NumPy, SciPy (Finite-difference radial heat conduction PDE, 1D Wave Equation)
* **Machine Learning:** PyTorch (Dynamometer CNN), LightGBM, TreeSHAP
* **Data Processing:** Pandas, PyArrow (Lossless Parquet compression for high-volume logs)
* **Storage:** SQLite (tested, zero-config) / PostgreSQL + TimescaleDB

### Frontend & Dashboard
* **Framework:** React 19, TypeScript, Vite
* **Styling:** Modern Dark-Theme Design System, Tailwind CSS 4
* **Visualization:** Apache ECharts 6 (High-FPS synchronized multi-axis charts), Lucide Icons
* **Networking:** Native WebSockets + TanStack React Query v5

### DevOps & Infrastructure
* **Containerization:** Multi-stage Dockerfile, Docker Compose
* **CI/CD:** GitHub Actions (Lint, Typecheck, Pytest, Vitest, npm audit)

---

## 📁 Repository Structure

```
pulse-digital-twin/
├── .github/workflows/ci.yml       # Automated CI/CD pipeline
├── backend/
│   ├── app/
│   │   ├── api/                   # REST endpoints & WebSocket routers
│   │   ├── domain/                # Fleet schemas & well definitions
│   │   ├── engines/
│   │   │   ├── thermal/           # Radial PDE, Gibbs-Vogel, LightGBM residual
│   │   │   ├── srp/               # 1D wave equation, card CNN, Goodman stress
│   │   │   └── optimizer/         # Multi-objective MPC, limits, XAI
│   │   ├── field/                 # 300-well Baghewala ETL, store & models
│   │   └── services/              # SCADA simulator, StreamHub, Twin orchestration
│   ├── scripts/                   # Model training, dataset import & store builder
│   └── tests/                     # Unit & integration test suites
├── frontend/
│   ├── src/
│   │   ├── api/                   # React Query client hooks & typed API contracts
│   │   ├── components/            # Reusable UI, ECharts wrappers, WellPicker, Layout
│   │   ├── pages/                 # Field Overview, Well History, CSS Optimizer,
│   │   │                          # SPM Advisor, Early Warning, Diagnostics, Live Twin
│   │   └── lib/                   # ECharts themes, formatting, WebSocket stream
├── data/
│   ├── field/                     # Baghewala dataset (Parquet, reports, manifests)
│   └── samples/                   # Uploadable sample dyno cards & production logs
├── docs/                          # Architecture, API, Physics & Dataset docs
├── docker-compose.yml             # Single-command full-stack containerization
└── README.md                      # SIH Team Fluxora Documentation
```

---

## 🚀 Getting Started

### Prerequisites
* **Python 3.11+**
* **Node.js 20+** and **npm**
* (Optional) **Docker & Docker Compose**

---

### Option A: Local Development Setup

#### 1. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.txt

# (Optional) Pre-train models and build field store:
python -m scripts.train_models
python -m scripts.build_field_store

# Start FastAPI backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be live at: **http://localhost:8000/docs**

#### 2. Frontend Setup
In a new terminal window:
```bash
# Navigate to frontend directory
cd frontend

# Install npm packages
npm install

# Start Vite development server
npm run dev
```
Open your browser at: **http://localhost:5173**

---

### Option B: Docker Compose (One-Click Launch)

```bash
# Copy example environment variables
cp .env.example .env

# Build and start services
docker compose up --build
```
Access the application at: **http://localhost:8080**

---

## 🧪 Testing & Code Quality

```bash
# Backend unit & physics tests
cd backend
pytest -v

# Frontend typechecking and unit tests
cd ../frontend
npm run typecheck
npm run test
```

---

## 📤 Pushing to GitHub

If you are setting up or pushing this repository to GitHub for the first time:

```bash
# 1. Initialize git and stage all files (already prepared locally)
git add .
git commit -m "feat(sih): complete PULSE digital twin & SIH documentation by Team Fluxora"

# 2. Add your GitHub remote repository
# Replace <YOUR-GITHUB-USERNAME> with your GitHub username:
git remote add origin https://github.com/<YOUR-GITHUB-USERNAME>/pulse-digital-twin.git

# 3. Push to main branch
git branch -M main
git push -u origin main
```

> **Using GitHub CLI (`gh`):**
> If you have GitHub CLI installed:
> ```bash
> gh auth login
> gh repo create pulse-digital-twin --public --source=. --remote=origin --push
> ```

---

## 👥 Team Fluxora (Code: 178187)

* **Team Name:** Fluxora
* **Team ID / Code:** 178187
* **Hackathon:** Smart India Hackathon (SIH)
* **Problem Category:** Energy, Heavy-Oil Reservoir Optimization & Digital Twin

---

## 📄 License & Data Disclosure

* **License:** Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
* **Data Notice:** The 300-well fleet, telemetry, dynamometer cards, and reported metrics in `data/field` are synthesized using physics-calibrated reservoir simulators adhering to published Baghewala field parameters. Replace with plant SCADA/historian feeds for live production deployment.
