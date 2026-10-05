// Types and hooks for the field-dataset API (/api/v1/dataset, see backend/app/api/dataset.py).
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./client";

export type Rag = "green" | "amber" | "red" | "blue" | "grey";
export type DsPhase = "soak" | "production" | "workover" | "injection" | "stopped";
export type RiskLevel = "alert" | "watch" | "normal" | null;

export interface DsStatus {
  state: "idle" | "missing" | "loading" | "building" | "ready" | "stale" | "error";
  message: string;
  progress: { step: string; fraction: number };
  elapsed_s: number | null;
  data_dir_present: boolean;
  ready: boolean;
}

export interface DsSummary {
  dataset: {
    version?: string;
    case?: string;
    seed?: number;
    noise_level?: number;
    min_osr?: number;
    n_wells: number;
    n_cycles: number;
    n_rows: number;
    n_running_days: number;
    n_failures: number;
    max_day: number;
  };
  built_at: string;
  totals: {
    oil_m3: number;
    steam_m3: number;
    cum_sor: number;
    mean_oil_bbl_d_per_producing_well: number;
    median_cycle_sor: number;
    high_sor_cycles: number;
    wells_stopped_early: number;
    failures_by_mode: Record<string, number>;
  };
  headline_metrics: {
    early_warning_pr_auc: number;
    early_warning_caught_ge3d: number;
    nowcast_mae_prior: number;
    nowcast_mae_twin: number;
    cycle_model_r2: number;
    advisor_floating_days_logged_pct: number;
    advisor_floating_days_pct: number;
  };
  max_day: number;
}

export interface DsWellState {
  well_id: string;
  phase: DsPhase;
  status: Rag;
  reason: string;
  cycle: number | null;
  oil_bbl_d: number | null;
  water_cut_pct: number | null;
  spm: number | null;
  rfi: number | null;
  fillage_pct: number | null;
  motor_kw: number | null;
  risk: number | null;
}

export interface DsSnapshot {
  day: number;
  max_day: number;
  kpis: {
    total_oil_bbl_d: number;
    total_oil_m3_d: number;
    total_liquid_bbl_d: number;
    field_water_cut_pct: number | null;
    total_power_kw: number;
    steam_today_m3: number;
    cum_oil_m3: number;
    cum_steam_m3: number;
    cum_sor: number | null;
    n_producing: number;
    n_injecting: number;
    n_soaking: number;
    n_workover: number;
    n_stopped: number;
    n_red: number;
    n_amber: number;
    mean_spm: number | null;
    failures_to_date: number;
  };
  thresholds: { risk_alert: number; risk_watch: number; rfi: number; rfi_margin: number };
  wells: DsWellState[];
}

export type Num = number | null;

export interface DsTrend {
  day: number[];
  oil_bbl_d: Num[];
  water_bbl_d: Num[];
  steam_m3: Num[];
  motor_kw: Num[];
  n_producing: Num[];
  n_injecting: Num[];
  n_soaking: Num[];
  n_workover: Num[];
  n_stopped: Num[];
  n_alert: Num[];
  n_floating: Num[];
  failures: Num[];
  cum_oil_m3: Num[];
  cum_steam_m3: Num[];
  cum_sor: Num[];
  mean_risk: Num[];
}

export interface DsWellRow {
  well_id: string;
  n_cycles: number;
  last_day: number;
  cum_oil_m3: number;
  cum_steam_m3: number;
  cum_sor: number;
  last_cycle_sor: number;
  mean_oil_bbl_d: number;
  pct_days_floating: number;
  mean_spm: number;
  n_failures: number;
  failures_rod_part: number;
  failures_pump_unset: number;
  failures_worn_barrel: number;
  pad: string;
  grid_x: number;
  grid_y: number;
  stopped_early: boolean;
  high_sor_cycles: number;
  best_cycle_opd_m3d: number;
  fold: number;
}

export interface DsCycle {
  cycle: number;
  start_day: number;
  inj_days: number;
  steam_rate_m3d: number;
  steam_quality: number;
  steam_temp_c: number;
  inj_pressure_kpa: number;
  soak_days_planned: number;
  soak_start_day: number;
  end_day: number;
  soak_days: number;
  workover_days: number;
  prod_days: number;
  cycle_days: number;
  steam_m3: number;
  oil_m3: number;
  oil_bbl: number;
  prior_oil_m3: number;
  peak_oil_bbl_d: number;
  avg_water_cut_pct: number;
  sor: number;
  osr: number;
  opd_m3d: number;
  opd_pred_oof_m3d: number;
  high_sor_flag: boolean;
  n_failures: number;
  failure_modes: string;
}

export interface DsFailure {
  well_id: string;
  cycle: number;
  failure_date_day: number;
  failure_mode: string;
  downtime_days: number;
}

export interface DsWell {
  well_id: string;
  summary: DsWellRow;
  cycles: DsCycle[];
  failures: DsFailure[];
  first_day: number;
  last_day: number;
  fold: number;
}

export interface DsDaily {
  day: number[];
  cycle: number[];
  phase: string[];
  days_since_soak_start: number[];
  oil_rate_bbl_day: Num[];
  prior_oil_rate_bbl_day: Num[];
  twin_oil_bbl_d: Num[];
  water_cut_pct: Num[];
  gross_liquid_bbl_day: Num[];
  reservoir_temp_c: Num[];
  prior_reservoir_temp_c: Num[];
  oil_viscosity_cp: Num[];
  prior_oil_viscosity_cp: Num[];
  pump_running: number[];
  spm: Num[];
  motor_kw: Num[];
  pump_fillage_pct: Num[];
  rod_floating_index: Num[];
  polished_rod_peak_load_kn: Num[];
  polished_rod_min_load_kn: Num[];
  load_variability_pct: Num[];
  risk: Num[];
  condition_label: string[];
  injection_windows: [number, number, number][];
  thresholds: { risk_alert: number; risk_watch: number };
}

export interface DsDriver {
  feature: string;
  label: string;
  value: number | null;
  contribution: number;
}

export interface DsRisk {
  well_id: string;
  day: number;
  available?: boolean;
  risk?: number;
  level?: RiskLevel;
  base_log_odds?: number;
  drivers?: DsDriver[];
  fold?: number;
  note?: string;
}

export interface DsReason {
  severity: "high" | "medium" | "info";
  code: string;
  text: string;
}

export interface DsSpm {
  well_id: string;
  day: number;
  applicable: boolean;
  reason?: string;
  cycle?: number;
  risk?: number | null;
  risk_level?: RiskLevel;
  action?: "hold" | "reduce" | "increase";
  headline?: string;
  current_spm?: number;
  recommended_spm?: number;
  binding_constraint?: string;
  limits?: { spm_rod_float: number; spm_inflow: number | null; spm_min: number; spm_max: number; rfi_target: number; fillage_target_pct: number };
  state?: {
    oil_viscosity_prior_cp: number;
    oil_viscosity_latent_cp: number;
    water_cut_pct: number;
    mixture_viscosity_cp: number;
    rod_fall_speed_m_s: number;
    rod_speed_m_s: number;
    rfi_model: number;
    rfi_logged: number;
    fillage_pct: number;
    liquid_bbl_d: number;
    oil_bbl_d: number;
    capacity_bbl_d: number;
    motor_kw: number;
    vfd_hz: number;
    pump_limited: boolean;
  };
  expected?: { rfi: number; fillage_pct: number; liquid_bbl_d: number; oil_bbl_d: number; motor_kw: number; vfd_hz: number; note: string | null };
  reasons?: DsReason[];
  curve?: { spm: number[]; rfi: number[]; capacity_bbl_d: number[]; lifted_bbl_d: number[]; inflow_known: boolean };
  outlook?: { day: number[]; prior_oil_viscosity_cp: number[]; spm_rod_float_limit: number[]; spm_plan: number[]; note: string };
}

export interface DsSpmHistory {
  well_id: string;
  day: number[];
  spm_logged: Num[];
  spm_advised: Num[];
  spm_rod_float_limit: Num[];
  rfi_logged: Num[];
  rfi_advised: Num[];
  binding: string[];
  summary: { days: number; floating_days_logged: number; floating_days_advised: number; mean_spm_logged: number | null; mean_spm_advised: number | null };
}

export interface CycleControls {
  steam_rate_m3d: number;
  inj_days: number;
  steam_quality: number;
  steam_temp_c: number;
  soak_days: number;
  prod_days: number;
}

export interface CycleEval {
  label?: string | null;
  controls: CycleControls;
  opd_m3d: number;
  oil_m3: number;
  oil_bbl: number;
  steam_m3: number;
  cycle_days: number;
  sor: number;
  osr: number;
  steam_per_cycle_day_m3: number;
  net_usd_per_day: number;
  economic: boolean;
  contributions?: { feature: string; log_effect: number; effect_pct: number }[];
  base_m3d?: number;
}

export interface CycleContext {
  well_id: string;
  cycle: number;
  is_next_cycle: boolean;
  prev_opd_m3d: number | null;
  prev_peak_oil_bbl_d: number | null;
  prev_sor: number | null;
  cum_oil_before_m3: number;
  defaults: CycleControls;
  defaults_source: string;
  actual?: { opd_m3d: number; oil_m3: number; sor: number; opd_pred_oof_m3d: number };
  shape_cycle: number;
}

export type Sweeps = Record<keyof CycleControls, { x: number[]; opd_m3d: number[]; sor: number[]; net_usd_per_day: number[] }>;

export interface ProfileCurves {
  shape_cycle: number;
  curves: { label: string; day: number[]; cum_oil_m3: number[]; oil_m3: number; sor: number }[];
}

export interface CyclePlan {
  context: CycleContext;
  plan: {
    objective: string;
    steam_budget_m3_per_cycle_day: number | null;
    quality: number;
    temp_c: number;
    searched: number;
    feasible: number;
    best: CycleEval;
    baseline: CycleEval;
    uplift_pct: number | null;
    why: { feature: string; effect_pct: number; from: number | null; to: number | null }[];
    alternatives: CycleEval[];
    frontier: { steam_m3: number; oil_m3: number; opd_m3d: number; sor: number }[];
  };
  sweeps: Sweeps;
  profile: ProfileCurves;
  prices: { oil_usd_per_m3: number; steam_usd_per_m3: number };
}

export interface CycleWhatIf {
  context: CycleContext;
  results: CycleEval[];
  sweeps: Sweeps;
  profile: ProfileCurves;
}

export interface WarningResult {
  roc_auc: number;
  pr_auc: number;
  threshold?: number;
  flag_rate_on_healthy_days: number;
  failures_caught_with_ge3d_warning: number;
  median_lead_days: number;
  n_failures?: number;
  n_failures_test?: number;
  false_alert_days_per_running_well_year: number;
}

export interface DsEarlyWarning {
  horizon_days: number;
  threshold_alert: number;
  threshold_watch: number;
  results: Record<string, WarningResult>;
  pr_curves: Record<string, { recall: number; precision: number }[]>;
  fold_pr_auc: number[];
  top_features: { feature: string; importance: number }[];
  n_rows: number;
  n_positive_rows: number;
  evaluation: string;
  baseline_report: { horizon_days: number; results: Record<string, WarningResult>; top_features: Record<string, number> } | null;
}

export interface DsAlerts {
  day: number;
  n_alert: number;
  n_watch: number;
  thresholds: { alert: number; watch: number };
  wells: {
    well_id: string;
    risk: number;
    level: RiskLevel;
    rfi: number | null;
    fillage_pct: number | null;
    load_variability_pct: number | null;
    spm: number | null;
    failure_within_14d: boolean;
    days_to_next_failure: number | null;
  }[];
  note: string;
}

export interface RegMetrics {
  mae: number;
  rmse: number;
  bias: number;
  mape_pct: number;
  r2: number;
  n: number;
}

export interface DsModels {
  early_warning: Omit<DsEarlyWarning, "baseline_report">;
  nowcast: {
    horizon_days: number;
    target: string;
    features: string[];
    metrics_oil_rate_bbl_d: Record<string, RegMetrics>;
    residual_r2: number;
    top_features: { feature: string; importance: number }[];
    temperature_viscosity_note: string;
    n_rows: number;
    evaluation: string;
  };
  cycle_model: {
    target: string;
    features: string[];
    controls: string[];
    monotone_constraints: Record<string, number>;
    metrics: Record<string, RegMetrics>;
    metrics_cycles_2plus: Record<string, RegMetrics>;
    importance: { feature: string; mean_abs_effect_pct: number }[];
    linear_cross_check: { r2_oof: number; coefficients: Record<string, number>; effects: { control: string; step: number; effect_pct: number }[]; soak_optimum_days: number | null };
    partial_dependence: Record<string, { x: number[]; opd_m3d: number[] }>;
    n_cycles: number;
    evaluation: string;
  };
  spm_advisor: {
    rfi_target: number;
    fillage_target_pct: number;
    n_days: number;
    logged: Record<string, number>;
    advisor: Record<string, number>;
    share_of_days: Record<string, number>;
    assumptions: string[];
  };
  folds: { n_folds: number; by: string; wells_per_fold: number[] };
  lightgbm: string;
  built_at: string;
  build_seconds: number;
}

export interface DsValidation {
  checks: { file: string; check: string; status: string; detail: string }[];
  status_counts: Record<string, number>;
  manifest: {
    imported_at?: string;
    tables?: Record<string, { file: string; rows: number; source_sha256: string; source_bytes: number; parquet_bytes: number; lossless_roundtrip: boolean }>;
    files?: Record<string, { sha256: string; bytes: number }>;
    dataset?: Record<string, unknown>;
  };
  field_case: {
    version: string;
    case: string;
    seed: number;
    noise_level: number;
    min_osr: number;
    pump: Record<string, unknown>;
    rod_float: Record<string, number>;
    reservoir: Record<string, unknown>;
    controls: Record<string, number[]>;
  } | null;
  files: { name: string; bytes: number; downloadable: boolean }[];
  plot: string | null;
  rules: string[];
}

// ------------------------------------------------------------------ hooks
const STATIC = { staleTime: Infinity, gcTime: 10 * 60_000 } as const;

export const useDsStatus = (poll: boolean) =>
  useQuery({ queryKey: ["ds-status"], queryFn: () => api.get<DsStatus>("/dataset/status"), refetchInterval: poll ? 2000 : false, retry: 30, retryDelay: (n) => Math.min(1500 * 2 ** n, 10000) });

export const useDsSummary = (enabled = true) => useQuery({ queryKey: ["ds-summary"], queryFn: () => api.get<DsSummary>("/dataset/summary"), enabled, ...STATIC });
export const useDsSnapshot = (day: number, enabled = true) =>
  useQuery({ queryKey: ["ds-field", day], queryFn: () => api.get<DsSnapshot>(`/dataset/field?day=${day}`), enabled, placeholderData: keepPreviousData, ...STATIC });
export const useDsTrend = (step = 1) => useQuery({ queryKey: ["ds-trend", step], queryFn: () => api.get<DsTrend>(`/dataset/field/trend?step=${step}`), ...STATIC });
export const useDsWells = () => useQuery({ queryKey: ["ds-wells"], queryFn: () => api.get<DsWellRow[]>("/dataset/wells"), ...STATIC });
export const useDsWell = (id: string) => useQuery({ queryKey: ["ds-well", id], queryFn: () => api.get<DsWell>(`/dataset/wells/${id}`), enabled: !!id, ...STATIC });
export const useDsDaily = (id: string, cycle?: number | null) =>
  useQuery({
    queryKey: ["ds-daily", id, cycle ?? "all"],
    queryFn: () => api.get<DsDaily>(`/dataset/wells/${id}/daily${cycle ? `?cycle=${cycle}` : ""}`),
    enabled: !!id,
    placeholderData: keepPreviousData,
    ...STATIC,
  });
export const useDsRisk = (id: string, day: number | null) =>
  useQuery({ queryKey: ["ds-risk", id, day], queryFn: () => api.get<DsRisk>(`/dataset/wells/${id}/risk?day=${day}`), enabled: !!id && day !== null, placeholderData: keepPreviousData, ...STATIC });
export const useDsSpm = (id: string, day: number | null) =>
  useQuery({ queryKey: ["ds-spm", id, day], queryFn: () => api.get<DsSpm>(`/dataset/wells/${id}/spm${day !== null ? `?day=${day}` : ""}`), enabled: !!id, placeholderData: keepPreviousData, ...STATIC });
export const useDsSpmHistory = (id: string, cycle?: number | null) =>
  useQuery({ queryKey: ["ds-spm-hist", id, cycle ?? "all"], queryFn: () => api.get<DsSpmHistory>(`/dataset/wells/${id}/spm-history${cycle ? `?cycle=${cycle}` : ""}`), enabled: !!id, ...STATIC });
export const useDsCyclePlan = (id: string, cycle: number | null, objective: string, unconstrained: boolean) =>
  useQuery({
    queryKey: ["ds-plan", id, cycle, objective, unconstrained],
    queryFn: () => api.get<CyclePlan>(`/dataset/wells/${id}/cycle-plan?objective=${objective}${cycle ? `&cycle=${cycle}` : ""}${unconstrained ? "&unconstrained=true" : ""}`),
    enabled: !!id,
    placeholderData: keepPreviousData,
    ...STATIC,
  });
export const useDsWhatIf = (id: string, cycle: number | null, controls: CycleControls | null) =>
  useQuery({
    queryKey: ["ds-whatif", id, cycle, controls],
    queryFn: () => api.post<CycleWhatIf>(`/dataset/wells/${id}/cycle-whatif`, { cycle: cycle ?? undefined, scenarios: [{ label: "your settings", ...controls }] }),
    enabled: !!id && !!controls,
    placeholderData: keepPreviousData,
    ...STATIC,
  });
export const useDsModels = () => useQuery({ queryKey: ["ds-models"], queryFn: () => api.get<DsModels>("/dataset/models"), ...STATIC });
export const useDsEarlyWarning = () => useQuery({ queryKey: ["ds-ew"], queryFn: () => api.get<DsEarlyWarning>("/dataset/early-warning"), ...STATIC });
export const useDsAlerts = (day: number, limit = 25) =>
  useQuery({ queryKey: ["ds-alerts", day, limit], queryFn: () => api.get<DsAlerts>(`/dataset/early-warning/alerts?day=${day}&limit=${limit}`), placeholderData: keepPreviousData, ...STATIC });
export const useDsValidation = () => useQuery({ queryKey: ["ds-validation"], queryFn: () => api.get<DsValidation>("/dataset/validation"), ...STATIC });

export function useDsRebuild() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post<DsStatus>("/dataset/rebuild"),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["ds-status"] }),
  });
}
