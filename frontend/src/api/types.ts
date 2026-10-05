// Response shapes of the PULSE REST API (see backend/app/api and docs/API.md).

export type Status = "green" | "amber" | "red";
export type Phase = "INJECTION" | "SOAK" | "PRODUCTION" | "CYCLE_END";
export type DynoClass = "NORMAL" | "ROD_FLOATING" | "PUMP_UNSETTING_RISK" | "FLUID_POUND" | "GAS_INTERFERENCE";

export interface WellSummary {
  id: string;
  name: string;
  pad: string;
  x_m: number;
  y_m: number;
  ts: number;
  phase: Phase;
  cycle_no: number;
  day_in_cycle: number;
  prod_day: number;
  producing: boolean;
  status: Status;
  reasons: string[];
  spm: number;
  setpoint_spm: number;
  oil_rate_m3d: number;
  liquid_rate_m3d: number;
  water_cut: number;
  visc_cp: number;
  mu_eff_cp: number;
  t_avg_c: number;
  t_wellbore_c: number;
  thermal_health: number;
  kw: number;
  cum_oil_m3: number;
  steam_injected_m3: number;
  sor: number | null;
  float_index: number | null;
  spm_ceiling: number | null;
  diag: { label: DynoClass; probability: number } | null;
}

export interface Kpis {
  total_oil_m3d: number;
  total_oil_bbld: number;
  total_liquid_m3d: number;
  avg_sor: number | null;
  total_power_kw: number;
  wells: number;
  pumping_wells: number;
  status_counts: Record<Status, number>;
  active_alerts: number;
  sim_ts: number;
}

export interface Overview {
  kpis: Kpis;
  wells: WellSummary[];
  control_mode: "advisory" | "closed_loop";
}

export interface FieldTrend {
  ts: number[];
  oil_m3d: number[];
  liquid_m3d: number[];
  power_kw: number[];
  bin_seconds: number;
}

export interface AlertRow {
  id: number;
  ts: number;
  well_id: string;
  kind: DynoClass;
  severity: "red" | "amber";
  message: string;
  probability: number;
}
export interface AlertsResponse {
  active: { well_id: string; kind: DynoClass }[];
  history: AlertRow[];
}

export interface CycleRow {
  well_id: string;
  cycle_no: number;
  start_ts: number;
  steam_m3: number;
  quality: number;
  inj_rate_m3d: number;
  inj_pressure_mpa: number;
  soak_days: number;
  prod_days: number;
  source: string;
}

export interface SpmLimits {
  drive: number;
  rod_float: number;
  rod_fatigue: number;
  unit_structure: number;
  max_allowed: number;
  binding: string;
  feasible: boolean;
  min: number;
}

export interface WellState {
  phase: Phase;
  cycle_no: number;
  day_in_cycle: number;
  prod_day: number;
  t_avg_c: number;
  t_wellbore_c: number;
  r_heated_m: number;
  mu_eff_phys_cp: number;
  mu_eff_cp: number;
  q_oil_phys_m3d: number;
  q_oil_m3d: number;
  water_cut: number;
  q_liquid_m3d: number;
  cum_oil_m3: number;
  thermal_health: number;
  steam_injected_m3: number;
  steam_total_m3: number;
  ml_factor: number;
  t_wellhead_c: number;
  mu_tub_cp: number;
  t_steam_c: number;
}

export interface MlFeature {
  feature: string;
  contribution: number;
  value: number;
}

export interface WellDetail {
  summary: WellSummary;
  state: WellState;
  config: {
    design: {
      pump_depth_m: number;
      plunger_mm: number;
      stroke_m: number;
      min_spm: number;
      max_spm: number;
      unit_rating_kn: number;
      motor_kw_rated: number;
      rods: { length_m: number; dia_mm: number }[];
      buoyant_rod_weight_kn: number;
      pump_capacity_m3d_at_1spm: number;
    };
    reservoir: { thickness_m: number; t_res_c: number; perm_md: number; depth_m: number; drainage_radius_m: number };
  };
  limits: SpmLimits;
  setpoint_spm: number;
  cycles: CycleRow[];
  ml_top_features: MlFeature[];
  sim_ts: number;
}

export interface Telemetry {
  well_id: string;
  columns: string[];
  series: Record<string, (number | null)[]>;
  now: number;
}

export interface WellProfile {
  day_in_cycle: number;
  wellbore: {
    depth_m: number[];
    t_geotherm_c: number[];
    t_flowing_c: number[];
    pressure_kpa: number[];
    pump_depth_m: number;
    pay_top_m: number;
    pay_bottom_m: number;
    fluid_level_m: number;
  };
  radial: { r_m: number[]; t_c: number[]; mu_cp: number[]; r_heated_m: number; r_drain_m: number };
  cross_section: { r_m: number[]; z_m: number[]; t_c: number[][]; note: string };
  state: WellState;
}

export interface ViscosityCurves {
  t_days: number[];
  phase: number[];
  mu_eff_phys_cp: number[];
  mu_eff_cp: number[];
  t_avg_c: number[];
  q_oil_phys_m3d: number[];
  q_oil_m3d: number[];
  now_day: number;
  inj_days: number;
  soak_days: number;
  prod_days: number;
  ml_used: boolean;
}

export interface DynoCard {
  well_id: string;
  ts: number;
  source: string;
  spm: number;
  position: number[];
  load: number[]; // kN
  dh_position: number[];
  dh_load: number[]; // kN
  label: DynoClass;
  probability: number;
  probabilities: Record<DynoClass, number>;
  latency_ms: number;
  buoyant_rod_weight_kn: number;
  truth?: string;
  latency_ms_total?: number;
  within_budget?: boolean;
}

export interface Baseline {
  position: number[];
  load: number[];
  dh_position: number[];
  dh_load: number[];
}
export interface LatestCard {
  well_id: string;
  card: DynoCard | null;
  baseline: Baseline | null;
}

export interface StoredCard {
  id: number;
  well_id: string;
  ts: number;
  source: string;
  spm: number;
  position: number[];
  load: number[]; // N
  label: DynoClass;
  probability: number;
}

export interface Driver {
  factor: string;
  detail: string;
  effect: string;
  weight: number;
}
export interface Contribution {
  name: string;
  usd_per_day: number;
}
export interface PlanDay {
  day: number;
  spm: number;
  oil_m3d: number;
  kw: number;
  float_index: number;
  fillage: number;
  mu_tub_cp: number;
  q_liquid_m3d: number;
}
export interface Candidate {
  spm: number;
  oil_m3d: number;
  kw: number;
  fillage: number;
  float_index: number;
  fatigue_utilisation: number;
  reward_usd: number;
  violations: string[];
}
export interface Recommendation {
  well_id: string;
  ts: number;
  applicable: boolean;
  reason?: string;
  mode: "advisory" | "closed_loop";
  current_spm?: number;
  recommended_spm?: number;
  action?: "hold" | "adjust";
  needs_resteam?: boolean;
  limits?: SpmLimits;
  plan?: PlanDay[];
  candidates?: Candidate[];
  envelope?: {
    spm: number[];
    float_index: number[];
    fatigue_utilisation: number[];
    fillage: number[];
    oil_m3d: number[];
    kw: number[];
    reward_usd: number[];
    float_margin: number;
  };
  xai?: {
    action: string;
    narrative: string;
    drivers: Driver[];
    contributions: Contribution[];
    uplift_usd_per_day: number;
    ml_correction: MlFeature[];
  };
  safety?: { passes: boolean; violations: string[] };
  notes?: string[];
  objective_usd?: number;
  hold_objective_usd?: number;
}

export interface Advisory {
  id: number;
  ts: number;
  well_id: string;
  current_spm: number;
  recommended_spm: number;
  status: "pending" | "applied" | "rejected" | "expired";
  payload: {
    narrative: string;
    drivers: Driver[];
    contributions: Contribution[];
    needs_resteam: boolean;
    uplift_usd_per_day: number;
    safety: { passes: boolean; violations: string[] };
  };
  decided_ts: number | null;
  decided_by: string | null;
  note: string | null;
}

export interface AuditRow {
  id: number;
  ts: number;
  actor: string;
  action: string;
  well_id: string | null;
  detail: Record<string, unknown> | null;
}

export interface WhatIfScenario {
  label: string;
  spec: { cycle_no: number; steam_m3: number; quality: number; inj_rate_m3d: number; inj_pressure_mpa: number; soak_days: number; prod_days: number; inj_days: number };
  t_days: number[];
  phase: number[];
  q_oil_m3d: number[];
  q_oil_phys_m3d: number[];
  cum_oil_m3: number[];
  cum_oil_phys_m3: number[];
  mu_eff_cp: number[];
  t_avg_c: number[];
  summary: {
    cum_oil_m3: number;
    cum_oil_phys_m3: number;
    peak_rate_m3d: number;
    sor: number | null;
    r_heated_m: number;
    economic_limit_day: number | null;
    steam_bh_quality: number;
    wellbore_loss_pct: number;
  };
}
export interface WhatIfResponse {
  well_id: string;
  base: string;
  scenarios: WhatIfScenario[];
  ml_used: boolean;
  latency_ms: number;
}
export interface ScenarioIn {
  label?: string;
  steam_m3?: number;
  quality?: number;
  inj_rate_m3d?: number;
  inj_pressure_mpa?: number;
  soak_days?: number;
  prod_days?: number;
}

export interface PlanResult {
  well_id: string;
  ml_used: boolean;
  rows: { steam_m3: number; cum_oil_m3: number; peak_rate_m3d: number; sor: number | null; profit_usd: number; marginal_oil_per_1000m3: number | null }[];
  best_profit_volume_m3: number;
  best_profit_usd: number;
  lowest_sor_volume_m3: number;
  note: string;
}

export interface Job<T = unknown> {
  id: string;
  kind: string;
  status: "queued" | "running" | "done" | "failed";
  result: T | null;
  error: string | null;
}

export interface ColumnReport {
  missing_before: number;
  missing_after: number;
  out_of_range: number;
  statistical_outliers: number;
  filled: number;
  mean: number | null;
  min: number | null;
  max: number | null;
}
export interface ProductionReport {
  rows_in: number;
  rows_out: number;
  rows_written: number;
  measures: string[];
  wells: Record<string, number>;
  warnings: string[];
  dropped_bad_timestamp: number;
  dropped_unknown_well: number;
  dropped_duplicates: number;
  columns: Record<string, ColumnReport>;
  time_range: [number, number];
  median_interval_s: number | null;
  time_shift_seconds?: number;
}
export interface CardIngestReport {
  format: string;
  cards_in: number;
  cards_ok: number;
  cards_rejected: number;
  rejected: { card: string; reason: string }[];
  outliers_fixed: number;
  wells: string[];
}
export interface CardUploadResult {
  cards: number;
  label_counts: Record<string, number>;
  wells: string[];
}

export interface SystemInfo {
  control_mode: "advisory" | "closed_loop";
  auth_required_for_writes: boolean;
  simulator: { enabled: boolean; tick_seconds: number; sim_minutes_per_tick: number; sim_time: number; ticks: number };
  database: { dialect: string; telemetry_retention_hours: number };
  stream_clients: number;
  models: {
    cnn: {
      classes: DynoClass[];
      val_accuracy: number;
      per_class_recall: Record<string, number>;
      confusion_matrix: number[][];
      parameters: number;
      latency_ms_p50: number;
      latency_ms_p95: number;
      n_train: number;
      n_val: number;
      data: string;
    } | null;
    thermal_ml: {
      source: string;
      n_rows: number;
      n_cycles: number;
      holdout_mape_physics: number;
      holdout_mape_corrected: number;
      holdout_r2_log_factor: number;
      max_correction_factor: number;
      feature_importance: Record<string, number>;
    } | null;
    inference_budget_ms: number;
  };
  safety: { hard_limits: string[]; mpc_horizon_days: number; max_speed_increase_spm_per_day: number; every_setpoint_revalidated_at_write: boolean };
  economics: { oil_usd_per_m3: number; power_usd_per_kwh: number; steam_usd_per_m3: number };
  data_provenance: string;
}

export interface SampleFile {
  name: string;
  bytes: number;
}
