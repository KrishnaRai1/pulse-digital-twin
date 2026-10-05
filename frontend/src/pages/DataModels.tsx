import { useMemo, useState } from "react";
import { api, getApiKey, sampleUrl, setApiKey } from "../api/client";
import { useSamples, useSystemInfo } from "../api/hooks";
import type { ProductionReport } from "../api/types";
import { Chart } from "../components/Chart";
import { Btn, ErrorNote, KpiTile, Loading, Panel, inputCls } from "../components/ui";
import { axisBase, tooltipBase } from "../lib/chartHelpers";
import { COLORS } from "../lib/echarts";
import { classLabel, fmt, fmtBytes, fmtDateTime, fmtInt, fmtPct } from "../lib/format";

function ProductionUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [align, setAlign] = useState(true);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  const [rep, setRep] = useState<ProductionReport | null>(null);

  const send = async (f: File) => {
    setBusy(true);
    setErr(null);
    setRep(null);
    try {
      setRep(await api.upload<ProductionReport>("/ingest/production", f, { align_to_now: String(align) }));
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
    }
  };
  const sample = async () => {
    try {
      const res = await fetch(sampleUrl("sample_production_log.csv"));
      if (!res.ok) throw new Error("sample file not available");
      const f = new File([await res.blob()], "sample_production_log.csv", { type: "text/csv" });
      setFile(f);
      await send(f);
    } catch (e) {
      setErr(e);
    }
  };

  return (
    <Panel title="Production & sensor log ingestion" subtitle="CSV or Excel → validation, unit/range checks, outlier rejection, gap filling → time-series store">
      <div className="flex flex-wrap items-center gap-3">
        <input type="file" accept=".csv,.xlsx" aria-label="Production log file" onChange={(e) => setFile(e.target.files?.[0] ?? null)} className="text-xs text-muted file:mr-3 file:rounded-md file:border file:border-line file:bg-panel2 file:px-3 file:py-1.5 file:text-xs file:text-ink" />
        <label className="flex items-center gap-2 text-xs text-muted" title="Shifts the file's timestamps so its last sample is 'now': lets you replay historical or sample files on the live charts">
          <input type="checkbox" checked={align} onChange={(e) => setAlign(e.target.checked)} /> align timestamps to now
        </label>
        <Btn variant="primary" disabled={!file || busy} onClick={() => file && send(file)}>
          {busy ? "Processing…" : "Upload & clean"}
        </Btn>
        <Btn disabled={busy} onClick={sample}>
          Use the bundled dirty sample
        </Btn>
      </div>
      <p className="mt-2 text-[11px] text-faint">Columns are matched by name/alias (timestamp, well_id, spm, oil_rate_m3d, water_cut, visc_cp, motor_kw, pprl_kn, …). Nothing is changed silently: the full cleaning report appears below.</p>
      {err != null && <div className="mt-3"><ErrorNote error={err} /></div>}
      {rep && (
        <div className="mt-4 space-y-3">
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
            <KpiTile label="Rows in" value={fmtInt(rep.rows_in)} />
            <KpiTile label="Rows stored" value={fmtInt(rep.rows_written)} tone="ok" />
            <KpiTile label="Duplicates dropped" value={rep.dropped_duplicates} />
            <KpiTile label="Unknown wells" value={rep.dropped_unknown_well} tone={rep.dropped_unknown_well ? "warn" : "neutral"} />
            <KpiTile label="Wells" value={Object.keys(rep.wells).length} sub={Object.keys(rep.wells).join(", ")} />
          </div>
          {rep.warnings.map((w) => (
            <div key={w} className="rounded-md border border-warn/40 bg-warn/10 px-3 py-1.5 text-xs text-warn">{w}</div>
          ))}
          <div className="overflow-x-auto rounded-md border border-line">
            <table className="w-full min-w-[720px] text-xs">
              <thead className="border-b border-line bg-panel2 text-[11px] uppercase tracking-wide text-muted">
                <tr>
                  {["Column", "Missing before", "Out of range", "Outliers (Hampel)", "Gaps filled", "Missing after", "Mean", "Min", "Max"].map((h, i) => (
                    <th key={h} className={`px-3 py-2 font-semibold ${i === 0 ? "text-left" : "text-right"}`}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(rep.columns).map(([c, s]) => (
                  <tr key={c} className="border-b border-line/50">
                    <td className="px-3 py-1.5 font-medium text-ink">{c}</td>
                    <td className="num px-3 py-1.5 text-right">{s.missing_before}</td>
                    <td className={`num px-3 py-1.5 text-right ${s.out_of_range ? "text-warn" : ""}`}>{s.out_of_range}</td>
                    <td className={`num px-3 py-1.5 text-right ${s.statistical_outliers ? "text-warn" : ""}`}>{s.statistical_outliers}</td>
                    <td className="num px-3 py-1.5 text-right text-ok">{s.filled}</td>
                    <td className="num px-3 py-1.5 text-right">{s.missing_after}</td>
                    <td className="num px-3 py-1.5 text-right">{fmt(s.mean, 2)}</td>
                    <td className="num px-3 py-1.5 text-right">{fmt(s.min, 2)}</td>
                    <td className="num px-3 py-1.5 text-right">{fmt(s.max, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-[11px] text-faint">
            Time range {fmtDateTime(rep.time_range[0])} → {fmtDateTime(rep.time_range[1])}
            {rep.median_interval_s ? ` · median interval ${fmt(rep.median_interval_s / 60, 1)} min` : ""}
            {rep.time_shift_seconds ? ` · shifted by ${fmt(rep.time_shift_seconds / 86400, 1)} days` : ""}. Open a well's digital twin to see the data on the 24 h charts.
          </p>
        </div>
      )}
    </Panel>
  );
}

function ConfusionMatrix({ classes, m }: { classes: string[]; m: number[][] }) {
  const opt = useMemo(() => {
    const data: [number, number, number][] = [];
    m.forEach((row, i) => row.forEach((v, j) => data.push([j, i, v])));
    const max = Math.max(...m.flat(), 1);
    return {
      tooltip: { ...tooltipBase, trigger: "item", formatter: (p: { data: number[] }) => `true ${classLabel(classes[p.data[1]])}<br/>predicted ${classLabel(classes[p.data[0]])}: <b>${p.data[2]}</b>` },
      grid: { left: 130, right: 12, top: 8, bottom: 78 },
      xAxis: { type: "category", data: classes.map(classLabel), name: "Predicted", nameLocation: "middle", nameGap: 62, ...axisBase, splitLine: { show: false }, axisLabel: { ...axisBase.axisLabel, rotate: 30, interval: 0 } },
      yAxis: { type: "category", data: classes.map(classLabel), name: "", inverse: true, ...axisBase, splitLine: { show: false } },
      visualMap: { show: false, min: 0, max, inRange: { color: ["#0b1928", "#1a324b", "#f59e0b"] } },
      series: [{ type: "heatmap", data, label: { show: true, color: COLORS.ink, fontSize: 11 } }],
    };
  }, [classes, m]);
  return <Chart option={opt} height={280} ariaLabel="CNN confusion matrix on held-out synthetic cards" />;
}

export function DataModels() {
  const { data: info, error, isLoading } = useSystemInfo();
  const { data: samples } = useSamples();
  const [key, setKey] = useState(getApiKey());
  const [saved, setSaved] = useState(false);
  if (isLoading) return <Loading what="system information" />;
  if (error) return <ErrorNote error={error} />;
  if (!info) return null;
  const cnn = info.models.cnn;
  const ml = info.models.thermal_ml;

  return (
    <div className="space-y-4">
      <ProductionUpload />

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Dynamometer CNN" subtitle="1-D convolutional network on surface + downhole cards (PyTorch)">
          {!cnn ? (
            <p className="text-sm text-bad">Model not loaded. Run <code className="num">python -m scripts.train_models</code>.</p>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <KpiTile label="Held-out accuracy" value={fmtPct(cnn.val_accuracy, 1)} sub={`${cnn.n_val} synthetic cards`} />
                <KpiTile label="Latency p50 / p95" value={`${fmt(cnn.latency_ms_p50, 2)}`} unit="ms" sub={`p95 ${fmt(cnn.latency_ms_p95, 2)} ms · budget ${info.models.inference_budget_ms} ms`} tone="ok" />
                <KpiTile label="Parameters" value={fmtInt(cnn.parameters)} />
                <KpiTile label="Classes" value={cnn.classes.length} />
              </div>
              <div className="mt-3">
                <ConfusionMatrix classes={cnn.classes} m={cnn.confusion_matrix} />
              </div>
              <div className="mt-3 rounded-md border border-warn/30 bg-warn/8 px-3 py-2 text-[11px] leading-relaxed text-warn/90">
                Trained on {cnn.data}. Accuracy on synthetic cards says the pipeline works, not how it will perform on field data: retrain with labelled field cards before relying on it operationally.
              </div>
            </>
          )}
        </Panel>

        <Panel title="Thermal residual model" subtitle="LightGBM learns what the physics misses (ln q_observed / q_physics)">
          {!ml ? (
            <p className="text-sm text-warn">Not loaded: the twin is running on physics only.</p>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                <KpiTile label="Hold-out error, physics" value={fmtPct(ml.holdout_mape_physics, 1)} sub="MAPE of oil rate" />
                <KpiTile label="Hold-out error, corrected" value={fmtPct(ml.holdout_mape_corrected, 1)} sub="MAPE of oil rate" tone="ok" />
                <KpiTile label="R² of residual" value={fmt(ml.holdout_r2_log_factor, 2)} sub={`${fmtInt(ml.n_rows)} rows · ${ml.n_cycles} cycles`} />
              </div>
              <h3 className="mb-2 mt-4 text-[11px] font-semibold uppercase tracking-wider text-muted">Feature importance (gain)</h3>
              <ul className="space-y-1.5">
                {Object.entries(ml.feature_importance)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 8)
                  .map(([f, v]) => (
                    <li key={f} className="grid grid-cols-[150px_1fr_44px] items-center gap-2 text-xs">
                      <span className="truncate text-muted">{f.replace(/_/g, " ")}</span>
                      <div className="h-2 rounded bg-panel2">
                        <div className="h-full rounded bg-accent/70" style={{ width: `${Math.max(2, v * 100)}%` }} />
                      </div>
                      <span className="num text-right text-muted">{(v * 100).toFixed(0)}%</span>
                    </li>
                  ))}
              </ul>
              <div className="mt-3 rounded-md border border-warn/30 bg-warn/8 px-3 py-2 text-[11px] leading-relaxed text-warn/90">
                Source: {ml.source} data. The correction is clipped to ×{ml.max_correction_factor === 2 ? "0.5–×2" : ml.max_correction_factor} of the physics, so a poor model can never dominate the twin.
              </div>
            </>
          )}
        </Panel>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Panel title="Safety & control" subtitle={info.control_mode === "advisory" ? "Advisory mode (default)" : "Closed-loop mode"}>
          <ul className="space-y-1.5 text-xs text-muted">
            {info.safety.hard_limits.map((l) => (
              <li key={l} className="flex gap-2">
                <span className="text-ok">✓</span>
                <span>Hard limit: {l}</span>
              </li>
            ))}
            <li className="flex gap-2"><span className="text-ok">✓</span><span>{info.safety.mpc_horizon_days}-day receding-horizon MPC; speed increases limited to {info.safety.max_speed_increase_spm_per_day} SPM/day</span></li>
            <li className="flex gap-2"><span className="text-ok">✓</span><span>Every set-point is re-validated against the limits at the moment it is written</span></li>
            <li className="flex gap-2"><span className="text-ok">✓</span><span>Full audit trail of recommendations, decisions and writes</span></li>
          </ul>
          <p className="mt-3 text-[11px] text-faint">
            Economics used by the optimiser: oil ${info.economics.oil_usd_per_m3}/m³ · power ${info.economics.power_usd_per_kwh}/kWh · steam ${info.economics.steam_usd_per_m3}/m³ (illustrative defaults, set via environment).
          </p>
        </Panel>

        <Panel title="API access" subtitle={info.auth_required_for_writes ? "Write operations require an API key" : "Open demo mode: no key required"}>
          <label className="block text-xs text-muted">
            X-API-Key
            <input type="password" autoComplete="off" className={`${inputCls} mt-1`} value={key} onChange={(e) => { setKey(e.target.value); setSaved(false); }} placeholder={info.auth_required_for_writes ? "required for approvals, uploads, edits" : "not required"} />
          </label>
          <div className="mt-2 flex items-center gap-2">
            <Btn small variant="primary" onClick={() => { setApiKey(key); setSaved(true); }}>Save for this browser session</Btn>
            {saved && <span className="text-xs text-ok">Saved</span>}
          </div>
          <p className="mt-2 text-[11px] text-faint">Kept in sessionStorage only (cleared when the tab closes) and sent solely on state-changing requests.</p>
        </Panel>

        <Panel title="System" subtitle="Runtime status">
          <dl className="space-y-1 text-xs">
            {[
              ["Database", info.database.dialect],
              ["Telemetry retention", `${info.database.telemetry_retention_hours} h`],
              ["Historian source", info.simulator.enabled ? `simulator (${info.simulator.sim_minutes_per_tick} min per ${info.simulator.tick_seconds} s)` : "external feed"],
              ["Simulator ticks", fmtInt(info.simulator.ticks)],
              ["Dashboards connected", info.stream_clients],
            ].map(([k, v]) => (
              <div key={String(k)} className="flex justify-between border-b border-line/50 py-1">
                <dt className="text-muted">{k}</dt>
                <dd className="num text-ink">{v}</dd>
              </div>
            ))}
          </dl>
          {samples && samples.length > 0 && (
            <>
              <h3 className="mb-1 mt-4 text-[11px] font-semibold uppercase tracking-wider text-muted">Sample files</h3>
              <ul className="space-y-1 text-xs">
                {samples.map((s) => (
                  <li key={s.name}>
                    <a className="text-accent hover:underline" href={sampleUrl(s.name)} download>{s.name}</a> <span className="text-faint">{fmtBytes(s.bytes)}</span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </Panel>
      </div>
    </div>
  );
}
