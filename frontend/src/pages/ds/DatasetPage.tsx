import { useMemo, useState } from "react";
import { datasetFileUrl, getApiKey } from "../../api/client";
import { useDsModels, useDsRebuild, useDsStatus, useDsSummary, useDsValidation } from "../../api/dataset";
import { SourceNote } from "../../components/dataset";
import { Btn, ErrorNote, KpiTile, Loading, Panel, selectCls } from "../../components/ui";
import { fmt, fmtBytes, fmtInt } from "../../lib/format";

const STATUS_CLS: Record<string, string> = { PASS: "border-ok/40 bg-ok/10 text-ok", INFO: "border-accent/40 bg-accent/10 text-accent", WARN: "border-warn/40 bg-warn/10 text-warn", FAIL: "border-bad/50 bg-bad/10 text-bad" };

function KV({ data }: { data: Record<string, unknown> }) {
  return (
    <dl className="grid grid-cols-[1fr_auto] gap-x-4 gap-y-1 text-[13px]">
      {Object.entries(data).map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-muted">{k.replace(/_/g, " ")}</dt>
          <dd className="num text-right text-ink">{Array.isArray(v) ? v.join(" – ") : typeof v === "object" && v !== null ? Object.values(v).join(" / ") : String(v)}</dd>
        </div>
      ))}
    </dl>
  );
}

export function DatasetPage() {
  const { data: v, error } = useDsValidation();
  const { data: s } = useDsSummary();
  const { data: m } = useDsModels();
  const { data: status } = useDsStatus(false);
  const rebuild = useDsRebuild();
  const [filter, setFilter] = useState("");
  const checks = useMemo(
    () => (v?.checks ?? []).filter((c) => !filter || `${c.file} ${c.check} ${c.status} ${c.detail}`.toLowerCase().includes(filter.toLowerCase())),
    [v, filter],
  );
  if (error) return <ErrorNote error={error} />;
  if (!v || !s) return <Loading what="dataset" />;
  const calib = v.checks.filter((c) => c.file.startsWith("calibration"));
  const fc = v.field_case;

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-lg font-semibold text-ink">Dataset & validation</h1>
        <p className="text-xs text-muted">
          PULSE synthetic dataset v{s.dataset.version} · case {s.dataset.case} · seed {s.dataset.seed} · noise level {s.dataset.noise_level}
        </p>
      </div>

      <section className="rounded-lg border border-warn/40 bg-warn/6 p-4 text-[13px] leading-relaxed text-ink">
        <p className="font-semibold text-warn">Read this first</p>
        <p className="mt-1 text-muted">
          The data is synthetic: a physics-based generator calibrated to published Baghewala figures produced it, and no Oil India well data was used. Models therefore learn the generator&apos;s assumptions,
          not real well behaviour. There are no dynamometer cards in this dataset: card-based detection (the live twin&apos;s CNN) is trained on simulated cards only. The pipeline is built to retrain on real data
          in the same schema.
        </p>
        <ul className="mt-2 list-disc space-y-0.5 pl-5 text-muted">
          {v.rules.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
      </section>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <KpiTile label="Wells" value={s.dataset.n_wells} sub={`${s.totals.wells_stopped_early} stopped at the economic limit`} />
        <KpiTile label="Steam cycles" value={fmtInt(s.dataset.n_cycles)} sub={`median cycle SOR ${fmt(s.totals.median_cycle_sor, 2)}`} />
        <KpiTile label="Well-days" value={fmtInt(s.dataset.n_rows)} sub={`${fmtInt(s.dataset.n_running_days)} with the pump running`} />
        <KpiTile label="Failures" value={s.dataset.n_failures} sub={Object.entries(s.totals.failures_by_mode).map(([k, n]) => `${n} ${k.replace("_", " ")}`).join(" · ")} />
        <KpiTile label="Oil produced" value={fmtInt(s.totals.oil_m3 / 1000)} unit="k m³" sub={`${fmt(s.totals.mean_oil_bbl_d_per_producing_well, 1)} bbl/d per producing well`} />
        <KpiTile label="Checks" value={`${v.status_counts.PASS ?? 0} pass`} sub={`${v.status_counts.INFO ?? 0} info · ${v.status_counts.WARN ?? 0} warn · ${v.status_counts.FAIL ?? 0} fail`} tone={(v.status_counts.FAIL ?? 0) ? "bad" : "ok"} />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.4fr_1fr]">
        <Panel title="Calibration against published figures" pad={false}>
          <table className="w-full text-[13px]">
            <tbody>
              {calib.map((c) => (
                <tr key={c.check} className="border-b border-line/60">
                  <td className="px-3 py-1.5 text-muted">{c.check}</td>
                  <td className="px-3 py-1.5 text-ink">{c.detail}</td>
                  <td className="px-3 py-1.5 text-right">
                    <span className={`rounded border px-1.5 py-0.5 text-[10px] font-semibold ${STATUS_CLS[c.status] ?? ""}`}>{c.status}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
        <Panel title="Files" subtitle={v.manifest.imported_at ? `imported ${v.manifest.imported_at}` : undefined} pad={false}>
          <table className="w-full text-[13px]">
            <tbody>
              {v.files.map((f) => {
                const t = v.manifest.tables?.[f.name.replace(".parquet", "")];
                return (
                  <tr key={f.name} className="border-b border-line/60">
                    <td className="px-3 py-1.5">
                      {f.downloadable ? (
                        <a className="text-accent hover:underline" href={datasetFileUrl(f.name)}>
                          {f.name}
                        </a>
                      ) : (
                        <span className="text-ink">{f.name}</span>
                      )}
                      {t && (
                        <div className="text-[11px] text-faint">
                          {fmtInt(t.rows)} rows · from {fmtBytes(t.source_bytes)} CSV · lossless · sha256 {t.source_sha256.slice(0, 10)}…
                        </div>
                      )}
                    </td>
                    <td className="num px-3 py-1.5 text-right text-muted">{fmtBytes(f.bytes)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Panel>
      </div>

      {v.plot && (
        <Panel title="Generator validation plot" subtitle="Three cycles of WELL-001: temperature decays after the heat pulse, viscosity climbs back toward ~15,000 cP, oil peaks shortly after soak, water cut falls">
          <img src={datasetFileUrl(v.plot)} alt="Validation plot: temperature, viscosity, oil rate and water cut for three cycles of WELL-001" className="w-full rounded border border-line bg-white" loading="lazy" />
        </Panel>
      )}

      {fc && (
        <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-4">
          <Panel title="Pump & drive">
            <KV data={fc.pump} />
          </Panel>
          <Panel title="Rod-float criterion">
            <KV data={fc.rod_float} />
            <SourceNote>mu_water was not in the config: it was recovered from the data (RFI reproduced with 0.4% median error).</SourceNote>
          </Panel>
          <Panel title="Reservoir & fluid">
            <KV data={fc.reservoir} />
          </Panel>
          <Panel title="Control ranges">
            <KV data={fc.controls} />
          </Panel>
        </div>
      )}

      {m && (
        <Panel title="Models trained on this dataset" subtitle={`LightGBM ${m.lightgbm} · built ${m.built_at} in ${fmt(m.build_seconds, 0)} s · ${m.folds.n_folds}-fold cross-validation grouped by ${m.folds.by}`} pad={false}>
          <table className="w-full text-[13px]">
            <thead className="text-xs text-muted">
              <tr className="border-b border-line">
                <th className="px-3 py-1.5 text-left font-medium">Model</th>
                <th className="px-3 py-1.5 text-left font-medium">Target</th>
                <th className="px-3 py-1.5 text-left font-medium">Held-out result</th>
                <th className="px-3 py-1.5 text-left font-medium">Baseline</th>
              </tr>
            </thead>
            <tbody className="text-muted">
              <tr className="border-b border-line/60">
                <td className="px-3 py-1.5 text-ink">Thermal residual (twin nowcast)</td>
                <td className="px-3 py-1.5">oil rate, 7 days ahead</td>
                <td className="num px-3 py-1.5 text-ink">MAE {fmt(m.nowcast.metrics_oil_rate_bbl_d["twin: prior + GBM residual"].mae, 2)} bbl/d</td>
                <td className="num px-3 py-1.5">
                  physics prior {fmt(m.nowcast.metrics_oil_rate_bbl_d["physics prior only"].mae, 2)} · persistence {fmt(m.nowcast.metrics_oil_rate_bbl_d["persistence (last 7 d ratio)"].mae, 2)}
                </td>
              </tr>
              <tr className="border-b border-line/60">
                <td className="px-3 py-1.5 text-ink">CSS cycle response</td>
                <td className="px-3 py-1.5">oil per cycle-day</td>
                <td className="num px-3 py-1.5 text-ink">R² {fmt(m.cycle_model.metrics["GBM (out-of-fold)"].r2, 3)}</td>
                <td className="num px-3 py-1.5">field mean by cycle {fmt(m.cycle_model.metrics["field mean for cycle number"].r2, 3)}</td>
              </tr>
              <tr className="border-b border-line/60">
                <td className="px-3 py-1.5 text-ink">Failure early warning</td>
                <td className="px-3 py-1.5">failure within 14 d</td>
                <td className="num px-3 py-1.5 text-ink">PR-AUC {fmt(m.early_warning.results["GBM (all features)"].pr_auc, 3)}</td>
                <td className="num px-3 py-1.5">best rule {fmt(Math.max(...Object.entries(m.early_warning.results).filter(([k]) => k.startsWith("rule")).map(([, r]) => r.pr_auc)), 3)}</td>
              </tr>
              <tr>
                <td className="px-3 py-1.5 text-ink">SPM advisor (physics)</td>
                <td className="px-3 py-1.5">days with rods floating</td>
                <td className="num px-3 py-1.5 text-ink">{fmt(m.spm_advisor.advisor.days_floating_pct, 2)}%</td>
                <td className="num px-3 py-1.5">logged operation {fmt(m.spm_advisor.logged.days_floating_pct, 2)}%</td>
              </tr>
            </tbody>
          </table>
          <p className="px-3 py-2 text-[11px] text-faint">{m.nowcast.temperature_viscosity_note}</p>
        </Panel>
      )}

      <Panel
        title="Automated validation checks"
        subtitle="validation_report.csv, as delivered with the dataset"
        pad={false}
        actions={<input aria-label="Filter checks" value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Filter…" className={`${selectCls} w-40`} />}
      >
        <div className="max-h-[420px] overflow-auto">
          <table className="w-full text-[13px]">
            <thead className="sticky top-0 bg-panel text-xs text-muted">
              <tr className="border-b border-line">
                <th className="px-3 py-1.5 text-left font-medium">File</th>
                <th className="px-3 py-1.5 text-left font-medium">Check</th>
                <th className="px-3 py-1.5 text-left font-medium">Status</th>
                <th className="px-3 py-1.5 text-left font-medium">Detail</th>
              </tr>
            </thead>
            <tbody>
              {checks.map((c, i) => (
                <tr key={i} className="border-b border-line/60 align-top">
                  <td className="num px-3 py-1.5 text-muted">{c.file}</td>
                  <td className="px-3 py-1.5 text-ink">{c.check}</td>
                  <td className="px-3 py-1.5">
                    <span className={`rounded border px-1.5 py-0.5 text-[10px] font-semibold ${STATUS_CLS[c.status] ?? ""}`}>{c.status}</span>
                  </td>
                  <td className="px-3 py-1.5 text-muted">{c.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel title="Updating the data" subtitle={`Store state: ${status?.state ?? "–"}`}>
        <div className="grid gap-4 text-[13px] lg:grid-cols-2">
          <div className="space-y-2 text-muted">
            <p>Drop a new version (or real field data in the same schema) into the data folder and rebuild:</p>
            <pre className="num overflow-x-auto rounded border border-line bg-bg p-3 text-xs text-ink">
              {`cd backend
python -m scripts.import_dataset --src /path/to/csv/folder
python -m scripts.build_field_store`}
            </pre>
            <p>The API also rebuilds automatically on its next start when the data changed.</p>
          </div>
          <div className="space-y-2 text-muted">
            <p>Or rebuild the store and retrain the models from the files already on the server (runs in the background, about 1-3 minutes).</p>
            <Btn variant="primary" disabled={rebuild.isPending || status?.state === "building"} onClick={() => rebuild.mutate()}>
              {status?.state === "building" ? "Building…" : "Rebuild & retrain"}
            </Btn>
            {!getApiKey() && <p className="text-[11px] text-faint">If the server has an API key, set it on the live twin&apos;s Ingest & Models page first.</p>}
            {rebuild.error && <ErrorNote error={rebuild.error} />}
            {rebuild.isSuccess && <p className="text-ok">Rebuild started. Pages reload the new results once it finishes.</p>}
          </div>
        </div>
      </Panel>
    </div>
  );
}
