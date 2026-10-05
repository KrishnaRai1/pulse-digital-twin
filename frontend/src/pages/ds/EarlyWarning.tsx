import { useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useDsAlerts, useDsEarlyWarning, useDsRisk, useDsTrend, type WarningResult } from "../../api/dataset";
import { Chart } from "../../components/Chart";
import { DayControl, RagPill, SourceNote, fmtDay, useAsOfDay } from "../../components/dataset";
import { Btn, ErrorNote, Loading, Panel } from "../../components/ui";
import { axisBase, gridBase, legendBase, tooltipBase } from "../../lib/chartHelpers";
import { COLORS, SERIES } from "../../lib/echarts";
import { fmt } from "../../lib/format";

const METRICS: [keyof WarningResult, string, (v: number) => string][] = [
  ["pr_auc", "PR-AUC", (v) => fmt(v, 3)],
  ["roc_auc", "ROC-AUC", (v) => fmt(v, 3)],
  ["failures_caught_with_ge3d_warning", "Caught ≥ 3 d ahead", (v) => `${fmt(v * 100, 0)}%`],
  ["median_lead_days", "Median lead", (v) => `${fmt(v, 0)} d`],
  ["false_alert_days_per_running_well_year", "False-alert days / well-yr", (v) => fmt(v, 1)],
];

export function EarlyWarning() {
  const { day } = useAsOfDay();
  const [params] = useSearchParams();
  const { data: ew, error } = useDsEarlyWarning();
  const { data: alerts } = useDsAlerts(day, 30);
  const { data: trend } = useDsTrend(3);
  const [sel, setSel] = useState<string | null>(params.get("well"));
  const well = sel ?? alerts?.wells[0]?.well_id ?? "";
  const { data: risk } = useDsRisk(well, well ? day : null);
  const nav = useNavigate();

  const prOpt = useMemo(() => {
    if (!ew) return null;
    return {
      tooltip: { ...tooltipBase, trigger: "item", formatter: (p: { seriesName: string; value: number[] }) => `${p.seriesName}<br/>recall ${fmt(p.value[0] * 100, 0)}% · precision ${fmt(p.value[1] * 100, 1)}%` },
      legend: { ...legendBase, type: "scroll" },
      grid: gridBase({ left: 50, right: 16, bottom: 36 }),
      xAxis: { type: "value", name: "recall", nameLocation: "middle", nameGap: 22, min: 0, max: 1, ...axisBase },
      yAxis: { type: "value", name: "precision", min: 0, max: 1, ...axisBase },
      series: Object.entries(ew.pr_curves).map(([name, pts], i) => ({
        name,
        type: "line",
        showSymbol: false,
        data: pts.map((p) => [p.recall, p.precision]),
        lineStyle: { width: i === 0 ? 2.4 : 1.3, color: i === 0 ? COLORS.accent : SERIES[(i + 1) % SERIES.length] },
        itemStyle: { color: i === 0 ? COLORS.accent : SERIES[(i + 1) % SERIES.length] },
      })),
    };
  }, [ew]);

  const impOpt = useMemo(() => {
    if (!ew) return null;
    const top = [...ew.top_features].slice(0, 12).reverse();
    return {
      tooltip: { ...tooltipBase, trigger: "item", valueFormatter: (v: number) => `${fmt(v * 100, 1)}% of gain` },
      grid: gridBase({ left: 170, right: 20, top: 8, bottom: 22 }),
      xAxis: { type: "value", ...axisBase, axisLabel: { ...axisBase.axisLabel, formatter: (v: number) => `${Math.round(v * 100)}%` } },
      yAxis: { type: "category", data: top.map((t) => t.feature), ...axisBase },
      series: [{ type: "bar", data: top.map((t) => t.importance), barWidth: 10, itemStyle: { color: COLORS.violet } }],
    };
  }, [ew]);

  const alertTrend = useMemo(() => {
    if (!trend) return null;
    return {
      tooltip: { ...tooltipBase },
      legend: { ...legendBase },
      grid: gridBase({ left: 46, right: 46, bottom: 50 }),
      dataZoom: [{ type: "inside" }, { type: "slider", height: 16, bottom: 6, borderColor: COLORS.line, textStyle: { color: COLORS.muted } }],
      xAxis: { type: "category", data: trend.day, ...axisBase },
      yAxis: [
        { type: "value", name: "wells", ...axisBase },
        { type: "value", name: "failures", ...axisBase, splitLine: { show: false }, minInterval: 1 },
      ],
      series: [
        {
          name: "Wells on alert",
          type: "line",
          data: trend.n_alert,
          showSymbol: false,
          lineStyle: { color: COLORS.bad, width: 1.3 },
          itemStyle: { color: COLORS.bad },
          markLine: { symbol: "none", silent: true, data: [{ xAxis: String(day - (day % 3)), lineStyle: { color: COLORS.warn }, label: { show: false } }] },
        },
        { name: "Wells with rods floating", type: "line", data: trend.n_floating, showSymbol: false, lineStyle: { color: COLORS.warn, width: 1 }, itemStyle: { color: COLORS.warn } },
        { name: "Failures", type: "bar", yAxisIndex: 1, data: trend.failures, itemStyle: { color: COLORS.muted }, barWidth: 2 },
      ],
    };
  }, [trend, day]);

  if (error) return <ErrorNote error={error} />;
  if (!ew) return <Loading what="early-warning model" />;
  const base = ew.baseline_report?.results ?? {};

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-lg font-semibold text-ink">Failure early warning</h1>
        <p className="text-xs text-muted">
          Gradient boosting on daily pump signals (no dynamometer cards in this dataset): probability of a rod part, pump unset or worn barrel within {ew.horizon_days} days. Labels come from the failure log
          and are never used as inputs.
        </p>
      </div>

      <Panel title="Model vs. simple rules" subtitle={ew.evaluation} pad={false}>
        <div className="overflow-x-auto">
          <table className="w-full text-[13px]">
            <thead className="text-xs text-muted">
              <tr className="border-b border-line">
                <th className="px-3 py-1.5 text-left font-medium">Model</th>
                {METRICS.map(([, l]) => (
                  <th key={l} className="px-3 py-1.5 text-right font-medium">
                    {l}
                  </th>
                ))}
                <th className="px-3 py-1.5 text-right font-medium">Dataset baseline PR-AUC</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(ew.results).map(([name, r], i) => (
                <tr key={name} className={`border-b border-line/60 ${i === 0 ? "text-ink" : "text-muted"}`}>
                  <td className="px-3 py-1.5">{i === 0 ? <span className="font-semibold text-accent">PULSE · {name}</span> : name}</td>
                  {METRICS.map(([k, l, f]) => (
                    <td key={l} className="num px-3 py-1.5 text-right">
                      {f(r[k] as number)}
                    </td>
                  ))}
                  <td className="num px-3 py-1.5 text-right text-faint">{base[name] ? fmt(base[name].pr_auc, 3) : "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="px-3 py-2 text-[11px] leading-relaxed text-faint">
          Threshold set to flag 2% of healthy days, as in the dataset&apos;s baseline report (early_warning_report.json, scored on its own held-out wells). Here every one of the 300 wells is scored out-of-fold, so
          the numbers are comparable in method but not on identical wells. A random model has PR-AUC ≈ {fmt(ew.n_positive_rows / ew.n_rows, 3)}. Fold spread of PR-AUC:{" "}
          {ew.fold_pr_auc.map((v) => fmt(v, 2)).join(" · ")}. Scores are an optimistic ceiling: in the generator, failures are caused by the same exposures used as features.
        </p>
      </Panel>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Precision-recall curves" subtitle="Out-of-fold scores on all running well-days">
          {prOpt && <Chart option={prOpt} height={280} ariaLabel="Precision recall curves" />}
        </Panel>
        <Panel title="What the model relies on" subtitle="Share of total split gain across folds">
          {impOpt && <Chart option={impOpt} height={280} ariaLabel="Feature importance" />}
        </Panel>
      </div>

      <DayControl />
      <div className="grid gap-4 xl:grid-cols-[1fr_400px]">
        <Panel title={`Ranked wells · ${fmtDay(day)}`} subtitle={`${alerts?.n_alert ?? 0} on alert · ${alerts?.n_watch ?? 0} on watch`} pad={false}>
          <div className="max-h-[420px] overflow-auto">
            <table className="w-full text-[13px]">
              <thead className="sticky top-0 bg-panel text-xs text-muted">
                <tr className="border-b border-line">
                  {["Well", "Level", "Risk", "RFI", "Fill %", "Load var %", "SPM", "Hindsight: failed ≤14 d"].map((h, i) => (
                    <th key={h} className={`px-2 py-1.5 font-medium ${i < 2 ? "text-left" : "text-right"}`}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(alerts?.wells ?? []).map((w) => (
                  <tr key={w.well_id} onClick={() => setSel(w.well_id)} className={`cursor-pointer border-b border-line/60 hover:bg-panel2 ${well === w.well_id ? "bg-accent/8" : ""}`}>
                    <td className="num px-2 py-1.5 text-ink">{w.well_id}</td>
                    <td className="px-2 py-1.5">{w.level && <RagPill status={w.level === "alert" ? "red" : w.level === "watch" ? "amber" : "green"} label={w.level} />}</td>
                    <td className="num px-2 py-1.5 text-right">{fmt(w.risk * 100, 2)}%</td>
                    <td className={`num px-2 py-1.5 text-right ${(w.rfi ?? 0) > 1 ? "text-bad" : ""}`}>{fmt(w.rfi, 2)}</td>
                    <td className="num px-2 py-1.5 text-right">{fmt(w.fillage_pct, 0)}</td>
                    <td className="num px-2 py-1.5 text-right">{fmt(w.load_variability_pct, 1)}</td>
                    <td className="num px-2 py-1.5 text-right">{fmt(w.spm, 1)}</td>
                    <td className={`num px-2 py-1.5 text-right ${w.failure_within_14d ? "text-bad" : "text-faint"}`}>{w.failure_within_14d ? `yes, in ${w.days_to_next_failure} d` : "no"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="px-3 py-2 text-[11px] text-faint">{alerts?.note}</p>
        </Panel>
        <Panel
          title={well ? `${well} · why` : "Select a well"}
          subtitle="TreeSHAP contributions in log-odds (red raises risk)"
          actions={well ? <Btn small onClick={() => nav(`/history/${well}?day=${day}`)}>History →</Btn> : undefined}
        >
          {!risk?.drivers ? (
            <p className="text-sm text-muted">Pump not running on this day.</p>
          ) : (
            <>
              <div className="num mb-3 text-2xl font-semibold text-ink">{fmt((risk.risk ?? 0) * 100, 2)}%</div>
              <ul className="space-y-2">
                {risk.drivers.map((d) => (
                  <li key={d.feature} className="text-xs">
                    <div className="flex justify-between">
                      <span className="text-muted">{d.label}</span>
                      <span className="num text-ink">{fmt(d.value, 2)}</span>
                    </div>
                    <div className="relative mt-1 h-2 rounded bg-panel2">
                      <span
                        className="absolute inset-y-0 rounded"
                        style={{
                          left: d.contribution >= 0 ? "50%" : `${50 - Math.min(50, Math.abs(d.contribution) * 12)}%`,
                          width: `${Math.min(50, Math.abs(d.contribution) * 12)}%`,
                          background: d.contribution >= 0 ? COLORS.bad : COLORS.ok,
                        }}
                      />
                      <span className="absolute inset-y-[-2px] left-1/2 w-px bg-line" />
                    </div>
                  </li>
                ))}
              </ul>
              <SourceNote>{risk.note}</SourceNote>
            </>
          )}
        </Panel>
      </div>

      <Panel title="Alerts over the field history" subtitle="Wells above the alert threshold each day, with failures for reference">
        {alertTrend && <Chart option={alertTrend} height={240} ariaLabel="Alert count over time" />}
      </Panel>
    </div>
  );
}
