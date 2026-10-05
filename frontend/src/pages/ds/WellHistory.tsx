import { useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useDsDaily, useDsRisk, useDsWell, type DsDaily, type Num } from "../../api/dataset";
import { Chart } from "../../components/Chart";
import { RagPill, SourceNote, WellSelect, fmtDay, useAsOfDay } from "../../components/dataset";
import { Btn, ErrorNote, KpiTile, Loading, Panel, selectCls } from "../../components/ui";
import { axisBase, gridBase, legendBase, tooltipBase } from "../../lib/chartHelpers";
import { COLORS } from "../../lib/echarts";
import { fmt, fmtInt, titleCase } from "../../lib/format";

const GROUP = "well-history";

/** [day, value] pairs with a null inserted at gaps so lines break over injection days. */
function pairs(day: number[], v: Num[]): [number, number | null][] {
  const out: [number, number | null][] = [];
  for (let i = 0; i < day.length; i++) {
    if (i > 0 && day[i] - day[i - 1] > 1) out.push([day[i] - 1, null]);
    out.push([day[i], v[i]]);
  }
  return out;
}

function runs(d: DsDaily, phase: string): [number, number][] {
  const out: [number, number][] = [];
  let start: number | null = null;
  for (let i = 0; i < d.day.length; i++) {
    const on = d.phase[i] === phase;
    if (on && start === null) start = d.day[i];
    if (!on && start !== null) {
      out.push([start, d.day[i - 1]]);
      start = null;
    }
  }
  if (start !== null) out.push([start, d.day[d.day.length - 1]]);
  return out;
}

export function WellHistory() {
  const { wellId = "WELL-001" } = useParams();
  const id = wellId.toUpperCase();
  const { day, setDay } = useAsOfDay();
  const [cycle, setCycle] = useState<number | null>(null);
  const { data: well, error } = useDsWell(id);
  const { data: d, isFetching } = useDsDaily(id, cycle);
  const { data: risk } = useDsRisk(id, day);
  const nav = useNavigate();

  const decor = useMemo(() => {
    if (!d || !well) return null;
    const inj = d.injection_windows.map(([a, b, c]) => [
      { xAxis: a, itemStyle: { color: "#5b9cf014" }, name: `injection, cycle ${c}` },
      { xAxis: b + 1 },
    ]);
    const wo = runs(d, "workover").map(([a, b]) => [{ xAxis: a, itemStyle: { color: "#e5484d1f" } }, { xAxis: b + 1 }]);
    const fails = well.failures
      .filter((f) => d.day.length && f.failure_date_day >= d.day[0] && f.failure_date_day <= d.day[d.day.length - 1])
      .map((f) => ({ xAxis: f.failure_date_day, lineStyle: { color: COLORS.bad, type: "dashed", width: 1 }, label: { formatter: titleCase(f.failure_mode), color: COLORS.bad, fontSize: 10 } }));
    const asof = d.day.length && day >= d.day[0] - 30 && day <= d.day[d.day.length - 1] + 30 ? [{ xAxis: day, lineStyle: { color: COLORS.warn, type: "solid", width: 1.2 }, label: { show: false } }] : [];
    return {
      markArea: { silent: true, label: { show: false }, data: [...inj, ...wo] },
      markLine: { symbol: "none", silent: true, lineStyle: { color: COLORS.bad, type: "dashed", width: 1 }, label: { position: "insideEndTop" }, data: [...fails, ...asof] },
    };
  }, [d, well, day]);

  const charts = useMemo(() => {
    if (!d || !decor) return null;
    const xMin = d.day[0] ?? 0;
    const xMax = d.day[d.day.length - 1] ?? 1;
    const x = { type: "value" as const, min: xMin, max: xMax, ...axisBase, axisLabel: { ...axisBase.axisLabel, formatter: (v: number) => `${v}` } };
    const base = (extra = {}) => ({
      tooltip: { ...tooltipBase, valueFormatter: (v: number | null) => (v === null || v === undefined ? "–" : fmt(v, 2)) },
      legend: { ...legendBase },
      grid: gridBase({ left: 56, right: 56, top: 36, bottom: 26 }),
      dataZoom: [{ type: "inside", xAxisIndex: 0 }],
      xAxis: x,
      ...extra,
    });
    // failure labels are drawn once (on the oil chart); the other charts show only the lines
    const quiet = { ...decor.markLine, data: (decor.markLine.data as object[]).map((m) => ({ ...m, label: { show: false } })) };
    const line = (name: string, data: [number, number | null][], color: string, opts: Record<string, unknown> = {}) => ({ name, type: "line", data, showSymbol: false, lineStyle: { width: 1.4, color }, itemStyle: { color }, ...opts });
    const dashed = (name: string, data: [number, number | null][], color: string, opts: Record<string, unknown> = {}) => line(name, data, color, { lineStyle: { width: 1.1, color, type: "dashed" }, ...opts });

    const oil = base({
      yAxis: { type: "value", name: "bbl/d", ...axisBase },
      series: [
        { ...line("Observed oil", pairs(d.day, d.oil_rate_bbl_day), COLORS.accent), ...decor },
        dashed("Physics prior", pairs(d.day, d.prior_oil_rate_bbl_day), COLORS.muted),
        line("Twin (prior + ML, 7 d ahead)", pairs(d.day, d.twin_oil_bbl_d), COLORS.violet, { lineStyle: { width: 1.2, color: COLORS.violet, opacity: 0.9 } }),
      ],
    });
    const thermo = base({
      yAxis: [
        { type: "value", name: "°C", ...axisBase },
        { type: "log", name: "cP", ...axisBase, splitLine: { show: false }, min: 1, max: 100000 },
      ],
      series: [
        { ...line("Temperature (state)", pairs(d.day, d.reservoir_temp_c), COLORS.orange), markLine: quiet },
        dashed("Temperature (prior)", pairs(d.day, d.prior_reservoir_temp_c), "#b07040"),
        line("Viscosity (state)", pairs(d.day, d.oil_viscosity_cp), COLORS.blue, { yAxisIndex: 1 }),
        dashed("Viscosity (prior)", pairs(d.day, d.prior_oil_viscosity_cp), "#4d78b3", { yAxisIndex: 1 }),
      ],
    });
    const pump = base({
      yAxis: [
        { type: "value", name: "SPM", min: 0, max: 10, ...axisBase },
        { type: "value", name: "RFI", min: 0, ...axisBase, splitLine: { show: false } },
      ],
      series: [
        { ...line("SPM", pairs(d.day, d.spm), COLORS.accent), markLine: quiet },
        line("Rod-floating index", pairs(d.day, d.rod_floating_index), COLORS.warn, {
          yAxisIndex: 1,
          markLine: { symbol: "none", silent: true, lineStyle: { color: COLORS.bad, type: "dotted" }, label: { formatter: "RFI 1 (rods float)", color: COLORS.bad, fontSize: 10, position: "insideEndTop" }, data: [{ yAxis: 1 }] },
        }),
      ],
    });
    const load = base({
      yAxis: [
        { type: "value", name: "fillage %", min: 0, max: 100, ...axisBase },
        { type: "value", name: "load var. %", min: 0, ...axisBase, splitLine: { show: false } },
      ],
      series: [
        { ...line("Pump fillage", pairs(d.day, d.pump_fillage_pct), COLORS.ok), markLine: quiet },
        line("Load variability", pairs(d.day, d.load_variability_pct), COLORS.violet, { yAxisIndex: 1 }),
      ],
    });
    const riskPct = d.risk.map((r) => (r === null ? null : Math.max(r * 100, 0.001)));
    const riskOpt = base({
      grid: gridBase({ left: 56, right: 56, top: 36, bottom: 54 }),
      dataZoom: [{ type: "inside", xAxisIndex: 0 }, { type: "slider", height: 16, bottom: 6, borderColor: COLORS.line, textStyle: { color: COLORS.muted } }],
      yAxis: { type: "log", name: "risk %", min: 0.001, max: 100, ...axisBase, axisLabel: { ...axisBase.axisLabel, formatter: (v: number) => (v >= 1 ? `${v}` : v.toString()) } },
      series: [
        {
          ...line("Failure risk, next 14 d", pairs(d.day, riskPct), COLORS.bad, { areaStyle: { color: "#e5484d14" } }),
          markLine: {
            symbol: "none",
            silent: true,
            data: [
              { yAxis: d.thresholds.risk_alert * 100, lineStyle: { color: COLORS.bad, type: "dotted" }, label: { formatter: "alert", color: COLORS.bad, fontSize: 10, position: "insideEndTop" } },
              { yAxis: d.thresholds.risk_watch * 100, lineStyle: { color: COLORS.warn, type: "dotted" }, label: { formatter: "watch", color: COLORS.warn, fontSize: 10, position: "insideEndTop" } },
              ...(quiet.data as object[]),
            ],
          },
        },
      ],
    });
    return { oil, thermo, pump, load, riskOpt };
  }, [d, decor]);

  if (error) return <ErrorNote error={error} />;
  if (!well || !d || !charts) return <Loading what={`history of ${id}`} />;
  const s = well.summary;
  const clickDay = (p: unknown) => {
    const v = (p as { value?: [number, number] }).value;
    if (Array.isArray(v)) setDay(v[0]);
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-lg font-semibold text-ink">Well history</h1>
          <WellSelect value={id} basePath="/history" />
          <select aria-label="Cycle" value={cycle ?? ""} onChange={(e) => setCycle(e.target.value ? Number(e.target.value) : null)} className={`${selectCls} w-36`}>
            <option value="">All cycles</option>
            {well.cycles.map((c) => (
              <option key={c.cycle} value={c.cycle}>
                Cycle {c.cycle} (day {c.start_day})
              </option>
            ))}
          </select>
          {isFetching && <span className="text-xs text-muted">loading…</span>}
        </div>
        <div className="flex gap-2">
          <Btn small onClick={() => nav(`/spm/${id}?day=${day}`)}>SPM advisor →</Btn>
          <Btn small onClick={() => nav(`/css/${id}?day=${day}`)}>Plan next cycle →</Btn>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <KpiTile label="Steam cycles" value={s.n_cycles} sub={s.stopped_early ? "stopped at the economic limit" : "full programme"} tone={s.stopped_early ? "warn" : "neutral"} />
        <KpiTile label="Cumulative oil" value={fmtInt(s.cum_oil_m3)} unit="m³" sub={`${fmt(s.mean_oil_bbl_d, 1)} bbl/d while producing`} />
        <KpiTile label="Cumulative SOR" value={fmt(s.cum_sor, 2)} sub={`last cycle ${fmt(s.last_cycle_sor, 2)}`} tone={s.cum_sor > 4 ? "warn" : "neutral"} />
        <KpiTile label="Rod/pump failures" value={s.n_failures} sub={`${s.failures_rod_part} rod part · ${s.failures_pump_unset} unset · ${s.failures_worn_barrel} barrel`} tone={s.n_failures > 3 ? "bad" : s.n_failures ? "warn" : "ok"} />
        <KpiTile label="Days floating" value={fmt(s.pct_days_floating, 1)} unit="%" sub={`mean ${fmt(s.mean_spm, 1)} SPM`} tone={s.pct_days_floating > 15 ? "bad" : s.pct_days_floating > 5 ? "warn" : "ok"} />
        <KpiTile
          label={`Risk on ${fmtDay(day)}`}
          value={risk?.risk !== undefined ? `${fmt(risk.risk * 100, 2)}%` : "–"}
          tone={risk?.level === "alert" ? "bad" : risk?.level === "watch" ? "warn" : "ok"}
          sub={risk?.risk !== undefined ? `${risk.level} · as of ${fmtDay(risk.day)}` : "pump not running"}
        />
      </div>

      <Panel title="Production: observed vs. physics vs. digital twin" subtitle="Blue bands: steam injection · red bands: workover · dashed red lines: failures · amber line: as-of day (click a chart to move it)">
        <Chart option={charts.oil} height={230} group={GROUP} onClick={clickDay} ariaLabel="Oil rate history" />
        <SourceNote>
          The twin line is the physics prior corrected by the ML residual model using production observed up to 7 days earlier (out-of-fold: this well was not in its training set).
        </SourceNote>
      </Panel>
      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Heated-zone temperature and oil viscosity">
          <Chart option={charts.thermo} height={220} group={GROUP} onClick={clickDay} ariaLabel="Temperature and viscosity history" />
          <SourceNote>&quot;State&quot; columns are the generator&apos;s latent states (not field measurements); the prior is the physics model. Viscosity is on a log scale.</SourceNote>
        </Panel>
        <Panel title="Pump speed and rod-floating index">
          <Chart option={charts.pump} height={220} group={GROUP} onClick={clickDay} ariaLabel="SPM and rod floating index history" />
        </Panel>
        <Panel title="Pump fillage and rod-load variability">
          <Chart option={charts.load} height={220} group={GROUP} onClick={clickDay} ariaLabel="Fillage and load variability history" />
        </Panel>
        <Panel title="Early-warning risk (out-of-fold)">
          <Chart option={charts.riskOpt} height={220} group={GROUP} onClick={clickDay} ariaLabel="Failure risk history" />
        </Panel>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_380px]">
        <Panel title="Steam cycles" subtitle="Controls and results per cycle (SOR above 10 is flagged as near the economic limit)" pad={false}>
          <div className="overflow-x-auto">
            <table className="w-full text-[13px]">
              <thead className="text-xs text-muted">
                <tr className="border-b border-line">
                  {["Cycle", "Start", "Steam m³/d", "Inj d", "Quality", "Temp °C", "Soak d", "Prod d", "Oil m³", "SOR", "Oil/cycle-day", "Model (OOF)", "Failures"].map((h, i) => (
                    <th key={h} className={`px-2 py-1.5 font-medium ${i ? "text-right" : "text-left"}`}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {well.cycles.map((c) => (
                  <tr key={c.cycle} className={`cursor-pointer border-b border-line/60 hover:bg-panel2 ${cycle === c.cycle ? "bg-accent/8" : ""}`} onClick={() => setCycle(cycle === c.cycle ? null : c.cycle)}>
                    <td className="num px-2 py-1.5 text-ink">{c.cycle}</td>
                    <td className="num px-2 py-1.5 text-right">{c.start_day}</td>
                    <td className="num px-2 py-1.5 text-right">{fmt(c.steam_rate_m3d, 0)}</td>
                    <td className="num px-2 py-1.5 text-right">{c.inj_days}</td>
                    <td className="num px-2 py-1.5 text-right">{fmt(c.steam_quality, 2)}</td>
                    <td className="num px-2 py-1.5 text-right">{fmt(c.steam_temp_c, 0)}</td>
                    <td className="num px-2 py-1.5 text-right">{c.soak_days}</td>
                    <td className="num px-2 py-1.5 text-right">{c.prod_days}</td>
                    <td className="num px-2 py-1.5 text-right">{fmtInt(c.oil_m3)}</td>
                    <td className={`num px-2 py-1.5 text-right ${c.high_sor_flag ? "text-bad" : c.sor > 5 ? "text-warn" : ""}`}>{fmt(c.sor, 2)}</td>
                    <td className="num px-2 py-1.5 text-right">{fmt(c.opd_m3d, 2)}</td>
                    <td className="num px-2 py-1.5 text-right text-muted">{fmt(c.opd_pred_oof_m3d, 2)}</td>
                    <td className={`num px-2 py-1.5 text-right ${c.n_failures ? "text-bad" : ""}`}>{c.n_failures || ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
        <div className="space-y-4">
          <Panel title={`Why this risk? · ${fmtDay(risk?.day)}`} subtitle="TreeSHAP contributions (log-odds) of the fold model that never saw this well">
            {!risk?.drivers ? (
              <p className="text-sm text-muted">The pump is not running on this day.</p>
            ) : (
              <>
                <div className="mb-2 flex items-center gap-2">
                  {risk.level && <RagPill status={risk.level === "alert" ? "red" : risk.level === "watch" ? "amber" : "green"} label={risk.level} />}
                  <span className="num text-sm text-ink">{fmt((risk.risk ?? 0) * 100, 2)}% within 14 days</span>
                </div>
                <ul className="space-y-1.5">
                  {risk.drivers.map((dr) => (
                    <li key={dr.feature} className="flex items-center gap-2 text-xs">
                      <span className="w-44 truncate text-muted" title={dr.feature}>
                        {dr.label}
                      </span>
                      <span className="num w-14 text-right text-ink">{fmt(dr.value, 2)}</span>
                      <span className="relative h-2 flex-1 rounded bg-panel2">
                        <span
                          className="absolute inset-y-0 rounded"
                          style={{
                            left: dr.contribution >= 0 ? "50%" : `${50 - Math.min(50, Math.abs(dr.contribution) * 12)}%`,
                            width: `${Math.min(50, Math.abs(dr.contribution) * 12)}%`,
                            background: dr.contribution >= 0 ? COLORS.bad : COLORS.ok,
                          }}
                        />
                      </span>
                      <span className={`num w-12 text-right ${dr.contribution >= 0 ? "text-bad" : "text-ok"}`}>{dr.contribution >= 0 ? "+" : ""}{fmt(dr.contribution, 2)}</span>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </Panel>
          <Panel title="Failure log" pad={false}>
            {well.failures.length === 0 ? (
              <p className="px-4 py-3 text-sm text-muted">No rod or pump failures recorded.</p>
            ) : (
              <ul className="divide-y divide-line text-[13px]">
                {well.failures.map((f) => (
                  <li key={f.failure_date_day} className="flex items-center gap-3 px-4 py-1.5">
                    <button className="num text-accent hover:underline" onClick={() => setDay(f.failure_date_day)}>
                      {fmtDay(f.failure_date_day)}
                    </button>
                    <span className="text-ink">{titleCase(f.failure_mode)}</span>
                    <span className="ml-auto text-xs text-muted">
                      cycle {f.cycle} · {f.downtime_days} d workover
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}
