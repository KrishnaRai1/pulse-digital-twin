import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useDsCyclePlan, useDsModels, useDsWell, useDsWhatIf, type CycleControls, type Sweeps } from "../../api/dataset";
import { Chart } from "../../components/Chart";
import { SourceNote, WellSelect, useAsOfDay } from "../../components/dataset";
import { Slider } from "../../components/Slider";
import { Btn, ErrorNote, KpiTile, Loading, Panel, selectCls } from "../../components/ui";
import { axisBase, gridBase, legendBase, tooltipBase } from "../../lib/chartHelpers";
import { COLORS } from "../../lib/echarts";
import { fmt, fmtInt, fmtUsd } from "../../lib/format";

const LABEL: Record<string, string> = {
  steam_rate_m3d: "Steam rate",
  inj_days: "Injection days",
  steam_quality: "Steam quality",
  steam_temp_c: "Steam temperature",
  soak_days: "Soak days",
  prod_days: "Production days",
  cycle: "Cycle number (depletion)",
  prev_opd_m3d: "Previous cycle oil/day",
  prev_peak_oil_bbl_d: "Previous cycle peak",
  prev_sor: "Previous cycle SOR",
  cum_oil_before_m3: "Cumulative oil so far",
  interactions: "Interactions",
};
const UNIT: Record<keyof CycleControls, string> = { steam_rate_m3d: "m³/d", inj_days: "d", steam_quality: "", steam_temp_c: "°C", soak_days: "d", prod_days: "d" };

function useDebounced<T>(value: T, ms = 300): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

function controlsText(c: CycleControls): string {
  return `${fmt(c.steam_rate_m3d, 0)} m³/d × ${c.inj_days} d at ${fmt(c.steam_quality, 2)} quality / ${fmt(c.steam_temp_c, 0)} °C, soak ${c.soak_days} d, produce ${c.prod_days} d`;
}

function SweepChart({ sweeps, k, current }: { sweeps: Sweeps; k: keyof CycleControls; current: number }) {
  const s = sweeps[k];
  const option = useMemo(
    () => ({
      tooltip: { ...tooltipBase, valueFormatter: (v: number) => fmt(v, 2) },
      grid: gridBase({ left: 38, right: 10, top: 12, bottom: 22 }),
      xAxis: { type: "value", min: s.x[0], max: s.x[s.x.length - 1], ...axisBase, axisLabel: { ...axisBase.axisLabel, fontSize: 10 } },
      yAxis: { type: "value", scale: true, ...axisBase, axisLabel: { ...axisBase.axisLabel, fontSize: 10 } },
      series: [
        {
          type: "line",
          data: s.x.map((x, i) => [x, s.opd_m3d[i]]),
          showSymbol: false,
          lineStyle: { color: COLORS.accent, width: 1.8 },
          markLine: { symbol: "none", silent: true, data: [{ xAxis: current, lineStyle: { color: COLORS.warn }, label: { show: false } }] },
        },
      ],
    }),
    [s, current],
  );
  return (
    <div>
      <div className="text-[11px] font-medium text-muted">
        {LABEL[k]} <span className="text-faint">→ oil m³ per cycle-day</span>
      </div>
      <Chart option={option} height={120} ariaLabel={`Oil per cycle-day versus ${LABEL[k]}`} />
    </div>
  );
}

export function CycleOptimizer() {
  const { wellId = "WELL-001" } = useParams();
  const id = wellId.toUpperCase();
  const { day } = useAsOfDay();
  const nav = useNavigate();
  const [cycle, setCycle] = useState<number | null>(null);
  const [objective, setObjective] = useState<"oil" | "margin" | "osr">("oil");
  const [unconstrained, setUnconstrained] = useState(false);
  const { data: well } = useDsWell(id);
  const { data: plan, error: planErr, isFetching: planBusy } = useDsCyclePlan(id, cycle, objective, unconstrained);
  const { data: models } = useDsModels();
  const [controls, setControls] = useState<CycleControls | null>(null);
  const ctxKey = plan ? `${plan.context.well_id}-${plan.context.cycle}` : "";
  useEffect(() => {
    if (plan) setControls(plan.context.defaults);
    // reset the sliders only when the well or the cycle changes
  }, [ctxKey]);
  useEffect(() => setCycle(null), [id]);
  const debounced = useDebounced(controls, 250);
  const { data: wi, isFetching: wiBusy } = useDsWhatIf(id, cycle, debounced);
  const mine = wi?.results[0];

  const set = (k: keyof CycleControls) => (v: number) => setControls((c) => (c ? { ...c, [k]: v } : c));

  const contribOpt = useMemo(() => {
    if (!mine?.contributions) return null;
    const top = mine.contributions.slice(0, 9).reverse();
    return {
      tooltip: { ...tooltipBase, trigger: "item", valueFormatter: (v: number) => `${v > 0 ? "+" : ""}${fmt(v, 1)}%` },
      grid: gridBase({ left: 170, right: 40, top: 10, bottom: 24 }),
      xAxis: { type: "value", ...axisBase, axisLabel: { ...axisBase.axisLabel, formatter: "{value}%" } },
      yAxis: { type: "category", data: top.map((c) => LABEL[c.feature] ?? c.feature), ...axisBase },
      series: [{ type: "bar", data: top.map((c) => ({ value: c.effect_pct, itemStyle: { color: c.effect_pct >= 0 ? COLORS.ok : COLORS.bad } })), barWidth: 12 }],
    };
  }, [mine]);

  const recovery = useMemo(() => {
    if (!plan || !wi) return null;
    const curves = [...plan.profile.curves, ...wi.profile.curves];
    const colors = [COLORS.accent, COLORS.muted, COLORS.warn];
    return {
      tooltip: { ...tooltipBase, valueFormatter: (v: number) => `${fmtInt(v)} m³` },
      legend: { ...legendBase },
      grid: gridBase({ left: 56, right: 18, bottom: 34 }),
      xAxis: { type: "value", name: "days from start of injection", nameLocation: "middle", nameGap: 24, ...axisBase },
      yAxis: { type: "value", name: "cumulative oil m³", ...axisBase },
      series: curves.map((c, i) => ({
        name: c.label === plan.context.defaults_source ? `reference (${c.label})` : c.label,
        type: "line",
        showSymbol: false,
        data: c.day.map((d, j) => [d, c.cum_oil_m3[j]]),
        lineStyle: { width: i === 0 ? 2.2 : 1.6, color: colors[i % 3], type: i === 1 ? "dashed" : "solid" },
        itemStyle: { color: colors[i % 3] },
      })),
    };
  }, [plan, wi]);

  const frontier = useMemo(() => {
    if (!plan) return null;
    const p = plan.plan;
    return {
      tooltip: { ...tooltipBase, trigger: "item", formatter: (x: { value: number[] }) => `steam ${fmtInt(x.value[0])} m³<br/>oil ${fmtInt(x.value[1])} m³` },
      grid: gridBase({ left: 56, right: 18, bottom: 34 }),
      xAxis: { type: "value", name: "steam per cycle m³", nameLocation: "middle", nameGap: 24, scale: true, ...axisBase },
      yAxis: { type: "value", name: "oil per cycle m³", scale: true, ...axisBase },
      series: [
        { name: "best for each steam total", type: "scatter", symbolSize: 5, data: p.frontier.map((f) => [f.steam_m3, f.oil_m3]), itemStyle: { color: COLORS.blue, opacity: 0.7 } },
        { name: "reference", type: "scatter", symbol: "circle", symbolSize: 13, data: [[p.baseline.steam_m3, p.baseline.oil_m3]], itemStyle: { color: COLORS.muted } },
        { name: "recommended", type: "scatter", symbol: "diamond", symbolSize: 16, data: [[p.best.steam_m3, p.best.oil_m3]], itemStyle: { color: COLORS.accent } },
      ],
    };
  }, [plan]);

  if (planErr) return <ErrorNote error={planErr} />;
  if (!plan || !well) return <Loading what="cycle planner" />;
  const ctx = plan.context;
  const best = plan.plan.best;
  const lin = models?.cycle_model.linear_cross_check;
  const met = models?.cycle_model.metrics;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-lg font-semibold text-ink">CSS cycle optimizer</h1>
          <WellSelect value={id} basePath="/css" />
          <select aria-label="Cycle to plan" value={cycle ?? ""} onChange={(e) => setCycle(e.target.value ? Number(e.target.value) : null)} className={`${selectCls} w-52`}>
            <option value="">Next cycle ({well.cycles.length + 1})</option>
            {well.cycles.map((c) => (
              <option key={c.cycle} value={c.cycle}>
                Re-plan cycle {c.cycle} (actual known)
              </option>
            ))}
          </select>
          {(planBusy || wiBusy) && <span className="text-xs text-muted">computing…</span>}
        </div>
        <Btn small onClick={() => nav(`/history/${id}?day=${day}`)}>Well history →</Btn>
      </div>

      <section className="rounded-lg border border-accent/50 bg-accent/6 p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-muted">
              Recommended settings · {id} · cycle {ctx.cycle} {ctx.is_next_cycle ? "(next)" : "(re-plan)"}
            </div>
            <div className="mt-1 text-[20px] font-semibold leading-snug text-ink">{controlsText(best.controls)}</div>
            <div className="mt-1 text-sm text-muted">
              {fmt(best.opd_m3d, 2)} m³ oil per cycle-day vs. {fmt(plan.plan.baseline.opd_m3d, 2)} with the reference ({ctx.defaults_source})
              {plan.plan.uplift_pct !== null && (
                <span className={plan.plan.uplift_pct >= 0 ? "text-ok" : "text-bad"}>
                  {" "}
                  · {plan.plan.uplift_pct >= 0 ? "+" : ""}
                  {fmt(plan.plan.uplift_pct, 1)}%
                </span>
              )}
              {" "}· SOR {fmt(best.sor, 2)} · {fmtUsd(best.net_usd_per_day)}/cycle-day margin
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <select aria-label="Objective" value={objective} onChange={(e) => setObjective(e.target.value as typeof objective)} className={`${selectCls} w-52`}>
              <option value="oil">Max oil per cycle-day</option>
              <option value="margin">Max margin per cycle-day</option>
              <option value="osr">Max oil-steam ratio</option>
            </select>
            <label className="flex items-center gap-1.5 text-xs text-muted">
              <input type="checkbox" checked={!unconstrained} onChange={(e) => setUnconstrained(!e.target.checked)} /> same steam per cycle-day
            </label>
            <Btn small variant="primary" onClick={() => setControls(best.controls)}>
              Load into what-if
            </Btn>
          </div>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          {plan.plan.why.slice(0, 7).map((w) => (
            <span key={w.feature} className={`rounded-md border px-2 py-1 text-xs ${w.effect_pct >= 0 ? "border-ok/40 text-ok" : "border-bad/40 text-bad"}`}>
              {LABEL[w.feature] ?? w.feature}
              {w.from !== null && w.to !== null ? ` ${fmt(w.from, w.feature === "steam_quality" ? 2 : 0)} → ${fmt(w.to, w.feature === "steam_quality" ? 2 : 0)}` : ""}: {w.effect_pct >= 0 ? "+" : ""}
              {fmt(w.effect_pct, 1)}%
            </span>
          ))}
        </div>
        <SourceNote>
          Searched {fmtInt(plan.plan.searched)} settings
          {plan.plan.steam_budget_m3_per_cycle_day ? ` within ${fmt(plan.plan.steam_budget_m3_per_cycle_day, 1)} m³ steam per cycle-day (the reference cycle's)` : " without a steam budget"}; quality and temperature at the
          generator maximum (positive effect in the data). Effects are TreeSHAP differences of the monotone-constrained GBM. Economic limit OSR ≥ {fmt(0.12, 2)}.
        </SourceNote>
      </section>

      <div className="grid gap-4 xl:grid-cols-[340px_1fr]">
        <Panel title="What-if" subtitle="Ranges are the ones the data covers">
          {controls && (
            <div className="space-y-3">
              <Slider label="Steam rate" unit={UNIT.steam_rate_m3d} value={controls.steam_rate_m3d} min={150} max={250} step={5} onChange={set("steam_rate_m3d")} />
              <Slider label="Injection days" unit={UNIT.inj_days} value={controls.inj_days} min={10} max={20} step={1} onChange={set("inj_days")} />
              <Slider label="Steam quality" value={controls.steam_quality} min={0.45} max={0.7} step={0.01} digits={2} onChange={set("steam_quality")} />
              <Slider label="Steam temperature" unit={UNIT.steam_temp_c} value={controls.steam_temp_c} min={302} max={330} step={1} onChange={set("steam_temp_c")} hint="pressure follows temperature (saturated steam)" />
              <Slider label="Soak days" unit={UNIT.soak_days} value={controls.soak_days} min={2} max={14} step={1} onChange={set("soak_days")} />
              <Slider label="Production days" unit={UNIT.prod_days} value={controls.prod_days} min={120} max={300} step={5} onChange={set("prod_days")} />
              <div className="flex gap-2">
                <Btn small onClick={() => setControls(ctx.defaults)}>Reset to reference</Btn>
                <Btn small onClick={() => setControls(best.controls)}>Recommended</Btn>
              </div>
            </div>
          )}
        </Panel>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <KpiTile label="Oil per cycle-day" value={fmt(mine?.opd_m3d, 2)} unit="m³/d" sub={ctx.actual ? `actual ${fmt(ctx.actual.opd_m3d, 2)} · model ${fmt(ctx.actual.opd_pred_oof_m3d, 2)} (OOF)` : `previous cycle ${fmt(ctx.prev_opd_m3d, 2)}`} />
            <KpiTile label="Cycle oil" value={fmtInt(mine?.oil_m3)} unit="m³" sub={`${fmtInt(mine?.oil_bbl)} bbl over ${mine?.cycle_days ?? "–"} d`} />
            <KpiTile label="Steam-oil ratio" value={fmt(mine?.sor, 2)} sub={`${fmtInt(mine?.steam_m3)} m³ steam`} tone={mine && mine.sor > 5 ? "warn" : "neutral"} />
            <KpiTile label="Margin" value={fmtUsd(mine?.net_usd_per_day)} unit="/cycle-day" sub={`oil $${plan.prices.oil_usd_per_m3}/m³ · steam $${plan.prices.steam_usd_per_m3}/m³`} tone={mine && !mine.economic ? "bad" : "neutral"} />
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <Panel title="What drives this prediction" subtitle="% effect on oil per cycle-day vs. the field average (TreeSHAP)">
              {contribOpt && <Chart option={contribOpt} height={250} ariaLabel="Feature contributions" />}
            </Panel>
            <Panel title="Expected recovery vs. time" subtitle={`Shape from this well's cycle ${plan.profile.shape_cycle}, scaled to each prediction`}>
              {recovery && <Chart option={recovery} height={250} ariaLabel="Cumulative oil versus time for scenarios" />}
            </Panel>
          </div>
        </div>
      </div>

      {wi && controls && (
        <Panel title="Sensitivities around your settings" subtitle="One control varied at a time; amber line = current value">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
            {(Object.keys(UNIT) as (keyof CycleControls)[]).map((k) => (
              <SweepChart key={k} sweeps={wi.sweeps} k={k} current={controls[k]} />
            ))}
          </div>
        </Panel>
      )}

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Oil vs. steam frontier" subtitle="Best predicted cycle for every steam total searched">
          {frontier && <Chart option={frontier} height={260} ariaLabel="Oil versus steam frontier" />}
        </Panel>
        <Panel title="What the field data says" subtitle="Cross-check: transparent log-linear model on all 2,739 cycles (well history held fixed)">
          {lin && met ? (
            <>
              <table className="w-full text-[13px]">
                <tbody>
                  {lin.effects.map((e) => (
                    <tr key={e.control} className="border-b border-line/60">
                      <td className="py-1.5 text-muted">
                        +{e.step} {e.control === "steam_quality" ? "" : UNIT[e.control as keyof CycleControls] ?? (e.control === "cycle" ? "cycle" : "")} {LABEL[e.control] ?? e.control}
                      </td>
                      <td className={`num py-1.5 text-right ${e.effect_pct >= 0 ? "text-ok" : "text-bad"}`}>
                        {e.effect_pct >= 0 ? "+" : ""}
                        {fmt(e.effect_pct, 1)}% oil/cycle-day
                      </td>
                    </tr>
                  ))}
                  <tr>
                    <td className="py-1.5 text-muted">Soak optimum (quadratic)</td>
                    <td className="num py-1.5 text-right text-ink">{fmt(lin.soak_optimum_days, 1)} days</td>
                  </tr>
                </tbody>
              </table>
              <p className="mt-3 text-xs text-muted">
                Held-out-well accuracy (R², oil per cycle-day): GBM {fmt(met["GBM (out-of-fold)"].r2, 2)} · log-linear {fmt(lin.r2_oof, 2)} · field mean by cycle number {fmt(met["field mean for cycle number"].r2, 2)} · repeat previous cycle{" "}
                {fmt(met["previous cycle repeated"].r2, 2)}.
              </p>
            </>
          ) : (
            <Loading what="model card" />
          )}
        </Panel>
      </div>
    </div>
  );
}
