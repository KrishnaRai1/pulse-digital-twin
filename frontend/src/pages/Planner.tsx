import { useEffect, useMemo, useState } from "react";
import { Navigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { useCycleMutations, useJob, useWellDetail } from "../api/hooks";
import type { CycleRow, PlanResult, ScenarioIn, WhatIfResponse } from "../api/types";
import { Chart } from "../components/Chart";
import { Slider } from "../components/Slider";
import { WellPicker } from "../components/WellPicker";
import { Btn, ErrorNote, Loading, Panel, inputCls } from "../components/ui";
import { axisBase, gridBase, legendBase, tooltipBase, zip } from "../lib/chartHelpers";
import { COLORS, SERIES } from "../lib/echarts";
import { fmt, fmtDate, fmtInt, fmtUsd } from "../lib/format";

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

interface Params {
  steam_m3: number;
  quality: number;
  inj_rate_m3d: number;
  inj_pressure_mpa: number;
  soak_days: number;
  prod_days: number;
}

function WhatIf({ wellId, last }: { wellId: string; last: CycleRow }) {
  const [p, setP] = useState<Params>(() => ({
    steam_m3: last.steam_m3,
    quality: last.quality,
    inj_rate_m3d: last.inj_rate_m3d,
    inj_pressure_mpa: last.inj_pressure_mpa,
    soak_days: last.soak_days,
    prod_days: last.prod_days,
  }));
  const [mode, setMode] = useState<"single" | "ladder">("ladder");
  const [showPhysics, setShowPhysics] = useState(false);
  const dp = useDebounced(p, 450);
  const set = <K extends keyof Params>(k: K) => (v: number) => setP((s) => ({ ...s, [k]: v }));

  const scenarios: ScenarioIn[] = useMemo(() => {
    if (mode === "ladder") {
      const vols = [0.7, 1, 1.3].map((f) => Math.round((dp.steam_m3 * f) / 100) * 100);
      return vols.map((v, i) => ({ ...dp, steam_m3: v, label: `${fmtInt(v)} m³${i === 1 ? " (slider)" : ""}` }));
    }
    return [{ label: "Last cycle's parameters" }, { ...dp, label: "Your scenario" }];
  }, [dp, mode]);

  const q = useQuery({
    queryKey: ["whatif", wellId, scenarios],
    queryFn: () => api.post<WhatIfResponse>("/thermal/whatif", { well_id: wellId, scenarios, base: "next" }),
    placeholderData: (prev) => prev,
  });

  const charts = useMemo(() => {
    if (!q.data) return null;
    const sc = q.data.scenarios;
    const mk = (key: "cum_oil_m3" | "q_oil_m3d", physKey: "cum_oil_phys_m3" | "q_oil_phys_m3d", ylabel: string) => ({
      tooltip: { ...tooltipBase, valueFormatter: (v: number) => v.toFixed(1) },
      legend: { ...legendBase, type: "scroll" },
      grid: gridBase({ top: 44, bottom: 40 }),
      xAxis: { type: "value", name: "Days since start of steam injection", nameLocation: "middle", nameGap: 26, ...axisBase },
      yAxis: { type: "value", name: ylabel, ...axisBase, min: 0 },
      series: sc.flatMap((s, i) => [
        { name: s.label, type: "line", showSymbol: false, lineStyle: { width: 2.5 }, color: SERIES[i % SERIES.length], data: zip(s.t_days, s[key]) },
        ...(showPhysics ? [{ name: `${s.label} (physics only)`, type: "line", showSymbol: false, lineStyle: { width: 1.2, type: "dashed" }, color: SERIES[i % SERIES.length], data: zip(s.t_days, s[physKey]) }] : []),
      ]),
    });
    return { cum: mk("cum_oil_m3", "cum_oil_phys_m3", "Cumulative oil, m³"), rate: mk("q_oil_m3d", "q_oil_phys_m3d", "Oil rate, m³/d") };
  }, [q.data, showPhysics]);

  return (
    <div className="grid gap-4 xl:grid-cols-[340px_1fr]">
      <Panel title="Next-cycle design" subtitle="Move the sliders: the twin re-simulates the whole cycle">
        <div className="space-y-4">
          <Slider label="Steam volume (CWE)" unit="m³" value={p.steam_m3} min={1000} max={12000} step={100} onChange={set("steam_m3")} hint={`${fmt(p.steam_m3 / p.inj_rate_m3d, 0)} days of injection`} />
          <Slider label="Surface steam quality" unit="" value={p.quality} min={0.4} max={1} step={0.01} digits={2} onChange={set("quality")} />
          <Slider label="Injection rate" unit="m³/d" value={p.inj_rate_m3d} min={60} max={400} step={5} onChange={set("inj_rate_m3d")} />
          <Slider label="Bottom-hole injection pressure" unit="MPa" value={p.inj_pressure_mpa} min={0.8} max={4.5} step={0.1} digits={1} onChange={set("inj_pressure_mpa")} />
          <Slider label="Soak time" unit="days" value={p.soak_days} min={0} max={30} step={1} onChange={set("soak_days")} />
          <Slider label="Production window" unit="days" value={p.prod_days} min={60} max={300} step={5} onChange={set("prod_days")} />
          <div className="flex gap-2 pt-1">
            <Btn small onClick={() => setP({ steam_m3: last.steam_m3, quality: last.quality, inj_rate_m3d: last.inj_rate_m3d, inj_pressure_mpa: last.inj_pressure_mpa, soak_days: last.soak_days, prod_days: last.prod_days })}>
              Reset to last cycle
            </Btn>
          </div>
          <fieldset className="border-t border-line pt-3">
            <legend className="text-[11px] font-semibold uppercase tracking-wide text-muted">Compare</legend>
            <label className="mt-1 flex items-center gap-2 text-xs text-ink">
              <input type="radio" name="mode" checked={mode === "ladder"} onChange={() => setMode("ladder")} /> Steam-volume ladder (−30 % / slider / +30 %)
            </label>
            <label className="mt-1 flex items-center gap-2 text-xs text-ink">
              <input type="radio" name="mode" checked={mode === "single"} onChange={() => setMode("single")} /> Scenario vs last cycle
            </label>
            <label className="mt-2 flex items-center gap-2 text-xs text-muted">
              <input type="checkbox" checked={showPhysics} onChange={(e) => setShowPhysics(e.target.checked)} /> Also show physics-only curves
            </label>
          </fieldset>
        </div>
      </Panel>

      <div className="space-y-4">
        {q.error && <ErrorNote error={q.error} />}
        {!charts && q.isLoading && <Loading what="simulation" />}
        {charts && q.data && (
          <>
            <div className="grid gap-4 lg:grid-cols-2">
              <Panel title="Recovery vs time" subtitle={`Cumulative oil per scenario · ${q.data.ml_used ? "physics + ML residual" : "physics only"}${q.isFetching ? " · updating…" : ""}`}>
                <Chart option={charts.cum} height={290} ariaLabel="Cumulative oil versus time for each scenario" />
              </Panel>
              <Panel title="Oil rate vs time" subtitle="Decline through the production window">
                <Chart option={charts.rate} height={290} ariaLabel="Oil rate versus time for each scenario" />
              </Panel>
            </div>
            <Panel title="Scenario comparison" pad={false}>
              <div className="overflow-x-auto">
                <table className="w-full min-w-[760px] text-xs">
                  <thead className="border-b border-line text-[11px] uppercase tracking-wide text-muted">
                    <tr>
                      {["Scenario", "Steam m³", "Cum. oil m³", "SOR", "Peak m³/d", "Heated radius m", "Rate < 2 m³/d at day", "BH quality", "Wellbore loss"].map((h, i) => (
                        <th key={h} className={`px-3 py-2 font-semibold ${i === 0 ? "text-left" : "text-right"}`}>
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {q.data.scenarios.map((s, i) => (
                      <tr key={s.label} className="border-b border-line/50">
                        <td className="px-3 py-1.5 font-semibold" style={{ color: SERIES[i % SERIES.length] }}>
                          {s.label}
                        </td>
                        <td className="num px-3 py-1.5 text-right">{fmtInt(s.spec.steam_m3)}</td>
                        <td className="num px-3 py-1.5 text-right">{fmtInt(s.summary.cum_oil_m3)}</td>
                        <td className="num px-3 py-1.5 text-right">{fmt(s.summary.sor, 2)}</td>
                        <td className="num px-3 py-1.5 text-right">{fmt(s.summary.peak_rate_m3d, 1)}</td>
                        <td className="num px-3 py-1.5 text-right">{fmt(s.summary.r_heated_m, 1)}</td>
                        <td className="num px-3 py-1.5 text-right">{s.summary.economic_limit_day != null ? fmtInt(s.summary.economic_limit_day) : "> window"}</td>
                        <td className="num px-3 py-1.5 text-right">{fmt(s.summary.steam_bh_quality * 100, 0)}%</td>
                        <td className="num px-3 py-1.5 text-right">{fmt(s.summary.wellbore_loss_pct, 0)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          </>
        )}
      </div>
    </div>
  );
}

function Sweep({ wellId }: { wellId: string }) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const job = useJob<PlanResult>(jobId);
  const start = async () => {
    setErr(null);
    setJobId(null);
    try {
      const r = await api.post<{ job_id: string }>(`/wells/${wellId}/cycle-plan`, {});
      setJobId(r.job_id);
    } catch (e) {
      setErr(e);
    }
  };
  useEffect(() => setJobId(null), [wellId]);
  const res = job.data?.status === "done" ? job.data.result : null;
  const opt = useMemo(() => {
    if (!res) return null;
    const rows = res.rows;
    return {
      tooltip: { ...tooltipBase, valueFormatter: (v: number) => (v == null ? "–" : v.toFixed(2)) },
      legend: { ...legendBase },
      grid: gridBase({ right: 60, top: 44, bottom: 40 }),
      xAxis: { type: "category", name: "Steam volume, m³", nameLocation: "middle", nameGap: 26, data: rows.map((r) => r.steam_m3), ...axisBase },
      yAxis: [
        { type: "value", name: "Net value, $", ...axisBase },
        { type: "value", name: "SOR · m³ oil / 1000 m³", ...axisBase, splitLine: { show: false } },
      ],
      series: [
        { name: "Net value (oil revenue − steam cost)", type: "bar", barMaxWidth: 34, data: rows.map((r) => ({ value: Math.round(r.profit_usd), itemStyle: { color: r.steam_m3 === res.best_profit_volume_m3 ? COLORS.ok : "#2c4a5a" } })) },
        { name: "SOR", type: "line", yAxisIndex: 1, color: COLORS.warn, lineStyle: { width: 2 }, data: rows.map((r) => r.sor) },
        { name: "Marginal oil per +1000 m³ steam", type: "line", yAxisIndex: 1, color: COLORS.violet, lineStyle: { width: 2, type: "dashed" }, data: rows.map((r) => r.marginal_oil_per_1000m3) },
      ],
    };
  }, [res]);
  return (
    <Panel
      title="Steam-volume sweep"
      subtitle="Background job: runs the full cycle simulation for a ladder of volumes and finds the best net value"
      actions={
        <Btn variant="primary" onClick={start} disabled={job.data?.status === "running" || job.data?.status === "queued"}>
          {job.data && (job.data.status === "running" || job.data.status === "queued") ? "Running…" : "Run sweep"}
        </Btn>
      }
    >
      {err != null && <ErrorNote error={err} />}
      {job.data?.status === "failed" && <ErrorNote error={new Error(job.data.error ?? "job failed")} />}
      {!res && !err && jobId && <Loading what="sweep results" />}
      {!jobId && !res && <p className="text-sm text-muted">Press “Run sweep” to compare eight steam volumes from 2 500 to 9 500 m³ for this well.</p>}
      {res && opt && (
        <>
          <Chart option={opt} height={300} ariaLabel="Net value and steam-oil ratio versus steam volume" />
          <div className="mt-2 flex flex-wrap gap-4 text-xs">
            <span className="text-muted">
              Best net value: <b className="num text-ok">{fmtInt(res.best_profit_volume_m3)} m³</b> ({fmtUsd(res.best_profit_usd)})
            </span>
            <span className="text-muted">
              Lowest SOR: <b className="num text-ink">{fmtInt(res.lowest_sor_volume_m3)} m³</b>
            </span>
          </div>
          <p className="mt-2 text-[11px] text-faint">{res.note}</p>
        </>
      )}
    </Panel>
  );
}

function CycleEditor({ wellId, cycles }: { wellId: string; cycles: CycleRow[] }) {
  const m = useCycleMutations(wellId);
  const [draft, setDraft] = useState<Record<number, Partial<CycleRow>>>({});
  const [err, setErr] = useState<unknown>(null);
  const [adding, setAdding] = useState(false);
  const [nw, setNw] = useState({ steam_m3: 4500, quality: 0.75, inj_rate_m3d: 200, inj_pressure_mpa: 1.8, soak_days: 7, prod_days: 180 });
  const cols: { key: keyof CycleRow; label: string; step: number; min: number; max: number }[] = [
    { key: "steam_m3", label: "Steam m³", step: 100, min: 200, max: 25000 },
    { key: "quality", label: "Quality", step: 0.01, min: 0.3, max: 1 },
    { key: "inj_rate_m3d", label: "Inj. m³/d", step: 5, min: 40, max: 800 },
    { key: "inj_pressure_mpa", label: "Inj. MPa", step: 0.1, min: 0.4, max: 8 },
    { key: "soak_days", label: "Soak d", step: 1, min: 0, max: 90 },
    { key: "prod_days", label: "Prod d", step: 5, min: 20, max: 500 },
  ];
  const lastNo = cycles[cycles.length - 1]?.cycle_no;
  const body = (c: CycleRow) => ({ ...Object.fromEntries(cols.map((k) => [k.key, c[k.key]])), ...draft[c.cycle_no] }) as Partial<CycleRow>;
  const run = async (fn: () => Promise<unknown>) => {
    setErr(null);
    try {
      await fn();
    } catch (e) {
      setErr(e);
    }
  };
  return (
    <Panel title="Cycle history overrides" subtitle="Correct what was actually injected: the twin recomputes the reservoir state from your edit (needs the API key when one is configured)">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[820px] text-xs">
          <thead className="border-b border-line text-[11px] uppercase tracking-wide text-muted">
            <tr>
              <th className="px-2 py-2 text-left">Cycle</th>
              <th className="px-2 py-2 text-left">Start</th>
              {cols.map((c) => (
                <th key={c.key} className="px-2 py-2 text-left">{c.label}</th>
              ))}
              <th className="px-2 py-2" />
            </tr>
          </thead>
          <tbody>
            {cycles.map((c) => {
              const dirty = !!draft[c.cycle_no] && Object.keys(draft[c.cycle_no]!).length > 0;
              return (
                <tr key={c.cycle_no} className="border-b border-line/50">
                  <td className="num px-2 py-1.5">{c.cycle_no}{c.source === "operator" && <span className="ml-1 text-[10px] text-accent">edited</span>}</td>
                  <td className="px-2 py-1.5 text-muted">{fmtDate(c.start_ts)}</td>
                  {cols.map((k) => (
                    <td key={k.key} className="px-1 py-1">
                      <input
                        type="number"
                        aria-label={`${k.label} cycle ${c.cycle_no}`}
                        className={`${inputCls} num !w-24 !py-1`}
                        step={k.step}
                        min={k.min}
                        max={k.max}
                        value={(draft[c.cycle_no]?.[k.key] as number | undefined) ?? Number((c[k.key] as number).toFixed(2))}
                        onChange={(e) => setDraft((d) => ({ ...d, [c.cycle_no]: { ...d[c.cycle_no], [k.key]: Number(e.target.value) } }))}
                      />
                    </td>
                  ))}
                  <td className="whitespace-nowrap px-2 py-1 text-right">
                    <Btn small variant="primary" disabled={!dirty || m.update.isPending} onClick={() => run(async () => { await m.update.mutateAsync({ cycle: c.cycle_no, body: body(c) }); setDraft((d) => { const n = { ...d }; delete n[c.cycle_no]; return n; }); })}>
                      Save
                    </Btn>{" "}
                    {c.cycle_no === lastNo && cycles.length > 1 && (
                      <Btn small variant="ghost" onClick={() => run(() => m.remove.mutateAsync(c.cycle_no))} title="Delete the most recent cycle">
                        Delete
                      </Btn>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {err != null && <div className="mt-3"><ErrorNote error={err} /></div>}
      <div className="mt-3 border-t border-line pt-3">
        {!adding ? (
          <Btn small onClick={() => setAdding(true)}>+ Record a new cycle (re-steam started now)</Btn>
        ) : (
          <div className="flex flex-wrap items-end gap-3">
            {cols.map((k) => (
              <label key={k.key} className="text-[11px] text-muted">
                {k.label}
                <input type="number" className={`${inputCls} num mt-1 !w-24`} step={k.step} min={k.min} max={k.max} value={nw[k.key as keyof typeof nw]} onChange={(e) => setNw((s) => ({ ...s, [k.key]: Number(e.target.value) }))} />
              </label>
            ))}
            <Btn variant="primary" disabled={m.add.isPending} onClick={() => run(async () => { await m.add.mutateAsync(nw); setAdding(false); })}>Add cycle</Btn>
            <Btn variant="ghost" onClick={() => setAdding(false)}>Cancel</Btn>
          </div>
        )}
        <p className="mt-2 text-[11px] text-faint">A new cycle starts when the previous production window ends (or now, if that is still in the future) and must begin after the previous steam injection has finished.</p>
      </div>
    </Panel>
  );
}

export function Planner() {
  const { wellId } = useParams();
  const id = (wellId ?? "").toUpperCase();
  const { data, error } = useWellDetail(id || "BGW-01");
  if (!id) return <Navigate to="/planner/BGW-01" replace />;
  return (
    <div className="space-y-4">
      <WellPicker value={id} basePath="/planner" />
      {error && <ErrorNote error={error} />}
      {!data && !error && <Loading what="well" />}
      {data && (
        <>
          <WhatIf key={id} wellId={id} last={data.cycles[data.cycles.length - 1]} />
          <Sweep wellId={id} />
          <CycleEditor key={`ed-${id}`} wellId={id} cycles={data.cycles} />
        </>
      )}
    </div>
  );
}
