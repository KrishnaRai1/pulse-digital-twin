import { useMemo, useState } from "react";
import { Navigate, useParams } from "react-router-dom";
import { api } from "../api/client";
import { useAdvisories, useAudit, useDecision, useOverview, useRecommendation } from "../api/hooks";
import type { Advisory, Recommendation } from "../api/types";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Chart } from "../components/Chart";
import { Slider } from "../components/Slider";
import { WellPicker } from "../components/WellPicker";
import { Btn, Empty, ErrorNote, Loading, Panel, StatusPill, inputCls } from "../components/ui";
import { axisBase, gridBase, legendBase, tooltipBase, zip } from "../lib/chartHelpers";
import { COLORS } from "../lib/echarts";
import { fmt, fmtDateTime, fmtUsd, fmtVisc } from "../lib/format";

function Hero({ rec, pending, wellId }: { rec: Recommendation; pending: Advisory | undefined; wellId: string }) {
  const decide = useDecision();
  const qc = useQueryClient();
  const [actor, setActor] = useState("operator");
  const create = useMutation({
    mutationFn: () => api.post<Advisory>(`/wells/${wellId}/advisories?force=true`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["advisories"] }),
  });
  const x = rec.xai!;
  const adjust = rec.action === "adjust";
  const unsafe = rec.safety && !rec.safety.passes;
  return (
    <div className={`rounded-lg border p-5 ${rec.needs_resteam ? "border-bad/60 bg-bad/8" : adjust ? "border-accent/50 bg-accent/8" : "border-line bg-panel"}`}>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-muted">
            {rec.mode === "advisory" ? "Advisory · operator approval required" : "Closed loop · safe advisories auto-applied"}
          </div>
          <div className="mt-1 text-[26px] font-semibold leading-tight text-ink">{x.action}</div>
          <p className="mt-2 max-w-3xl text-[13.5px] leading-relaxed text-ink/90">{x.narrative}</p>
          <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
            <span className={`rounded-full border px-2 py-0.5 ${rec.safety?.passes ? "border-ok/40 text-ok" : "border-bad/50 text-bad"}`}>{rec.safety?.passes ? "✓ inside physics safety envelope" : "✕ outside the safety envelope"}</span>
            <span className="rounded-full border border-line px-2 py-0.5 text-muted">expected {fmtUsd(x.uplift_usd_per_day)}/day vs holding speed</span>
            {rec.needs_resteam && <span className="rounded-full border border-bad/50 px-2 py-0.5 text-bad">no safe speed at this viscosity: plan a re-steam</span>}
          </div>
          {unsafe && <ul className="mt-2 list-disc pl-5 text-xs text-bad">{rec.safety!.violations.map((v) => <li key={v}>{v}</li>)}</ul>}
        </div>
        <div className="flex min-w-[240px] flex-col gap-2">
          {pending ? (
            <>
              <label className="text-[11px] text-muted">
                Approving as
                <input className={`${inputCls} mt-1`} value={actor} maxLength={64} onChange={(e) => setActor(e.target.value.replace(/[^\w .@-]/g, ""))} aria-label="Operator name" />
              </label>
              <div className="flex gap-2">
                <Btn variant="primary" disabled={decide.isPending || !actor} onClick={() => decide.mutate({ id: pending.id, approve: true, actor })}>
                  Approve &amp; apply {pending.recommended_spm.toFixed(1)} SPM
                </Btn>
                <Btn variant="danger" disabled={decide.isPending || !actor} onClick={() => decide.mutate({ id: pending.id, approve: false, actor })}>
                  Reject
                </Btn>
              </div>
              <p className="text-[11px] text-faint">Advisory #{pending.id} · drafted {fmtDateTime(pending.ts)}. The set-point is re-checked against the hard limits at the moment it is written.</p>
            </>
          ) : adjust ? (
            <>
              <Btn variant="primary" disabled={create.isPending} onClick={() => create.mutate()}>
                Draft advisory for approval
              </Btn>
              <p className="text-[11px] text-faint">No advisory is pending for this well.</p>
              {create.error && <ErrorNote error={create.error} />}
            </>
          ) : (
            <p className="text-xs text-muted">Nothing to approve: the current speed is already the best safe choice.</p>
          )}
          {decide.error && <ErrorNote error={decide.error} />}
        </div>
      </div>
    </div>
  );
}

function ContributionChart({ rec }: { rec: Recommendation }) {
  const opt = useMemo(() => {
    const c = rec.xai!.contributions;
    return {
      tooltip: { ...tooltipBase, trigger: "item", valueFormatter: (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(0)} $/day` },
      grid: gridBase({ left: 120, right: 40, top: 8, bottom: 26 }),
      xAxis: { type: "value", name: "$ per day vs holding speed", nameLocation: "middle", nameGap: 22, ...axisBase },
      yAxis: { type: "category", inverse: true, data: c.map((x) => x.name), ...axisBase, splitLine: { show: false } },
      series: [{ type: "bar", data: c.map((x) => ({ value: Math.round(x.usd_per_day * 10) / 10, itemStyle: { color: x.usd_per_day >= 0 ? COLORS.ok : COLORS.bad } })), barWidth: 16, label: { show: true, position: "right", color: COLORS.muted, fontSize: 11, formatter: (p: { value: number }) => `${p.value >= 0 ? "+" : ""}${p.value.toFixed(0)}` } }],
    };
  }, [rec]);
  return <Chart option={opt} height={210} ariaLabel="Contribution of each factor to the recommendation, dollars per day" />;
}

function PlanChart({ rec }: { rec: Recommendation }) {
  const opt = useMemo(() => {
    const p = rec.plan!;
    const d = p.map((x) => x.day);
    return {
      tooltip: { ...tooltipBase, valueFormatter: (v: number) => v.toFixed(2) },
      legend: { ...legendBase },
      grid: gridBase({ right: 56, top: 44, bottom: 40 }),
      xAxis: { type: "category", name: "Days ahead", nameLocation: "middle", nameGap: 24, data: d, ...axisBase },
      yAxis: [
        { type: "value", name: "SPM", ...axisBase, min: 0 },
        { type: "value", name: "float index", ...axisBase, splitLine: { show: false }, min: 0, max: 1.2 },
      ],
      series: [
        { name: "Planned SPM", type: "line", step: "end", lineStyle: { width: 3 }, symbolSize: 6, color: COLORS.accent, data: p.map((x) => x.spm) },
        {
          name: "Rod float index at plan",
          type: "line",
          yAxisIndex: 1,
          showSymbol: false,
          lineStyle: { width: 2 },
          color: COLORS.warn,
          data: p.map((x) => x.float_index),
          markLine: { symbol: "none", silent: true, lineStyle: { color: COLORS.bad, type: "dashed" }, label: { formatter: "limit 0.85", color: COLORS.bad, fontSize: 10, position: "insideEndTop" }, data: [{ yAxis: rec.envelope?.float_margin ?? 0.85 }] },
        },
      ],
    };
  }, [rec]);
  return <Chart option={opt} height={270} ariaLabel="14-day MPC speed plan with rod-float index" />;
}

function WhatIfSpeed({ rec, wellId }: { rec: Recommendation; wellId: string }) {
  const env = rec.envelope!;
  const lim = rec.limits!;
  const [spm, setSpm] = useState(rec.recommended_spm ?? rec.current_spm ?? 2);
  const [actor] = useState("operator");
  const qc = useQueryClient();
  const apply = useMutation({
    mutationFn: () => api.post(`/wells/${wellId}/setpoint`, { spm, actor }),
    onSuccess: () => {
      for (const k of ["recommendation", "overview", "well-detail", "audit"]) qc.invalidateQueries({ queryKey: [k] });
    },
  });
  const i = env.spm.reduce((best, v, k) => (Math.abs(v - spm) < Math.abs(env.spm[best] - spm) ? k : best), 0);
  const over = spm > lim.max_allowed + 1e-6;
  const opt = useMemo(() => {
    const xMax = env.spm[env.spm.length - 1];
    // quadratic wear penalties explode at high speed: clip the axis so the useful region stays readable
    const peak = Math.max(...env.reward_usd);
    const yFloor = -Math.max(1500, 4 * Math.abs(peak));
    return {
      tooltip: { ...tooltipBase, valueFormatter: (v: number) => v.toFixed(2) },
      legend: { ...legendBase },
      grid: gridBase({ right: 60, top: 44, bottom: 40 }),
      xAxis: { type: "value", name: "Pumping speed, SPM", nameLocation: "middle", nameGap: 24, ...axisBase, min: env.spm[0], max: xMax },
      yAxis: [
        { type: "value", name: "$/day", ...axisBase, min: yFloor },
        { type: "value", name: "index", ...axisBase, splitLine: { show: false }, min: 0, max: 1.5 },
      ],
      series: [
        {
          name: "Net value",
          type: "line",
          showSymbol: false,
          lineStyle: { width: 2.6 },
          color: COLORS.accent,
          data: zip(env.spm, env.reward_usd),
          markArea: { silent: true, itemStyle: { color: "rgba(229,72,77,0.10)" }, label: { color: COLORS.bad, fontSize: 10, position: "insideTop" }, data: lim.max_allowed < xMax ? [[{ xAxis: lim.max_allowed, name: "unsafe" }, { xAxis: xMax }]] : [] },
          markLine: {
            symbol: "none",
            silent: true,
            label: { fontSize: 10, color: COLORS.ink },
            data: [
              { xAxis: rec.current_spm, label: { formatter: "now" }, lineStyle: { color: COLORS.muted, type: "dotted" } },
              { xAxis: rec.recommended_spm, label: { formatter: "MPC" }, lineStyle: { color: COLORS.ok, type: "solid" } },
              { xAxis: spm, label: { formatter: "test" }, lineStyle: { color: COLORS.warn, type: "dashed" } },
            ],
          },
        },
        { name: "Rod float index", type: "line", yAxisIndex: 1, showSymbol: false, lineStyle: { width: 1.8 }, color: COLORS.warn, data: zip(env.spm, env.float_index) },
        { name: "Pump fillage", type: "line", yAxisIndex: 1, showSymbol: false, lineStyle: { width: 1.8 }, color: COLORS.violet, data: zip(env.spm, env.fillage) },
        { name: "Fatigue utilisation", type: "line", yAxisIndex: 1, showSymbol: false, lineStyle: { width: 1.5, type: "dashed" }, color: COLORS.orange, data: zip(env.spm, env.fatigue_utilisation) },
      ],
    };
  }, [env, lim, rec.current_spm, rec.recommended_spm, spm]);
  return (
    <Panel title="What-if: pumping speed" subtitle="Slide to test a speed against oil, energy, rod loading and the safety envelope (today's viscosity)">
      <Chart option={opt} height={280} ariaLabel="Net value, float index, fillage and fatigue versus pumping speed" />
      <div className="mt-3 grid gap-4 md:grid-cols-[1fr_1.2fr]">
        <div>
          <Slider label="Test speed" unit="SPM" value={spm} min={env.spm[0]} max={env.spm[env.spm.length - 1]} step={0.25} digits={2} onChange={setSpm} />
          <div className="mt-3 flex items-center gap-2">
            <Btn variant={over ? "default" : "primary"} onClick={() => apply.mutate()} disabled={apply.isPending} title="Manual override: still checked by the physics safety layer on the server">
              Apply {spm.toFixed(2)} SPM manually
            </Btn>
            {over && <span className="text-xs text-bad">above the {fmt(lim.max_allowed, 2)} SPM ceiling: the server will refuse</span>}
          </div>
          {apply.error && <div className="mt-2"><ErrorNote error={apply.error} /></div>}
          {apply.isSuccess && <p className="mt-2 text-xs text-ok">Set-point written.</p>}
        </div>
        <dl className="grid grid-cols-3 gap-2 text-xs">
          {[
            ["Oil", `${fmt(env.oil_m3d[i], 1)} m³/d`],
            ["Power", `${fmt(env.kw[i], 1)} kW`],
            ["Net value", `${fmtUsd(env.reward_usd[i])}/d`],
            ["Float index", `${fmt(env.float_index[i], 2)} / ${env.float_margin}`],
            ["Fillage", `${fmt(env.fillage[i] * 100, 0)}%`],
            ["Fatigue", `${fmt(env.fatigue_utilisation[i] * 100, 0)}%`],
          ].map(([k, v]) => (
            <div key={k} className="rounded-md border border-line bg-bg/50 px-2.5 py-2">
              <dt className="text-[10px] uppercase tracking-wide text-muted">{k}</dt>
              <dd className="num mt-0.5 text-[15px] text-ink">{v}</dd>
            </div>
          ))}
        </dl>
      </div>
    </Panel>
  );
}

export function Optimizer() {
  const { wellId } = useParams();
  const { data: ov } = useOverview();
  const id = (wellId ?? "").toUpperCase();
  const { data: rec, error, isLoading } = useRecommendation(id || "BGW-01");
  const { data: pending } = useAdvisories("pending");
  const { data: history } = useAdvisories();
  const { data: audit } = useAudit();

  if (!id) {
    const first = ov?.wells.find((w) => w.status !== "green" && w.producing) ?? ov?.wells.find((w) => w.producing);
    return <Navigate to={`/optimizer/${first?.id ?? "BGW-01"}`} replace />;
  }
  const pendingHere = pending?.find((a) => a.well_id === id);

  return (
    <div className="space-y-4">
      <WellPicker value={id} basePath="/optimizer" />
      {error && <ErrorNote error={error} />}
      {isLoading && <Loading what="recommendation" />}
      {rec && !rec.applicable && (
        <Panel title={`${id}: no speed optimisation`}>
          <p className="text-sm text-muted">{rec.reason}</p>
        </Panel>
      )}
      {rec && rec.applicable && rec.xai && rec.plan && rec.envelope && rec.limits && (
        <>
          <Hero rec={rec} pending={pendingHere} wellId={id} />
          <div className="grid gap-4 xl:grid-cols-3">
            <Panel title="Why this recommendation" subtitle="Drivers ranked by influence: every number comes from the twin, not from a template">
              <ul className="space-y-3">
                {rec.xai.drivers.map((d) => (
                  <li key={d.factor}>
                    <div className="flex items-baseline justify-between gap-3">
                      <span className="text-[13px] font-semibold text-ink">{d.factor}</span>
                      <span className="shrink-0 text-[11px] text-accent">{d.effect}</span>
                    </div>
                    <div className="mt-1 h-1.5 rounded bg-panel2">
                      <div className="h-full rounded bg-accent/70" style={{ width: `${Math.max(4, d.weight * 100)}%` }} />
                    </div>
                    <p className="mt-1 text-xs leading-relaxed text-muted">{d.detail}</p>
                  </li>
                ))}
              </ul>
              {rec.xai.ml_correction.length > 0 && (
                <p className="mt-4 border-t border-line pt-3 text-[11px] text-faint">
                  ML residual model is currently driven by: {rec.xai.ml_correction.slice(0, 3).map((m) => m.feature.replace(/_/g, " ")).join(", ")}.
                </p>
              )}
            </Panel>
            <Panel title="Value of the change" subtitle="What moves the objective when you leave today's speed">
              <ContributionChart rec={rec} />
              <table className="mt-2 w-full text-xs">
                <tbody>
                  {[
                    ["Safe ceiling now", `${fmt(rec.limits.max_allowed, 2)} SPM (${rec.limits.binding.replace("_", " ")})`],
                    ["Float / fatigue / structure", `${fmt(rec.limits.rod_float, 1)} / ${fmt(rec.limits.rod_fatigue, 1)} / ${fmt(rec.limits.unit_structure, 1)} SPM`],
                    ["Tubing viscosity (day 0 → 14)", `${fmtVisc(rec.plan[0].mu_tub_cp)} → ${fmtVisc(rec.plan[rec.plan.length - 1].mu_tub_cp)} cP`],
                  ].map(([k, v]) => (
                    <tr key={k} className="border-t border-line/50">
                      <td className="py-1 text-muted">{k}</td>
                      <td className="num py-1 text-right text-ink">{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Panel>
            <Panel title="14-day MPC plan" subtitle="Receding-horizon plan: only the first move is ever applied">
              <PlanChart rec={rec} />
            </Panel>
          </div>

          <WhatIfSpeed key={`${id}-${rec.recommended_spm}`} rec={rec} wellId={id} />

          <Panel title="Candidate speeds" subtitle="Alternatives the optimiser compared for today" pad={false}>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[700px] text-xs">
                <thead className="border-b border-line text-[11px] uppercase tracking-wide text-muted">
                  <tr>
                    {["SPM", "Oil m³/d", "Power kW", "Fillage", "Float idx", "Fatigue", "Net $/day", "Safety"].map((h, k) => (
                      <th key={h} className={`px-3 py-2 font-semibold ${k === 0 || k === 7 ? "text-left" : "text-right"}`}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rec.candidates?.map((c) => (
                    <tr key={c.spm} className={`border-b border-line/50 ${Math.abs(c.spm - (rec.recommended_spm ?? -1)) < 0.01 ? "bg-accent/8" : ""}`}>
                      <td className="num px-3 py-1.5 font-semibold text-ink">
                        {c.spm.toFixed(1)}
                        {Math.abs(c.spm - (rec.current_spm ?? -1)) < 0.26 && <span className="ml-1.5 text-[10px] font-normal text-muted">current</span>}
                        {Math.abs(c.spm - (rec.recommended_spm ?? -1)) < 0.01 && <span className="ml-1.5 text-[10px] font-normal text-accent">recommended</span>}
                      </td>
                      <td className="num px-3 py-1.5 text-right">{fmt(c.oil_m3d, 1)}</td>
                      <td className="num px-3 py-1.5 text-right">{fmt(c.kw, 1)}</td>
                      <td className="num px-3 py-1.5 text-right">{fmt(c.fillage * 100, 0)}%</td>
                      <td className="num px-3 py-1.5 text-right">{fmt(c.float_index, 2)}</td>
                      <td className="num px-3 py-1.5 text-right">{fmt(c.fatigue_utilisation * 100, 0)}%</td>
                      <td className="num px-3 py-1.5 text-right">{fmt(c.reward_usd, 0)}</td>
                      <td className="px-3 py-1.5">{c.violations.length ? <span className="text-bad">✕ {c.violations.join(", ")}</span> : <span className="text-ok">✓ inside limits</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Advisories" subtitle="Field-wide queue and history" pad={false}>
          {!history || history.length === 0 ? (
            <Empty>No advisories yet.</Empty>
          ) : (
            <ul className="max-h-80 divide-y divide-line/60 overflow-y-auto">
              {history.map((a) => (
                <li key={a.id} className="flex items-center gap-3 px-4 py-2 text-xs">
                  <StatusPill status={a.status === "pending" ? "amber" : a.status === "applied" ? "green" : "idle"} label={a.status} />
                  <span className="font-semibold text-ink">{a.well_id}</span>
                  <span className="num text-muted">
                    {a.current_spm.toFixed(1)} → {a.recommended_spm.toFixed(1)} SPM
                  </span>
                  <span className="ml-auto text-faint">{a.decided_by ? `${a.decided_by} · ` : ""}{fmtDateTime(a.decided_ts ?? a.ts)}</span>
                </li>
              ))}
            </ul>
          )}
        </Panel>
        <Panel title="Audit trail" subtitle="Every recommendation, decision and set-point write" pad={false}>
          {!audit || audit.length === 0 ? (
            <Empty>No entries.</Empty>
          ) : (
            <ul className="max-h-80 divide-y divide-line/60 overflow-y-auto">
              {audit.map((a) => (
                <li key={a.id} className="flex items-baseline gap-3 px-4 py-1.5 text-xs">
                  <span className="num shrink-0 text-faint">{fmtDateTime(a.ts)}</span>
                  <span className="text-ink">{a.action.replace(/_/g, " ")}</span>
                  <span className="text-muted">{a.well_id ?? ""}</span>
                  <span className="ml-auto truncate text-faint">{a.actor}</span>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>
    </div>
  );
}
