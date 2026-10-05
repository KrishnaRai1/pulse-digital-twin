import { useMemo } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useDsModels, useDsSpm, useDsSpmHistory } from "../../api/dataset";
import { Chart } from "../../components/Chart";
import { DayControl, SourceNote, WellSelect, fmtDay, useAsOfDay } from "../../components/dataset";
import { Gauge } from "../../components/Gauge";
import { Btn, ErrorNote, Loading, Panel } from "../../components/ui";
import { axisBase, gridBase, legendBase, tooltipBase } from "../../lib/chartHelpers";
import { COLORS } from "../../lib/echarts";
import { fmt, fmtInt } from "../../lib/format";

const BINDING_TEXT: Record<string, string> = {
  rod_float: "Rod-float limit is binding",
  inflow: "Matched to the well's inflow",
  ramp_up: "Pump-limited: ramping up",
  max_speed: "At maximum speed",
  min_speed: "At minimum speed (low inflow)",
  float_unavoidable: "Rods float even at minimum speed",
};
const SEV: Record<string, string> = { high: "border-bad/50 bg-bad/8 text-bad", medium: "border-warn/50 bg-warn/8 text-warn", info: "border-line bg-panel2 text-muted" };

function Delta({ label, now, next, unit, digits = 1, goodDown = false }: { label: string; now: number; next: number; unit: string; digits?: number; goodDown?: boolean }) {
  const diff = next - now;
  const better = Math.abs(diff) < 1e-6 ? null : goodDown ? diff < 0 : diff > 0;
  return (
    <div className="rounded-md border border-line bg-panel2/50 px-3 py-2">
      <div className="text-[11px] uppercase tracking-wider text-muted">{label}</div>
      <div className="num mt-0.5 text-[15px] text-ink">
        {fmt(now, digits)} <span className="text-faint">→</span> <span className={better === null ? "" : better ? "text-ok" : "text-warn"}>{fmt(next, digits)}</span> <span className="text-xs text-muted">{unit}</span>
      </div>
    </div>
  );
}

export function SpmAdvisor() {
  const { wellId = "WELL-010" } = useParams();
  const id = wellId.toUpperCase();
  const { day, setDay } = useAsOfDay();
  const { data: a, error, isFetching } = useDsSpm(id, day);
  const { data: hist } = useDsSpmHistory(id);
  const { data: models } = useDsModels();
  const nav = useNavigate();

  const nearest = useMemo(() => {
    if (!hist?.day.length) return null;
    let best = hist.day[0];
    for (const d of hist.day) if (Math.abs(d - day) < Math.abs(best - day)) best = d;
    return best;
  }, [hist, day]);

  const envelope = useMemo(() => {
    if (!a?.curve || a.current_spm === undefined || a.recommended_spm === undefined || !a.state || !a.expected) return null;
    const c = a.curve;
    return {
      tooltip: { ...tooltipBase, valueFormatter: (v: number) => fmt(v, 2) },
      legend: { ...legendBase },
      grid: gridBase({ left: 50, right: 60, bottom: 40 }),
      xAxis: { type: "value", name: "SPM", nameLocation: "middle", nameGap: 24, min: 2, max: 9.6, ...axisBase },
      yAxis: [
        { type: "value", name: "RFI", min: 0, ...axisBase },
        { type: "value", name: "bbl/d", min: 0, ...axisBase, splitLine: { show: false } },
      ],
      series: [
        {
          name: "Rod-floating index",
          type: "line",
          showSymbol: false,
          data: c.spm.map((s, i) => [s, c.rfi[i]]),
          lineStyle: { color: COLORS.warn, width: 2 },
          itemStyle: { color: COLORS.warn },
          markArea: { silent: true, data: [[{ yAxis: 1, itemStyle: { color: "#e5484d14" } }, { yAxis: 10 }]] },
          markLine: {
            symbol: "none",
            silent: true,
            data: [
              { yAxis: 1, lineStyle: { color: COLORS.bad }, label: { formatter: "RFI 1: rods float", color: COLORS.bad, fontSize: 10, position: "insideStartTop" } },
              { yAxis: 0.9, lineStyle: { color: COLORS.warn, type: "dotted" }, label: { formatter: "0.9 target", color: COLORS.warn, fontSize: 10, position: "insideStartTop" } },
            ],
          },
          markPoint: {
            symbolSize: 12,
            data: [
              { coord: [a.current_spm, a.state.rfi_model], symbol: "circle", itemStyle: { color: COLORS.muted }, label: { formatter: "now", color: COLORS.ink, position: "top", fontSize: 10 } },
              { coord: [a.recommended_spm, a.expected.rfi], symbol: "diamond", itemStyle: { color: COLORS.accent }, label: { formatter: "advised", color: COLORS.accent, position: "bottom", fontSize: 10 } },
            ],
          },
        },
        { name: "Pump capacity", type: "line", yAxisIndex: 1, showSymbol: false, data: c.spm.map((s, i) => [s, c.capacity_bbl_d[i]]), lineStyle: { color: COLORS.muted, type: "dashed", width: 1 }, itemStyle: { color: COLORS.muted } },
        { name: c.inflow_known ? "Liquid lifted" : "Liquid lifted (≥, pump full)", type: "line", yAxisIndex: 1, showSymbol: false, data: c.spm.map((s, i) => [s, c.lifted_bbl_d[i]]), lineStyle: { color: COLORS.accent, width: 1.6 }, itemStyle: { color: COLORS.accent } },
      ],
    };
  }, [a]);

  const outlook = useMemo(() => {
    if (!a?.outlook) return null;
    const o = a.outlook;
    return {
      tooltip: { ...tooltipBase, valueFormatter: (v: number) => fmt(v, 2) },
      legend: { ...legendBase },
      grid: gridBase({ left: 46, right: 60, bottom: 30 }),
      xAxis: { type: "category", data: o.day, ...axisBase },
      yAxis: [
        { type: "value", name: "SPM", min: 0, max: 10, ...axisBase },
        { type: "log", name: "cP", ...axisBase, splitLine: { show: false } },
      ],
      series: [
        { name: "Rod-float limit (RFI 0.9)", type: "line", data: o.spm_rod_float_limit, showSymbol: false, lineStyle: { color: COLORS.warn, type: "dashed" }, itemStyle: { color: COLORS.warn } },
        { name: "Planned SPM", type: "line", step: "end", data: o.spm_plan, showSymbol: false, lineStyle: { color: COLORS.accent, width: 2 }, itemStyle: { color: COLORS.accent } },
        { name: "Oil viscosity (physics prior)", type: "line", yAxisIndex: 1, data: o.prior_oil_viscosity_cp, showSymbol: false, lineStyle: { color: COLORS.blue, width: 1 }, itemStyle: { color: COLORS.blue } },
      ],
    };
  }, [a]);

  const history = useMemo(() => {
    if (!hist) return null;
    const pts = (v: (number | null)[]) => hist.day.map((d, i) => [d, v[i]]);
    return {
      tooltip: { ...tooltipBase, valueFormatter: (v: number) => fmt(v, 2) },
      legend: { ...legendBase },
      grid: gridBase({ left: 46, right: 50, bottom: 50 }),
      dataZoom: [{ type: "inside" }, { type: "slider", height: 16, bottom: 6, borderColor: COLORS.line, textStyle: { color: COLORS.muted } }],
      xAxis: { type: "value", min: hist.day[0], max: hist.day[hist.day.length - 1], ...axisBase },
      yAxis: [
        { type: "value", name: "SPM", min: 0, max: 10, ...axisBase },
        { type: "value", name: "RFI", min: 0, ...axisBase, splitLine: { show: false } },
      ],
      series: [
        {
          name: "Logged SPM",
          type: "scatter",
          symbolSize: 2.5,
          data: pts(hist.spm_logged),
          itemStyle: { color: COLORS.muted },
          markLine: { symbol: "none", silent: true, data: [{ xAxis: day, lineStyle: { color: COLORS.warn }, label: { show: false } }] },
        },
        { name: "Advised SPM", type: "scatter", symbolSize: 2.5, data: pts(hist.spm_advised), itemStyle: { color: COLORS.accent } },
        { name: "RFI logged", type: "scatter", yAxisIndex: 1, symbolSize: 2, data: pts(hist.rfi_logged), itemStyle: { color: COLORS.bad, opacity: 0.6 } },
        { name: "RFI with advice", type: "scatter", yAxisIndex: 1, symbolSize: 2, data: pts(hist.rfi_advised), itemStyle: { color: COLORS.ok, opacity: 0.7 } },
      ],
    };
  }, [hist, day]);

  const bt = models?.spm_advisor;
  const tone = a?.action === "reduce" ? "border-warn/60 bg-warn/8" : a?.action === "increase" ? "border-accent/60 bg-accent/8" : "border-ok/50 bg-ok/8";

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-lg font-semibold text-ink">SPM advisor</h1>
          <WellSelect value={id} basePath="/spm" />
          {isFetching && <span className="text-xs text-muted">updating…</span>}
        </div>
        <Btn small onClick={() => nav(`/history/${id}?day=${day}`)}>Well history →</Btn>
      </div>
      <DayControl />
      {error && <ErrorNote error={error} />}

      {a && !a.applicable ? (
        <Panel title={`${id} · ${fmtDay(day)}`}>
          <p className="text-sm text-muted">No advice: {a.reason}.</p>
          {nearest !== null && (
            <Btn small variant="primary" onClick={() => setDay(nearest)}>
              Go to the nearest production day ({fmtDay(nearest)})
            </Btn>
          )}
        </Panel>
      ) : !a || !a.state || !a.expected ? (
        <Loading what="advice" />
      ) : (
        <>
          <div className="grid gap-4 xl:grid-cols-[1.1fr_1fr]">
            <section className={`rounded-lg border p-4 ${tone}`} aria-live="polite">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-muted">
                Recommendation · {id} · {fmtDay(a.day)} · cycle {a.cycle}
              </div>
              <div className="mt-1 text-[28px] font-semibold leading-tight text-ink">{a.headline}</div>
              <div className="mt-1 text-sm text-muted">
                {BINDING_TEXT[a.binding_constraint ?? ""] ?? a.binding_constraint} · VFD {fmt(a.state.vfd_hz, 1)} → {fmt(a.expected.vfd_hz, 1)} Hz · advisory: an operator applies the set-point
              </div>
              <div className="mt-3 grid grid-cols-2 gap-2 md:grid-cols-4">
                <Delta label="Rod-floating index" now={a.state.rfi_model} next={a.expected.rfi} unit="" digits={2} goodDown />
                <Delta label="Pump fillage" now={a.state.fillage_pct} next={a.expected.fillage_pct} unit="%" digits={0} />
                <Delta label="Oil" now={a.state.oil_bbl_d} next={a.expected.oil_bbl_d} unit="bbl/d" />
                <Delta label="Motor power" now={a.state.motor_kw} next={a.expected.motor_kw} unit="kW" goodDown />
              </div>
              {a.expected.note && <p className="mt-2 text-[11px] text-faint">Note: {a.expected.note}.</p>}
              <h3 className="mt-4 text-xs font-semibold uppercase tracking-wider text-muted">Why</h3>
              <ul className="mt-1.5 space-y-1.5">
                {(a.reasons ?? []).map((r, i) => (
                  <li key={i} className={`rounded-md border px-3 py-1.5 text-[13px] ${SEV[r.severity]}`}>
                    {r.text}
                  </li>
                ))}
              </ul>
            </section>
            <Panel title="Operating state" subtitle={`Physics-prior oil viscosity ${fmtInt(a.state.oil_viscosity_prior_cp)} cP · fluid ${fmt(a.state.mixture_viscosity_cp, 0)} cP · water cut ${fmt(a.state.water_cut_pct, 0)}%`}>
              <div className="grid grid-cols-2 gap-1">
                <Gauge title="Pump speed" value={a.current_spm ?? null} unit="SPM" max={9.6} limit={a.limits ? Math.min(a.limits.spm_rod_float, 9.6) : null} bands={[[0.6, COLORS.ok], [0.85, COLORS.warn], [1, COLORS.bad]]} />
                <Gauge title="Rod-floating index" value={a.state.rfi_model} unit="RFI" max={2} digits={2} limit={1} bands={[[0.45, COLORS.ok], [0.5, COLORS.warn], [1, COLORS.bad]]} />
                <Gauge title="Pump fillage" value={a.state.fillage_pct} unit="%" max={100} digits={0} bands={[[0.5, COLORS.bad], [0.75, COLORS.warn], [1, COLORS.ok]]} />
                <Gauge title="Motor load" value={a.state.motor_kw} unit="kW" max={18.5} bands={[[0.75, COLORS.ok], [0.9, COLORS.warn], [1, COLORS.bad]]} />
              </div>
              <SourceNote>
                Rod-fall speed {fmt(a.state.rod_fall_speed_m_s, 2)} m/s vs. polished-rod speed {fmt(a.state.rod_speed_m_s, 2)} m/s. Logged RFI {fmt(a.state.rfi_logged, 2)} (dataset) vs. {fmt(a.state.rfi_model, 2)} from the
                advisor&apos;s physics-prior viscosity.
              </SourceNote>
            </Panel>
          </div>

          <div className="grid gap-4 xl:grid-cols-2">
            <Panel title="Operating envelope" subtitle="Rod-floating index and liquid lifted across the speed range at today's fluid">
              {envelope && <Chart option={envelope} height={280} ariaLabel="RFI and lifted liquid versus SPM" />}
            </Panel>
            <Panel title="Next 14 days" subtitle="As the heated zone cools, viscosity rises and the safe speed falls">
              {outlook && <Chart option={outlook} height={280} ariaLabel="14-day SPM outlook" />}
              <SourceNote>{a.outlook?.note}.</SourceNote>
            </Panel>
          </div>
        </>
      )}

      {history && hist && (
        <Panel
          title={`Replay over ${id}'s production days`}
          subtitle={`Floating days: ${hist.summary.floating_days_logged} logged → ${hist.summary.floating_days_advised} with the advisor · mean SPM ${fmt(hist.summary.mean_spm_logged, 2)} → ${fmt(hist.summary.mean_spm_advised, 2)}`}
        >
          <Chart option={history} height={260} ariaLabel="Logged versus advised SPM over the well's history" onClick={(p) => setDay(((p as { value?: number[] }).value ?? [day])[0])} />
        </Panel>
      )}

      {bt && (
        <Panel title="Field back-test: logged operation vs. advisor" subtitle={`${fmtInt(bt.n_days)} running production days, all 300 wells, decided day by day with what was known that day`}>
          <div className="grid gap-4 lg:grid-cols-[1.2fr_1fr]">
            <table className="w-full text-[13px]">
              <thead className="text-xs text-muted">
                <tr className="border-b border-line">
                  <th className="py-1.5 text-left font-medium">Metric</th>
                  <th className="py-1.5 text-right font-medium">Logged</th>
                  <th className="py-1.5 text-right font-medium">Advisor</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ["Days with rods floating (RFI > 1)", "days_floating_pct", "%", 2],
                  ["Float exposure (Σ RFI − 1)", "float_exposure_sum", "", 0],
                  ["Avoidable fluid pound (fill < 50% above min speed)", "days_pound_above_min_speed_pct", "%", 1],
                  ["All days with fill < 50%", "days_low_fillage_pct", "%", 1],
                  ["Oil lifted (conservative)", "oil_bbl", "bbl", 0],
                  ["Pump energy", "energy_mwh", "MWh", 0],
                  ["Mean speed", "mean_spm", "SPM", 2],
                ].map(([label, key, unit, dg]) => (
                  <tr key={key as string} className="border-b border-line/60">
                    <td className="py-1.5 text-muted">{label}</td>
                    <td className="num py-1.5 text-right">
                      {fmt(bt.logged[key as string], dg as number)} {unit}
                    </td>
                    <td className="num py-1.5 text-right text-ink">
                      {fmt(bt.advisor[key as string], dg as number)} {unit}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="text-xs text-muted">
              <p>
                Advice changes speed on {fmt(bt.share_of_days.reduce, 1)}% of days (down) and {fmt(bt.share_of_days.increase, 1)}% (up). Targets: RFI ≤ {bt.rfi_target}, fillage ≈ {bt.fillage_target_pct}%.
              </p>
              <p className="mt-2 font-medium text-ink">Assumptions</p>
              <ul className="mt-1 list-disc space-y-0.5 pl-4">
                {bt.assumptions.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ul>
            </div>
          </div>
        </Panel>
      )}
    </div>
  );
}
