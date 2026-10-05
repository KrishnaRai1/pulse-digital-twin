import { useMemo } from "react";
import { Link, Navigate, useParams } from "react-router-dom";
import { useProfile, useTelemetry, useViscosity, useWellDetail } from "../api/hooks";
import type { Telemetry, WellDetail } from "../api/types";
import { Chart } from "../components/Chart";
import { Gauge } from "../components/Gauge";
import { WellPicker } from "../components/WellPicker";
import { ErrorNote, KpiTile, Loading, Panel, StatusPill } from "../components/ui";
import { axisBase, gridBase, legendBase, timeAxisData, tooltipBase, zip } from "../lib/chartHelpers";
import { COLORS } from "../lib/echarts";
import { PHASE_LABEL, classLabel, fmt, fmtDate, fmtInt, fmtPct, fmtVisc } from "../lib/format";

function lastValue(t: Telemetry | undefined, key: string): number | null {
  const arr = t?.series[key];
  if (!arr) return null;
  for (let i = arr.length - 1; i >= 0; i--) if (arr[i] !== null && arr[i] !== undefined) return arr[i] as number;
  return null;
}

function WellboreChart({ well }: { well: ReturnType<typeof useProfile>["data"] }) {
  const opt = useMemo(() => {
    if (!well) return null;
    const w = well.wellbore;
    return {
      tooltip: { ...tooltipBase, trigger: "axis", axisPointer: { type: "cross", label: { backgroundColor: "#1a222c" } } },
      legend: { ...legendBase },
      grid: gridBase({ top: 70, left: 60, right: 24, bottom: 20 }),
      xAxis: [
        { type: "value", name: "Temperature °C", nameLocation: "middle", nameGap: 26, position: "bottom", ...axisBase, min: 20 },
        { type: "value", name: "Pressure kPa", nameLocation: "middle", nameGap: 26, position: "top", ...axisBase, splitLine: { show: false }, min: 0 },
      ],
      yAxis: { type: "value", name: "Depth m", inverse: true, ...axisBase, min: 0, max: Math.ceil((w.pay_bottom_m + 20) / 100) * 100 },
      series: [
        { name: "Geotherm", type: "line", showSymbol: false, lineStyle: { type: "dashed", width: 1.5 }, color: COLORS.faint, data: zip(w.t_geotherm_c, w.depth_m) },
        {
          name: "Flowing temperature",
          type: "line",
          showSymbol: false,
          lineStyle: { width: 2.5 },
          color: COLORS.orange,
          data: zip(w.t_flowing_c, w.depth_m),
          markLine: {
            symbol: "none",
            silent: true,
            label: { color: COLORS.muted, fontSize: 10, position: "insideEndTop" },
            lineStyle: { color: COLORS.faint, type: "dotted" },
            data: [
              { yAxis: w.pump_depth_m, label: { formatter: "pump", position: "insideEndTop" } },
              { yAxis: w.fluid_level_m, label: { formatter: "fluid level", position: "insideEndTop" }, lineStyle: { color: COLORS.accent, type: "dotted" } },
            ],
          },
        },
        { name: "Pressure", type: "line", xAxisIndex: 1, showSymbol: false, lineStyle: { width: 2 }, color: COLORS.accent, data: zip(w.pressure_kpa, w.depth_m) },
      ],
    };
  }, [well]);
  return opt ? <Chart option={opt} height={330} ariaLabel="Wellbore depth versus temperature and pressure" /> : <Loading what="wellbore profile" />;
}

function RadialChart({ profile }: { profile: ReturnType<typeof useProfile>["data"] }) {
  const opt = useMemo(() => {
    if (!profile) return null;
    const r = profile.radial;
    return {
      tooltip: { ...tooltipBase, valueFormatter: (v: number) => v.toFixed(1) },
      legend: { ...legendBase },
      grid: gridBase({ right: 60, top: 44, bottom: 40 }),
      xAxis: { type: "log", name: "Radius from wellbore, m", nameLocation: "middle", nameGap: 26, ...axisBase, min: 0.1, max: 200 },
      yAxis: [
        { type: "value", name: "°C", ...axisBase },
        { type: "log", name: "cP", ...axisBase, splitLine: { show: false }, min: 1 },
      ],
      series: [
        {
          name: "Temperature",
          type: "line",
          showSymbol: false,
          lineStyle: { width: 2.5 },
          color: COLORS.orange,
          data: zip(r.r_m, r.t_c),
          markLine: { symbol: "none", silent: true, lineStyle: { color: COLORS.steam, type: "dashed" }, label: { formatter: "heated radius", color: COLORS.muted, fontSize: 10 }, data: [{ xAxis: Math.max(r.r_heated_m, 0.11) }] },
        },
        { name: "Oil viscosity", type: "line", yAxisIndex: 1, showSymbol: false, lineStyle: { width: 2.5 }, color: COLORS.violet, data: zip(r.r_m, r.mu_cp) },
      ],
    };
  }, [profile]);
  return opt ? <Chart option={opt} height={280} ariaLabel="Radial temperature and viscosity decay" /> : <Loading what="radial profile" />;
}

function CrossSection({ profile }: { profile: ReturnType<typeof useProfile>["data"] }) {
  const opt = useMemo(() => {
    if (!profile) return null;
    const cs = profile.cross_section;
    const data: [number, number, number][] = [];
    cs.t_c.forEach((row, zi) => row.forEach((t, ri) => data.push([ri, zi, Math.round(t * 10) / 10])));
    const tmin = Math.min(...cs.t_c.flat());
    const tmax = Math.max(...cs.t_c.flat());
    return {
      tooltip: { ...tooltipBase, trigger: "item", formatter: (p: { data: number[] }) => `r ${cs.r_m[p.data[0]].toFixed(1)} m · z ${cs.z_m[p.data[1]].toFixed(1)} m<br/><b>${p.data[2].toFixed(0)} °C</b>` },
      grid: gridBase({ top: 12, bottom: 84, left: 56, right: 16 }),
      xAxis: { type: "category", name: "Radius m", nameLocation: "middle", nameGap: 28, data: cs.r_m.map((v) => v.toFixed(1)), ...axisBase, splitLine: { show: false }, axisLabel: { ...axisBase.axisLabel, interval: Math.ceil(cs.r_m.length / 8) } },
      yAxis: { type: "category", name: "Pay depth m", inverse: true, data: cs.z_m.map((v) => v.toFixed(0)), ...axisBase, splitLine: { show: false }, axisLabel: { ...axisBase.axisLabel, interval: 2 } },
      visualMap: { min: tmin, max: tmax, calculable: false, orient: "horizontal", left: "center", bottom: 0, itemWidth: 12, itemHeight: 140, textStyle: { color: COLORS.muted, fontSize: 10 }, inRange: { color: ["#0c1b2c", "#163350", "#f59e0b", "#f97316", "#ef4444"] }, text: ["°C", ""] },
      series: [{ type: "heatmap", data, progressive: 0 }],
    };
  }, [profile]);
  return opt ? (
    <>
      <Chart option={opt} height={260} ariaLabel="Reservoir cross-section temperature" />
      <p className="mt-1 text-[11px] text-faint">{profile?.cross_section.note}</p>
    </>
  ) : (
    <Loading what="cross-section" />
  );
}

function CycleTimeline({ well }: { well: string }) {
  const { data } = useViscosity(well);
  const opt = useMemo(() => {
    if (!data) return null;
    const soakStart = data.inj_days;
    const prodStart = data.inj_days + data.soak_days;
    const end = data.t_days[data.t_days.length - 1] ?? 0;
    const area = (a: number, b: number, c: string, name: string) => [{ name, xAxis: a, itemStyle: { color: c } }, { xAxis: b }];
    return {
      tooltip: { ...tooltipBase, valueFormatter: (v: number) => v.toFixed(1) },
      legend: { ...legendBase, data: ["Viscosity (physics)", data.ml_used ? "Viscosity (physics + ML)" : "", "Oil rate"].filter(Boolean) },
      grid: gridBase({ right: 56, top: 44, bottom: 40 }),
      xAxis: { type: "value", name: "Days since start of injection", nameLocation: "middle", nameGap: 26, ...axisBase, min: 0, max: end },
      yAxis: [
        { type: "log", name: "cP", ...axisBase, min: 1 },
        { type: "value", name: "m³/d", ...axisBase, splitLine: { show: false }, min: 0 },
      ],
      series: [
        {
          name: "Viscosity (physics)",
          type: "line",
          showSymbol: false,
          lineStyle: { width: 1.5, type: "dashed" },
          color: COLORS.faint,
          data: zip(data.t_days, data.mu_eff_phys_cp),
          markArea: {
            silent: true,
            label: { color: COLORS.muted, fontSize: 10, position: "insideTop" },
            data: [area(0, soakStart, "rgba(229,72,77,0.10)", "Injection"), area(soakStart, prodStart, "rgba(227,167,47,0.10)", "Soak"), area(prodStart, end, "rgba(76,195,217,0.05)", "Production")],
          },
          markLine: { symbol: "none", silent: true, lineStyle: { color: COLORS.ink }, label: { formatter: "now", color: COLORS.ink, fontSize: 10 }, data: [{ xAxis: Math.min(data.now_day, end) }] },
        },
        ...(data.ml_used ? [{ name: "Viscosity (physics + ML)", type: "line", showSymbol: false, lineStyle: { width: 2.5 }, color: COLORS.violet, data: zip(data.t_days, data.mu_eff_cp) }] : []),
        { name: "Oil rate", type: "line", yAxisIndex: 1, showSymbol: false, lineStyle: { width: 2 }, color: COLORS.accent, data: zip(data.t_days, data.q_oil_m3d) },
      ],
    };
  }, [data]);
  return opt ? <Chart option={opt} height={280} ariaLabel="Viscosity and oil rate through the steam cycle" /> : <Loading what="cycle curves" />;
}

function SensorGrid({ t }: { t: Telemetry }) {
  const x = timeAxisData(t.series.ts);
  const line = (name: string, key: string, color: string, extra: Record<string, unknown> = {}) => ({
    name,
    type: "line",
    showSymbol: false,
    lineStyle: { width: 1.8 },
    color,
    data: x.map((v, i) => [v, t.series[key]?.[i] ?? null]),
    connectNulls: false,
    ...extra,
  });
  const panels: { title: string; unit: string; series: ReturnType<typeof line>[]; log?: boolean }[] = [
    { title: "Pumping speed", unit: "SPM", series: [line("SPM", "spm", COLORS.accent)] },
    { title: "Motor load / power", unit: "% · kW", series: [line("Load %", "motor_load_pct", COLORS.warn), line("kW", "motor_kw", COLORS.violet)] },
    { title: "Polished-rod load", unit: "kN", series: [line("PPRL", "pprl_kn", COLORS.bad), line("MPRL", "mprl_kn", COLORS.blue)] },
    { title: "Wellhead pressure", unit: "kPa", series: [line("WHP", "whp_kpa", COLORS.steam)] },
    { title: "Wellhead temperature", unit: "°C", series: [line("Tubing T", "tubing_temp_c", COLORS.orange)] },
    { title: "Oil & liquid rate", unit: "m³/d", series: [line("Oil", "oil_rate_m3d", COLORS.ok), line("Liquid", "liquid_rate_m3d", COLORS.faint)] },
    { title: "Tubing viscosity", unit: "cP", log: true, series: [line("Viscosity", "visc_cp", COLORS.violet)] },
    { title: "Pump fillage", unit: "%", series: [line("Fillage", "fillage_pct", COLORS.accent)] },
  ];
  return (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
      {panels.map((p) => (
        <div key={p.title} className="rounded-md border border-line bg-bg/40 p-2">
          <div className="flex items-baseline justify-between px-1">
            <span className="text-[11px] font-semibold uppercase tracking-wide text-muted">{p.title}</span>
            <span className="text-[10px] text-faint">{p.unit}</span>
          </div>
          <Chart
            height={140}
            ariaLabel={`${p.title}, last 24 hours`}
            option={{
              tooltip: { ...tooltipBase, valueFormatter: (v: number) => (v == null ? "–" : v.toFixed(2)) },
              grid: { left: 40, right: 8, top: 8, bottom: 22 },
              xAxis: { type: "time", ...axisBase, axisLabel: { ...axisBase.axisLabel, hideOverlap: true, fontSize: 10 } },
              yAxis: { type: p.log ? "log" : "value", scale: true, ...axisBase, axisLabel: { ...axisBase.axisLabel, fontSize: 10 } },
              series: p.series,
            }}
          />
        </div>
      ))}
    </div>
  );
}

export function WellTwin() {
  const { wellId = "" } = useParams();
  const id = wellId.toUpperCase();
  const { data: d, error, isLoading } = useWellDetail(id);
  const { data: tele } = useTelemetry(id, 24);
  const { data: profile } = useProfile(id);

  if (!id) return <Navigate to="/wells/BGW-01" replace />;
  if (error) return (<div className="space-y-3"><WellPicker value={id} basePath="/wells" /><ErrorNote error={error} /></div>);
  if (isLoading || !d) return <Loading what="well" />;
  const s = d.summary;
  const st = d.state;
  const design = d.config.design;
  const spmNow = s.producing ? s.spm : 0;
  const ceiling = d.limits.max_allowed;
  const vfdLoad = lastValue(tele, "motor_load_pct");
  const stroke = lastValue(tele, "stroke_m");

  return (
    <div className="space-y-4">
      <WellPicker value={id} basePath="/wells" />
      <Header d={d} />

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-6">
        <KpiTile label="Oil rate" value={s.producing ? fmt(s.oil_rate_m3d, 1) : "–"} unit="m³/d" sub={`water cut ${fmtPct(s.water_cut)}`} />
        <KpiTile label="Tubing viscosity" value={fmtVisc(st.mu_tub_cp)} unit="cP" sub={`wellhead ${fmt(st.t_wellhead_c, 0)} °C`} tone={st.mu_tub_cp > 800 ? "warn" : "neutral"} />
        <KpiTile label="Heated-zone temp" value={fmt(st.t_avg_c, 0)} unit="°C" sub={`sandface ${fmt(st.t_wellbore_c, 0)} °C`} />
        <KpiTile label="Thermal health" value={fmtPct(st.thermal_health)} sub="of injected heat still stored" tone={st.thermal_health < 0.22 ? "bad" : st.thermal_health < 0.45 ? "warn" : "ok"} />
        <KpiTile label="Heated radius" value={fmt(st.r_heated_m, 1)} unit="m" sub={`steam ${fmtInt(st.steam_total_m3)} m³ · ${fmt(st.t_steam_c, 0)} °C`} />
        <KpiTile label="Cum. oil / SOR" value={fmtInt(st.cum_oil_m3)} unit="m³" sub={`SOR ${fmt(s.sor, 2)}`} />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Panel title="Lift equipment" subtitle={`Set-point ${fmt(d.setpoint_spm, 1)} SPM · ceiling ${fmt(ceiling, 1)} SPM (binding: ${d.limits.binding.replace("_", " ")})`} className="lg:col-span-1">
          <div className="grid grid-cols-3 gap-1">
            <Gauge
              title="Speed"
              value={spmNow}
              unit="SPM"
              max={Math.ceil(design.max_spm)}
              bands={[
                [Math.min(0.99, (ceiling * 0.85) / Math.ceil(design.max_spm)), COLORS.ok],
                [Math.min(0.995, ceiling / Math.ceil(design.max_spm)), COLORS.warn],
                [1, COLORS.bad],
              ]}
              limit={ceiling}
            />
            <Gauge title="Stroke" value={stroke ?? design.stroke_m} unit="m" max={Math.ceil(design.stroke_m + 1)} digits={2} bands={[[1, COLORS.accent]]} />
            <Gauge title="VFD load" value={vfdLoad} unit="%" max={120} digits={0} bands={[[0.7, COLORS.ok], [0.9, COLORS.warn], [1, COLORS.bad]]} />
          </div>
          <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
            {[
              ["Pump depth", `${fmt(design.pump_depth_m, 0)} m`],
              ["Plunger", `${fmt(design.plunger_mm, 0)} mm`],
              ["Unit rating", `${fmt(design.unit_rating_kn, 0)} kN`],
              ["Motor", `${fmt(design.motor_kw_rated, 0)} kW`],
              ["Rod taper", design.rods.map((r) => `${r.dia_mm.toFixed(0)}`).join(" / ") + " mm"],
              ["Float index", s.float_index != null ? `${s.float_index.toFixed(2)} (limit 0.85)` : "–"],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between border-b border-line/50 py-0.5">
                <dt className="text-muted">{k}</dt>
                <dd className="num text-ink">{v}</dd>
              </div>
            ))}
          </dl>
        </Panel>

        <Panel title="Wellbore profile" subtitle="Depth versus temperature and pressure" className="lg:col-span-2">
          <WellboreChart well={profile} />
        </Panel>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Radial temperature &amp; viscosity decay" subtitle="How far the heat reached and how thin the oil is at each radius">
          <RadialChart profile={profile} />
        </Panel>
        <Panel title="Reservoir cross-section" subtitle="Temperature field around the wellbore">
          <CrossSection profile={profile} />
        </Panel>
      </div>

      <Panel title="Steam-cycle timeline" subtitle="Viscosity (physics vs physics + ML residual correction) and oil rate through the current cycle">
        <CycleTimeline well={id} />
      </Panel>

      <Panel title="Live sensors, last 24 h" subtitle="Updates every few seconds from the time-series store">
        {tele ? <SensorGrid t={tele} /> : <Loading what="telemetry" />}
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="ML correction drivers" subtitle="Which features move the LightGBM residual (SHAP contributions to ln q_obs/q_phys)">
          {d.ml_top_features.length === 0 ? (
            <p className="text-sm text-muted">No ML contribution for this phase.</p>
          ) : (
            <ul className="space-y-2">
              {d.ml_top_features.map((f) => {
                const max = Math.max(...d.ml_top_features.map((x) => Math.abs(x.contribution)), 1e-6);
                const w = (Math.abs(f.contribution) / max) * 100;
                return (
                  <li key={f.feature} className="text-xs">
                    <div className="mb-0.5 flex justify-between">
                      <span className="text-ink">{f.feature.replace(/_/g, " ")}</span>
                      <span className="num text-muted">
                        {f.contribution >= 0 ? "+" : ""}
                        {f.contribution.toFixed(3)} <span className="text-faint">(value {fmt(f.value, 2)})</span>
                      </span>
                    </div>
                    <div className="h-1.5 rounded bg-panel2">
                      <div className="h-full rounded" style={{ width: `${w}%`, background: f.contribution >= 0 ? COLORS.ok : COLORS.bad }} />
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
          <p className="mt-3 text-[11px] text-faint">
            Model output is clipped to ×0.5–×2 of the physics prediction, so the physics stays in charge. Current correction factor: ×{fmt(st.ml_factor, 2)}.
          </p>
        </Panel>

        <Panel title="Steam-cycle history" subtitle="Recorded cycles (edit or extend them on the CSS Optimization page)" actions={<Link className="text-xs text-accent hover:underline" to={`/planner/${id}`}>Edit &amp; what-if →</Link>} pad={false}>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="border-b border-line text-[11px] uppercase tracking-wide text-muted">
                <tr>
                  {["Cycle", "Start", "Steam m³", "Quality", "Soak d", "Prod d"].map((h) => (
                    <th key={h} className="px-3 py-2 text-left font-semibold">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {d.cycles.map((c) => (
                  <tr key={c.cycle_no} className="border-b border-line/50">
                    <td className="num px-3 py-1.5">{c.cycle_no}</td>
                    <td className="px-3 py-1.5 text-muted">{fmtDate(c.start_ts)}</td>
                    <td className="num px-3 py-1.5">{fmtInt(c.steam_m3)}</td>
                    <td className="num px-3 py-1.5">{fmtPct(c.quality)}</td>
                    <td className="num px-3 py-1.5">{fmt(c.soak_days, 0)}</td>
                    <td className="num px-3 py-1.5">{fmt(c.prod_days, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>
    </div>
  );
}

function Header({ d }: { d: WellDetail }) {
  const s = d.summary;
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border border-line bg-panel px-4 py-3">
      <div>
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-semibold text-ink">{s.name}</h1>
          <StatusPill status={s.status === "green" && s.phase !== "PRODUCTION" ? "idle" : s.status} label={s.status === "green" && s.phase !== "PRODUCTION" ? PHASE_LABEL[s.phase] : undefined} />
        </div>
        <div className="mt-0.5 text-xs text-muted">
          Pad {s.pad} · cycle {s.cycle_no}, day {fmtInt(s.day_in_cycle)} · {PHASE_LABEL[s.phase]}
          {s.diag && (
            <>
              {" "}
              · CNN: <span className={s.diag.label === "NORMAL" ? "text-muted" : "font-semibold text-ink"}>{classLabel(s.diag.label)}</span> ({(s.diag.probability * 100).toFixed(0)}%)
            </>
          )}
        </div>
      </div>
      {s.reasons.length > 0 && (
        <ul className="ml-auto max-w-2xl space-y-0.5 text-xs">
          {s.reasons.map((r) => (
            <li key={r} className={s.status === "red" ? "text-bad" : "text-warn"}>
              ▸ {r}
            </li>
          ))}
        </ul>
      )}
      <div className="flex gap-2 sm:ml-auto">
        <Link className="rounded-md border border-line bg-panel2 px-3 py-1.5 text-xs hover:text-accent" to={`/diagnostics/${s.id}`}>
          Dyno cards
        </Link>
        <Link className="rounded-md border border-line bg-panel2 px-3 py-1.5 text-xs hover:text-accent" to={`/optimizer/${s.id}`}>
          Optimizer
        </Link>
      </div>
    </div>
  );
}
