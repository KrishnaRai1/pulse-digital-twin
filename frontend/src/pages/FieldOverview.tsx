import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import type { Status, WellSummary } from "../api/types";
import { useAlerts, useFieldTrend, useOverview } from "../api/hooks";
import { Chart } from "../components/Chart";
import { Empty, ErrorNote, KpiTile, Loading, Panel, STATUS_COLOR, StatusPill } from "../components/ui";
import { axisBase, gridBase, legendBase, timeAxisData, tooltipBase } from "../lib/chartHelpers";
import { COLORS } from "../lib/echarts";
import { PHASE_LABEL, classLabel, fmt, fmtClock, fmtDateTime, fmtInt, fmtVisc } from "../lib/format";
import { useLive } from "../lib/stream";

const PAD_BOX: Record<string, { x: number; w: number }> = { A: { x: 50, w: 430 }, B: { x: 590, w: 380 }, C: { x: 1080, w: 400 } };

function PadMap({ wells, onOpen }: { wells: WellSummary[]; onOpen: (id: string) => void }) {
  return (
    <svg viewBox="30 10 1470 380" className="w-full" role="group" aria-label="Field pad map">
      {Object.entries(PAD_BOX).map(([pad, b]) => (
        <g key={pad}>
          <rect x={b.x} y={30} width={b.w} height={270} rx={12} fill="#0b1928" stroke="#1a324b" strokeDasharray="5 5" />
          <text x={b.x + 14} y={52} fill="#8ea5bd" fontSize="15" fontWeight="600" letterSpacing="2">
            PAD {pad}
          </text>
        </g>
      ))}
      {/* gathering line to the group gathering station */}
      <path d="M50 340 H1480" stroke="#1a324b" strokeWidth="3" />
      <text x="50" y="368" fill="#5a738e" fontSize="13">
        Gathering line → Group Gathering Station
      </text>
      {wells.map((w) => {
        const idle = w.phase !== "PRODUCTION";
        const colour = idle && w.status === "green" ? STATUS_COLOR.idle : STATUS_COLOR[w.status];
        return (
          <g key={w.id} transform={`translate(${w.x_m + 40}, ${w.y_m + 30})`} className="cursor-pointer" onClick={() => onOpen(w.id)} role="link" tabIndex={0} onKeyDown={(e) => e.key === "Enter" && onOpen(w.id)} aria-label={`${w.id}, ${w.status}, ${PHASE_LABEL[w.phase] ?? w.phase}`}>
            <line x1={0} y1={26} x2={0} y2={340 - (w.y_m + 30)} stroke="#1a324b" strokeWidth={2} />
            {w.status === "red" && <circle r={34} fill="none" stroke={colour} strokeOpacity={0.35} strokeWidth={2} className="blink-red" />}
            <circle r={26} fill={colour} fillOpacity={0.16} stroke={colour} strokeWidth={2.5} />
            <text y={-2} textAnchor="middle" fill="#e5edf5" fontSize="16" fontWeight="700">
              {w.id.replace("BGW-", "")}
            </text>
            <text y={13} textAnchor="middle" fill="#8796a8" fontSize="11">
              {w.phase === "PRODUCTION" ? `${w.spm.toFixed(1)} SPM` : w.phase === "INJECTION" ? "INJ" : w.phase === "SOAK" ? "SOAK" : "END"}
            </text>
            <text y={46} textAnchor="middle" fill="#8796a8" fontSize="13">
              {w.phase === "PRODUCTION" ? `${w.oil_rate_m3d.toFixed(1)} m³/d` : `cycle ${w.cycle_no}`}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

type SortKey = "status" | "id" | "oil" | "spm" | "visc" | "sor";
const RANK: Record<Status, number> = { red: 0, amber: 1, green: 2 };

export function FieldOverview() {
  const { data, error, isLoading } = useOverview();
  const { data: trend } = useFieldTrend(24);
  const { data: alerts } = useAlerts();
  const live = useLive();
  const nav = useNavigate();
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: "status", dir: 1 });

  const wells = data?.wells ?? [];
  const otherEvents = live.events.filter((e) => e.kind !== "alert");
  const k = live.kpis ?? data?.kpis;
  const sorted = useMemo(() => {
    const val: Record<SortKey, (w: WellSummary) => number | string> = {
      status: (w) => RANK[w.status] * 10 + (w.phase === "PRODUCTION" ? 0 : 1),
      id: (w) => w.id,
      oil: (w) => w.oil_rate_m3d,
      spm: (w) => w.spm,
      visc: (w) => w.visc_cp,
      sor: (w) => w.sor ?? -1,
    };
    return [...wells].sort((a, b) => {
      const x = val[sort.key](a);
      const y = val[sort.key](b);
      return (x < y ? -1 : x > y ? 1 : 0) * sort.dir || a.id.localeCompare(b.id);
    });
  }, [wells, sort]);

  const trendOpt = useMemo(() => {
    if (!trend || !trend.ts.length) return null;
    const t = timeAxisData(trend.ts);
    return {
      tooltip: { ...tooltipBase, valueFormatter: (v: number) => v.toFixed(1) },
      legend: { ...legendBase },
      grid: gridBase({ right: 56, top: 44 }),
      xAxis: { type: "time", ...axisBase, axisLabel: { ...axisBase.axisLabel, hideOverlap: true } },
      yAxis: [
        { type: "value", name: "m³/d", ...axisBase, min: 0 },
        { type: "value", name: "kW", ...axisBase, splitLine: { show: false }, min: 0 },
      ],
      series: [
        { name: "Oil", type: "line", showSymbol: false, smooth: true, lineStyle: { width: 2 }, areaStyle: { opacity: 0.12 }, color: COLORS.accent, data: t.map((x, i) => [x, trend.oil_m3d[i]]) },
        { name: "Power", type: "line", showSymbol: false, smooth: true, yAxisIndex: 1, lineStyle: { width: 1.5 }, color: COLORS.warn, data: t.map((x, i) => [x, trend.power_kw[i]]) },
      ],
    };
  }, [trend]);

  const head = (key: SortKey, label: string, cls = "") => (
    <th scope="col" className={`px-3 py-2 text-left font-semibold ${cls}`} aria-sort={sort.key === key ? (sort.dir === 1 ? "ascending" : "descending") : "none"}>
      <button onClick={() => setSort((s) => ({ key, dir: s.key === key ? (-s.dir as 1 | -1) : key === "status" || key === "id" ? 1 : -1 }))} className="uppercase tracking-wide hover:text-ink">
        {label}
        {sort.key === key ? (sort.dir === 1 ? " ▲" : " ▼") : ""}
      </button>
    </th>
  );

  if (isLoading) return <Loading what="field overview" />;
  if (error) return <ErrorNote error={error} />;
  if (!k) return null;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <KpiTile label="Total production" value={fmt(k.total_oil_m3d, 1)} unit="m³/d oil" sub={`${fmtInt(k.total_oil_bbld)} bbl/d · liquid ${fmt(k.total_liquid_m3d, 0)} m³/d`} />
        <KpiTile label="Average SOR" value={fmt(k.avg_sor, 2)} unit="m³ steam / m³ oil" sub="cumulative, current cycles" tone={k.avg_sor && k.avg_sor > 5 ? "warn" : "neutral"} />
        <KpiTile label="Total lift power" value={fmt(k.total_power_kw, 1)} unit="kW" sub={`${k.pumping_wells} of ${k.wells} wells pumping`} />
        <KpiTile
          label="Well status"
          value={
            <span className="flex items-baseline gap-3">
              <span className="text-ok">{k.status_counts.green}</span>
              <span className="text-warn">{k.status_counts.amber}</span>
              <span className="text-bad">{k.status_counts.red}</span>
            </span>
          }
          sub="normal · warning · critical"
        />
        <KpiTile label="Active CNN alerts" value={k.active_alerts} tone={k.active_alerts ? "bad" : "ok"} sub={k.active_alerts ? "dynamometer faults confirmed on 2 cards" : "no confirmed faults"} />
      </div>

      <Panel title="Field map" subtitle="Traffic-light status per well: click a well to open its digital twin">
        <PadMap wells={wells} onOpen={(id) => nav(`/wells/${id}`)} />
        <div className="mt-1 flex flex-wrap gap-4 text-xs text-muted">
          {(["green", "amber", "red"] as Status[]).map((s) => (
            <span key={s} className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: STATUS_COLOR[s] }} />
              {s === "green" ? "Normal" : s === "amber" ? "Warning (act soon)" : "Critical (act now)"}
            </span>
          ))}
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: STATUS_COLOR.idle }} />
            Injecting / soaking
          </span>
        </div>
      </Panel>

      <div className="grid gap-4 xl:grid-cols-3">
        <Panel title="Field production, last 24 h" subtitle="Oil rate and total lift power" className="xl:col-span-2">
          {trendOpt ? <Chart option={trendOpt} height={250} ariaLabel="Field production trend" /> : <Loading what="trend" />}
        </Panel>
        <Panel title="Alerts & events" subtitle="Confirmed on two consecutive cards (debounced)">
          <ul className="max-h-[250px] space-y-1.5 overflow-y-auto pr-1 text-xs">
            {otherEvents.length === 0 && (alerts?.history.length ?? 0) === 0 && <li className="text-faint">No alerts yet.</li>}
            {otherEvents.slice(0, 5).map((e) => (
              <li key={e.key} className="flex gap-2">
                <span className="num shrink-0 text-faint">{fmtClock(e.ts)}</span>
                <span className="text-muted">{e.text}</span>
              </li>
            ))}
            {alerts?.history.slice(0, 12).map((a) => (
              <li key={`h${a.id}`} className="flex gap-2">
                <span className="num shrink-0 text-faint">{fmtDateTime(a.ts)}</span>
                <span className={a.severity === "red" ? "text-bad" : "text-warn"}>{a.message}</span>
              </li>
            ))}
          </ul>
        </Panel>
      </div>

      <Panel title="Wells" subtitle="Sorted by urgency: reasons come from the CNN, the physics limits and the thermal state" pad={false}>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[980px] text-[13px]">
            <thead className="border-b border-line text-[11px] text-muted">
              <tr>
                {head("status", "Status")}
                {head("id", "Well")}
                <th className="px-3 py-2 text-left font-semibold uppercase tracking-wide">Phase</th>
                {head("oil", "Oil m³/d", "text-right")}
                {head("spm", "SPM", "text-right")}
                {head("visc", "Visc cP", "text-right")}
                {head("sor", "SOR", "text-right")}
                <th className="px-3 py-2 text-left font-semibold uppercase tracking-wide">CNN diagnosis</th>
                <th className="px-3 py-2 text-left font-semibold uppercase tracking-wide">Why</th>
              </tr>
            </thead>
            <tbody>
              {sorted.length === 0 && (
                <tr>
                  <td colSpan={9}>
                    <Empty>No wells reported yet.</Empty>
                  </td>
                </tr>
              )}
              {sorted.map((w) => (
                <tr key={w.id} className="cursor-pointer border-b border-line/60 hover:bg-panel2" onClick={() => nav(`/wells/${w.id}`)}>
                  <td className="px-3 py-2">
                    <StatusPill status={w.status === "green" && w.phase !== "PRODUCTION" ? "idle" : w.status} label={w.status === "green" && w.phase !== "PRODUCTION" ? PHASE_LABEL[w.phase] : undefined} />
                  </td>
                  <td className="px-3 py-2 font-semibold text-ink">
                    {w.id} <span className="ml-1 text-[11px] font-normal text-faint">pad {w.pad}</span>
                  </td>
                  <td className="px-3 py-2 text-muted">
                    {PHASE_LABEL[w.phase]} <span className="text-faint">· cycle {w.cycle_no}, day {fmtInt(w.day_in_cycle)}</span>
                  </td>
                  <td className="num px-3 py-2 text-right">{w.producing ? fmt(w.oil_rate_m3d, 1) : "–"}</td>
                  <td className="num px-3 py-2 text-right">{w.producing ? fmt(w.spm, 1) : "–"}</td>
                  <td className="num px-3 py-2 text-right">{fmtVisc(w.visc_cp)}</td>
                  <td className="num px-3 py-2 text-right">{fmt(w.sor, 2)}</td>
                  <td className="px-3 py-2">{w.diag ? <span className={w.diag.label === "NORMAL" ? "text-muted" : "font-semibold text-ink"}>{classLabel(w.diag.label)} <span className="num text-faint">{(w.diag.probability * 100).toFixed(0)}%</span></span> : <span className="text-faint">–</span>}</td>
                  <td className="max-w-[360px] truncate px-3 py-2 text-muted" title={w.reasons.join(" · ")}>
                    {w.reasons[0] ?? ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
