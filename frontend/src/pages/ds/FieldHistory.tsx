import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useDsAlerts, useDsSnapshot, useDsSummary, useDsTrend, useDsWells, type DsWellRow, type DsWellState, type Rag } from "../../api/dataset";
import { Chart } from "../../components/Chart";
import { DayControl, RAG_COLOR, RagPill, SourceNote, fmtDay, useAsOfDay } from "../../components/dataset";
import { ErrorNote, KpiTile, Loading, Panel, selectCls } from "../../components/ui";
import { axisBase, gridBase, legendBase, tooltipBase } from "../../lib/chartHelpers";
import { COLORS } from "../../lib/echarts";
import { fmt, fmtInt } from "../../lib/format";

type MapMode = "status" | "oil" | "risk" | "rfi";

/** Axis maximum over several series, ignoring the first ``skip`` points, rounded up to 5,000. */
function niceMax(series: (number | null)[][], skip: number): number {
  let m = 0;
  for (const s of series) for (let i = skip; i < s.length; i++) m = Math.max(m, s[i] ?? 0);
  return Math.ceil(m / 5000) * 5000 || 5000;
}
const CELL = 30;

function ramp(v: number | null, lo: number, hi: number, colors: string[]): string {
  if (v === null || !Number.isFinite(v)) return RAG_COLOR.grey;
  const t = Math.max(0, Math.min(0.9999, (v - lo) / (hi - lo)));
  return colors[Math.floor(t * colors.length)];
}
const OIL_RAMP = ["#102235", "#163350", "#854d0e", "#b45309", "#f59e0b", "#fde047"];
const RISK_RAMP = ["#24423a", "#3fb56b", "#a8c23a", "#e3a72f", "#ef7a3a", "#e5484d"];
const RFI_RAMP = ["#21566a", "#2f8fa8", "#3fb56b", "#e3a72f", "#e5484d", "#ff7b8a"];

function wellColor(w: DsWellState, mode: MapMode, thr: number): string {
  if (mode === "status") return RAG_COLOR[w.status];
  if (w.phase !== "production") return w.phase === "injection" || w.phase === "soak" ? RAG_COLOR.blue : RAG_COLOR.grey;
  if (mode === "oil") return ramp(w.oil_bbl_d, 0, 120, OIL_RAMP);
  if (mode === "risk") return ramp(w.risk === null ? null : Math.log10(Math.max(w.risk, 1e-4)), -3.5, Math.log10(Math.max(thr * 3, 0.05)), RISK_RAMP);
  return ramp(w.rfi, 0.2, 1.4, RFI_RAMP);
}

function FieldMap({ wells, layout, mode, thr, onOpen }: { wells: DsWellState[]; layout: DsWellRow[]; mode: MapMode; thr: number; onOpen: (id: string) => void }) {
  const pos = useMemo(() => new Map(layout.map((w) => [w.well_id, w])), [layout]);
  const pads = useMemo(() => {
    const m = new Map<string, { x0: number; y0: number; x1: number; y1: number }>();
    for (const w of layout) {
      const b = m.get(w.pad) ?? { x0: 1e9, y0: 1e9, x1: -1e9, y1: -1e9 };
      m.set(w.pad, { x0: Math.min(b.x0, w.grid_x), y0: Math.min(b.y0, w.grid_y), x1: Math.max(b.x1, w.grid_x), y1: Math.max(b.y1, w.grid_y) });
    }
    return [...m.entries()];
  }, [layout]);
  const maxX = Math.max(...layout.map((w) => w.grid_x), 0);
  const maxY = Math.max(...layout.map((w) => w.grid_y), 0);
  return (
    <svg viewBox={`-14 -26 ${(maxX + 1) * CELL + 28} ${(maxY + 1) * CELL + 40}`} className="w-full" role="group" aria-label="Field map, 300 wells in 12 pads (schematic)">
      {pads.map(([pad, b]) => (
        <g key={pad}>
          <rect x={b.x0 * CELL - 10} y={b.y0 * CELL - 10} width={(b.x1 - b.x0 + 1) * CELL + 2} height={(b.y1 - b.y0 + 1) * CELL + 2} rx={8} fill="#0b1928" stroke="#1a324b" strokeDasharray="4 4" />
          <text x={b.x0 * CELL - 6} y={b.y0 * CELL - 14} fill="#5a738e" fontSize="10" fontWeight="600" letterSpacing="1.5">
            PAD {pad}
          </text>
        </g>
      ))}
      {wells.map((w) => {
        const p = pos.get(w.well_id);
        if (!p) return null;
        const c = wellColor(w, mode, thr);
        const alert = w.risk !== null && w.risk >= thr;
        return (
          <g
            key={w.well_id}
            transform={`translate(${p.grid_x * CELL + 5}, ${p.grid_y * CELL + 5})`}
            className="cursor-pointer"
            role="link"
            tabIndex={0}
            onClick={() => onOpen(w.well_id)}
            onKeyDown={(e) => e.key === "Enter" && onOpen(w.well_id)}
            aria-label={`${w.well_id}: ${w.reason}`}
          >
            <title>
              {`${w.well_id} · ${w.phase}${w.cycle ? ` · cycle ${w.cycle}` : ""}\n${w.reason}` +
                (w.phase === "production" ? `\noil ${fmt(w.oil_bbl_d, 1)} bbl/d · SPM ${fmt(w.spm, 1)} · RFI ${fmt(w.rfi, 2)} · risk ${fmt((w.risk ?? 0) * 100, 1)}%` : "")}
            </title>
            {alert && <circle r={13} fill="none" stroke={RAG_COLOR.red} strokeOpacity={0.6} strokeWidth={1.5} />}
            <circle r={9} fill={c} fillOpacity={w.phase === "stopped" ? 0.25 : 0.85} stroke={w.status === "red" ? RAG_COLOR.red : "#06111c"} strokeWidth={w.status === "red" ? 2 : 1} />
          </g>
        );
      })}
    </svg>
  );
}

const MODE_LEGEND: Record<MapMode, { label: string; items: [string, string][] }> = {
  status: { label: "Status", items: [["Normal", RAG_COLOR.green], ["Warning", RAG_COLOR.amber], ["Critical", RAG_COLOR.red], ["Injection / soak", RAG_COLOR.blue], ["Stopped", RAG_COLOR.grey]] },
  oil: { label: "Oil rate", items: [["0", OIL_RAMP[0]], ["40", OIL_RAMP[2]], ["80", OIL_RAMP[4]], ["120+ bbl/d", OIL_RAMP[5]]] },
  risk: { label: "14-day failure risk", items: [["<0.1%", RISK_RAMP[0]], ["1%", RISK_RAMP[2]], ["alert", RISK_RAMP[5]]] },
  rfi: { label: "Rod-floating index", items: [["0.2", RFI_RAMP[0]], ["0.8", RFI_RAMP[2]], ["1.0", RFI_RAMP[3]], ["1.4+", RFI_RAMP[5]]] },
};

type SortKey = "status" | "well_id" | "oil_bbl_d" | "spm" | "rfi" | "fillage_pct" | "risk" | "water_cut_pct";
const RANK: Record<Rag, number> = { red: 0, amber: 1, green: 2, blue: 3, grey: 4 };

export function FieldHistory() {
  const { day, setDay } = useAsOfDay();
  const { data: snap, error, isLoading } = useDsSnapshot(day);
  const { data: layout } = useDsWells();
  const { data: trend } = useDsTrend(1);
  const { data: summary } = useDsSummary();
  const { data: alerts } = useDsAlerts(day, 12);
  const nav = useNavigate();
  const [mode, setMode] = useState<MapMode>("status");
  const [filter, setFilter] = useState<"all" | "red" | "amber" | "producing">("all");
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: "status", dir: 1 });
  const open = (id: string) => nav(`/history/${id}?day=${day}`);

  const trendOpt = useMemo(() => {
    if (!trend) return null;
    return {
      tooltip: { ...tooltipBase },
      legend: { ...legendBase, data: ["Oil", "Water", "Steam injected", "Pump power", "Failures"] },
      grid: gridBase({ left: 58, right: 58, bottom: 58 }),
      dataZoom: [{ type: "inside" }, { type: "slider", height: 18, bottom: 6, borderColor: COLORS.line, textStyle: { color: COLORS.muted } }],
      xAxis: { type: "category", data: trend.day, ...axisBase, name: "field day", nameLocation: "middle", nameGap: 26, axisLabel: { ...axisBase.axisLabel, interval: 364 } },
      yAxis: [
        // cap the axis at the post-start-up maximum: day 0 has all 300 wells injecting at once
        { type: "value", name: "bbl/d · m³/d", max: niceMax([trend.oil_bbl_d, trend.water_bbl_d, trend.steam_m3], 60), ...axisBase },
        { type: "value", name: "kW", ...axisBase, splitLine: { show: false } },
        { type: "value", show: false, max: 14 },
      ],
      series: [
        {
          name: "Oil",
          type: "line",
          data: trend.oil_bbl_d,
          showSymbol: false,
          lineStyle: { width: 1.6, color: COLORS.accent },
          itemStyle: { color: COLORS.accent },
          areaStyle: { color: "#f59e0b1f" },
          markLine: { symbol: "none", silent: true, label: { formatter: fmtDay(day), color: COLORS.ink, fontSize: 11 }, lineStyle: { color: COLORS.warn, type: "solid", width: 1.5 }, data: [{ xAxis: String(day) }] },
        },
        { name: "Water", type: "line", data: trend.water_bbl_d, showSymbol: false, lineStyle: { width: 1, color: COLORS.blue, opacity: 0.6 }, itemStyle: { color: COLORS.blue } },
        { name: "Steam injected", type: "line", data: trend.steam_m3, showSymbol: false, lineStyle: { width: 1, color: COLORS.steam, opacity: 0.5 }, itemStyle: { color: COLORS.steam } },
        { name: "Pump power", type: "line", yAxisIndex: 1, data: trend.motor_kw, showSymbol: false, lineStyle: { width: 1.2, color: COLORS.violet }, itemStyle: { color: COLORS.violet } },
        { name: "Failures", type: "bar", yAxisIndex: 2, data: trend.failures.map((f) => (f ? f : null)), barWidth: 1.5, itemStyle: { color: COLORS.bad, opacity: 0.55 } },
      ],
    };
  }, [trend, day]);

  const rows = useMemo(() => {
    const ws = snap?.wells ?? [];
    const f = ws.filter((w) => {
      if (filter === "red" && w.status !== "red") return false;
      if (filter === "amber" && w.status !== "amber") return false;
      if (filter === "producing" && w.phase !== "production") return false;
      return !q || w.well_id.includes(q.toUpperCase());
    });
    const val = (w: DsWellState): number | string => (sort.key === "status" ? RANK[w.status] * 10 + (w.risk ? -w.risk : 0) : sort.key === "well_id" ? w.well_id : (w[sort.key] ?? -1));
    return [...f].sort((a, b) => {
      const x = val(a);
      const y = val(b);
      return (x < y ? -1 : x > y ? 1 : 0) * sort.dir || a.well_id.localeCompare(b.well_id);
    });
  }, [snap, filter, q, sort]);

  if (isLoading && !snap) return <Loading what="field history" />;
  if (error && !snap) return <ErrorNote error={error} />;
  if (!snap) return null;
  const k = snap.kpis;
  const thr = snap.thresholds.risk_alert;
  const th = (key: SortKey, label: string, right = true) => (
    <th className={`cursor-pointer select-none px-2 py-1.5 font-medium ${right ? "text-right" : "text-left"}`} onClick={() => setSort((s) => ({ key, dir: s.key === key ? ((-s.dir) as 1 | -1) : 1 }))}>
      {label}
      {sort.key === key ? (sort.dir === 1 ? " ▲" : " ▼") : ""}
    </th>
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold text-ink">Field history · Baghewala CSS</h1>
          <p className="text-xs text-muted">
            {summary ? `${summary.dataset.n_wells} wells · ${fmtInt(summary.dataset.n_cycles)} steam cycles · ${fmtInt(summary.dataset.n_rows)} well-days · ${summary.dataset.n_failures} rod/pump failures` : "…"}
          </p>
        </div>
      </div>
      <DayControl />

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <KpiTile label="Total oil" value={fmtInt(k.total_oil_bbl_d)} unit="bbl/d" sub={`${fmt(k.total_oil_m3_d, 0)} m³/d · WC ${fmt(k.field_water_cut_pct, 0)}%`} />
        <KpiTile label="Field SOR (cumulative)" value={fmt(k.cum_sor, 2)} unit="m³/m³" sub={`${fmtInt(k.cum_steam_m3)} m³ steam · ${fmtInt(k.cum_oil_m3)} m³ oil`} tone={k.cum_sor && k.cum_sor > 3 ? "warn" : "neutral"} />
        <KpiTile label="Pump power" value={fmtInt(k.total_power_kw)} unit="kW" sub={`mean ${fmt(k.mean_spm, 1)} SPM`} />
        <KpiTile label="Wells producing" value={k.n_producing} unit={`/ ${snap.wells.length}`} sub={`${k.n_injecting} inj · ${k.n_soaking} soak · ${k.n_workover} workover · ${k.n_stopped} stopped`} />
        <KpiTile label="Critical / warning" value={`${k.n_red} / ${k.n_amber}`} tone={k.n_red ? "bad" : k.n_amber ? "warn" : "ok"} sub="risk, rod float, pound, workover" />
        <KpiTile label="Failures to date" value={k.failures_to_date} sub={`steam today ${fmtInt(k.steam_today_m3)} m³`} />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_360px]">
        <Panel
          title={`Field map · ${fmtDay(snap.day)}`}
          subtitle="Schematic layout: the dataset has no coordinates, wells are grouped 25 per pad. Click a well for its history."
          actions={
            <div className="flex gap-1" role="tablist" aria-label="Colour wells by">
              {(Object.keys(MODE_LEGEND) as MapMode[]).map((m) => (
                <button key={m} role="tab" aria-selected={mode === m} onClick={() => setMode(m)} className={`rounded-md border px-2 py-0.5 text-[11px] ${mode === m ? "border-accent bg-accent/12 text-ink" : "border-line text-muted hover:text-ink"}`}>
                  {MODE_LEGEND[m].label}
                </button>
              ))}
            </div>
          }
        >
          {layout && <FieldMap wells={snap.wells} layout={layout} mode={mode} thr={thr} onOpen={open} />}
          <div className="mt-2 flex flex-wrap items-center gap-3 text-[11px] text-muted">
            {MODE_LEGEND[mode].items.map(([l, c]) => (
              <span key={l} className="flex items-center gap-1.5">
                <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: c }} /> {l}
              </span>
            ))}
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-3 w-3 rounded-full border border-bad" /> early-warning alert
            </span>
          </div>
        </Panel>

        <Panel title="Highest failure risk" subtitle={`Early-warning model, 14-day horizon · ${alerts?.n_alert ?? 0} alert, ${alerts?.n_watch ?? 0} watch`} pad={false}>
          <ul className="divide-y divide-line">
            {(alerts?.wells ?? []).map((a) => (
              <li key={a.well_id}>
                <button onClick={() => nav(`/warning?day=${day}&well=${a.well_id}`)} className="flex w-full items-center gap-3 px-4 py-2 text-left hover:bg-panel2">
                  <span className="num w-[72px] text-[13px] text-ink">{a.well_id}</span>
                  <span className="relative h-2 flex-1 overflow-hidden rounded bg-panel2">
                    <span className="absolute inset-y-0 left-0 rounded" style={{ width: `${Math.min(100, a.risk * 100)}%`, background: a.level === "alert" ? COLORS.bad : a.level === "watch" ? COLORS.warn : COLORS.ok }} />
                  </span>
                  <span className={`num w-12 text-right text-xs ${a.level === "alert" ? "text-bad" : a.level === "watch" ? "text-warn" : "text-muted"}`}>{fmt(a.risk * 100, 1)}%</span>
                </button>
              </li>
            ))}
          </ul>
          <p className="px-4 py-2 text-[11px] text-faint">Scores are out-of-fold: each well is scored by a model trained without it.</p>
        </Panel>
      </div>

      {trendOpt && (
        <Panel title="Field production history" subtitle="Daily totals over the whole record. Click the chart to move the as-of day.">
          <Chart option={trendOpt} height={300} ariaLabel="Field oil, water, steam and power over time" onClick={(p) => setDay(Number((p as { name?: string }).name ?? day))} />
          <SourceNote>Red bars mark days with rod or pump failures (count in the tooltip). Steam injected is cold-water-equivalent m³/d.</SourceNote>
        </Panel>
      )}

      <Panel
        title={`Wells · ${fmtDay(snap.day)}`}
        pad={false}
        actions={
          <div className="flex items-center gap-2">
            <select aria-label="Filter wells" value={filter} onChange={(e) => setFilter(e.target.value as typeof filter)} className={`${selectCls} w-36`}>
              <option value="all">All wells</option>
              <option value="red">Critical only</option>
              <option value="amber">Warning only</option>
              <option value="producing">Producing</option>
            </select>
            <input aria-label="Search well" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search…" className={`${selectCls} w-28`} />
          </div>
        }
      >
        <div className="max-h-[460px] overflow-auto">
          <table className="w-full text-[13px]">
            <thead className="sticky top-0 bg-panel text-xs text-muted">
              <tr className="border-b border-line">
                {th("well_id", "Well", false)}
                {th("status", "Status", false)}
                <th className="px-2 py-1.5 text-left font-medium">Phase</th>
                {th("oil_bbl_d", "Oil bbl/d")}
                {th("water_cut_pct", "WC %")}
                {th("spm", "SPM")}
                {th("rfi", "RFI")}
                {th("fillage_pct", "Fill %")}
                {th("risk", "Risk 14 d")}
              </tr>
            </thead>
            <tbody>
              {rows.map((w) => (
                <tr key={w.well_id} onClick={() => open(w.well_id)} className="cursor-pointer border-b border-line/60 hover:bg-panel2">
                  <td className="num px-2 py-1.5 text-ink">{w.well_id}</td>
                  <td className="px-2 py-1.5">
                    <div className="flex items-center gap-2">
                      <RagPill status={w.status} />
                      <span className="truncate text-xs text-muted">{w.status === "green" || w.status === "blue" ? "" : w.reason}</span>
                    </div>
                  </td>
                  <td className="px-2 py-1.5 text-muted">
                    {w.phase}
                    {w.cycle ? ` · c${w.cycle}` : ""}
                  </td>
                  <td className="num px-2 py-1.5 text-right">{w.phase === "production" ? fmt(w.oil_bbl_d, 1) : "–"}</td>
                  <td className="num px-2 py-1.5 text-right">{fmt(w.water_cut_pct, 0)}</td>
                  <td className="num px-2 py-1.5 text-right">{w.spm ? fmt(w.spm, 1) : "–"}</td>
                  <td className={`num px-2 py-1.5 text-right ${(w.rfi ?? 0) > 1 ? "text-bad" : (w.rfi ?? 0) > 0.9 ? "text-warn" : ""}`}>{fmt(w.rfi, 2)}</td>
                  <td className={`num px-2 py-1.5 text-right ${w.fillage_pct !== null && w.fillage_pct < 50 ? "text-warn" : ""}`}>{fmt(w.fillage_pct, 0)}</td>
                  <td className={`num px-2 py-1.5 text-right ${w.risk !== null && w.risk >= thr ? "text-bad" : ""}`}>{w.risk !== null ? `${fmt(w.risk * 100, 2)}%` : "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
