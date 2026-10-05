import { useEffect, useMemo, useRef, useState } from "react";
import { Link, Navigate, useParams } from "react-router-dom";
import { api, sampleUrl } from "../api/client";
import { useCardHistory, useJob, useLatestCard, useOverview } from "../api/hooks";
import type { CardIngestReport, CardUploadResult, DynoCard, DynoClass } from "../api/types";
import { Chart } from "../components/Chart";
import { WellPicker } from "../components/WellPicker";
import { Btn, Empty, ErrorNote, Loading, Panel, StatusDot, inputCls } from "../components/ui";
import { axisBase, gridBase, legendBase, loop, tooltipBase, zip } from "../lib/chartHelpers";
import { COLORS } from "../lib/echarts";
import { PHASE_LABEL, classLabel, fmt, fmtClock, fmtDateTime } from "../lib/format";
import { useCardSubscription, useLive } from "../lib/stream";

const ADVICE: Record<Exclude<DynoClass, "NORMAL">, { text: string; sev: "red" | "amber" }> = {
  ROD_FLOATING: { sev: "red", text: "Rods are not falling freely against the viscous oil. Slow the unit (see Optimizer) or re-heat the well." },
  PUMP_UNSETTING_RISK: { sev: "red", text: "Severe float with compressive rod loads: reduce speed now to protect the hold-down and the pump seating." },
  FLUID_POUND: { sev: "amber", text: "The pump is not filling on the upstroke. Slow down or shorten the stroke to match the inflow." },
  GAS_INTERFERENCE: { sev: "amber", text: "Gas is compressing in the barrel and cutting fillage. Consider gas separation or a lower speed." },
};
const CLASSES: DynoClass[] = ["NORMAL", "ROD_FLOATING", "PUMP_UNSETTING_RISK", "FLUID_POUND", "GAS_INTERFERENCE"];

function CardChart({ title, x, y, bx, by, refLine, color }: { title: string; x: number[]; y: number[]; bx?: number[]; by?: number[]; refLine?: { value: number; label: string }; color: string }) {
  const opt = useMemo(
    () => ({
      animation: false,
      tooltip: { ...tooltipBase, trigger: "item", valueFormatter: (v: number) => v.toFixed(2) },
      legend: { ...legendBase },
      grid: gridBase({ top: 30, left: 56, bottom: 38 }),
      xAxis: { type: "value", name: "Position, m", nameLocation: "middle", nameGap: 24, ...axisBase, scale: true },
      yAxis: { type: "value", name: "Load, kN", ...axisBase, scale: true },
      series: [
        ...(bx && by
          ? [{ name: "Healthy baseline", type: "line", showSymbol: false, lineStyle: { width: 1.5, type: "dashed" }, color: COLORS.faint, z: 1, data: zip(loop(bx), loop(by)) }]
          : []),
        {
          name: "Measured",
          type: "line",
          showSymbol: false,
          lineStyle: { width: 2.6 },
          color,
          z: 3,
          data: zip(loop(x), loop(y)),
          markLine: refLine ? { symbol: "none", silent: true, lineStyle: { color: COLORS.muted, type: "dotted" }, label: { formatter: refLine.label, color: COLORS.muted, fontSize: 10, position: "insideEndTop" }, data: [{ yAxis: refLine.value }] } : undefined,
        },
      ],
    }),
    [x, y, bx, by, refLine, color],
  );
  return (
    <div>
      <h3 className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-muted">{title}</h3>
      <Chart option={opt} height={300} ariaLabel={title} />
    </div>
  );
}

function UploadPanel({ wellIds }: { wellIds: string[] }) {
  const [file, setFile] = useState<File | null>(null);
  const [posUnit, setPosUnit] = useState("m");
  const [loadUnit, setLoadUnit] = useState("kN");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  const [report, setReport] = useState<CardIngestReport | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob<CardUploadResult>(jobId);
  const input = useRef<HTMLInputElement>(null);

  const submit = async (f: File) => {
    setBusy(true);
    setErr(null);
    setReport(null);
    setJobId(null);
    try {
      const r = await api.upload<{ job_id: string; ingest: CardIngestReport }>("/ingest/dynamometer", f, { position_unit: posUnit, load_unit: loadUnit });
      setReport(r.ingest);
      setJobId(r.job_id);
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
    }
  };
  const useSample = async () => {
    setErr(null);
    try {
      const res = await fetch(sampleUrl("sample_dyno_cards.csv"));
      if (!res.ok) throw new Error("sample file not available");
      const f = new File([await res.blob()], "sample_dyno_cards.csv", { type: "text/csv" });
      setPosUnit("m");
      setLoadUnit("kN");
      setFile(f);
      await submit(f);
    } catch (e) {
      setErr(e);
    }
  };

  return (
    <Panel title="Feed dynamometer cards" subtitle="Upload a CSV/Excel file: the console classifies every card and replays them live above" >
      <div className="grid gap-3 sm:grid-cols-[1fr_auto_auto]">
        <input
          ref={input}
          type="file"
          accept=".csv,.xlsx"
          aria-label="Dynamometer card file"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="block w-full text-xs text-muted file:mr-3 file:rounded-md file:border file:border-line file:bg-panel2 file:px-3 file:py-1.5 file:text-xs file:text-ink"
        />
        <select aria-label="Position unit" className={`${inputCls} !w-auto`} value={posUnit} onChange={(e) => setPosUnit(e.target.value)}>
          {["m", "in", "mm", "ft"].map((u) => (
            <option key={u} value={u}>
              position: {u}
            </option>
          ))}
        </select>
        <select aria-label="Load unit" className={`${inputCls} !w-auto`} value={loadUnit} onChange={(e) => setLoadUnit(e.target.value)}>
          {["N", "kN", "lbf", "klbf"].map((u) => (
            <option key={u} value={u}>
              load: {u}
            </option>
          ))}
        </select>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Btn variant="primary" disabled={!file || busy} onClick={() => file && submit(file)}>
          {busy ? "Uploading…" : "Upload & classify"}
        </Btn>
        <Btn onClick={useSample} disabled={busy}>
          Try the bundled sample (15 cards, 5 fault classes)
        </Btn>
        <span className="text-[11px] text-faint">
          Expected columns: well_id, position, load (JSON arrays per row, or one point per row with card_id), optional spm/timestamp. Wells: {wellIds.slice(0, 3).join(", ")}…
        </span>
      </div>
      {err != null && <div className="mt-3"><ErrorNote error={err} /></div>}
      {report && (
        <div className="mt-3 rounded-md border border-line bg-bg/50 p-3 text-xs text-muted">
          <b className="text-ink">{report.cards_ok}</b> of {report.cards_in} cards accepted ({report.format} format), {report.outliers_fixed} spikes repaired
          {report.cards_rejected > 0 && (
            <ul className="mt-1 list-disc pl-4 text-warn">
              {report.rejected.slice(0, 5).map((r) => (
                <li key={r.card}>
                  {r.card}: {r.reason}
                </li>
              ))}
            </ul>
          )}
          {job.data?.status === "done" && job.data.result && (
            <div className="mt-2 flex flex-wrap gap-2">
              {Object.entries(job.data.result.label_counts).map(([k, v]) => (
                <span key={k} className="rounded-full border border-line px-2 py-0.5 text-ink">
                  {classLabel(k)}: <b>{v}</b>
                </span>
              ))}
            </div>
          )}
          {jobId && job.data && job.data.status !== "done" && <p className="mt-2">Job {job.data.status}…</p>}
          {job.data?.status === "failed" && <p className="mt-2 text-bad">{job.data.error}</p>}
        </div>
      )}
    </Panel>
  );
}

export function Diagnostics() {
  const { wellId } = useParams();
  const { data: ov } = useOverview();
  const ids = useMemo(() => (ov?.wells ?? []).map((w) => w.id).sort(), [ov]);
  const live = useLive();
  useCardSubscription(ids);
  const selected = (wellId ?? "").toUpperCase();

  // "follow the upload": show uploaded cards as they replay, otherwise the selected well's live card
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  const replaying = live.lastCard && live.lastCard.card.source === "upload" && now - live.lastCard.at < 6000 ? live.lastCard.card : null;
  const showWell = replaying?.well_id ?? selected;

  const { data: rest, error } = useLatestCard(showWell || "BGW-01");
  const { data: hist } = useCardHistory(showWell || "BGW-01");
  const card: DynoCard | null = replaying ?? (live.cards[showWell] as DynoCard | undefined) ?? rest?.card ?? null;
  const summary = ov?.wells.find((w) => w.id === showWell);

  if (!selected) return <Navigate to={`/diagnostics/${ids.find((i) => ov?.wells.find((w) => w.id === i)?.producing) ?? "BGW-01"}`} replace />;

  const fault = card && card.label !== "NORMAL" && card.probability >= 0.7 ? ADVICE[card.label as Exclude<DynoClass, "NORMAL">] : null;

  return (
    <div className="space-y-4">
      <WellPicker value={selected} basePath="/diagnostics" />

      {fault && card && (
        <div role="alert" className={`flex flex-wrap items-center gap-3 rounded-lg border px-4 py-3 ${fault.sev === "red" ? "border-bad/60 bg-bad/12" : "border-warn/60 bg-warn/10"}`}>
          <StatusDot status={fault.sev === "red" ? "red" : "amber"} pulse size={14} />
          <div className="min-w-0 flex-1">
            <div className={`text-[15px] font-semibold ${fault.sev === "red" ? "text-bad" : "text-warn"}`}>
              {classLabel(card.label)} on {card.well_id} <span className="num text-sm font-normal opacity-80">{(card.probability * 100).toFixed(0)}% CNN confidence</span>
            </div>
            <div className="text-[13px] text-ink/90">{fault.text}</div>
          </div>
          <Link to={`/optimizer/${card.well_id}`} className="rounded-md border border-line bg-panel2 px-3 py-1.5 text-xs font-medium text-ink hover:text-accent">
            Open recommendation →
          </Link>
        </div>
      )}

      {replaying && <div className="rounded-md border border-accent/40 bg-accent/10 px-3 py-1.5 text-xs text-accent">Replaying uploaded cards: showing {replaying.well_id} at {fmtClock(replaying.ts)}</div>}

      {error && <ErrorNote error={error} />}
      {summary && !summary.producing && !replaying && (
        <Empty>
          {summary.id} is in {PHASE_LABEL[summary.phase]?.toLowerCase()}: no dynamometer cards until pumping resumes.
        </Empty>
      )}

      {!card && (!summary || summary.producing) && !error && <Loading what="dynamometer card" />}

      {card && (
        <div className="grid gap-4 xl:grid-cols-3">
          <Panel title={`Surface & downhole cards · ${card.well_id}`} subtitle={`${card.source === "upload" ? "Uploaded card" : "Live"} · ${fmt(card.spm, 2)} SPM · ${fmtDateTime(card.ts)}`} className="xl:col-span-2">
            <div className="grid gap-4 md:grid-cols-2">
              <CardChart title="Surface card (polished rod)" x={card.position} y={card.load} bx={rest?.baseline?.position} by={rest?.baseline?.load} refLine={{ value: card.buoyant_rod_weight_kn, label: "buoyant rod weight" }} color={COLORS.accent} />
              <CardChart title="Downhole card (pump, wave equation)" x={card.dh_position} y={card.dh_load} bx={rest?.baseline?.dh_position} by={rest?.baseline?.dh_load} refLine={{ value: 0, label: "zero load" }} color={COLORS.orange} />
            </div>
            <p className="mt-2 text-[11px] text-faint">Dashed grey = model-expected healthy card for the same speed, viscosity and lift. The downhole card is computed from the surface card by solving the damped wave equation along the tapered rod string.</p>
          </Panel>

          <Panel title="CNN diagnosis" subtitle={`${card.latency_ms_total ?? card.latency_ms} ms inference · budget 200 ms`}>
            <div className="mb-3 flex items-center gap-2">
              <StatusDot status={card.label === "NORMAL" ? "green" : (ADVICE[card.label as Exclude<DynoClass, "NORMAL">]?.sev ?? "amber")} size={12} />
              <span className="text-lg font-semibold text-ink">{classLabel(card.label)}</span>
              <span className="num text-sm text-muted">{(card.probability * 100).toFixed(1)}%</span>
            </div>
            <ul className="space-y-2">
              {CLASSES.map((c) => {
                const p = card.probabilities[c] ?? 0;
                return (
                  <li key={c} className="text-xs">
                    <div className="mb-0.5 flex justify-between">
                      <span className={c === card.label ? "font-semibold text-ink" : "text-muted"}>{classLabel(c)}</span>
                      <span className="num text-muted">{(p * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-2 rounded bg-panel2" role="progressbar" aria-valuenow={Math.round(p * 100)} aria-valuemin={0} aria-valuemax={100} aria-label={classLabel(c)}>
                      <div className="h-full rounded" style={{ width: `${p * 100}%`, background: c === "NORMAL" ? COLORS.ok : c === card.label ? COLORS.bad : COLORS.faint }} />
                    </div>
                  </li>
                );
              })}
            </ul>
            {summary?.float_index != null && (
              <p className="mt-4 rounded-md bg-bg/60 px-3 py-2 text-xs text-muted">
                Rod float index <b className="num text-ink">{summary.float_index.toFixed(2)}</b> (limit 0.85) · speed ceiling <b className="num text-ink">{fmt(summary.spm_ceiling, 2)} SPM</b> at {fmt(summary.visc_cp, 0)} cP.
              </p>
            )}
          </Panel>
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Recorded fault cards" subtitle="Confirmed non-normal live cards and uploaded cards" pad={false}>
          {!hist || hist.length === 0 ? (
            <Empty>No stored cards for {showWell} yet.</Empty>
          ) : (
            <table className="w-full text-xs">
              <thead className="border-b border-line text-[11px] uppercase tracking-wide text-muted">
                <tr>
                  <th className="px-3 py-2 text-left">Time</th>
                  <th className="px-3 py-2 text-left">Source</th>
                  <th className="px-3 py-2 text-right">SPM</th>
                  <th className="px-3 py-2 text-left">Class</th>
                  <th className="px-3 py-2 text-right">Conf.</th>
                </tr>
              </thead>
              <tbody>
                {hist.map((h) => (
                  <tr key={h.id} className="border-b border-line/50">
                    <td className="num px-3 py-1.5 text-muted">{fmtDateTime(h.ts)}</td>
                    <td className="px-3 py-1.5 text-muted">{h.source}</td>
                    <td className="num px-3 py-1.5 text-right">{fmt(h.spm, 1)}</td>
                    <td className="px-3 py-1.5">{classLabel(h.label)}</td>
                    <td className="num px-3 py-1.5 text-right">{((h.probability ?? 0) * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>
        <UploadPanel wellIds={ids} />
      </div>
    </div>
  );
}
