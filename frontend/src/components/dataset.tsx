// Shared building blocks of the field-dataset pages: readiness gate, as-of day, well selector.
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useDsStatus, useDsSummary, useDsWells, type Rag } from "../api/dataset";
import { ApiError } from "../api/client";
import { Btn, Panel } from "./ui";

export const RAG_COLOR: Record<Rag, string> = { green: "#3fb56b", amber: "#e3a72f", red: "#e5484d", blue: "#5b9cf0", grey: "#4a5666" };
export const RAG_TEXT: Record<Rag, string> = { green: "Normal", amber: "Warning", red: "Critical", blue: "Steaming", grey: "Idle" };
export const DEFAULT_DAY = 1200;

export function fmtDay(day: number | null | undefined): string {
  if (day === null || day === undefined) return "–";
  return `Day ${day.toLocaleString("en-US")}`;
}
export function fmtYears(day: number): string {
  return `year ${(day / 365.25 + 1).toFixed(1)}`;
}

export function RagPill({ status, label }: { status: Rag; label?: string }) {
  const c = RAG_COLOR[status];
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide" style={{ borderColor: `${c}66`, background: `${c}1a`, color: c }}>
      <span className="inline-block h-[7px] w-[7px] rounded-full" style={{ background: c }} />
      {label ?? RAG_TEXT[status]}
    </span>
  );
}

// ------------------------------------------------------------------ readiness gate
export function DatasetGate({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const { data, error } = useDsStatus(!ready);
  useEffect(() => {
    if (data?.ready) setReady(true);
  }, [data?.ready]);
  if (ready) return <>{children}</>;
  const unreachable = error instanceof ApiError && error.status === 0;
  const pct = Math.round((data?.progress.fraction ?? 0) * 100);
  return (
    <Panel title="Field dataset" subtitle="Baghewala synthetic dataset v1.4 · 300 wells">
      {unreachable ? (
        <p className="text-sm text-warn">{(error as Error).message}</p>
      ) : !data ? (
        <p className="text-sm text-muted">Contacting the API…</p>
      ) : data.state === "building" || data.state === "loading" ? (
        <div className="max-w-xl">
          <p className="text-sm text-ink">
            {data.state === "building" ? "Preparing the dataset: joining tables and training the models (first start only, about 1-3 minutes)." : "Loading the field store…"}
          </p>
          <div className="mt-3 h-2 overflow-hidden rounded bg-panel2" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
            <div className="h-full bg-accent transition-all" style={{ width: `${Math.max(pct, 3)}%` }} />
          </div>
          <p className="num mt-2 text-xs text-muted">
            {pct}% · {data.progress.step || "starting"}
            {data.elapsed_s ? ` · ${Math.round(data.elapsed_s)} s` : ""}
          </p>
        </div>
      ) : (
        <div className="max-w-2xl text-sm">
          <p className={data.state === "error" ? "text-bad" : "text-warn"}>{data.message}</p>
          <p className="mt-2 text-muted">
            Import the dataset with <code className="num text-ink">python -m scripts.import_dataset --src &lt;folder with the CSV files&gt;</code> and build it with{" "}
            <code className="num text-ink">python -m scripts.build_field_store</code> (see docs/DATASET.md).
          </p>
        </div>
      )}
    </Panel>
  );
}

// ------------------------------------------------------------------ as-of day (shared across pages, mirrored in ?day=)
interface DayCtx {
  day: number;
  setDay: (d: number) => void;
  maxDay: number;
}
const Ctx = createContext<DayCtx>({ day: DEFAULT_DAY, setDay: () => {}, maxDay: 2851 });

export function AsOfDayProvider({ children }: { children: ReactNode }) {
  const [params, setParams] = useSearchParams();
  const { data: summary } = useDsSummary();
  const maxDay = summary?.max_day ?? 2851;
  const initial = Number(params.get("day"));
  const [day, setDayState] = useState<number>(Number.isFinite(initial) && params.get("day") ? initial : DEFAULT_DAY);
  const setDay = useCallback(
    (d: number) => {
      const v = Math.max(0, Math.min(maxDay, Math.round(d)));
      setDayState(v);
      setParams(
        (p) => {
          const n = new URLSearchParams(p);
          n.set("day", String(v));
          return n;
        },
        { replace: true },
      );
    },
    [maxDay, setParams],
  );
  const value = useMemo(() => ({ day: Math.min(day, maxDay), setDay, maxDay }), [day, setDay, maxDay]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const useAsOfDay = () => useContext(Ctx);

export function DayControl({ compact = false }: { compact?: boolean }) {
  const { day, setDay, maxDay } = useAsOfDay();
  const [playing, setPlaying] = useState(false);
  const dayRef = useRef(day);
  dayRef.current = day;
  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => {
      const next = dayRef.current + 7;
      if (next > maxDay) setPlaying(false);
      setDay(Math.min(next, maxDay));
    }, 450);
    return () => clearInterval(t);
  }, [playing, maxDay, setDay]);
  return (
    <div className={`flex flex-wrap items-center gap-2 ${compact ? "" : "rounded-lg border border-line bg-panel px-3 py-2"}`}>
      <span className="text-[11px] font-semibold uppercase tracking-wider text-muted">As of</span>
      <Btn small variant={playing ? "primary" : "default"} onClick={() => setPlaying((p) => !p)} title={playing ? "Pause replay" : "Replay the field history (7 days per step)"}>
        {playing ? "❚❚ Pause" : "▶ Replay"}
      </Btn>
      <Btn small variant="ghost" onClick={() => setDay(day - 30)} title="30 days back">
        −30 d
      </Btn>
      <Btn small variant="ghost" onClick={() => setDay(day - 1)} title="One day back">
        −1
      </Btn>
      <input
        aria-label="As-of day"
        type="range"
        min={0}
        max={maxDay}
        value={day}
        onChange={(e) => setDay(Number(e.target.value))}
        className="min-w-[140px] flex-1"
      />
      <Btn small variant="ghost" onClick={() => setDay(day + 1)} title="One day forward">
        +1
      </Btn>
      <Btn small variant="ghost" onClick={() => setDay(day + 30)} title="30 days forward">
        +30 d
      </Btn>
      <input
        aria-label="Day number"
        type="number"
        min={0}
        max={maxDay}
        value={day}
        onChange={(e) => Number.isFinite(Number(e.target.value)) && setDay(Number(e.target.value))}
        className="num w-20 rounded border border-line bg-bg px-1.5 py-0.5 text-right text-[13px] text-ink focus:border-accent focus:outline-none"
      />
      <span className="num text-xs text-muted">
        / {maxDay.toLocaleString("en-US")} · {fmtYears(day)}
      </span>
    </div>
  );
}

// ------------------------------------------------------------------ well selector (300 wells)
export function WellSelect({ value, basePath, keepDay = true }: { value: string; basePath: string; keepDay?: boolean }) {
  const { data: wells } = useDsWells();
  const nav = useNavigate();
  const { day } = useAsOfDay();
  const ids = useMemo(() => (wells ?? []).map((w) => w.well_id), [wells]);
  const [text, setText] = useState(value);
  useEffect(() => setText(value), [value]);
  const go = (id: string) => nav(`${basePath}/${id}${keepDay ? `?day=${day}` : ""}`);
  const idx = ids.indexOf(value);
  const submit = (t: string) => {
    const s = t.trim().toUpperCase();
    const hit = ids.find((w) => w === s) ?? ids.find((w) => w.endsWith(s.padStart(3, "0"))) ?? ids.find((w) => w.includes(s));
    if (hit) go(hit);
  };
  return (
    <div className="flex items-center gap-1.5">
      <Btn small variant="ghost" disabled={idx <= 0} onClick={() => go(ids[idx - 1])} title="Previous well">
        ‹
      </Btn>
      <input
        list="ds-well-list"
        aria-label="Well"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && submit(text)}
        onBlur={() => text !== value && submit(text)}
        className="num w-32 rounded-md border border-line bg-bg px-2 py-1 text-[13px] text-ink focus:border-accent focus:outline-none"
        placeholder="WELL-001"
      />
      <datalist id="ds-well-list">
        {ids.map((w) => (
          <option key={w} value={w} />
        ))}
      </datalist>
      <Btn small variant="ghost" disabled={idx < 0 || idx >= ids.length - 1} onClick={() => go(ids[idx + 1])} title="Next well">
        ›
      </Btn>
    </div>
  );
}

export function SourceNote({ children }: { children: ReactNode }) {
  return <p className="mt-2 text-[11px] leading-relaxed text-faint">{children}</p>;
}
