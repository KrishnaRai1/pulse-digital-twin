import { useNavigate } from "react-router-dom";
import { useOverview } from "../api/hooks";
import { PHASE_LABEL } from "../lib/format";
import { STATUS_COLOR } from "./ui";

/** Compact well selector: status-coloured chips (all wells visible at a glance). */
export function WellPicker({ value, onChange, basePath }: { value: string; onChange?: (id: string) => void; basePath?: string }) {
  const { data } = useOverview();
  const nav = useNavigate();
  const wells = [...(data?.wells ?? [])].sort((a, b) => a.id.localeCompare(b.id));
  return (
    <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Select well">
      {wells.map((w) => {
        const active = w.id === value;
        return (
          <button
            key={w.id}
            role="tab"
            aria-selected={active}
            title={`${w.name}: ${PHASE_LABEL[w.phase] ?? w.phase}`}
            onClick={() => (onChange ? onChange(w.id) : nav(`${basePath}/${w.id}`))}
            className={`flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs font-medium transition ${active ? "border-accent bg-accent/12 text-ink" : "border-line bg-panel text-muted hover:text-ink"}`}
          >
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: w.phase === "PRODUCTION" || w.status !== "green" ? STATUS_COLOR[w.status] : STATUS_COLOR.idle }} />
            {w.id}
          </button>
        );
      })}
    </div>
  );
}
