import type { ReactNode } from "react";
import type { Status } from "../api/types";

export const STATUS_COLOR: Record<Status | "idle", string> = { green: "#3fb56b", amber: "#e3a72f", red: "#e5484d", idle: "#6b7787" };

export function Panel({
  title,
  subtitle,
  actions,
  children,
  className = "",
  pad = true,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  pad?: boolean;
}) {
  return (
    <section className={`rounded-lg border border-line bg-panel ${className}`}>
      {(title || actions) && (
        <header className="flex items-start justify-between gap-3 border-b border-line px-4 py-2.5">
          <div className="min-w-0">
            <h2 className="truncate text-[13px] font-semibold uppercase tracking-wide text-ink">{title}</h2>
            {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={pad ? "p-4" : ""}>{children}</div>
    </section>
  );
}

export function KpiTile({ label, value, unit, sub, tone = "neutral" }: { label: string; value: ReactNode; unit?: string; sub?: ReactNode; tone?: "neutral" | "ok" | "warn" | "bad" }) {
  const toneCls = { neutral: "text-ink", ok: "text-ok", warn: "text-warn", bad: "text-bad" }[tone];
  return (
    <div className="rounded-lg border border-line bg-panel px-4 py-3">
      <div className="text-[11px] font-medium uppercase tracking-wider text-muted">{label}</div>
      <div className="mt-1 flex items-baseline gap-1.5">
        <span className={`num text-[28px] font-semibold leading-none ${toneCls}`}>{value}</span>
        {unit && <span className="text-xs text-muted">{unit}</span>}
      </div>
      {sub && <div className="mt-1.5 text-xs text-muted">{sub}</div>}
    </div>
  );
}

export function StatusDot({ status, pulse = false, size = 10 }: { status: Status | "idle"; pulse?: boolean; size?: number }) {
  return (
    <span
      aria-hidden
      className={`inline-block shrink-0 rounded-full ${pulse && status === "red" ? "blink-red" : ""}`}
      style={{ width: size, height: size, background: STATUS_COLOR[status] }}
    />
  );
}

const PILL: Record<Status | "idle", string> = {
  green: "border-ok/40 bg-ok/10 text-ok",
  amber: "border-warn/40 bg-warn/10 text-warn",
  red: "border-bad/50 bg-bad/10 text-bad",
  idle: "border-line bg-panel2 text-muted",
};
const STATUS_TEXT: Record<Status, string> = { green: "Normal", amber: "Warning", red: "Critical" };

/** Status is always shown with text as well as colour (colour-blind safe). */
export function StatusPill({ status, label }: { status: Status | "idle"; label?: string }) {
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide ${PILL[status]}`}>
      <StatusDot status={status} size={7} />
      {label ?? (status === "idle" ? "Idle" : STATUS_TEXT[status])}
    </span>
  );
}

export function Loading({ what = "data" }: { what?: string }) {
  return (
    <div className="flex items-center gap-2 py-6 text-sm text-muted" role="status">
      <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-line border-t-accent" />
      Loading {what}…
    </div>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  const msg = error instanceof Error ? error.message : "Something went wrong";
  return (
    <div role="alert" className="rounded-md border border-bad/40 bg-bad/10 px-3 py-2 text-sm text-bad">
      {msg}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="py-8 text-center text-sm text-muted">{children}</div>;
}

export function Btn({
  children,
  onClick,
  disabled,
  variant = "default",
  type = "button",
  title,
  small,
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "default" | "primary" | "danger" | "ghost";
  type?: "button" | "submit";
  title?: string;
  small?: boolean;
}) {
  const v = {
    default: "border-line bg-panel2 text-ink hover:bg-[#222c38]",
    primary: "border-accent/60 bg-accent/15 text-accent hover:bg-accent/25",
    danger: "border-bad/50 bg-bad/10 text-bad hover:bg-bad/20",
    ghost: "border-transparent text-muted hover:bg-panel2 hover:text-ink",
  }[variant];
  return (
    <button
      type={type}
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={`rounded-md border font-medium transition disabled:cursor-not-allowed disabled:opacity-45 ${small ? "px-2 py-1 text-xs" : "px-3 py-1.5 text-[13px]"} ${v}`}
    >
      {children}
    </button>
  );
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-muted">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-[11px] text-faint">{hint}</span>}
    </label>
  );
}

export const selectCls = "rounded-md border border-line bg-bg px-2.5 py-1 text-[13px] text-ink focus:border-accent focus:outline-none";
export const inputCls = "w-full rounded-md border border-line bg-bg px-2.5 py-1.5 text-[13px] text-ink placeholder:text-faint focus:border-accent focus:outline-none";
