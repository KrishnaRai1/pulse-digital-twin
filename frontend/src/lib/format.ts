export const M3_TO_BBL = 6.2898;

export function fmt(v: number | null | undefined, digits = 1, unit = ""): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  const s = v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  return unit ? `${s} ${unit}` : s;
}

export function fmtInt(v: number | null | undefined): string {
  return fmt(v, 0);
}

/** Viscosity spans decades: switch to thousands separators / k notation for readability. */
export function fmtVisc(cp: number | null | undefined): string {
  if (cp === null || cp === undefined || !Number.isFinite(cp)) return "–";
  if (cp >= 10000) return `${(cp / 1000).toFixed(1)}k`;
  if (cp >= 100) return Math.round(cp).toLocaleString("en-US");
  return cp.toFixed(1);
}

export function fmtPct(fraction: number | null | undefined, digits = 0): string {
  if (fraction === null || fraction === undefined || !Number.isFinite(fraction)) return "–";
  return `${(fraction * 100).toFixed(digits)}%`;
}

export function fmtUsd(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  const sign = v < 0 ? "-" : "";
  const a = Math.abs(v);
  if (a >= 1e6) return `${sign}$${(a / 1e6).toFixed(2)}M`;
  if (a >= 1e4) return `${sign}$${(a / 1e3).toFixed(1)}k`;
  return `${sign}$${Math.round(a).toLocaleString("en-US")}`;
}

export function fmtClock(epochSeconds: number | null | undefined): string {
  if (!epochSeconds) return "–";
  const d = new Date(epochSeconds * 1000);
  return d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function fmtDateTime(epochSeconds: number | null | undefined): string {
  if (!epochSeconds) return "–";
  const d = new Date(epochSeconds * 1000);
  return `${d.toLocaleDateString("en-GB", { day: "2-digit", month: "short" })} ${d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })}`;
}

export function fmtDate(epochSeconds: number | null | undefined): string {
  if (!epochSeconds) return "–";
  return new Date(epochSeconds * 1000).toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} kB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function titleCase(s: string): string {
  return s
    .toLowerCase()
    .split(/[_\s]+/)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

export const CLASS_LABEL: Record<string, string> = {
  NORMAL: "Normal",
  ROD_FLOATING: "Rod floating",
  PUMP_UNSETTING_RISK: "Pump unsetting risk",
  FLUID_POUND: "Fluid pound",
  GAS_INTERFERENCE: "Gas interference",
};

export function classLabel(c: string | null | undefined): string {
  if (!c) return "–";
  return CLASS_LABEL[c] ?? titleCase(c);
}

export const PHASE_LABEL: Record<string, string> = {
  INJECTION: "Steam injection",
  SOAK: "Soak",
  PRODUCTION: "Production",
  CYCLE_END: "Cycle complete",
};
