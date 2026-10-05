import { COLORS } from "./echarts";

export const axisBase = {
  axisLine: { lineStyle: { color: COLORS.line } },
  axisTick: { lineStyle: { color: COLORS.line } },
  axisLabel: { color: COLORS.muted, fontSize: 11 },
  splitLine: { lineStyle: { color: COLORS.grid } },
  nameTextStyle: { color: COLORS.muted, fontSize: 11 },
};

export const tooltipBase = {
  trigger: "axis" as const,
  backgroundColor: "#06111cee",
  borderColor: COLORS.line,
  textStyle: { color: COLORS.ink, fontSize: 12 },
  confine: true,
};

export const legendBase = { top: 0, left: "center", textStyle: { color: COLORS.muted, fontSize: 11 }, icon: "roundRect", itemWidth: 12, itemHeight: 4, itemGap: 14 };

export function gridBase(extra: Partial<{ left: number; right: number; top: number; bottom: number }> = {}) {
  return { left: 52, right: 16, top: 44, bottom: 34, containLabel: false, ...extra };
}

/** Close a periodic card so the polygon is drawn as a loop. */
export function loop<T>(a: T[]): T[] {
  return a.length ? [...a, a[0]] : a;
}

export function zip(x: number[], y: number[]): [number, number][] {
  return x.map((v, i) => [v, y[i]]);
}

export function timeAxisData(ts: (number | null)[]): number[] {
  return ts.map((t) => (t ?? 0) * 1000);
}
