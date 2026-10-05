import { BarChart, GaugeChart, HeatmapChart, LineChart, ScatterChart } from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  MarkPointComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([
  LineChart,
  BarChart,
  ScatterChart,
  GaugeChart,
  HeatmapChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  MarkLineComponent,
  MarkAreaComponent,
  VisualMapComponent,
  DataZoomComponent,
  MarkPointComponent,
  CanvasRenderer,
]);

export const COLORS = {
  ink: "#e5edf5",
  muted: "#8ea5bd",
  faint: "#5a738e",
  line: "#1a324b",
  grid: "#102235",
  accent: "#f59e0b", // Oil India Signature Golden Amber
  ok: "#10b981",
  warn: "#f59e0b",
  bad: "#ef4444",
  idle: "#64748b",
  violet: "#a78bfa",
  orange: "#fb923c",
  blue: "#38bdf8",
  steam: "#cbd5e1",
} as const;

export const SERIES = [COLORS.accent, COLORS.blue, COLORS.ok, COLORS.orange, COLORS.violet, "#e5edf5"];

echarts.registerTheme("pulse", {
  backgroundColor: "transparent",
  textStyle: { color: COLORS.muted, fontFamily: "ui-sans-serif, system-ui, sans-serif" },
  color: SERIES,
  animationDuration: 300,
  animationDurationUpdate: 250,
});

export { echarts };
export type { EChartsCoreOption } from "echarts/core";
