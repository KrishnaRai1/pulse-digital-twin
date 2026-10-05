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
  ink: "#d9e0e8",
  muted: "#8796a8",
  faint: "#5d6b7c",
  line: "#263140",
  grid: "#1e2733",
  accent: "#4cc3d9",
  ok: "#3fb56b",
  warn: "#e3a72f",
  bad: "#e5484d",
  idle: "#6b7787",
  violet: "#9d8cf0",
  orange: "#f08a4b",
  blue: "#5b9cf0",
  steam: "#c9d6e3",
} as const;

export const SERIES = [COLORS.accent, COLORS.warn, COLORS.violet, COLORS.ok, COLORS.orange, COLORS.blue];

echarts.registerTheme("pulse", {
  backgroundColor: "transparent",
  textStyle: { color: COLORS.muted, fontFamily: "ui-sans-serif, system-ui, sans-serif" },
  color: SERIES,
  animationDuration: 300,
  animationDurationUpdate: 250,
});

export { echarts };
export type { EChartsCoreOption } from "echarts/core";
