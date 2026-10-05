import { useMemo } from "react";
import { COLORS } from "../lib/echarts";
import { Chart } from "./Chart";

interface Props {
  title: string;
  value: number | null;
  unit: string;
  min?: number;
  max: number;
  digits?: number;
  /** [fraction of max, colour] stops for the arc */
  bands?: [number, string][];
  /** optional tick showing an operating limit */
  limit?: number | null;
}

export function Gauge({ title, value, unit, min = 0, max, digits = 1, bands, limit }: Props) {
  const option = useMemo(() => {
    const stops: [number, string][] = bands ?? [
      [0.7, COLORS.ok],
      [0.9, COLORS.warn],
      [1, COLORS.bad],
    ];
    return {
      series: [
        {
          type: "gauge",
          min,
          max,
          startAngle: 205,
          endAngle: -25,
          radius: "84%",
          center: ["50%", "58%"],
          progress: { show: false },
          axisLine: { lineStyle: { width: 10, color: stops } },
          axisTick: { show: false },
          splitLine: { length: 8, distance: -10, lineStyle: { color: "#0e1318", width: 2 } },
          axisLabel: { show: false, color: COLORS.muted, fontSize: 10, distance: 14, formatter: (v: number) => (max >= 100 ? v.toFixed(0) : v.toFixed(max < 10 ? 1 : 0)) },
          splitNumber: 5,
          pointer: { width: 4, length: "58%", itemStyle: { color: COLORS.ink } },
          anchor: { show: true, size: 8, itemStyle: { color: COLORS.ink } },
          title: { show: false },
          detail: { valueAnimation: true, offsetCenter: [0, "62%"], fontSize: 18, fontFamily: "ui-monospace, monospace", color: COLORS.ink, formatter: (v: number) => `${v.toFixed(digits)} ${unit}` },
          data: [{ value: value ?? 0 }],
        },
      ],
    };
  }, [value, unit, min, max, digits, bands, limit]);

  return (
    <div className="text-center">
      <Chart option={option} height={150} ariaLabel={`${title} gauge`} />
      <div className="-mt-1 text-[11px] font-medium uppercase tracking-wider text-muted">{title}</div>
      {limit != null && <div className="num mt-0.5 text-[11px] text-faint">ceiling {limit.toFixed(digits)} {unit}</div>}
    </div>
  );
}
