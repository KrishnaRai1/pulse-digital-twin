import { memo, useEffect, useRef } from "react";
import { echarts, type EChartsCoreOption } from "../lib/echarts";

interface Props {
  option: EChartsCoreOption;
  height?: number | string;
  className?: string;
  ariaLabel?: string;
  onClick?: (params: unknown) => void;
  /** charts sharing a group get a synchronised crosshair and zoom */
  group?: string;
}

/** Minimal ECharts wrapper: init once, update via setOption, resize with the container. */
function ChartInner({ option, height = 260, className = "", ariaLabel, onClick, group }: Props) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<ReturnType<typeof echarts.init> | null>(null);
  const clickRef = useRef(onClick);
  clickRef.current = onClick;

  useEffect(() => {
    if (!el.current) return;
    const c = echarts.init(el.current, "pulse", { renderer: "canvas" });
    chart.current = c;
    c.on("click", (p) => clickRef.current?.(p));
    if (group) {
      c.group = group;
      echarts.connect(group);
    }
    const ro = new ResizeObserver(() => c.resize());
    ro.observe(el.current);
    return () => {
      ro.disconnect();
      c.dispose();
      chart.current = null;
    };
  }, []);

  useEffect(() => {
    chart.current?.setOption(option, { notMerge: true, lazyUpdate: true });
  }, [option]);

  return <div ref={el} role="img" aria-label={ariaLabel} className={className} style={{ height, width: "100%" }} />;
}

export const Chart = memo(ChartInner);
