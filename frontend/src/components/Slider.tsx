interface Props {
  label: string;
  unit?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  digits?: number;
  hint?: string;
}

export function Slider({ label, unit, value, min, max, step, onChange, digits = 0, hint }: Props) {
  const id = `sl-${label.replace(/\W+/g, "-")}`;
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <label htmlFor={id} className="text-xs font-medium text-muted">
          {label}
        </label>
        <div className="flex items-baseline gap-1">
          <input
            type="number"
            aria-label={`${label} value`}
            className="num w-20 rounded border border-line bg-bg px-1.5 py-0.5 text-right text-[13px] text-ink focus:border-accent focus:outline-none"
            min={min}
            max={max}
            step={step}
            value={Number.isFinite(value) ? Number(value.toFixed(digits)) : ""}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (Number.isFinite(v)) onChange(Math.min(max, Math.max(min, v)));
            }}
          />
          {unit && <span className="w-9 text-xs text-muted">{unit}</span>}
        </div>
      </div>
      <input id={id} className="mt-1 w-full" type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      {hint && <p className="text-[11px] text-faint">{hint}</p>}
    </div>
  );
}
