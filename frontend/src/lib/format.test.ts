import { describe, expect, it } from "vitest";
import { classLabel, fmt, fmtBytes, fmtPct, fmtUsd, fmtVisc } from "./format";

describe("format helpers", () => {
  it("formats numbers and handles missing values", () => {
    expect(fmt(1234.567, 1)).toBe("1,234.6");
    expect(fmt(null)).toBe("–");
    expect(fmt(Number.NaN)).toBe("–");
    expect(fmt(3.14159, 2, "kW")).toBe("3.14 kW");
  });
  it("formats viscosity across decades", () => {
    expect(fmtVisc(45.234)).toBe("45.2");
    expect(fmtVisc(1500)).toBe("1,500");
    expect(fmtVisc(25000)).toBe("25.0k");
  });
  it("formats percentages, money and sizes", () => {
    expect(fmtPct(0.456)).toBe("46%");
    expect(fmtUsd(-1234)).toBe("-$1,234");
    expect(fmtUsd(2_500_000)).toBe("$2.50M");
    expect(fmtBytes(2048)).toBe("2.0 kB");
  });
  it("maps diagnostic classes to operator wording", () => {
    expect(classLabel("ROD_FLOATING")).toBe("Rod floating");
    expect(classLabel(null)).toBe("–");
  });
});
