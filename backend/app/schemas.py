"""Request / response schemas (Pydantic v2) with strict physical bounds."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CycleIn(Strict):
    steam_m3: float = Field(ge=200, le=25000, description="cold-water-equivalent steam volume")
    quality: float = Field(default=0.75, ge=0.3, le=1.0)
    inj_rate_m3d: float = Field(default=200.0, ge=40, le=800)
    inj_pressure_mpa: float = Field(default=1.8, ge=0.4, le=8.0)
    soak_days: float = Field(default=7.0, ge=0, le=90)
    prod_days: float = Field(default=180.0, ge=20, le=500)
    start_ts: int | None = Field(default=None, ge=0, description="epoch seconds; defaults to after the previous cycle")


class ScenarioIn(Strict):
    label: str | None = Field(default=None, max_length=40)
    steam_m3: float | None = Field(default=None, ge=200, le=25000)
    quality: float | None = Field(default=None, ge=0.3, le=1.0)
    inj_rate_m3d: float | None = Field(default=None, ge=40, le=800)
    inj_pressure_mpa: float | None = Field(default=None, ge=0.4, le=8.0)
    soak_days: float | None = Field(default=None, ge=0, le=90)
    prod_days: float | None = Field(default=None, ge=20, le=500)

    def overrides(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None and k != "label"}


class WhatIfRequest(Strict):
    well_id: str = Field(max_length=16)
    scenarios: list[ScenarioIn] = Field(min_length=1, max_length=6)
    base: Literal["next", "current"] = "next"


class ThermalPredictRequest(Strict):
    well_id: str = Field(max_length=16)
    ts: int | None = Field(default=None, ge=0)


class ClassifyRequest(Strict):
    well_id: str | None = Field(default=None, max_length=16)
    spm: float = Field(gt=0.2, le=20)
    position: list[float] | None = Field(default=None, min_length=32, max_length=4096)
    load: list[float] | None = Field(default=None, min_length=32, max_length=4096)
    points: list[list[float]] | None = Field(default=None, min_length=32, max_length=4096, description="[[position, load], ...]")
    position_unit: Literal["m", "in", "mm", "ft"] = "m"
    load_unit: Literal["N", "kN", "lbf", "klbf"] = "kN"
    viscosity_cp: float | None = Field(default=None, gt=0, le=1e7)

    @model_validator(mode="after")
    def _shape(self):
        if self.points is not None:
            if any(len(p) != 2 for p in self.points):
                raise ValueError("every point must be [position, load]")
            self.position = [p[0] for p in self.points]
            self.load = [p[1] for p in self.points]
        if self.position is None or self.load is None:
            raise ValueError("provide either points or position + load arrays")
        if len(self.position) != len(self.load):
            raise ValueError("position and load must have the same length")
        if not all(math.isfinite(v) for v in self.position + self.load):
            raise ValueError("arrays must contain finite numbers")
        return self


class DecisionIn(Strict):
    actor: str = Field(default="operator", min_length=1, max_length=64, pattern=r"^[\w .@\-]+$")
    note: str | None = Field(default=None, max_length=300)


class CyclePlanRequest(Strict):
    volumes: list[float] | None = Field(default=None, min_length=2, max_length=16)
    overrides: ScenarioIn | None = None

    @field_validator("volumes")
    @classmethod
    def _bounds(cls, v):
        if v is not None and any((x < 200 or x > 25000) for x in v):
            raise ValueError("volumes must be within 200-25000 m3")
        return v


class SetpointIn(Strict):
    spm: float = Field(ge=0, le=20)
    actor: str = Field(default="operator", min_length=1, max_length=64, pattern=r"^[\w .@\-]+$")
