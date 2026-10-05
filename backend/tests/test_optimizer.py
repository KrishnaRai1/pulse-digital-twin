"""Hard safety limits and the MPC optimiser."""
import numpy as np

from app.domain.wells import build_fleet
from app.engines.optimizer import mpc
from app.engines.optimizer.limits import check_setpoint, spm_limits
from app.engines.srp.loads import FLOAT_MARGIN, float_index

FLEET = {w.id: w for w in build_fleet()}
CFG = FLEET["BGW-01"]
D = CFG.design
F_LOAD = D.fluid_load_n(D.pump_depth_m * CFG.net_lift_fraction)


def test_ceiling_drops_as_the_oil_cools():
    ceilings = [spm_limits(D, mu, F_LOAD).spm_max for mu in (0.02, 0.05, 0.2, 1.0)]
    assert all(a >= b for a, b in zip(ceilings, ceilings[1:]))
    assert ceilings[0] > ceilings[-1]


def test_limit_is_the_boundary_of_the_float_criterion():
    lim = spm_limits(D, 0.2, F_LOAD)
    assert float_index(lim.spm_float, 0.2, D) <= FLOAT_MARGIN + 1e-3
    assert float_index(lim.spm_float * 1.05 + 0.05, 0.2, D) > FLOAT_MARGIN or lim.spm_float >= D.max_spm - 1e-6


def test_check_setpoint_flags_violations_and_passes_safe_speeds():
    lim = spm_limits(D, 0.5, F_LOAD)
    assert check_setpoint(min(lim.spm_max, D.max_spm) * 0.9, D, 0.5, F_LOAD) == []
    assert check_setpoint(D.max_spm + 3, D, 0.5, F_LOAD)
    assert check_setpoint(-1.0, D, 0.5, F_LOAD)


def _forecast(mu, q_liq=25.0, days=14, wc=0.4):
    n = days
    return mpc.Forecast(
        mu_tub_pa_s=np.full(n, mu) if np.isscalar(mu) else np.asarray(mu),
        q_liquid_m3d=np.full(n, q_liq),
        q_oil_m3d=np.full(n, q_liq * (1 - wc)),
        net_lift_m=D.pump_depth_m * CFG.net_lift_fraction,
    )


def test_mpc_never_recommends_an_unsafe_speed_across_conditions():
    econ = mpc.Economics()
    for mu in (0.01, 0.05, 0.2, 0.8):
        for q in (5.0, 20.0, 60.0):
            for start in (1.0, 4.0, 8.0):
                fc = _forecast(mu, q)
                res = mpc.solve(D, econ, fc, start)
                lim = spm_limits(D, mu, D.fluid_load_n(fc.net_lift_m))
                if lim.feasible:
                    assert res.recommended <= lim.spm_max + 1e-6, (mu, q, start, res.recommended, lim.spm_max)
                assert D.min_spm - 1e-9 <= res.recommended <= D.max_spm + 1e-9


def test_mpc_reduces_speed_when_oil_is_cooling_into_the_float_zone():
    mu = np.linspace(0.05, 0.6, 14)  # viscosity rising through the horizon
    fc = _forecast(mu, q_liq=15.0)
    now = spm_limits(D, mu[0], D.fluid_load_n(fc.net_lift_m)).spm_max
    res = mpc.solve(D, mpc.Economics(), fc, min(now, D.max_spm))
    assert res.recommended < min(now, D.max_spm) + 1e-9
    assert res.plan_spm[-1] <= res.plan_spm[0] + 1e-9


def test_mpc_speed_increase_is_rate_limited():
    fc = _forecast(0.02, q_liq=80.0)
    res = mpc.solve(D, mpc.Economics(), fc, 2.0)
    assert np.all(np.diff(res.plan_spm) <= mpc.MAX_RATE_UP + 1e-9)
    assert res.recommended <= 2.0 + mpc.MAX_RATE_UP + 1e-9


def test_mpc_holds_when_already_optimal():
    fc = _forecast(0.03, q_liq=6.0)
    first = mpc.solve(D, mpc.Economics(), fc, 3.0)
    again = mpc.solve(D, mpc.Economics(), fc, first.recommended)
    assert abs(again.recommended - first.recommended) < 0.51
