"""Thermal engine: steam properties, viscosity law and the cyclic-steam simulation."""
import numpy as np
import pytest

from app.engines.thermal import steam
from app.engines.thermal.cycle import CycleSpec, default_walther, simulate_cycle, simulate_history
from app.engines.thermal.radial import Reservoir
from app.engines.thermal.viscosity import Walther, emulsion_factor


def test_saturation_temperature_matches_steam_tables():
    # steam tables: 1 MPa -> 179.9 C, 2 MPa -> 212.4 C, 5 MPa -> 263.9 C
    for p, t in [(1.0, 179.9), (2.0, 212.4), (5.0, 263.9)]:
        assert steam.t_sat_c(p) == pytest.approx(t, abs=2.5)


def test_latent_heat_decreases_with_temperature():
    assert steam.latent_heat_j_kg(100) == pytest.approx(2.257e6, rel=0.01)
    assert steam.latent_heat_j_kg(200) < steam.latent_heat_j_kg(150) < steam.latent_heat_j_kg(100)


def test_walther_reproduces_calibration_points_and_is_monotonic():
    w = default_walther()
    assert w.mu_cp(50.0) == pytest.approx(1500.0, rel=0.02)
    assert w.mu_cp(100.0) == pytest.approx(45.0, rel=0.02)
    t = np.linspace(30, 250, 50)
    mu = w.mu_cp(t)
    assert np.all(np.diff(mu) < 0), "viscosity must fall with temperature"
    assert mu.min() > 0.3, "never below the physical range of light water-like fluids"


def test_walther_two_point_fit_generic():
    w = Walther.from_two_points(40, 800, 90, 60)
    assert w.mu_cp(40) == pytest.approx(800, rel=0.02)
    assert w.mu_cp(90) == pytest.approx(60, rel=0.02)


def test_emulsion_viscosity_peaks_then_inverts():
    wc = np.linspace(0, 0.95, 40)
    f = emulsion_factor(wc)
    assert f[0] == pytest.approx(1.0)
    assert f.max() > 2.0, "emulsion should be more viscous than dry oil somewhere"
    assert f[-1] < f.max(), "phase inversion at high water cut lowers the viscosity again"


RES = Reservoir()


@pytest.fixture(scope="module")
def base_cycle():
    return simulate_cycle(RES, CycleSpec(cycle_no=1, steam_m3=4500.0))


def test_cycle_phases_and_decline(base_cycle):
    r = base_cycle
    assert set(np.unique(r.phase)) == {0, 1, 2}
    prod = r.phase == 2
    q = r.q_oil_m3d[prod]
    assert q.max() > 5.0
    assert q[-1] < 0.5 * q.max(), "rate must decline through the production window"
    assert np.all(np.diff(r.cum_oil_m3) >= -1e-9)
    assert 1.5 < r.sor < 10.0, f"steam-oil ratio {r.sor:.2f} outside the plausible CSS range"


def test_reservoir_cools_and_viscosity_rises_after_injection(base_cycle):
    r = base_cycle
    i_peak = int(np.argmax(r.t_avg_c))
    assert r.t_avg_c[i_peak] > RES.t_res_c + 20
    assert r.t_avg_c[-1] < r.t_avg_c[i_peak]
    prod = np.flatnonzero(r.phase == 2)
    assert r.mu_eff_cp[prod[-1]] > r.mu_eff_cp[prod[0]]


def test_more_steam_gives_more_oil_but_worse_sor():
    vols = (2500, 5500, 8500, 11500, 13500)
    cum = np.array([simulate_cycle(RES, CycleSpec(cycle_no=1, steam_m3=v)).cum_oil_m3[-1] for v in vols])
    assert np.all(np.diff(cum) > 0)
    sor = np.array(vols) / cum
    assert np.all(np.diff(sor) > 0), "steam-oil ratio must worsen as volume grows"
    early = (cum[1] - cum[0]) / (vols[1] - vols[0])
    late = (cum[-1] - cum[-2]) / (vols[-1] - vols[-2])
    assert late < early, "marginal oil per extra m3 of steam must fall"


def test_cumulative_oil_is_smooth_in_steam_volume():
    """Regression: a discrete grid-node crossing once made cumulative oil jump by 6 %."""
    cum = np.array([simulate_cycle(RES, CycleSpec(cycle_no=1, steam_m3=v)).cum_oil_m3[-1] for v in np.arange(7300, 7900, 60)])
    assert np.all(np.diff(cum) > 0)
    assert np.max(np.abs(np.diff(cum) / cum[:-1])) < 0.02


def test_cycles_chain_with_residual_heat():
    specs = [CycleSpec(cycle_no=i + 1, steam_m3=4500.0) for i in range(2)]
    hist = simulate_history(RES, specs, [0.0, 230.0])
    first, second = hist.results
    # the second cycle starts from a warmer reservoir than the virgin one
    assert second.profiles[0].mean() > first.profiles[0].mean()
