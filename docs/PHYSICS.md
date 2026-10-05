# Physics and model notes

All parameters are illustrative; calibrate to the field before use.

**Steam and heat.** Saturation temperature from Antoine; latent heat by Watson correlation;
heat-injection efficiency by Marx–Langenheim. Radial conduction is solved implicitly on a
logarithmic grid; cycles chain via the residual temperature field.

**Viscosity.** Walther law calibrated to ~1500 cP at 50 °C and ~45 cP at 100 °C, with an emulsion
factor for water cut. The effective produced viscosity is the flow-weighted average over the
heated zone (continuous in radius).

**Inflow.** Darcy radial flow with temperature-dependent viscosity; LightGBM learns the residual
ln(q_obs / q_phys), clipped to ×0.5–×2 so ML cannot overrule physics. SHAP (pred_contrib) explains it.

**Rod pump.** A 2×2 transfer-matrix wave-equation model maps surface cards to downhole cards and
back. Fault cards (rod floating, pump-unsetting risk, fluid pound, gas interference) are generated
for training and for the simulator. Float index and Goodman fatigue give the rod-stress limits.

**CNN.** 1-D CNN (~16.7k parameters) on the resampled, normalised downhole card plus scalars.
Trained on synthetic cards; metrics are on synthetic held-out cards.

**Optimizer.** Bisection finds the maximum safe SPM; a receding-horizon dynamic program over a
0.5-SPM grid (14 days, +1.5 SPM/day rate limit) maximises modelled margin subject to those limits.
The XAI layer reports the driver, per-term contributions and a narrative; the unsafe-constraint
penalty is excluded from displayed economics.
