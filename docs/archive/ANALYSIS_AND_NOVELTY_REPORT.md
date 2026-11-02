# Project audit and research-positioning report

## Executive assessment

The project is a plausible **research prototype**, not yet a publishable result.
The implementation has a coherent hypothesis: use an advection-diffusion residual
with observed surface currents to improve SST forecasts and downstream DHW skill
over persistence, climatology, and a vanilla ConvLSTM. The 2017-02-13 to 2018-12-31
window is a defensible first ingestion window because it is bounded by the
available current product.

The work should not claim a new PINN architecture. ConvLSTM forecasting, ocean
physics constraints, marine-heatwave prediction, and NOAA DHW are established
areas. The potentially publishable contribution is narrower and testable:

> **Does an explicitly calibrated, current-forced advection-diffusion residual
> improve out-of-sample coral-thermal-stress forecasts, and can the residual
> provide a calibrated early-warning signal before DHW acceleration?**

That is a hypothesis requiring multi-year, event-based validation—not a result
that can be inferred from the code alone.

## Current-state audit

| Area | Status | Evidence and remaining risk |
|---|---|---|
| Real ingestion | Implemented | Strict ERDDAP/OPeNDAP FSM, monthly cache, Zarr output. Network execution still needs a completed 2017-2018 run in this environment. |
| Common canvas | Fixed | The grid is now inclusive and endpoint-stable; interpolation no longer extrapolates beyond source coverage. |
| Physics units | Implemented | `dx`, `dy`, `dt`, and `K` are SI-scaled finite differences. The learned `K` is constrained in log-space. |
| NOAA DHW | Implemented with guardrails | DHW uses MMM, a 1 C hotspot gate, full hotspot accumulation, and an 84-day rolling window. Product aggregation is explicit so DHW is not silently averaged. |
| Leakage control | Implemented | Dataset normalization and climatology use only the chronological training split. |
| Baselines | Implemented | Persistence, monthly climatology, vanilla ConvLSTM, K=0 physics ablation, and NOAA-style DHW baseline are present. |
| Reproducibility | Partial | Global seeding exists and a smoke test is explicit. A paper result still needs repeated seeds, saved environment metadata, and frozen data provenance. |
| Evaluation | Partial | Metrics and lead-time outputs exist. Confidence intervals, event-level splits, and skill significance tests remain required. |

## Novelty and publishability rating

These are **risk-adjusted estimates**, not claims of exhaustive literature
coverage:

| Criterion | Rating | Interpretation |
|---|---:|---|
| Technical novelty | 5/10 | The combination of current-forced residual + DHW is more specific than a generic PINN, but each ingredient is established. |
| Scientific question | 7/10 | The residual-as-early-warning hypothesis is clear and falsifiable. |
| Reproducibility potential | 7/10 | Public sources, one config, strict ingestion, and small sidecars are strong foundations. |
| Evidence today | 2/10 | No verified end-to-end metrics or repeated-seed results are committed. |
| Publishability today | 3/10 | Prototype only. |
| Publishability after the required experiments | 6-7/10 | A realistic workshop or applied ML/oceanography submission if gains are robust and calibrated. |

## Required experiments before making a novelty claim

1. Use at least 5-10 years of continuous data, with event-based holdouts rather
   than one random chronological split. Keep 2017-2018 as a smoke dataset.
2. Repeat every condition with at least three seeds and report mean, standard
   deviation, and paired bootstrap confidence intervals.
3. Compare persistence, monthly climatology, vanilla ConvLSTM, K=0 advection,
   K-fixed diffusion, learned K, and a current-free physics control.
4. Report SST skill, DHW skill, alert calibration, and event detection lead time
   separately. Do not use DHW derived from the same predicted maps as the only
   evidence of biological value.
5. Test whether the physics residual predicts future DHW-rate after controlling
   for present DHW, SST anomaly, and persistence. A correlation by itself is not
   an early-warning result.
6. Save exact source URLs, query windows, package versions, config hash, seed,
   model parameter count, and hardware in each run manifest.

## What was corrected in this scaffold

- Unified grid axes include both requested spatial bounds.
- Interpolation refuses to invent values outside a source's spatial support.
- Temporal aggregation is variable-specific: SST max, CRW DHW last value, and
  currents mean by default.
- Missing Zarr data no longer produces a success-shaped random-data training run;
  synthetic smoke behavior requires `run_pinns.py --smoke`.
- Core contracts now have a fast test suite.

## Literature-positioning note

The safe positioning is **method integration plus a new evaluation hypothesis**,
not “the first physics-informed coral bleaching model.” Existing work already
covers ConvLSTM-style environmental forecasting, physics-informed neural
networks, marine heatwave prediction, and NOAA Coral Reef Watch DHW products.
The manuscript should therefore lead with the controlled ablation and the
residual lead-warning test, and should explicitly distinguish:

- operational DHW reproduction from ML forecast skill;
- SST forecast accuracy from bleaching-alert utility;
- learned diffusivity as an interpretable parameter from a merely convenient
  trainable scalar; and
- exploratory correlation from validated early warning.

The report should be updated with a formal systematic-search table before
submission. The repository can support that process, but code inspection alone
cannot establish priority over every internet publication.
