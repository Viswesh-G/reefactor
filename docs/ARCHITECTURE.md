# How a GBR observation reaches a model

Run commands from the repository root with the Python 3.11 research lock installed. The root Python commands are stable entry points. New research logic belongs in `src/reef/`.

| Layer | Files | Responsibility |
|---|---|---|
| Study definition | `configs/research.yaml`, `configs/gbr_study.yaml`, `catalog.py`, `data/` inside `src/reef` | Small default pilot, GBR envelope, source roles, native units, vintages and official polygon |
| Planning | `region.py`, `planning.py` | Intersect source dates, validate resolution, prune geometry-only tiles, size native working sets and enforce budgets before network access |
| Acquisition | `sources.py`, `remote_zarr.py`, `opendap.py`, `cache.py` | Subset upstream arrays, bound transfers and deduplicate remote chunks in RAM |
| Scientific alignment | `harmonize.py`, `unity.py` | Check native units, time slots and vertical layers, compute spherical area means, enforce one finite all-field domain |
| Orchestration | `pipeline.py`, `observability.py` | Finite state transitions, bounded worker pool, atomic Zarr publication, resume checks and a single DuckDB writer with Polars-readable snapshots |
| Observations and evaluation design | `labels.py`, `colonies.py`, `coralwatch.py`, `study.py` | Keep percentages, colony scores and colour charts in distinct tables. Link valid containing cells and dated histories, then allocate temporal and buffered spatial folds |
| Forecasting | `features.py`, `physics.py`, `learning.py`, root `train_research.py` | Derive covariates, evaluate heat/salt residuals, build cropped windows, train and checkpoint the SST and optional survey-percent heads |
| Verification | `tests/`, `docs/evidence/` | Offline scientific and failure-injection tests, small versioned summaries of real runs |

The run FSM advances through INIT, PREFLIGHT, RUNNING, FINALIZING and READY. Exceptions enter ERROR. Workers fetch and align complete feature sets. The coordinator validates and stages a tile, verifies its marker, publishes it and journals the transition. A restarted run reuses verified partitions. FINALIZING intersects masks across all dates. READY is required for normal model access.

Each run has its own directory below the configured storage root. Compressed Zarr partitions hold scientific values. The manifest and markers identify coordinates, source provenance and hashes. DuckDB and Parquet hold small operational records. Sparse colony and survey tables live under separate label directories. Tests use temporary storage. Large materialized data, credentials and environments stay outside version control.

## Repository guide

* `run_research.py` handles catalog inspection, planning, ingestion, run reporting and observational audits. `train_research.py` is the training entry.
* The root `run_all.py`, other original `run_*.py` commands, `config.yaml`, `requirements.txt` and modules directly under `src/` support the earlier SST/DHW experiment. The research forecaster reuses its ConvLSTM cell. Historical documents live in [the archive](archive/README.md).
* `Dockerfile.research`, `compose.research.yaml` and `requirements-research.lock` define the research environment. `make research-plan` and `make test` use the active interpreter.
* Read the [study protocol](GBR_STUDY.md) for the latest state and test design, the [model contract](GBR_MODEL_CONTRACT.md) for the foundation, and the [first ingestion audit](RESEARCH_EXPANSION.md) for earlier benchmarks.
