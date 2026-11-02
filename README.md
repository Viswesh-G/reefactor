# Reefactor

Reefactor studies **Great Barrier Reef** coral bleaching through environmental observations, ocean
transport, atmospheric forcing and physics-informed forecasting.

The original GBR experiment remains in `config.yaml` and `run_all.py`. It uses
SST, DHW and geostrophic currents over 2017–2018. The expanded research pipeline
is separate so its grids, stores and checkpoints cannot be mixed accidentally.

Read [the GBR model contract and driver report](docs/GBR_MODEL_CONTRACT.md)
for the current coordinate guarantees, sources, parameter inventory and model
capacity. The [first-phase report](docs/RESEARCH_EXPANSION.md) retains the earlier audit.

```powershell
# Python 3.11 recommended; activate your environment first.
python -m pip install -r requirements-research.lock
python run_research.py catalog
python run_research.py drivers
python run_research.py model --profile gbr_rich --parameters 1000000
python run_research.py plan --profile daily --start 1993-01-01 --end 2026-08-31
python run_research.py ingest --profile ocean
python run_research.py ingest --profile gbr_rich_public
python run_research.py labels
python -m pytest tests -q
```

`plan` performs no data downloads. The checked-in default ingest request is only
three days over one degree near the GBR. `ocean` uses anonymous public NOAA and
Copernicus Zarr endpoints. `daily` also uses ERA5 through CDS, which requires a
free account, accepted data licence and `CDSAPI_KEY` in the environment.

```powershell
docker compose -f compose.research.yaml build
docker compose -f compose.research.yaml run --rm reef python -m pytest tests -q
docker compose -f compose.research.yaml run --rm reef python run_research.py plan --profile daily
```

The container runs as an unprivileged user. Ensure the mounted `data` directory
is writable by UID 10001 on Linux. Docker's engine must be running.

Work belongs on `codex/global-reef-data-fsm`; merge destination is **main only**,
after the requested merge signal. No merge or remote push is automatic.
