.PHONY: install test research-plan ingest derive_mmm train eval baseline ablation noaa_baseline early_warning plot smoke all clean help

# Use the active environment. Override with make PYTHON=/path/to/python.
PYTHON ?= python
PIP = $(PYTHON) -m pip

help:
	@echo.
	@echo  REEFS_ML — PINN Coral Bleaching Pipeline
	@echo  =========================================
	@echo  make install        Install dependencies into the virtual environment
	@echo  make test           Run the full suite with research dependencies installed
	@echo  make research-plan  Review full GBR coverage and resource budgets offline
	@echo  make all            Run the FULL pipeline: ingest+train+baseline+eval+plot
	@echo  make smoke          Run end-to-end smoke test (no real data needed, ~30s)
	@echo  make ingest         Stream satellite data and build the Zarr data store
	@echo  make derive_mmm     Recover the true NOAA CRW MMM climatology sidecar
	@echo  make train          Train the Physics-Informed ConvLSTM (PhyConvLSTM)
	@echo  make eval           Evaluate the best trained PINN and print metrics
	@echo  make baseline       Train the Vanilla ConvLSTM baseline (no physics loss)
	@echo  make ablation       Train the ablation: PINN with K=0 (advection only)
	@echo  make noaa_baseline  Run the NOAA CRW operational DHW algorithm baseline
	@echo  make early_warning  Physics-residual early-warning precursor analysis
	@echo  make plot           Generate all paper figures from saved checkpoints
	@echo  make clean          Remove Python cache files
	@echo.

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest tests -q

research-plan:
	$(PYTHON) run_research.py plan --config configs/gbr_study.yaml --common-window

all:
	$(PYTHON) run_all.py

ingest:
	$(PYTHON) run_ingestion.py

derive_mmm:
	$(PYTHON) run_derive_mmm.py

train:
	$(PYTHON) run_pinns.py

eval:
	$(PYTHON) run_eval.py

baseline:
	$(PYTHON) run_baseline.py

ablation:
	$(PYTHON) run_ablation.py

noaa_baseline:
	$(PYTHON) run_noaa_dhw_baseline.py

early_warning:
	$(PYTHON) run_early_warning.py

plot:
	$(PYTHON) run_plot_figures.py

smoke:
	$(PYTHON) run_smoke_test.py

clean:
	@echo Cleaning up Python cache files...
	powershell -Command "Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	powershell -Command "Get-ChildItem -Path . -Recurse -File -Filter *.pyc | Remove-Item -Force -ErrorAction SilentlyContinue"
