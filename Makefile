.PHONY: install ingest derive_mmm train eval baseline ablation noaa_baseline early_warning plot smoke all clean help

# Python executable from the virtual environment (Windows)
PYTHON = venv\Scripts\python.exe
PIP    = venv\Scripts\pip.exe

help:
	@echo.
	@echo  REEFS_ML — PINN Coral Bleaching Pipeline
	@echo  =========================================
	@echo  make install        Install dependencies into the virtual environment
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
