.PHONY: install ingest train eval clean help

# Python executable from the virtual environment (Windows)
PYTHON = venv\Scripts\python.exe
PIP    = venv\Scripts\pip.exe

help:
	@echo.
	@echo  REEFS_ML — PINN Coral Bleaching Pipeline
	@echo  =========================================
	@echo  make install   Install dependencies into the virtual environment
	@echo  make ingest    Stream satellite data and build the Zarr data store
	@echo  make train     Train the Physics-Informed Neural Network
	@echo  make eval      Evaluate the best trained model and print metrics
	@echo  make clean     Remove Python cache files
	@echo.

install:
	$(PIP) install -r requirements.txt

ingest:
	$(PYTHON) run_ingestion.py

train:
	$(PYTHON) run_pinns.py

eval:
	$(PYTHON) run_eval.py

clean:
	@echo Cleaning up Python cache files...
	powershell -Command "Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	powershell -Command "Get-ChildItem -Path . -Recurse -File -Filter *.pyc | Remove-Item -Force -ErrorAction SilentlyContinue"
