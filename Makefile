.PHONY: install ingest train clean help

# Python executable from the virtual environment (Windows)
PYTHON = venv\Scripts\python.exe
PIP = venv\Scripts\pip.exe

help:
	@echo Available commands:
	@echo   make install  - Install dependencies into the virtual environment
	@echo   make ingest   - Run the data ingestion and resolution pipeline
	@echo   make train    - Run the PINN model training
	@echo   make clean    - Remove cached Python files

install:
	$(PIP) install -r requirements.txt

ingest:
	$(PYTHON) run_ingestion.py

train:
	$(PYTHON) run_pinns.py

clean:
	@echo Cleaning up Python cache files...
	powershell -Command "Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	powershell -Command "Get-ChildItem -Path . -Recurse -File -Filter *.pyc | Remove-Item -Force -ErrorAction SilentlyContinue"
