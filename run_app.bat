@echo off
REM Starts the dashboard using the project's virtual environment.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Virtual environment not found. Follow README.md section 2 first.
  exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app.py
