@echo off
setlocal
echo === YouTube Comments Tool: Installer ===
echo.
where py >nul 2>&1
if %errorlevel% neq 0 (
  echo Python launcher not found. Please install Python 3 from https://www.python.org/downloads/windows/ and re-run.
  pause
  exit /b 1
)
echo Creating virtual environment (.venv)...
py -3 -m venv .venv
if %errorlevel% neq 0 (
  echo Failed to create virtual environment.
  pause
  exit /b 1
)
call .venv\Scripts\activate
echo Upgrading pip...
python -m pip install --upgrade pip
echo Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt
if %errorlevel% neq 0 (
  echo Failed to install dependencies.
  pause
  exit /b 1
)
echo.
echo All set! You can now run the tool using:
echo   run.bat --help
echo.
pause
call .venv\Scripts\activate
python -m textblob.download_corpora
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

