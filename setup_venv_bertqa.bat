@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "REQ_FILE=%SCRIPT_DIR%\src\qa_hf_package\requirements.txt"
set "VENV_DIR=%SCRIPT_DIR%\.venv_bertqa"

if not exist "%REQ_FILE%" (
  echo Error: requirements file not found at "%REQ_FILE%"
  exit /b 1
)

where py >nul 2>nul
if %errorlevel%==0 (
  set "PYTHON_CMD=py -3"
) else (
  where python >nul 2>nul
  if %errorlevel%==0 (
    set "PYTHON_CMD=python"
  ) else (
    echo Error: Python not found in PATH. Install Python and try again.
    exit /b 1
  )
)

echo Using interpreter: %PYTHON_CMD%

if not exist "%VENV_DIR%" (
  echo Creating virtual environment at "%VENV_DIR%"
  %PYTHON_CMD% -m venv "%VENV_DIR%"
) else (
  echo Virtual environment already exists at "%VENV_DIR%"
)

if not exist "%VENV_DIR%\Scripts\activate.bat" (
  echo Error: activation script not found at "%VENV_DIR%\Scripts\activate.bat"
  exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 exit /b 1

echo Installing pinned requirements from "%REQ_FILE%"
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b 1
python -m pip install -r "%REQ_FILE%"
if errorlevel 1 exit /b 1

echo.
echo Setup complete.
echo Virtual environment directory: "%VENV_DIR%"
echo Requirements installed from: "%REQ_FILE%"
echo.
echo To activate manually in a new Command Prompt:
echo   .venv_bertqa\Scripts\activate.bat
