@echo off
cd /d "%~dp0"
if exist "..\.python\python.exe" (
  "..\.python\python.exe" run.py %*
) else (
  python run.py %*
)
if errorlevel 1 exit /b 1
