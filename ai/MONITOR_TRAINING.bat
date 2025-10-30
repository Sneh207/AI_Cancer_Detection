@echo off
echo ============================================================
echo   Training Progress Monitor
echo ============================================================
echo.
echo Detecting latest experiment folder...
echo.
echo Press Ctrl+C to stop monitoring (training will continue)
echo.
for /f "delims=" %%F in ('powershell -NoProfile -Command "Get-ChildItem -Path 'experiments' -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name"') do set EXP=%%F

if "%EXP%"=="" (
  echo Could not find an experiments folder. Make sure training has started.
  goto :eof
)

echo Monitoring: experiments\%EXP%\logs\cancer_detection_training.log
echo.
echo Showing the last 50 lines and following in real-time...
echo.
powershell -NoProfile -Command "Get-Content -Path 'experiments/%EXP%/logs/cancer_detection_training.log' -Tail 50 -Wait"
