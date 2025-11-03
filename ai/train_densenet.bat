@echo off
echo ========================================
echo Training DenseNet121 Model
echo ========================================
echo.

cd /d "%~dp0"

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check if config uses densenet121
findstr /C:"densenet121" configs\config.yaml >nul
if errorlevel 1 (
    echo ERROR: config.yaml is not set to densenet121
    echo Please update model.architecture to "densenet121"
    pause
    exit /b 1
)

echo Starting training...
echo Model: DenseNet121
echo Experiment: densenet121_balanced
echo.

python main.py train ^
    --config configs/config.yaml ^
    --experiment-name densenet121_balanced ^
    --device auto ^
    --seed 42

echo.
echo ========================================
echo Training Complete!
echo ========================================
echo.
echo Results saved to: experiments\densenet121_balanced_*
echo.
pause
