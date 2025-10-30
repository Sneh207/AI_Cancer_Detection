@echo off
echo ============================================================
echo   Applying Real Model Configuration
echo ============================================================
echo.

echo Waiting for training to complete...
echo.
echo This script will:
echo   1. Copy the correct .env files
echo   2. Configure backend to use your REAL trained model
echo   3. Configure frontend to connect to backend
echo.

:CHECK_MODEL
if not exist "ai\experiments\real_model_20251019_194914\checkpoints\best_model.pth" (
    echo [%TIME%] Waiting for model training to complete...
    echo            Checking for: ai\experiments\real_model_20251019_194914\checkpoints\best_model.pth
    timeout /t 30 /nobreak >nul
    goto CHECK_MODEL
)

echo.
echo ============================================================
echo   Training Complete! Model Found!
echo ============================================================
echo.

REM Copy corrected .env files
echo Applying configuration...
copy /Y "backend\.env.CORRECTED" "backend\.env"
copy /Y "frontend\.env.CORRECTED" "frontend\.env"

echo.
echo ============================================================
echo   Configuration Applied!
echo ============================================================
echo.
echo Backend .env: Uses REAL trained model
echo   Path: ai/experiments/real_model_20251019_194914/checkpoints/best_model.pth
echo.
echo Frontend .env: Connects to backend at http://localhost:5000
echo.
echo ============================================================
echo   Next Steps:
echo ============================================================
echo.
echo 1. Start Backend (Terminal 1):
echo    cd backend
echo    npm start
echo.
echo 2. Start Frontend (Terminal 2):
echo    cd frontend
echo    npm run dev
echo.
echo 3. Open browser:
echo    http://localhost:3000
echo.
echo 4. Upload X-ray images and get REAL predictions!
echo.
echo Your model trained on 3025 images will now be used!
echo.
pause
