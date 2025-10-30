@echo off
echo.
echo ============================================================
echo   AI Lung Cancer Detection - Checkpoint Fix Script
echo ============================================================
echo.

cd ai
echo Creating dummy model checkpoint...
echo.

python create_dummy_checkpoint.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo   CHECKPOINT CREATED SUCCESSFULLY!
    echo ============================================================
    echo.
    echo Next Steps:
    echo.
    echo 1. Start Backend Server:
    echo    cd backend
    echo    node server.js
    echo.
    echo 2. Start Frontend in new terminal:
    echo    cd frontend
    echo    npm run dev
    echo.
    echo 3. Open browser: http://localhost:3000
    echo.
    echo WARNING: This is a DUMMY model for testing only!
    echo          Predictions will NOT be accurate.
    echo.
) else (
    echo.
    echo ============================================================
    echo   ERROR CREATING CHECKPOINT
    echo ============================================================
    echo.
    echo Please check the error messages above.
    echo.
)

cd ..
pause
