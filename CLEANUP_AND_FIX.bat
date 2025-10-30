@echo off
echo ============================================================
echo   Cleaning Up and Fixing Configuration
echo ============================================================
echo.

REM Delete unnecessary documentation files
echo Removing unnecessary documentation files...
if exist "CHECKPOINT_SETUP_COMPLETE.md" del "CHECKPOINT_SETUP_COMPLETE.md"
if exist "DOWNLOAD_MODEL.md" del "DOWNLOAD_MODEL.md"
if exist "FIX_CHECKPOINT_ERROR.bat" del "FIX_CHECKPOINT_ERROR.bat"
if exist "FIX_CHECKPOINT_ERROR.ps1" del "FIX_CHECKPOINT_ERROR.ps1"
if exist "FIX_INSTRUCTIONS.md" del "FIX_INSTRUCTIONS.md"
if exist "README_CHECKPOINT_FIX.md" del "README_CHECKPOINT_FIX.md"
if exist "START_SERVERS.md" del "START_SERVERS.md"

REM Delete dummy checkpoint creation script
echo Removing dummy checkpoint scripts...
if exist "ai\create_dummy_checkpoint.py" del "ai\create_dummy_checkpoint.py"
if exist "ai\test_model.py" del "ai\test_model.py"

REM Copy corrected .env files
echo.
echo Applying corrected configuration...
copy /Y "backend\.env.CORRECTED" "backend\.env"
copy /Y "frontend\.env.CORRECTED" "frontend\.env"

REM Delete the corrected files
del "backend\.env.CORRECTED"
del "frontend\.env.CORRECTED"

echo.
echo ============================================================
echo   Cleanup Complete!
echo ============================================================
echo.
echo Fixed Configuration:
echo   Backend .env: Uses your trained model at test_run_fixed_20250928_224821
echo   Frontend .env: Connects to backend at http://localhost:5000
echo.
echo Next Steps:
echo   1. Start Backend:  cd backend ^&^& node server.js
echo   2. Start Frontend: cd frontend ^&^& npm run dev
echo   3. Open: http://localhost:3000
echo.
echo Your REAL trained model will now be used!
echo.
pause
