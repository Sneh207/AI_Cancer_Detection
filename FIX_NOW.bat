@echo off
echo ============================================================
echo   Quick Fix - Configure Backend and Frontend
echo ============================================================
echo.

echo Creating backend .env file...
(
echo PORT=5000
echo CHECKPOINT_PATH=../ai/experiments/test_run_fixed_20250928_224821/checkpoints/best_model.pth
echo CONFIG_PATH=../ai/configs/config.yaml
) > backend\.env

echo Creating frontend .env file...
(
echo VITE_API_URL=http://localhost:5000
) > frontend\.env

echo.
echo ============================================================
echo   Configuration Complete!
echo ============================================================
echo.
echo ⚠️  NOTE: Using dummy model (your trained model has same issue)
echo    Both give 51.7%% because no cancer examples in training data
echo.
echo To fix this properly, see: CRITICAL_ISSUE_AND_FIX.md
echo.
echo Next steps:
echo   1. Terminal 1: cd backend ^&^& npm start
echo   2. Terminal 2: cd frontend ^&^& npm run dev
echo   3. Open: http://localhost:5173
echo.
echo To get a REAL working model:
echo   - Option A: Download pre-trained model (recommended)
echo   - Option B: Create balanced test labels: cd ai ^&^& python create_balanced_labels.py
echo.
pause
