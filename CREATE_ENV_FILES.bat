@echo off
echo Creating environment files...

REM Create frontend .env
(
echo VITE_API_URL=http://localhost:5000
) > frontend\.env

REM Create backend .env
(
echo PORT=5000
echo CHECKPOINT_PATH=../ai/experiments/test_run_fixed_20250928_224821/checkpoints/best_model.pth
echo CONFIG_PATH=../ai/configs/config.yaml
) > backend\.env

echo.
echo ✅ Environment files created successfully!
echo.
echo Frontend .env: VITE_API_URL=http://localhost:5000
echo Backend .env: PORT=5000
echo.
pause
