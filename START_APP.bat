@echo off
echo ============================================================
echo   Starting Cancer Detection AI Application
echo ============================================================
echo.

echo Step 1: Checking environment files...
if not exist "frontend\.env" (
    echo Creating frontend .env file...
    echo VITE_API_URL=http://localhost:5000 > frontend\.env
)

if not exist "backend\.env" (
    echo Creating backend .env file...
    (
        echo PORT=5000
        echo CHECKPOINT_PATH=../ai/experiments/test_run_fixed_20250928_224821/checkpoints/best_model.pth
        echo CONFIG_PATH=../ai/configs/config.yaml
    ) > backend\.env
)

echo.
echo Step 2: Starting Backend Server...
start "Cancer Detection Backend" cmd /k "cd backend && node server.js"

timeout /t 3 /nobreak >nul

echo.
echo Step 3: Starting Frontend Dev Server...
start "Cancer Detection Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ============================================================
echo   Application Starting!
echo ============================================================
echo.
echo Backend: http://localhost:5000
echo Frontend: http://localhost:5173
echo.
echo Two terminal windows will open:
echo   1. Backend Server (Node.js)
echo   2. Frontend Dev Server (Vite)
echo.
echo Wait for both servers to start, then open:
echo   http://localhost:5173
echo.
echo Press any key to exit this window...
pause >nul
