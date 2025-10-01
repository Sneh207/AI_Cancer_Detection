@echo off
echo ========================================
echo   Cancer Detection System - Dev Mode
echo ========================================
echo.
echo Starting Backend and Frontend servers...
echo.
echo Backend will run on: http://localhost:3000
echo Frontend will run on: http://localhost:5173
echo.
echo Press Ctrl+C to stop both servers
echo ========================================
echo.

REM Start backend in a new window
start "Backend Server" cmd /k "cd backend && npm start"

REM Wait 3 seconds for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend in a new window
start "Frontend Dev Server" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================
echo   Both servers are starting...
echo   Check the new terminal windows
echo ========================================
echo.
echo Open your browser to: http://localhost:5173
echo.
pause
