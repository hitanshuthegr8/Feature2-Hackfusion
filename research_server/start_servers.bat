@echo off
REM ============================================
REM Research Paper Analyzer - Start Server
REM ============================================

echo.
echo ========================================
echo  Starting Research Paper Analyzer
echo ========================================
echo.

REM Change to research_server directory
cd /d "%~dp0"

echo [1/2] Starting Backend API on port 5000...
start "Backend API" cmd /k ".\venv\Scripts\python.exe app.py"

timeout /t 3 /nobreak >nul

echo [2/2] Starting Frontend HTTP Server on port 8080...
start "Frontend Server" cmd /k "python -m http.server 8080 --directory simple_client"

echo.
echo ========================================
echo  Servers Started Successfully!
echo ========================================
echo.
echo Backend API:  http://localhost:5000
echo Frontend UI:  http://localhost:8080
echo.
echo Press any key to view server status...
pause >nul

REM Wait a bit for servers to start
timeout /t 5 /nobreak >nul

echo.
echo Checking server health...
curl -s http://localhost:5000/health

echo.
echo.
echo ========================================
echo To stop servers: Close both CMD windows
echo ========================================
echo.
pause
