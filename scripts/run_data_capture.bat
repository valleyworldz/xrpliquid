@echo off
REM Market Data Capture Scheduler for Windows
REM Launches both tick listener and funding logger as sidecar processes

setlocal enabledelayedexpansion

REM Configuration
set DATA_DIR=data
set LOG_DIR=logs
set PID_DIR=pids

REM Create directories
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%PID_DIR%" mkdir "%PID_DIR%"

REM Function to start a process
:start_process
set name=%1
set command=%2
set log_file=%LOG_DIR%\%name%.log
set pid_file=%PID_DIR%\%name%.pid

echo Starting %name%...

REM Check if already running
if exist "%pid_file%" (
    for /f %%i in (%pid_file%) do (
        tasklist /FI "PID eq %%i" 2>NUL | find /I "%%i" >NUL
        if not errorlevel 1 (
            echo %name% is already running ^(PID: %%i^)
            goto :eof
        )
    )
)

REM Start process in background
start /B %command% > "%log_file%" 2>&1
echo %name% started
echo Logs: %log_file%
goto :eof

REM Function to stop a process
:stop_process
set name=%1
set pid_file=%PID_DIR%\%name%.pid

if exist "%pid_file%" (
    for /f %%i in (%pid_file%) do (
        tasklist /FI "PID eq %%i" 2>NUL | find /I "%%i" >NUL
        if not errorlevel 1 (
            echo Stopping %name% ^(PID: %%i^)...
            taskkill /PID %%i /F >NUL 2>&1
            del "%pid_file%"
            echo %name% stopped
        ) else (
            echo %name% is not running
            del "%pid_file%"
        )
    )
) else (
    echo %name% is not running
)
goto :eof

REM Function to check process status
:check_status
set name=%1
set pid_file=%PID_DIR%\%name%.pid

if exist "%pid_file%" (
    for /f %%i in (%pid_file%) do (
        tasklist /FI "PID eq %%i" 2>NUL | find /I "%%i" >NUL
        if not errorlevel 1 (
            echo %name% is running ^(PID: %%i^)
        ) else (
            echo %name% is not running ^(stale PID file^)
            del "%pid_file%"
        )
    )
) else (
    echo %name% is not running
)
goto :eof

REM Main script logic
if "%1"=="start" goto :start
if "%1"=="stop" goto :stop
if "%1"=="restart" goto :restart
if "%1"=="status" goto :status
if "%1"=="logs" goto :logs
if "%1"=="convert" goto :convert
if "%1"=="summary" goto :summary
if "%1"=="clean" goto :clean
goto :usage

:start
echo Starting Market Data Capture System...

REM Start tick listener
call :start_process tick_listener "python src\data_capture\tick_listener.py"

REM Start funding logger
call :start_process funding_logger "python src\data_capture\funding_logger.py"

echo Market Data Capture System started
echo Use '%0 status' to check status
echo Use '%0 logs' to view logs
goto :eof

:stop
echo Stopping Market Data Capture System...

REM Stop tick listener
call :stop_process tick_listener

REM Stop funding logger
call :stop_process funding_logger

echo Market Data Capture System stopped
goto :eof

:restart
echo Restarting Market Data Capture System...
call :stop
timeout /t 2 /nobreak >NUL
call :start
goto :eof

:status
echo Market Data Capture System Status:
echo ==================================
call :check_status tick_listener
call :check_status funding_logger
goto :eof

:logs
echo Market Data Capture System Logs:
echo ================================
if exist "%LOG_DIR%\tick_listener.log" (
    echo === tick_listener logs ^(last 20 lines^) ===
    powershell "Get-Content '%LOG_DIR%\tick_listener.log' | Select-Object -Last 20"
) else (
    echo No logs found for tick_listener
)

echo.
if exist "%LOG_DIR%\funding_logger.log" (
    echo === funding_logger logs ^(last 20 lines^) ===
    powershell "Get-Content '%LOG_DIR%\funding_logger.log' | Select-Object -Last 20"
) else (
    echo No logs found for funding_logger
)
goto :eof

:convert
echo Converting tick data to Parquet...
set date=%2
if "%date%"=="" set date=%date:~0,10%
python -c "from src.data_capture.tick_listener import TickListener; listener = TickListener(); listener.convert_to_parquet('%date%'); print('Conversion completed')"
goto :eof

:summary
echo Market Data Capture Summary:
echo ============================

echo Data Directory Contents:
if exist "%DATA_DIR%" (
    dir "%DATA_DIR%" /B
) else (
    echo No data directory found
)

echo.
echo Tick Data Files:
if exist "%DATA_DIR%\ticks" (
    dir "%DATA_DIR%\ticks" /B
) else (
    echo No tick data found
)

echo.
echo Funding Data Files:
if exist "%DATA_DIR%\funding" (
    dir "%DATA_DIR%\funding" /B
) else (
    echo No funding data found
)

echo.
echo Warehouse Data:
if exist "%DATA_DIR%\warehouse" (
    dir "%DATA_DIR%\warehouse" /B
) else (
    echo No warehouse data found
)
goto :eof

:clean
echo Cleaning old data files...
set days=%2
if "%days%"=="" set days=7

REM Clean old files (Windows doesn't have find command, so we'll use PowerShell)
powershell "Get-ChildItem '%DATA_DIR%\ticks\*.jsonl' | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-%days%)} | Remove-Item -Force"
powershell "Get-ChildItem '%DATA_DIR%\ticks\*.parquet' | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-%days%)} | Remove-Item -Force"
powershell "Get-ChildItem '%DATA_DIR%\funding\*.json' | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-%days%)} | Remove-Item -Force"
powershell "Get-ChildItem '%LOG_DIR%\*.log' | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-%days%)} | Remove-Item -Force"

echo Cleaned files older than %days% days
goto :eof

:usage
echo Usage: %0 {start^|stop^|restart^|status^|logs^|convert^|summary^|clean}
echo.
echo Commands:
echo   start     - Start the market data capture system
echo   stop      - Stop the market data capture system
echo   restart   - Restart the market data capture system
echo   status    - Check the status of running processes
echo   logs      - Show recent logs from all processes
echo   convert   - Convert tick data to Parquet format
echo   summary   - Show data directory summary
echo   clean     - Clean old data files ^(default: 7 days^)
echo.
echo Examples:
echo   %0 start
echo   %0 status
echo   %0 logs
echo   %0 convert 2025-09-15
echo   %0 clean 14
exit /b 1
