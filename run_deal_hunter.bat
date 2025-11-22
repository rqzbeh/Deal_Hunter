@echo off
REM Deal Hunter VPS Service Script
REM This script runs the deal hunter in a way that survives RDP disconnects

cd /d "%~dp0"

REM Set environment variables for VPS mode
set VPS_MODE=true

REM Optional: Set Telegram credentials if needed
REM set TELEGRAM_BOT_TOKEN=your_bot_token
REM set TELEGRAM_CHAT_ID=your_chat_id

REM Optional: Set alert thresholds
REM set ALERT_THRESHOLD_PERCENT=40
REM set BASKET_THRESHOLD_PERCENT=50

REM Run the Python script
python main.py

REM If the script exits, wait 30 seconds before restarting (for auto-restart)
timeout /t 30 /nobreak > nul
goto :start