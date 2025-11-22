# Deal Hunter - VPS Setup Guide

## Issues Fixed

### 1. Logging Issues ✅

- Added comprehensive logging to `deal_hunter.log` file
- Console output now also goes to log file
- Monitoring cycles, price checks, and delay optimizations are logged

### 2. RDP Disconnect Issues ✅

- Improved browser configuration with session-resistant arguments
- Custom user data directory that works in disconnected sessions
- Added batch file for service-like execution

## VPS Setup Instructions

### Option 1: Run as Windows Service (Recommended for VPS)

1. **Download NSSM (Non-Sucking Service Manager)**:

   Download from: <https://nssm.cc/download>

   Extract `nssm.exe` to a folder in your PATH

2. **Install the service**:

   ```cmd
   nssm install DealHunter "C:\Users\Rqzbeh\Desktop\Deal_Hunter-rqzbeh\run_deal_hunter.bat"
   ```

3. **Configure the service**:

   - Open Services (services.msc)
   - Find "DealHunter" service
   - Right-click → Properties
   - Set Startup type to "Automatic"
   - Go to "Log On" tab → Select "Local System account"
   - Check "Allow service to interact with desktop" (if needed for initial setup)

4. **Start the service**:

   ```cmd
   net start DealHunter
   ```

### Option 2: Use Task Scheduler (Alternative)

1. **Create a scheduled task**:

   - Open Task Scheduler
   - Create new task
   - Set to run as SYSTEM or your user
   - Set to run "At system startup"
   - Action: Start program → `C:\Users\Rqzbeh\Desktop\Deal_Hunter-rqzbeh\run_deal_hunter.bat`

### Option 3: Manual VPS Mode

If you must use RDP, run the script with:

```cmd
set VPS_MODE=true
python main.py
```

## Monitoring Logs

- **Log file**: `deal_hunter.log` in the script directory
- **Console output**: Also visible in the service logs
- **Telegram notifications**: If configured, you'll get alerts on your phone

## Troubleshooting

### Script Stops on RDP Disconnect

- Use Option 1 (Windows Service) above
- The service runs independently of your RDP session

### No Logs Appearing

- Check that `deal_hunter.log` exists in the script directory
- Ensure the script has write permissions
- Look for any error messages in the console/service logs

### Browser Issues

- The script now uses a dedicated Chrome profile in `AppData\Local\DealHunter\ChromeProfile`
- Previous Edge profile conflicts are avoided

## Environment Variables

Set these in your environment or in the batch file:

```cmd
set VPS_MODE=true
set TELEGRAM_BOT_TOKEN=your_bot_token_here
set TELEGRAM_CHAT_ID=your_chat_id_here
set ALERT_THRESHOLD_PERCENT=40
set BASKET_THRESHOLD_PERCENT=50
