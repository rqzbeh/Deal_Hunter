# Deal Hunter

A **Windows-only** Python script to monitor product prices on websites, automatically add items to basket on significant drops, and log changes. Uses async concurrency for efficient multi-site monitoring with anti-blocking measures.

## Features

- Click-based selector detection for easy setup
- Automatic price parsing (supports Persian digits)
- Concurrent price checking per domain
- Adaptive delays per domain (2-60s) based on success rates to optimize speed without blocking
- Basket automation on ≥50% price drops
- Data persistence (saves/loads products from JSON)
- Logging to deal_hunter.log
- Optional Telegram notifications (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars)
- Smart notifications: only notifies on price changes ≥40%

## Setup

1. Install dependencies:
   - Python 3.x
   - Packages: playwright, asyncio

2. Install Playwright browsers:

   ```powershell
   python -m playwright install msedge
   ```

   **Note**: The script uses your system-installed Microsoft Edge browser, not Playwright's bundled version. Make sure Microsoft Edge is installed on your system.

3. Close Microsoft Edge if it's running, so the script can use your profile with saved logins.

## Usage

Run the script:

```powershell
python main.py
```

- On first run or if no products.json exists, enter product URLs one by one
- For each product:
  - Click on the price element in the opened browser when prompted (price is auto-detected)
  - Choose if you want to set add to basket (y/n), and click on the button if yes
- Type 'done' when finished adding products
- Products are saved to products.json for future runs

On subsequent runs, you can load previous products or start fresh.

The script will automatically open Microsoft Edge with your user profile. It monitors prices concurrently within domains, with adaptive delays per domain that learn from success/failure rates (starting at 5s, adapting 2-60s) to find the optimal check frequency without triggering blocks. On ≥50% drops, it adds to basket once per product and sends Telegram notifications if configured. Telegram notifications are only sent for price changes ≥40% to reduce noise.

## Telegram Setup (Optional)

Set environment variables for notifications:

- `TELEGRAM_BOT_TOKEN`: Your bot token from @BotFather
- `TELEGRAM_CHAT_ID`: Your chat ID (get from @userinfobot or bot API)

Install `requests` package if not present. Notifications are sent when adding products and on price changes ≥40%.

## Notes

- Groups products by domain for efficient concurrent checking
- Adapts check delays per domain based on reliability (decreases on success, increases on failures)
- Logs all price changes to deal_hunter.log

## VPS/Headless Mode Setup

The script now supports **hybrid mode**: GUI setup with headless monitoring for optimal VPS deployment.

- **Setup Phase**: Uses GUI mode to allow clicking on price elements and add-to-basket buttons
- **Monitoring Phase**: Runs headless for efficient 24/7 operation without graphics acceleration

Set the environment variable for VPS mode:

```powershell
$env:VPS_MODE = "true"
```

Or the script automatically detects headless capability.

### Setup Process (GUI Mode)

1. **Interactive Setup**: Browser opens for visual element selection
2. **Click Selection**: Click on price elements and add-to-basket buttons when prompted
3. **Auto-Detection**: Prices are automatically parsed from selected elements
4. **Products Saved**: Configuration saved to products.json for headless monitoring

### Monitoring Process (Headless Mode)

- ✅ Runs without display after setup
- ✅ Faster, lower resource usage
- ✅ 24/7 continuous monitoring
- ✅ No GUI dependencies for production

### VPS Deployment

Perfect for VPS environments - use GUI locally for setup, then deploy headless for monitoring.
