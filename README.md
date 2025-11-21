# Deal Hunter

A Python script to monitor product prices on websites, automatically add items to basket on significant drops, and log changes. Uses async concurrency for efficient multi-site monitoring with anti-blocking measures.

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

   ```bash
   python -m playwright install msedge
   ```

3. Close Microsoft Edge if it's running, so the script can use your profile with saved logins.

## Usage

Run the script:

```bash
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

## Linux/VPS Compatibility

Currently, this script is designed for Windows and uses Microsoft Edge with a user profile. To run on Linux/VPS:

1. **Browser Options**: Replace Edge with Chromium or Firefox
   - Install Playwright browsers: `python -m playwright install chromium`
   - Modify the script to use `chromium` instead of `msedge` channel
   - Update user data directory path for Linux (e.g., `~/.config/chromium/`)

2. **Headless Mode**: For VPS without display:
   - Set `headless=True` in `launch_persistent_context()`
   - Note: Click-based selector detection won't work in headless mode - you'll need to manually specify selectors or run setup on a desktop first

3. **Path Changes**: Update Windows-specific paths:
   - Replace `C:\\Users\\{username}\\...` with Linux paths
   - Use `os.path.expanduser('~')` for cross-platform home directory

4. **Process Management**: The `taskkill` command is Windows-specific:
   - Replace with `pkill msedge` or `killall chromium` on Linux
   - Or remove this section if running in a clean VPS environment

Example modifications for Linux would involve changing the browser launch code to use Chromium and appropriate Linux paths.
