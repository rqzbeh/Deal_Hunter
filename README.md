# Deal Hunter

A Python script to monitor product prices on websites, automatically add items to basket on significant drops, and log changes. Uses async concurrency for efficient multi-site monitoring with anti-blocking measures.

## Features

- Click-based selector detection for easy setup
- Automatic price parsing (supports Persian digits)
- Concurrent price checking per domain
- Adaptive delays per domain (2-60s) based on success rates to optimize speed without blocking
- Basket automation on >40% price drops
- Data persistence (saves/loads products from JSON)
- Logging to deal_hunter.log
- Optional Telegram notifications (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars)

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

The script will automatically open Microsoft Edge with your user profile. It monitors prices concurrently within domains, with adaptive delays per domain that learn from success/failure rates (starting at 5s, adapting 2-60s) to find the optimal check frequency without triggering blocks. On >40% drops, it adds to basket once per product and sends Telegram notifications if configured.

## Telegram Setup (Optional)

Set environment variables for notifications:

- `TELEGRAM_BOT_TOKEN`: Your bot token from @BotFather
- `TELEGRAM_CHAT_ID`: Your chat ID (get from @userinfobot or bot API)

Install `requests` package if not present. Notifications are sent when adding products and on >40% price drops.

## Notes

- Groups products by domain for efficient concurrent checking
- Adapts check delays per domain based on reliability (decreases on success, increases on failures)
- Logs all price changes to deal_hunter.log
