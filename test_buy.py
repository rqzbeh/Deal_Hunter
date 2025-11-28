"""
Test harness to exercise LLM detection and add-to-basket flows without placing an order.

Usage:
  python test_buy.py --url <product_url> --old-price <old_price> [--headful] [--force-drop]

- `--url` the product URL to test
- `--old-price` baseline price (used to compute discount percentage)
- `--headful` run with visible browser (useful to debug flows)
- `--force-drop` simulate a 70% price discount regardless of real page price

This script intentionally will not click the final "place order" button and avoids sending PII to any LLM.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
import os

from playwright.async_api import async_playwright

from main import (
    get_user_data_dir,
    llm_detect_selectors,
    get_price_with_page,
    add_to_basket_with_page,
    attempt_checkout_with_page,
    BROWSER_CHANNEL,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def run_test(url: str, old_price: float, headful: bool = True, force_drop: bool = False):
    async with async_playwright() as p:
        user_data_dir = get_user_data_dir()
        # Launch persistent profile using Playwright-provided Chromium
        # Avoid using the system 'chrome' channel in this test harness to keep execution portable
        context = await p.chromium.launch_persistent_context(user_data_dir=str(user_data_dir), headless=(not headful))
        page = await context.new_page()
        try:
            print(f'Opening page: {url}')
            await page.goto(url, timeout=60000)
            await page.wait_for_load_state('networkidle')

            print('Running LLM-based selector detection (if configured)...')
            selectors = await llm_detect_selectors(page, url)
            price_selector = selectors.get('price_selector') if selectors else None
            add_selector = selectors.get('add_selector') if selectors else None
            checkout_selector = selectors.get('checkout_selector') if selectors else None
            payment_indicator = selectors.get('payment_indicator') if selectors else None

            if not price_selector:
                print('LLM did not find a price selector — cannot proceed')
                return

            if not add_selector:
                print('LLM did not find add selector — cannot proceed')
                return

            # Get current price using detected selector
            current_price = None
            if price_selector:
                price_val = await get_price_with_page(page, url, price_selector)
                current_price = price_val
                print(f'Current price detected: {price_val}')
            else:
                print('No price selector found on the page')

            # If force_drop, simulate 70% off by recomputing current_price
            if force_drop and old_price:
                simulated_price = round(old_price * 0.3, 2)
                print(f'Forcing 70% discount: setting current price = {simulated_price}')
                current_price = simulated_price

            if current_price is None:
                print('No price data — cannot evaluate discount. Test stopping.')
                return

            pct_drop = ((old_price - current_price) / old_price) * 100.0 if old_price > 0 else 0
            print(f'Old price: {old_price}, Current price: {current_price}, Drop: {pct_drop:.2f}%')

            if pct_drop >= 70:
                print('Detected >=70% discount — attempting to add to cart (stopping before placing order)')
                if add_selector:
                    added = await add_to_basket_with_page(page, url, add_selector)
                    print(f'Add to basket attempted: {added}')
                else:
                    print('No add selector found — cannot add to cart automatically')

                # Prepare a minimal product dict as expected by attempt_checkout_with_page
                product = {
                    'url': url,
                    'title': (await page.title()) or 'product',
                    'price': old_price,
                    'add_selector': add_selector,
                    'checkout_selector': checkout_selector,
                    'payment_indicator': payment_indicator
                }

                print('Attempting to walk checkout flow (will stop at payment page)')
                try:
                    await attempt_checkout_with_page(page, product, playwright_instance=p, browser_context=context)
                except Exception as e:
                    print(f'Attempt_checkout failed: {e}')
            else:
                print('Price drop < 70% — no further action taken')
        finally:
            # Save a screenshot for debug
            out = 'test_output.png'
            try:
                await page.screenshot(path=out, full_page=True)
                print(f'Debug screenshot saved: {out}')
            except Exception:
                pass
            await context.close()


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True, help='Product URL')
    parser.add_argument('--old-price', required=True, type=float, help='Baseline price to compare against')
    parser.add_argument('--headful', action='store_true', help='Open a visible browser')
    parser.add_argument('--force-drop', action='store_true', help='Simulate 70% price drop')
    args = parser.parse_args()
    asyncio.get_event_loop().run_until_complete(run_test(args.url, args.old_price, headful=args.headful, force_drop=args.force_drop))


if __name__ == '__main__':
    cli()

