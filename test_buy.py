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
    detect_checkout_fields,
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
            if not price_selector:
                print('LLM did not find a price selector — falling back to heuristics')
                heuristics = ['[data-price]', '.price', '[class*="price"]', '#price', '.product-price', '.price-tag']
                for h in heuristics:
                    try:
                        el = await page.query_selector(h)
                        if el:
                            price_selector = h
                            break
                    except Exception:
                        continue

            if not add_selector:
                print('LLM did not find add selector — trying button heuristics')
                try:
                    # Search for nodes matching the exact text 'افزودن' using Playwright's text selector
                    try:
                        nodes = await page.query_selector_all('text=افزودن')
                        if nodes and len(nodes) > 0:
                            print(f'Found {len(nodes)} text=افزودن nodes; first outerHTML: {await nodes[0].evaluate("e=>e.outerHTML")[:200]}')
                    except Exception:
                        pass
                    try:
                        nodes2 = await page.query_selector_all('text="افزودن به سبد"')
                        if nodes2 and len(nodes2) > 0:
                            print(f'Found {len(nodes2)} text="افزودن به سبد" nodes; first outerHTML: {await nodes2[0].evaluate("e=>e.outerHTML")[:200]}')
                    except Exception:
                        pass
                    content = await page.content()
                    # Show if common Persian words for 'add' appear in the page content
                    for keyword in ['خرید', 'افزودن', 'سبد', 'خریدن', 'خرید', 'افزودن به سبد']:
                        if keyword in content:
                            idx = content.find(keyword)
                            snippet = content[max(0, idx-120):idx+120]
                            print(f"Found page keyword '{keyword}' near: ...{snippet}...")
                    # Print out some candidate buttons to help debug
                    cands = await page.query_selector_all('button, input[type=submit], a')
                    candidates_count = len(cands)
                    print(f'Found {candidates_count} candidate buttons/links; inspecting...')
                    # Print first 20 candidates debug info
                    for i, c in enumerate(cands):
                        try:
                            txt = (await c.inner_text()).lower() if c else ''
                            href = await c.get_attribute('href') or ''
                            data_pid = await c.get_attribute('data-product-id') or ''
                            outer = await c.evaluate('(el)=>el.outerHTML')
                            text_content = await c.evaluate('(el) => el.textContent')
                            if text_content and 'افزودن' in text_content:
                                print(f'Found "افزودن" in textContent of candidate #{i}: textContent={text_content[:120]}')
                            print(f'Cand #{i}: text={txt[:80]} href={href} id={await c.get_attribute("id") or ""} class={await c.get_attribute("class") or ""} data-product-id={data_pid} outer={outer[:200]}')
                        except Exception:
                            txt = ''
                        if any(k in txt for k in ['add', 'cart', 'basket', 'سبد', 'خرید', 'افزودن']):
                            print(f'Matching candidate text: {txt} attr_id={attr_id} attr_cls={attr_cls}')
                            attr_id = await c.get_attribute('id') or ''
                            attr_cls = await c.get_attribute('class') or ''
                            if attr_id:
                                add_selector = f'#{attr_id}'
                            elif attr_cls:
                                cls = '.'.join([s for s in attr_cls.split() if s])
                                add_selector = f'button.{cls}' if cls else 'button'
                            else:
                                add_selector = 'button'
                            break
                except Exception:
                    pass

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
                product = {'url': url, 'title': (await page.title()) or 'product', 'price': old_price, 'add_selector': add_selector, 'checkout_enabled': True, 'checkout_selectors': {}}

                # Load checkout info if present — we do not send this to LLM or auto-place
                checkout_info = {}
                if Path('checkout_info.json').exists():
                    with open('checkout_info.json', 'r', encoding='utf-8') as cf:
                        checkout_info = json.load(cf)

                print('Attempting to walk checkout flow (will not place order)')
                try:
                    await attempt_checkout_with_page(page, product, checkout_info, allow_place=False, playwright_instance=p, browser_context=context)
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

