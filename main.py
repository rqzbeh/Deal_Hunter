"""
Deal Hunter - clean rewrite (Chromium only)
"""

import sys
import asyncio
import os
import json
import time
import logging
import re
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional
from playwright.async_api import async_playwright, Page

try:
    import requests
except Exception:
    requests = None

logging.basicConfig(filename='deal_hunter.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(console_formatter)
logging.getLogger().addHandler(console)

# Use Chrome as the Playwright Chromium channel; we use 'chrome' to use system-installed Chrome
BROWSER_CHANNEL = 'chrome'
USE_HEADLESS_ENV = os.environ.get('VPS_MODE', '').lower() == 'true'

# Timeouts & dynamic behaviour - these are dynamically adjusted per domain based on failures
DEFAULT_PAGE_TIMEOUT = 60000
BASE_SELECTOR_TIMEOUT = 1200
ALERT_THRESHOLD_PERCENT = 40.0
BASKET_THRESHOLD_PERCENT = 50.0
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
TELEGRAM_ENABLED = (TELEGRAM_BOT_TOKEN is not None and TELEGRAM_CHAT_ID is not None and requests is not None)

# Health-check config
FAILURE_THRESHOLD_RESTART = 5
HEALTH_CHECK_INTERVAL = 60  # seconds

# Checkout automation defaults - we try to be helpful: auto-checkout is allowed, but auto-place order is off by default
# To keep this repo minimal, these are constants (no need for env flags):
ALLOW_AUTO_CHECKOUT = True
ALLOW_AUTO_PLACE_ORDER = False

# AI-only Groq integration (no heuristics, no provider selection)
GROQ_API_KEY = os.environ.get('GROQ_API_KEY') or os.environ.get('LLM_API_KEY')  # fallback env name
LLM_MODEL = 'llama-3.1-8b-instant'
ENABLE_LLM_AUTODETECT = bool(GROQ_API_KEY)
ENABLE_TELEGRAM_APPROVAL = TELEGRAM_ENABLED

# LLM cache file - basic per-URL caching to avoid repeated LLM calls
LLM_CACHE_FILE = 'llm_cache.json'


def load_llm_cache() -> Dict[str, dict]:
    try:
        if not os.path.exists(LLM_CACHE_FILE):
            return {}
        with open(LLM_CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_llm_cache(cache: Dict[str, dict]):
    try:
        with open(LLM_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_user_data_dir() -> str:
    path = Path(os.environ.get('USERPROFILE', str(Path.home()))) / 'AppData' / 'Local' / 'DealHunter' / 'ChromeProfile'
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def compute_selector_timeout(domain_delays: Dict[str, float], page_failures: Optional[Dict[str, int]], domain: str, url: str) -> int:
    """Compute a per-url selector wait timeout in milliseconds based on domain delay and failure count to balance speed and reliability."""
    base = BASE_SELECTOR_TIMEOUT
    delay_sec = float(domain_delays.get(domain, 0.5)) if domain_delays else 0.5
    failures = page_failures.get(url, 0) if page_failures else 0
    # Compose a timeout: base + penalty per failure + scale based on domain delay
    timeout = int(max(400, min(5000, base + failures * 250 + int(delay_sec * 200))))
    return timeout


def persian_to_english_digits(text: str) -> str:
    persian = '۰۱۲۳۴۵۶۷۸۹'
    english = '0123456789'
    return text.translate(str.maketrans(persian, english))


def parse_price(text: str):
    if not text:
        return None
    text = persian_to_english_digits(text)
    text = re.sub(r'[\s,\.]', '', text)
    m = re.search(r'\d+', text)
    if m:
        try:
            return float(m.group())
        except Exception:
            return None
    return None


async def send_telegram(message: str):
    if not TELEGRAM_ENABLED or not requests:
        return
    try:
        post = requests.post
        await asyncio.to_thread(post, f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage', data={'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'})
    except Exception as e:
        logging.warning(f'Failed to send Telegram: {e}')


async def wait_for_telegram_approval(code: str, timeout: int = 60) -> bool:
    if not TELEGRAM_ENABLED or not requests:
        return False
    try:
        url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates'
        end = time.time() + timeout
        offset = None
        while time.time() < end:
            params = {'timeout': 1}
            if offset:
                params['offset'] = offset
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            j = r.json()
            if not j.get('ok'):
                await asyncio.sleep(1)
                continue
            for update in j.get('result', []):
                offset = update['update_id'] + 1
                msg = update.get('message') or update.get('edited_message')
                if not msg:
                    continue
                if str(msg.get('chat', {}).get('id')) != str(TELEGRAM_CHAT_ID):
                    continue
                text = msg.get('text', '')
                if text and code in text:
                    return True
            await asyncio.sleep(1)
        return False
    except Exception as e:
        logging.warning(f'Telegram approval check failed: {e}')
        return False





async def extract_relevant_html(page: Page, max_chars: int = 4000) -> str:
    """Top-level snippet extractor for LLM prompts.
    Tries to capture a product container or a relevant element first, otherwise returns trimmed page content."""
    try:
        # Try to detect typical product containers and return focused inner HTML
        selectors = ['[itemtype]','[itemprop=offers]','[itemprop=price]','[data-product]','.product','[class*="product"]','[id*="product"]','main','article']
        for sel in selectors:
            try:
                el = await page.query_selector(sel)
                if el:
                    html = await el.inner_html()
                    tag = await el.evaluate('e => e.tagName')
                    snippet = f'<{tag}>\n' + (html[:max_chars])
                    return snippet
            except Exception:
                continue
        # fallback to price label quick snippet
        priceEl = await page.query_selector('[class*="price"], [id*="price"], [data-price]')
        if priceEl:
            snippet = await priceEl.evaluate('e => e.outerHTML')
            return snippet[:max_chars]
        content = await page.content()
        return content[:max_chars]
    except Exception as e:
        logging.warning(f'extract_relevant_html error: {e}')
        try:
            return (await page.content())[:max_chars]
        except Exception:
            return ''


def call_llm(prompt: str, system: Optional[str] = None) -> str:
    """Groq chat completion (llama-3.1-8b-instant). Returns assistant content or ''."""
    if not GROQ_API_KEY or not requests:
        logging.warning('GROQ_API_KEY missing or requests unavailable')
        return ''
    try:
        url = 'https://api.groq.com/openai/v1/chat/completions'
        headers = {
            'Authorization': f'Bearer {GROQ_API_KEY}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': LLM_MODEL,
            'messages': [
                {'role': 'system', 'content': system or 'You output ONLY raw JSON with CSS selectors.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.0,
            'max_tokens': 800
        }
        resp = requests.post(url, headers=headers, json=data, timeout=45)
        resp.raise_for_status()
        j = resp.json()
        if 'choices' in j and j['choices']:
            msg = j['choices'][0].get('message', {})
            if isinstance(msg, dict):
                return msg.get('content', '')
        return ''
    except Exception as e:
        logging.warning(f'Groq call failed: {e}')
        return ''


async def llm_detect_selectors(page: Page, url: str) -> Dict[str, str]:
    """AI-only selector detection: returns any of price_selector, add_selector, checkout_selector, payment_indicator, confidence."""
    try:
        html = await extract_relevant_html(page)
        prompt = (
            "Return ONLY JSON with keys: price_selector, add_selector, checkout_selector, payment_indicator, confidence.\n"
            "Rules:\n- JSON only (no markdown).\n- Minimal stable CSS (prefer id > attribute > short class).\n"
            "- payment_indicator: selector visible ONLY on payment page; empty if unknown.\n- Empty string for unknown selectors.\n"
            "Example HTML: <span id=\"price\">1,234</span><button id=\"addToCart\">افزودن به سبد خرید</button><a class=\"checkout-btn\">پرداخت</a>\n"
            "Example Output: {\"price_selector\":\"#price\",\"add_selector\":\"#addToCart\",\"checkout_selector\":\".checkout-btn\",\"payment_indicator\":\"\",\"confidence\":0.95}\n"
            f"URL: {url}\nHTML_SNIPPET:{html[:4000]}\n"
        )
        cache_key = f'selectors:{url}'
        cache = load_llm_cache()
        cached = cache.get(cache_key)
        if isinstance(cached, dict) and cached:
            return cached
        model_output = call_llm(prompt)
        logging.info(f'LLM selectors raw output for {url}: {model_output[:1000]}')
        if not model_output:
            return {}
        try:
            import json as _json
            start = model_output.find('{')
            if start >= 0:
                j = _json.loads(model_output[start:])
                valid = {}
                for k, sel in j.items():
                    if not isinstance(sel, str) or not sel or '//' in sel:
                        continue
                    try:
                        if await page.query_selector(sel):
                            valid[k] = sel
                    except Exception:
                        continue
                if valid:
                    cache[cache_key] = valid
                    save_llm_cache(cache)
                    logging.info(f'LLM detected selectors for {url}: {valid}')
                return valid
        except Exception:
            logging.warning('LLM output not valid JSON for selectors')
            return {}
    except Exception as e:
        logging.warning(f'AI selector detect failed: {e}')
    return {}


async def get_price_with_page(page: Page, url: str, selector: str, selector_timeout: int = BASE_SELECTOR_TIMEOUT):
    try:
        if not page.url or page.url.rstrip('/') != url.rstrip('/'):
            await page.goto(url, timeout=60000)
        else:
            try:
                await page.reload(timeout=60000)
            except Exception:
                await page.goto(url, timeout=60000)
        await page.wait_for_load_state('networkidle')
        await page.wait_for_timeout(selector_timeout)
        el = await page.query_selector(selector)
        if not el:
            return None
        text = await el.inner_text()
        return parse_price(text)
    except Exception as e:
        logging.warning(f'get_price_with_page error for {url}: {e}')
        return None


async def add_to_basket_with_page(page: Page, url: str, selector: str, selector_timeout: int = BASE_SELECTOR_TIMEOUT) -> bool:
    try:
        if not page.url or page.url.rstrip('/') != url.rstrip('/'):
            await page.goto(url, timeout=60000)
        else:
            try:
                await page.reload(timeout=60000)
            except Exception:
                await page.goto(url, timeout=60000)
        await page.wait_for_load_state('networkidle')
        el = await page.query_selector(selector)
        if not el:
            return False
        await el.scroll_into_view_if_needed()
        # Fast click: use evaluate to avoid potential page event overheads
        try:
            await page.evaluate('(el) => el.click()', el)
        except Exception:
            try:
                await el.click(timeout=selector_timeout)
            except Exception:
                pass
        await page.wait_for_timeout(selector_timeout)
        return True
    except Exception as e:
        logging.warning(f'add_to_basket_with_page error for {url}: {e}')
        return False


async def attempt_checkout_with_page(page: Page, product: dict, playwright_instance=None, browser_context=None, selector_timeout: int = BASE_SELECTOR_TIMEOUT):
    """AI-only checkout: click add_selector then checkout_selector; stop when payment indicator or shaparak domain detected."""
    try:
        if not page.url or product['url'].rstrip('/') != page.url.rstrip('/'):
            await page.goto(product['url'], timeout=60000)
            await page.wait_for_load_state('networkidle')
        if product.get('add_selector'):
            try:
                el = await page.query_selector(product['add_selector'])
                if el:
                    try:
                        await page.evaluate('(e)=>e.click()', el)
                    except Exception:
                        await el.click(timeout=selector_timeout)
                    await page.wait_for_timeout(selector_timeout)
            except Exception as e:
                logging.warning(f'Add click failed: {e}')
        if product.get('checkout_selector'):
            try:
                elc = await page.query_selector(product['checkout_selector'])
                if elc:
                    try:
                        await page.evaluate('(e)=>e.click()', elc)
                    except Exception:
                        await elc.click(timeout=selector_timeout)
                    await page.wait_for_load_state('networkidle')
                    await page.wait_for_timeout(1200)
            except Exception as e:
                logging.warning(f'Checkout click failed: {e}')
        indicator = product.get('payment_indicator', '')
        reached = False
        if indicator:
            try:
                await page.wait_for_selector(indicator, timeout=15000)
                reached = True
            except Exception:
                reached = False
        if not reached and 'shaparak' in (page.url or ''):
            reached = True
        if reached:
            product['reached_payment'] = True
            fname = f'payment_stage_{product.get("title","product").replace(" ","_")}_{int(time.time())}.png'
            try:
                await page.screenshot(path=fname, full_page=True)
                logging.info(f'Payment stage screenshot saved: {fname}')
                await send_telegram(f'Payment stage for {product.get("title")}: {page.url}')
            except Exception as e:
                logging.warning(f'Payment screenshot failed: {e}')
            if playwright_instance and browser_context:
                try:
                    await open_headful_view(playwright_instance, browser_context, page.url)
                except Exception as e:
                    logging.warning(f'Headful open failed: {e}')
        return reached
    except Exception as e:
        logging.warning(f'AI checkout error: {e}')
        return False
    # (legacy checkout code removed)


async def check_domain(domain: str, prods: List[dict], pages: Dict[str, Page], domain_delays: Dict[str, float], page_failures: Optional[Dict[str, int]] = None, playwright_instance=None, browser_context=None):
    delay = domain_delays.get(domain, 0.5)
    logging.info(f'Checking {len(prods)} products for {domain}, delay {delay:.1f}s')
    failures = 0
    for idx, product in enumerate(prods):
        if idx > 0:
            await asyncio.sleep(0.5)
        url = product['url']
        selector = product['selector']
        page = pages.get(url)
        # If we don't have a selector and LLM detection is enabled, try to auto detect selectors
        if (not product.get('selector')) and ENABLE_LLM_AUTODETECT and page and not product.get('llm_tried', False):
            try:
                detected = await llm_detect_selectors(page, url)
                if detected:
                    sel = detected.get('price_selector')
                    addsel = detected.get('add_selector')
                    if sel:
                        product['selector'] = sel
                    if addsel:
                        product['add_selector'] = addsel
                    product['llm_tried'] = True
                    logging.info(f'LLM detected selectors for {url}: {detected}')
            except Exception as e:
                logging.warning(f'LLM detection failed for {url}: {e}')
        price = None
        selector_timeout = compute_selector_timeout(domain_delays, page_failures, domain, url)
        if page:
            price = await get_price_with_page(page, url, selector, selector_timeout=selector_timeout)
        if price is None:
            failures += 1
            if page_failures is not None:
                page_failures[url] = page_failures.get(url, 0) + 1
            logging.warning(f'Failed to get price for {url}')
        else:
            logging.info(f"{product.get('title','')} price {int(price):,} (was {int(product['price']):,})")
            if price < product.get('price', float('inf')):
                old = product.get('price', 0)
                pct = (old - price) / old * 100 if old > 0 else 100
                if pct >= BASKET_THRESHOLD_PERCENT and product.get('add_selector') and not product.get('added'):
                    added = False
                    if page:
                        # When a promising deal is detected, reduce recheck interval for this domain for speed
                        try:
                            domain_delays[domain] = max(0.05, domain_delays.get(domain, 0.1))
                        except Exception:
                            pass
                        # Open headful view so the user can monitor payment steps (e.g., Shaparak OTP)
                        if playwright_instance and browser_context:
                            try:
                                await open_headful_view(playwright_instance, browser_context, product['url'])
                            except Exception as e:
                                logging.warning(f'Failed to open GUI for {product["url"]}: {e}')
                        added = await add_to_basket_with_page(page, url, product['add_selector'], selector_timeout=selector_timeout)
                    product['added'] = bool(added)
                    await send_telegram(f"Deal alert: {product.get('title','')} {old:,} → {price:,} added_to_basket={added} {url}")
                    # AI-only checkout attempt (stop at payment stage)
                    if ALLOW_AUTO_CHECKOUT and product.get('checkout_selector'):
                        try:
                            if page:
                                await attempt_checkout_with_page(page, product, playwright_instance=playwright_instance, browser_context=browser_context, selector_timeout=selector_timeout)
                        except Exception as e:
                            logging.warning(f'AI checkout attempt failed for {url}: {e}')
                product['price'] = price
                product.setdefault('history', []).append({'price': price, 'timestamp': time.time()})
                if len(product['history']) > 100:
                    product['history'] = product['history'][-100:]
    if failures == 0:
        domain_delays[domain] = max(0.1, delay * 0.95)
    else:
        domain_delays[domain] = min(30.0, delay + failures * 0.5)


async def interactive_setup(context):
    page = await context.new_page()
    result = []
    print('AI setup — enter product URLs (type done to finish).')
    if not ENABLE_LLM_AUTODETECT:
        print('ERROR: GROQ_API_KEY missing. Set and rerun --setup.')
        await page.close()
        return result
    while True:
        url = input('Product URL (or done): ').strip()
        if url.lower() == 'done':
            break
        try:
            await page.goto(url, timeout=60000)
            await page.wait_for_load_state('networkidle')
        except Exception as e:
            logging.warning(f'Navigation failed for {url}: {e}')
            continue
        detected = await llm_detect_selectors(page, url)
        price_sel = detected.get('price_selector')
        add_sel = detected.get('add_selector')
        checkout_sel = detected.get('checkout_selector')
        payment_ind = detected.get('payment_indicator')
        price_val = None
        if price_sel:
            try:
                elp = await page.query_selector(price_sel)
                if elp:
                    price_val = parse_price(await elp.inner_text())
            except Exception:
                price_val = None
        if price_val is None:
            logging.warning(f'AI could not parse price for {url}; skipping.')
            continue
        title = await page.title()
        product_entry = {
            'url': url,
            'selector': price_sel,
            'price': price_val,
            'add_selector': add_sel,
            'checkout_selector': checkout_sel,
            'payment_indicator': payment_ind,
            'title': title,
            'added': False,
            'history': [],
            'llm_tried': True
        }
        result.append(product_entry)
        logging.info(f'Added (AI) {title} @ {int(price_val):,}')
    await page.close()
    return result


async def open_headful_view(playwright_instance, source_context, url: str, ttl: int = 300):
    """Spawn a headful browser context with the same storage state so the user can monitor a page.
       The new headful browser will stay open for `ttl` seconds (default 5 minutes) then close.
    """
    try:
        state = await source_context.storage_state()
        # Launch a new headful browser and create a context with the saved storage_state
        br = await playwright_instance.chromium.launch(headless=False, channel=BROWSER_CHANNEL)
        ctx = await br.new_context(storage_state=state)
        p = await ctx.new_page()
        await p.goto(url, timeout=60000)
        # Do not block; close after ttl seconds
        async def _close_later(pctx, brw, delay):
            await asyncio.sleep(delay)
            try:
                await pctx.close()
            except Exception:
                pass
            try:
                await brw.close()
            except Exception:
                pass
        asyncio.create_task(_close_later(ctx, br, ttl))
        logging.info(f'Opened headful browser view for {url} (ttl={ttl}s)')
    except Exception as e:
        logging.warning(f'Failed to open headful view: {e}')


async def main():
    # Config & mode
    is_setup = '--setup' in sys.argv
    headless = USE_HEADLESS_ENV and not is_setup
    user_data_dir = get_user_data_dir()

    products_file = Path('products.json')
    domain_delays_file = Path('domain_delays.json')

    products = []
    if products_file.exists() and not is_setup:
        with open(products_file, 'r', encoding='utf-8') as f:
            products = json.load(f)
            logging.info(f'Loaded {len(products)} products')

    if not products or is_setup:
        async with async_playwright() as p:
            context = await p.chromium.launch_persistent_context(user_data_dir, channel=BROWSER_CHANNEL, headless=False)
            products = await interactive_setup(context)
            await context.close()
            if products:
                with open(products_file, 'w', encoding='utf-8') as f:
                    json.dump(products, f, ensure_ascii=False, indent=2)
    if not products:
        logging.info('No products configured. Exiting')
        return

    domain_delays = {}
    if domain_delays_file.exists():
        with open(domain_delays_file, 'r', encoding='utf-8') as f:
            domain_delays = json.load(f)

    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(user_data_dir, channel=BROWSER_CHANNEL, headless=headless, args=['--no-sandbox'])
        pages: Dict[str, Page] = {}
        for prod in products:
            try:
                pg = await context.new_page()
                await pg.goto(prod['url'], timeout=60000)
                await pg.wait_for_load_state('networkidle')
                pages[prod['url']] = pg
            except Exception as e:
                logging.warning(f'Could not create page for {prod["url"]}: {e}')

        async def recreate_context_and_pages(old_context, pages_map):
            logging.info('Recreating browser context and pages due to persistent page failures...')
            try:
                for pg in list(pages_map.values()):
                    try:
                        await pg.close()
                    except Exception:
                        pass
                try:
                    await old_context.close()
                except Exception:
                    pass
            except Exception as e:
                logging.warning(f'Error while closing pages/context: {e}')
            # Create new context
            new_context = await p.chromium.launch_persistent_context(user_data_dir, channel=BROWSER_CHANNEL, headless=headless, args=['--no-sandbox'])
            new_pages = {}
            for prod in products:
                try:
                    pg = await new_context.new_page()
                    await pg.goto(prod['url'], timeout=60000)
                    await pg.wait_for_load_state('networkidle')
                    new_pages[prod['url']] = pg
                except Exception as e:
                    logging.warning(f'Could not create page for {prod["url"]} during recreate: {e}')
            return new_context, new_pages

        # Initialize per-page failure counters
        page_failures: Dict[str,int] = {prod['url']: 0 for prod in products}

        try:
            cycle = 0
            while True:
                cycle += 1
                logging.info(f'=== Monitoring cycle {cycle} ===')
                domains: Dict[str, List[dict]] = {}
                for prod in products:
                    d = urllib.parse.urlparse(prod['url']).netloc
                    domains.setdefault(d, []).append(prod)
                    domain_delays.setdefault(d, 0.5)
                tasks = [asyncio.create_task(check_domain(d, prods, pages, domain_delays, page_failures, p, context)) for d, prods in domains.items()]
                await asyncio.gather(*tasks)
                # Check for persistent failures to decide if we need to restart context
                max_fail = max(page_failures.values()) if page_failures else 0
                if max_fail >= FAILURE_THRESHOLD_RESTART:
                    logging.warning(f'Max page failures {max_fail} reached threshold {FAILURE_THRESHOLD_RESTART}; restarting browser context...')
                    context, pages = await recreate_context_and_pages(context, pages)
                    # reset failure counters
                    for k in list(page_failures.keys()):
                        page_failures[k] = 0
                with open(domain_delays_file, 'w', encoding='utf-8') as f:
                    json.dump(domain_delays, f, indent=2)
                with open(products_file, 'w', encoding='utf-8') as f:
                    json.dump(products, f, ensure_ascii=False, indent=2)
                await asyncio.sleep(max(min(domain_delays.values()) if domain_delays else 1, 0.1))
        finally:
            for pg in pages.values():
                try:
                    await pg.close()
                except Exception:
                    pass
            await context.close()


if __name__ == '__main__':
    asyncio.run(main())
