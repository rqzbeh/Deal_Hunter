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

# LLM integration environment
LLM_PROVIDER = os.environ.get('LLM_PROVIDER', '').lower()  # 'groq' or 'openai' or ''
LLM_API_KEY = os.environ.get('LLM_API_KEY')
ENABLE_LLM_AUTODETECT = bool(LLM_PROVIDER and LLM_API_KEY)
ENABLE_LLM_FIELD_MAP = bool(LLM_PROVIDER and LLM_API_KEY)
# Safety: Do NOT send checkout_info (PII) to LLM. This is intentionally disabled.
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
    """Call configured LLM provider with a simple prompt. Returns model text output as string.
    For now, supports Groq and OpenAI compatible API styles. If no provider configured, returns ''"""
    if not LLM_PROVIDER or not LLM_API_KEY or not requests:
        logging.info('LLM not configured or requests not available')
        return ''

    # Note: extract_relevant_html is implemented at top-level to avoid nested functions
    try:
        headers = {'Authorization': f'Bearer {LLM_API_KEY}', 'Content-Type': 'application/json'}
        if LLM_PROVIDER == 'groq':
            # Groq inference endpoint example (not tested) - user should set GROQ specifics
            url = os.environ.get('GROQ_ENDPOINT', 'https://api.groq.com/v1/inference')
            data = {'prompt': prompt, 'max_tokens': 2048}
            resp = requests.post(url, headers=headers, json=data, timeout=20)
            resp.raise_for_status()
            try:
                j = resp.json()
                # Groq returns `output` array or `choices` similar to OpenAI - prefer `output` then `choices`
                if 'output' in j and isinstance(j['output'], list) and len(j['output'])>0 and isinstance(j['output'][0], dict):
                    # Groq may return {'type': 'message', 'content': '...'} objects in output
                    out = j['output'][0]
                    if 'content' in out:
                        return out['content']
                    # fallback - join stringified output
                    return ' '.join(str(x) for x in j['output'])
                if 'choices' in j and len(j['choices'])>0:
                    c = j['choices'][0]
                    if isinstance(c, dict) and 'text' in c:
                        return c['text']
                    return str(c)
                # fallback
            except Exception as e:
                logging.info(f'Could not parse Groq JSON response: {e}; returning text fallback')
                return resp.text
        elif LLM_PROVIDER == 'openai' or LLM_PROVIDER == 'oai':
            # OpenAI completion (chat/completions) example
            url = os.environ.get('OPENAI_ENDPOINT', 'https://api.openai.com/v1/completions')
            data = {
                'model': os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'),
                'prompt': prompt,
                'max_tokens': 512,
                'temperature': 0.0
            }
            resp = requests.post(url, headers=headers, json=data, timeout=30)
            resp.raise_for_status()
            j = resp.json()
            # Extract a text depending on API type
            if 'choices' in j and len(j['choices']) > 0:
                choice = j['choices'][0]
                # Chat completion style
                if isinstance(choice, dict):
                    if 'message' in choice and isinstance(choice['message'], dict):
                        return choice['message'].get('content', '')
                    return choice.get('text', '')
                return str(choice)
            if 'text' in j:
                return j.get('text', '')
            # Some OpenAI endpoints return choices[0].message.content; try to flatten
            if 'messages' in j and isinstance(j['messages'], list) and len(j['messages'])>0:
                m = j['messages'][0]
                if isinstance(m, dict) and 'content' in m:
                    return m['content']
            return json.dumps(j)
        else:
            logging.warning(f'LLM_PROVIDER {LLM_PROVIDER} not supported')
            return ''
    except Exception as e:
        logging.warning(f'LLM call failed: {e}')
        return ''
    return ''


async def llm_detect_selectors(page: Page, url: str) -> Dict[str, str]:
    """Use LLM to detect price and add-to-basket selectors given a product page. Returns selectors dict.
    We pass page HTML and a short prompt and expect a JSON-like answer. This is a best-effort attempt and requires configured LLM.
    """
    try:
        html = await extract_relevant_html(page)
        prompt = (
            "You are a web scraper developer. Given the HTML snippet and the URL, return a JSON object with these keys: \n"
            "{\"price_selector\": \"<css>\", \"add_selector\": \"<css>\", \"confidence\": 0.0}\n"
            "Rules: \n"
            "- Return only valid JSON and nothing else. Do NOT include markdown, code fences, or extra commentary.\n"
            "- Provide a CSS selector that uniquely selects the element whenever possible. Prefer id (e.g., #price) when available, then attribute selectors (e.g., input[name=\'price\']), then concise class selectors (e.g., .price).\n"
            "- If add-to-basket button is not found, return an empty string for add_selector.\n"
            "- Keep the selectors short and stable (avoid auto-generated classes when possible).\n"
            "Examples:\n"
            "Input: <span id=\"price\">$9.99</span><button class=\"btn add\">Add</button>\nOutput: {\"price_selector\":\"#price\",\"add_selector\":\"button.btn.add\",\"confidence\":0.96}\n"
            "Input: <div class=\"price-tag\"><span class=\"value\">9,999</span></div>\nOutput: {\"price_selector\":\".price-tag .value\",\"add_selector\":\"\",\"confidence\":0.75}\n"
            "Example for a Persian site: Input: <button class=\"btn btn-primary\">\u0628\u062E\u0634\u06CC\u062F\u0627\u062F</button><span class=\"price\">1,234</span>\nOutput: {\"price_selector\":\".price\",\"add_selector\":\"button.btn.btn-primary\",\"confidence\":0.90}\n"
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
        # Attempt to extract JSON from model_output
        try:
            import json as _json
            # find first '{' and parse JSON substring (best-effort)
            start = model_output.find('{')
            if start >= 0:
                j = _json.loads(model_output[start:])
                # Validate selectors exist on the page
                candidates = {k: v for k, v in j.items() if isinstance(v, str)}
                valid = {}
                for k, sel in candidates.items():
                    try:
                        if not sel:
                            continue
                        # Reject obvious non-css selectors (e.g., XPath candidate using //)
                        if '//' in sel or sel.strip().startswith('//'):
                            continue
                        q = await page.query_selector(sel)
                        if q:
                            count = 1
                            try:
                                count = len(await page.query_selector_all(sel))
                            except Exception:
                                pass
                            # Accept selectors that match at least 1 element (prefer unique)
                            if count >= 1:
                                valid[k] = sel
                    except Exception:
                        continue
                # If we have at least one valid selector, cache and return
                if valid:
                    cache[cache_key] = valid
                    try:
                        save_llm_cache(cache)
                    except Exception:
                        pass
                    logging.info(f'LLM detected selectors for {url}: {valid}')
                return valid
        except Exception:
            logging.warning('LLM output not valid JSON for selectors')
            return {}
    except Exception as e:
        logging.warning(f'llm_detect_selectors failed: {e}')
    return {}


async def llm_map_checkout_fields(page: Page, checkout_info: dict) -> Dict[str, str]:
    """Use an LLM to map user `checkout_info` keys (like name/email/phone/address) to CSS selectors on the given page.
             Returns mapping of key->selector. NOTE: this script does not send `checkout_info` (PII) to the LLM.
    """
    try:
        html = await page.content()
        safe_info = {}
        prompt = (
            f"You are a web scraper developer. Map these fields {list(safe_info.keys())} to CSS selectors on this page.\n"
            "Return only valid JSON and nothing else. Do not include markdown or extra text.\n"
            "If a field cannot be detected, return an empty string for that field.\n"
            "Examples:\n"
            "HTML: <input id=\"name\" name=\"name\"/><input id=\"email\" name=\"email\"/>\n"
            "Output: {\"name\":\"#name\",\"email\":\"#email\"}\n"
            "HTML: <input class=\"first_name\"/><input class=\"last_name\"/>\n"
            "Output: {\"first_name\":\".first_name\",\"last_name\":\".last_name\"}\n"
            "HTML: <div><label>Email</label><input name=\"email\"></div>\n"
            "Output: {\"email\":\"input[name=\\\"email\\\"]\"}\n"
            f"HTML:\n{html[:4000]}\n"
        )
        model_output = call_llm(prompt)
        logging.info(f'LLM field mapping raw output for {getattr(page, "url", "unknown")} : {model_output[:1000]}')
        if not model_output:
            return {}
        try:
            import json as _json
            page_url = page.url if (page and getattr(page, 'url', None)) else ''
            cache_key = f'fields:{page_url}' if page_url else 'fields:unknown'
            cache = load_llm_cache()
            start = model_output.find('{')
            if start >= 0:
                j = _json.loads(model_output[start:])
                # Validate selectors exist on page
                valid = {}
                for k, v in j.items():
                    if not isinstance(v, str) or not v.strip():
                        continue
                    try:
                        if '//' in v:
                            continue
                        q = await page.query_selector(v)
                        if q:
                            valid[k] = v
                    except Exception:
                        continue
                if valid:
                    cache[cache_key] = valid
                    try:
                        save_llm_cache(cache)
                    except Exception:
                        pass
                    logging.info(f'LLM detected field mapping for {getattr(page, "url", "unknown")} : {valid}')
                return valid
        except Exception:
            logging.warning('LLM output not valid JSON for field mapping')
            return {}
    except Exception as e:
        logging.warning(f'llm_map_checkout_fields failed: {e}')
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


async def detect_checkout_fields(page: Page) -> Dict[str, str]:
    """Heuristics to detect common checkout fields (name, email, phone, address, zip, city).
       Returns a mapping of field names to CSS selectors when detected."""
    mapping = {}
    try:
        inputs = await page.query_selector_all('input, textarea, select')
        for inp in inputs:
            try:
                name = await inp.get_attribute('name') or ''
                _id = await inp.get_attribute('id') or ''
                placeholder = await inp.get_attribute('placeholder') or ''
                aria = await inp.get_attribute('aria-label') or ''
                attributes = ' '.join([name, _id, placeholder, aria]).lower()
                selector = ''
                if _id:
                    selector = f'#{_id}'
                elif name:
                    selector = f'input[name="{name}"]'
                else:
                    # fallback to xpath-based CSS: nth-child not ideal, so skip
                    continue

                if any(k in attributes for k in ['email', 'e-mail']):
                    mapping['email'] = selector
                elif any(k in attributes for k in ['phone', 'mobile', 'tel']):
                    mapping['phone'] = selector
                elif any(k in attributes for k in ['first', 'firstname', 'name']):
                    mapping['name'] = selector
                elif any(k in attributes for k in ['address', 'street', 'addr']):
                    mapping['address'] = selector
                elif any(k in attributes for k in ['zip', 'postal']):
                    mapping['zip'] = selector
                elif any(k in attributes for k in ['city', 'town']):
                    mapping['city'] = selector
            except Exception:
                continue
    except Exception as e:
        logging.warning(f'detect_checkout_fields error: {e}')
    return mapping


async def attempt_checkout_with_page(page: Page, product: dict, checkout_info: dict, allow_place: bool = False, playwright_instance=None, browser_context=None, selector_timeout: int = BASE_SELECTOR_TIMEOUT):
    """Attempt to walk through the checkout flow using known selectors in product['checkout_selectors'].
       Defaults to partial flow (stop before place order) unless `allow_place` is True.
    """
    if 'checkout_selectors' not in product:
        logging.info('No checkout selectors configured for product')
        return False
    selectors = product['checkout_selectors']
    try:
        # Ensure product page loaded
        if not page.url or product['url'].rstrip('/') != page.url.rstrip('/'):
            await page.goto(product['url'], timeout=60000)
            await page.wait_for_load_state('networkidle')

        # Add to basket if necessary
        if product.get('add_selector'):
            try:
                el = await page.query_selector(product['add_selector'])
                if el:
                    try:
                        await page.evaluate('(el)=>el.click()', el)
                    except Exception:
                        await el.click(timeout=selector_timeout)
                await page.wait_for_timeout(selector_timeout)
            except Exception as e:
                logging.warning(f'add to basket click failed: {e}')

        # Click checkout start
        if selectors.get('checkout_button'):
            try:
                elch = await page.query_selector(selectors['checkout_button'])
                if elch:
                    try:
                        await page.evaluate('(el)=>el.click()', elch)
                    except Exception:
                        await elch.click(timeout=selector_timeout)
                await page.wait_for_load_state('networkidle')
                await page.wait_for_timeout(1000)
            except Exception as e:
                logging.warning(f'Failed clicking checkout_button: {e}')

        # Fill address/fields
        fields = selectors.get('fields', {})
        # If no explicit fields and LLM mapping enabled, attempt via LLM
        if not fields and ENABLE_LLM_FIELD_MAP and False:
            try:
                mapped = await llm_map_checkout_fields(page, checkout_info)
                if mapped:
                    fields.update(mapped)
                    logging.info(f'LLM mapped checkout fields: {mapped}')
            except Exception as e:
                logging.warning(f'LLM checkout field mapping failed: {e}')
        effective_fields = dict(fields)
        # If fields incomplete, try to detect
        if not set(effective_fields.keys()).issuperset({'name','address','email','phone'}):
            detected = await detect_checkout_fields(page)
            for k,v in detected.items():
                if k not in effective_fields:
                    effective_fields[k] = v

        # Fill the fields with checkout_info if available
        for key, sel in effective_fields.items():
            if not sel or key not in checkout_info:
                continue
            try:
                await page.fill(sel, str(checkout_info[key]))
                await page.wait_for_timeout(200)
            except Exception as e:
                logging.warning(f'Failed to fill field {key} ({sel}): {e}')

        # Continue through checkout if selectors provided
        if selectors.get('continue_button'):
            try:
                await page.click(selectors['continue_button'])
                await page.wait_for_load_state('networkidle')
                await page.wait_for_timeout(1000)
            except Exception as e:
                logging.warning(f'Failed clicking continue_button: {e}')

        # At payment step. We intentionally stop before placing an order unless allow_place True
        if allow_place and selectors.get('place_order'):
            try:
                # If tele-approval is enabled, wait for human approval
                proceed = True
                if ENABLE_TELEGRAM_APPROVAL and TELEGRAM_ENABLED:
                    code = f'CONFIRM-{int(time.time())}'
                    await send_telegram(f'Approve purchase for {product.get("title")} by replying with code: {code}')
                    approved = await wait_for_telegram_approval(code, timeout=300)
                    if not approved:
                        logging.info('Telegram approval not received - aborting auto place.')
                        proceed = False

                if proceed:
                    # If payment method is shaparak or GUI-flag requested, open a headful browser so user can complete OTP
                    try:
                        pm = None
                        if isinstance(checkout_info, dict):
                            pm = checkout_info.get('payment_method')
                        elif isinstance(product.get('checkout_info', {}), dict):
                            pm = product.get('checkout_info', {}).get('payment_method')
                        if (pm and str(pm).lower() == 'shaparak'):
                            if playwright_instance and browser_context:
                                logging.info('Opening headful GUI for user to complete bank payment')
                                await open_headful_view(playwright_instance, browser_context, page.url)

                    except Exception:
                        pass

                    elplace = await page.query_selector(selectors['place_order'])
                    if elplace:
                        try:
                            await page.evaluate('(el)=>el.click()', elplace)
                        except Exception:
                            await elplace.click(timeout=selector_timeout)
                        await page.wait_for_timeout(selector_timeout)
                    logging.info('Placed order (auto)')
                    await send_telegram(f'Auto-purchase attempted for {product.get("title")}.')
                    return True
                return False
            except Exception as e:
                logging.warning(f'Failed to click place_order: {e}')
                return False

        # Save a screenshot of the final checkout page and notify user for manual completion
        fname = f'debug_checkout_{product.get("title","product").replace(" ","_")}_{int(time.time())}.png'
        try:
            await page.screenshot(path=fname, full_page=True)
            logging.info(f'Checkout screenshot saved: {fname}')
            # Notify via Telegram that checkout is ready (if enabled)
            await send_telegram(f'Checkout ready for {product.get("title","product")}. Screenshot: {fname}\n{product.get("url")}')
        except Exception as e:
            logging.warning(f'Failed to save checkout screenshot: {e}')
        return True
    except Exception as e:
        logging.warning(f'Checkout automation error: {e}')
        return False


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
                    # Attempt to continue to checkout if configured for this product
                    if ALLOW_AUTO_CHECKOUT and product.get('checkout_enabled') and product.get('checkout_selectors'):
                        checkout_info_path = Path('checkout_info.json')
                        checkout_info = {}
                        if checkout_info_path.exists():
                            try:
                                with open(checkout_info_path, 'r', encoding='utf-8') as cf:
                                    checkout_info = json.load(cf)
                            except Exception as e:
                                logging.warning(f'Failed to load checkout_info.json: {e}')
                        try:
                            # pass ALLOW_AUTO_PLACE_ORDER flag to function to prevent accidental ordering
                            if page:
                                await attempt_checkout_with_page(page, product, checkout_info, allow_place=ALLOW_AUTO_PLACE_ORDER, playwright_instance=playwright_instance, browser_context=browser_context, selector_timeout=selector_timeout)
                            else:
                                logging.warning('No persistent page to perform checkout flow (skipping)')
                        except Exception as e:
                            logging.warning(f'Checkout attempt failed for {url}: {e}')
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
    checkout_info = {}
    cpath = Path('checkout_info.json')
    if cpath.exists():
        try:
            with open(cpath, 'r', encoding='utf-8') as cf:
                checkout_info = json.load(cf)
        except Exception:
            checkout_info = {}
    print('Interactive setup — add products (enter URLs). Type done to finish')
    while True:
        url = input('Product URL (or done): ').strip()
        if url.lower() == 'done':
            break
        await page.goto(url, timeout=60000)
        await page.wait_for_load_state('networkidle')
        # Autodetect selectors using LLM (or fallback heuristics) - no manual click selection
        selector = None
        add_sel = None
        if ENABLE_LLM_AUTODETECT:
            detected = await llm_detect_selectors(page, url)
            selector = detected.get('price_selector')
            add_sel = detected.get('add_selector')
            print(f'Detected selectors from LLM for price: {selector}, add: {add_sel}')
        else:
            logging.warning('LLM autodetect disabled; using heuristic detection for price and add selectors')
            # Heuristic: Try some known price patterns
            heuristics = ['[data-price]', '.price', '[class*="price"]', '#price', '.product-price', '.price-tag']
            for h in heuristics:
                try:
                    el = await page.query_selector(h)
                    if el:
                        selector = h
                        break
                except Exception:
                    continue
        el = await page.query_selector(selector) if selector else None
        price = None
        if el:
            text = await el.inner_text()
            price = parse_price(text)
        if price is None:
            price = float(input('Enter current price: ').replace(',',''))
        # if LLM did not detect an add-to-cart selector, try heuristics
        if not add_sel:
            try:
                candidates = await page.query_selector_all('button, input[type=submit], a')
                for c in candidates:
                    txt = (await c.inner_text()).lower() if c else ''
                    if txt and any(k in txt for k in ['add', 'cart', 'basket', 'سبد', 'خرید']):
                        # create a simple CSS based on classes/ids
                        attr_id = await c.get_attribute('id') or ''
                        attr_cls = await c.get_attribute('class') or ''
                        if attr_id:
                            add_sel = f'#{attr_id}'
                        elif attr_cls:
                            cls = '.'.join([s for s in attr_cls.split() if s])
                            add_sel = f'button.{cls}' if cls else 'button'
                        else:
                            add_sel = 'button'
                        break
            except Exception:
                pass
        title = await page.title()
        product_entry = {'url': url, 'selector': selector, 'price': price, 'add_selector': add_sel, 'title': title, 'added': False, 'history': [], 'llm_tried': bool(selector)}
        # Optional checkout configuration
        checkout_enable = input('Enable checkout automation for this product? (y/n): ').strip().lower() == 'y'
        if checkout_enable:
            checkout_selectors = {}
            # Ensure we have user checkout_info; if not, prompt user to provide it now for mapping
            if not checkout_info:
                print('No global checkout_info saved. Please enter checkout details for mapping (this will be saved if you choose at the end).')
                checkout_info = {}
                checkout_info['name'] = input('Full name: ').strip()
                checkout_info['email'] = input('Email: ').strip()
                checkout_info['phone'] = input('Phone: ').strip()
                checkout_info['address'] = input('Address line: ').strip()
                checkout_info['city'] = input('City: ').strip()
                checkout_info['zip'] = input('ZIP/postal code: ').strip()
            # Try to let LLM map known checkout buttons and fields
            # We try the LLM mapping only if enabled
            if ENABLE_LLM_FIELD_MAP:
                fields_map = await llm_map_checkout_fields(page, checkout_info)
                if fields_map:
                    checkout_selectors['fields'] = fields_map
                    logging.info(f'LLM detected checkout fields: {fields_map}')
            # For checkout buttons (checkout / continue / place), try heuristics
            try:
                for btn_text in ['checkout', 'continue', 'place order', 'submit', 'پرداخت', 'نهایی']:
                    q = await page.query_selector(f'button:has-text("{btn_text}")')
                    if q:
                        text = (await q.inner_text()).strip()
                        if 'checkout' in btn_text or 'بسته' in text:
                            checkout_selectors['checkout_button'] = await q.get_attribute('class') or ''
                        else:
                            # If not matched, set as generic selector
                            checkout_selectors.setdefault('continue_button', btn_text)
            except Exception:
                pass
            # fields
            fields = {}
            # If LLM field mapping is enabled, try that first (no manual selection required)
            if ENABLE_LLM_FIELD_MAP:
                mapped = await llm_map_checkout_fields(page, checkout_info)
                if mapped:
                    fields.update(mapped)
                    print('Detected fields by LLM:', mapped)
            # If not complete, try heuristics
            if not set(fields.keys()).issuperset({'name', 'email', 'phone', 'address'}):
                detected = await detect_checkout_fields(page)
                for k, v in detected.items():
                    if k not in fields:
                        fields[k] = v
                if fields:
                    print('Detected fields by heuristics:', detected)
            if fields:
                checkout_selectors['fields'] = fields
            product_entry['checkout_selectors'] = checkout_selectors
            product_entry['checkout_enabled'] = True
        result.append(product_entry)
        logging.info(f'Added {title} @ {price:,}')
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
                # Ask if user wants to save global checkout info
                if input('Save global checkout info (name/address/email/phone)? (y/n): ').strip().lower() == 'y':
                    info = {}
                    info['name'] = input('Full name: ').strip()
                    info['email'] = input('Email: ').strip()
                    info['phone'] = input('Phone: ').strip()
                    info['address'] = input('Address line: ').strip()
                    info['city'] = input('City: ').strip()
                    info['zip'] = input('ZIP/postal code: ').strip()
                    # Payment method hints: in Iran, 'shaparak' / OTP via SMS is common; ask if user wants to enable interactive bank flow
                    info['payment_method'] = input('Payment method (e.g., shaparak, card, other). If shaparak, we will open GUI for payment step: ').strip()
                    info['sms_phone'] = input('Phone for bank SMS/OTP (if different from phone): ').strip()
                    info['note'] = input('Any special checkout note (e.g., postal code format, national ID): ').strip()
                    with open('checkout_info.json', 'w', encoding='utf-8') as cf:
                        json.dump(info, cf, ensure_ascii=False, indent=2)
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
