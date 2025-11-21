import time
import random
import asyncio
from playwright.async_api import async_playwright
import re
import os
import urllib.parse
import json
import logging
import subprocess
import platform

try:
    import requests
except ImportError:
    requests = None

logging.basicConfig(filename='deal_hunter.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def get_browser_config(is_setup=False):
    # Use system-installed Microsoft Edge for setup, Chromium for headless monitoring
    if is_setup:
        # Use GUI mode for setup to allow clicking
        edge_paths = [
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
        ]
        
        executable_path = None
        for path in edge_paths:
            if os.path.exists(path):
                executable_path = path
                break
        
        if executable_path:
            print(f"Using Edge executable for setup: {executable_path}")
        else:
            print("Warning: Could not find Edge executable for setup!")
        
        return {"executable_path": executable_path, "headless": False}
    else:
        # Use headless Chromium for monitoring (better VPS compatibility)
        print("Using Chromium for headless monitoring")
        return {"channel": "chromium", "headless": True}

def get_user_data_dir():
    return f"C:\\Users\\{os.environ['USERNAME']}\\AppData\\Local\\Microsoft\\Edge\\User Data"

def kill_edge_processes():
    system = platform.system()
    try:
        if system == "Windows":
            subprocess.run(['taskkill', '/f', '/im', 'msedge.exe'], capture_output=True, check=False)
        elif system == "Linux":
            subprocess.run(['pkill', '-f', 'microsoft-edge'], capture_output=True, check=False)
        print("Closed any running Edge instances.")
        return True
    except Exception as e:
        print(f"Could not close Edge: {e}")
        return False

# Telegram setup
telegram_enabled = False
if requests and 'TELEGRAM_BOT_TOKEN' in os.environ and 'TELEGRAM_CHAT_ID' in os.environ:
    telegram_enabled = True
    bot_token = os.environ['TELEGRAM_BOT_TOKEN']
    chat_id = os.environ['TELEGRAM_CHAT_ID']
    print("Telegram notifications enabled.")
else:
    print("Telegram notifications disabled (missing env vars or requests).")

# Configurable thresholds
ALERT_THRESHOLD_PERCENT = float(os.environ.get('ALERT_THRESHOLD_PERCENT', '40'))  # Default 40%
BASKET_THRESHOLD_PERCENT = float(os.environ.get('BASKET_THRESHOLD_PERCENT', '50'))  # Default 50%

def parse_price(text):
    text = persian_to_english_digits(text)
    text = re.sub(r'[,\s]', '', text)
    match = re.search(r'\d+', text)
    if match:
        return float(match.group())
    return None

def persian_to_english_digits(text):
    persian_digits = '۰۱۲۳۴۵۶۷۸۹'
    english_digits = '0123456789'
    trans = str.maketrans(persian_digits, english_digits)
    return text.translate(trans)

async def get_price(url, selector, context):
    try:
        page = await context.new_page()
        await page.goto(url, timeout=30000)
        await page.wait_for_load_state('networkidle')
        await page.wait_for_timeout(3000)  # wait for JS
        element = await page.query_selector(selector)
        if element:
            text = await element.inner_text()
            # Convert Persian digits
            text = persian_to_english_digits(text)
            # Remove commas and spaces
            text = re.sub(r'[,\s]', '', text)
            # Extract number
            match = re.search(r'\d+', text)
            if match:
                price = float(match.group())
                await page.close()
                return price
        await page.close()
    except Exception as e:
        print(f"Error getting price for {url}: {e}")
    return None

async def get_selector_from_click(page, element_name):
    await page.evaluate("""
    window.clickedElement = null;
    document.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        window.clickedElement = e.target;
    }, true);
    """)
    input(f"Click on the {element_name} in the browser and press Enter")
    selector = await page.evaluate("""
    let el = window.clickedElement;
    let selector = '';
    if (el) {
        selector = el.tagName.toLowerCase();
        if (el.id) selector += '#' + el.id;
        if (el.className) {
            let classes = el.className.trim().split(/\\s+/).map(c => c.replace(/:/g, '\\\\:'));
            selector += '.' + classes.join('.');
        }
    }
    selector
    """)
    if selector:
        print(f"Detected selector for {element_name}: {selector}")
        return selector
    else:
        return input(f"Could not detect selector for {element_name}, enter manually: ")

async def add_to_basket(url, selector, context):
    try:
        print(f"Attempting to add to basket for {url} with selector: {selector}")
        page = await context.new_page()
        await page.goto(url, timeout=30000)
        await page.wait_for_load_state('networkidle')
        await page.wait_for_timeout(3000)  # wait for JS
        print(f"Page title: {await page.title()}")
        # Take screenshot for debugging
        safe_name = url.split('/')[-1].replace('?', '_').replace('&', '_')[:50]
        await page.screenshot(path=f"debug_screenshot_{safe_name}.png")
        print(f"Screenshot saved as debug_screenshot_{safe_name}.png")
        button = await page.query_selector(selector)
        if button:
            print("Add to basket button found, attempting to click...")
            await button.wait_for_element_state('visible')
            await button.scroll_into_view_if_needed()
            await button.click()
            print(f"Clicked add to basket for {url}")
            await page.wait_for_timeout(2000)  # wait after click
        else:
            print(f"Add to basket button not found for {url} with selector {selector}")
        await page.close()
    except Exception as e:
        print(f"Error adding to basket for {url}: {e}")

async def check_domain(domain, prods, context, domain_delays, products):
    count = len(prods)
    delay = domain_delays[domain]
    print(f"Checking {count} products on {domain}, current delay: {delay:.1f}s")
    # Concurrently check all products in the domain
    tasks = [get_price(product['url'], product['selector'], context) for product in prods]
    new_prices = await asyncio.gather(*tasks)
    failures = sum(1 for p in new_prices if p is None)
    if failures == 0:
        # All successful, decrease delay aggressively
        old_delay = delay
        domain_delays[domain] = max(0.05, delay * 0.7)  # 30% decrease each time
        if domain_delays[domain] != old_delay:
            print(f"⚡ Optimized delay for {domain}: {old_delay:.2f}s → {domain_delays[domain]:.2f}s")
    else:
        # Some failures, increase slowly and incrementally
        # Add 0.5 seconds for each failure, but cap at reasonable max
        old_delay = delay
        increment = failures * 0.5
        domain_delays[domain] = min(30.0, delay + increment)
        if domain_delays[domain] != old_delay:
            print(f"Increased delay for {domain} by {increment:.1f}s due to {failures} failures (new delay: {domain_delays[domain]:.1f}s)")
    for i, new_price in enumerate(new_prices):
        product = prods[i]
        if new_price and new_price != product['price']:
            old_price = product['price']
            
            # Handle division by zero for price change calculation
            if old_price > 0:
                price_change_percent = abs(new_price - old_price) / old_price * 100
            else:
                price_change_percent = 100.0  # Treat as 100% change if old price was 0
            
            message = f"Price changed for {product['url']}: from {old_price} to {new_price} ({price_change_percent:.1f}% change)"
            print(message)
            logging.info(message)
            
            # Add to basket if price drops by threshold or more
            if product['add_selector'] and not product['added'] and new_price <= old_price * (1 - BASKET_THRESHOLD_PERCENT/100):
                await add_to_basket(product['url'], product['add_selector'], context)
                product['added'] = True
                added_text = "yes"
                print(f"Added to basket due to ≥{BASKET_THRESHOLD_PERCENT}% price drop for {product['url']}")
            else:
                added_text = "no"
            
            # Send Telegram notification only for significant price DROPS with basket action
            if new_price < old_price and price_change_percent >= BASKET_THRESHOLD_PERCENT:
                basket_emoji = "✅ Added to Basket" if added_text == "yes" else "❌ Basket Full/Limit Reached"
                trend_info = analyze_price_trend(product)
                trend_text = f"\n{trend_info}" if trend_info else ""
                message = f"🛒 DEAL ALERT!\n📦 {product['title']}\n💰 {old_price:,} → {new_price:,}\n📉 Drop: {price_change_percent:.1f}%\n{basket_emoji}{trend_text}\n🔗 {product['url']}"
                await send_telegram(message)
            
            product['price'] = new_price
            
            # Track price history for trend analysis
            if 'history' not in product:
                product['history'] = []
            product['history'].append({
                'price': new_price,
                'timestamp': time.time(),
                'change_percent': price_change_percent if 'old_price' in locals() else 0
            })
            # Keep only last 100 entries to avoid memory issues
            if len(product['history']) > 100:
                product['history'] = product['history'][-100:]

async def send_telegram(message):
    if not telegram_enabled:
        return
    try:
        await asyncio.to_thread(
            requests.post,
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            data={'chat_id': chat_id, 'text': message, 'parse_mode': 'HTML'}
        )
    except Exception as e:
        print(f"Failed to send Telegram: {e}")

def analyze_price_trend(product):
    """Analyze price trend and return insights"""
    if 'history' not in product or len(product['history']) < 3:
        return None
    
    history = product['history'][-10:]  # Last 10 price points
    prices = [h['price'] for h in history]
    
    # Calculate trend
    if len(prices) >= 2:
        first_price = prices[0]
        last_price = prices[-1]
        trend_percent = ((last_price - first_price) / first_price) * 100
        
        # Check for consistent downward trend
        downward_trend = all(prices[i] >= prices[i+1] for i in range(len(prices)-1))
        
        if trend_percent <= -10 and downward_trend:  # 10%+ drop with consistent trend
            return f"📈 Strong downward trend detected! {trend_percent:.1f}% over {len(history)} checks"
    
    return None

async def main():
    print("Starting setup... (Microsoft Edge will open - ensuring no conflicts)")

    # Kill any running Edge processes to avoid conflicts
    kill_edge_processes()
    await asyncio.sleep(2)  # wait for cleanup

    user_data_dir = get_user_data_dir()
    config = get_browser_config(is_setup=True)  # GUI mode for setup

    # Check if running on VPS (no GUI) - use headless setup
    import os
    is_vps = os.environ.get('VPS_MODE', '').lower() == 'true' or config["headless"]

    if is_vps:
        print("Running in VPS mode - setup will use GUI, monitoring will be headless")
        # For VPS, setup still uses GUI for element selection, monitoring is headless

    # Load products if exists
    if os.path.exists('products.json'):
        load = input("Load previous products from products.json? (y/n): ").strip().lower()
        if load == 'y':
            with open('products.json', 'r') as f:
                products = json.load(f)
            print(f"Loaded {len(products)} products.")
        else:
            products = []
    else:
        products = []

    # Load domain delays if exists
    if os.path.exists('domain_delays.json'):
        with open('domain_delays.json', 'r') as f:
            domain_delays = json.load(f)
        print("Loaded domain delays.")
    else:
        domain_delays = {}

    if not products:
        # Setup new products - always use GUI mode for element selection
        async with async_playwright() as p:
            config = get_browser_config(is_setup=True)  # GUI mode for setup
            context = await p.chromium.launch_persistent_context(
                user_data_dir,
                executable_path=config["executable_path"],
                headless=config["headless"],
                args=["--remote-debugging-port=9222"]
            )

            while True:
                url = input("Enter product URL (or 'done' to finish): ").strip()
                if url.lower() == 'done':
                    break
                try:
                    page = await context.new_page()
                    await page.goto(url)
                    await page.wait_for_load_state('networkidle')
                    await page.wait_for_timeout(3000)
                    selector = await get_selector_from_click(page, "price")
                    element = await page.query_selector(selector)
                    current_price = None
                    if element:
                        price_text = await element.inner_text()
                        current_price = parse_price(price_text)
                        if current_price:
                            print(f"Detected current price: {current_price}")
                        else:
                            print("Could not parse price from element")
                    if not current_price:
                        current_price = float(input("Enter current price: ").replace(',', ''))
                    add_sel = input("Do you want to set add to basket for this product? (y/n): ").strip().lower()
                    if add_sel == 'y':
                        add_selector = await get_selector_from_click(page, "add to basket button")
                    else:
                        add_selector = None
                    title = await page.title()
                    await page.close()
                    product_index = len(products) + 1
                    products.append({'url': url, 'selector': selector, 'price': current_price, 'add_selector': add_selector, 'added': False, 'title': title, 'index': product_index})

                    # Send first Telegram notification
                    message = f"🆕 New Product Added!\n📦 {title}\n💰 Current Price: {current_price:,}\n🔗 {url}"
                    await send_telegram(message)
                    print(f"Product added successfully! Total products: {len(products)}")
                except Exception as e:
                    error_msg = str(e)
                    if "closed" in error_msg.lower() or "context" in error_msg.lower():
                        print("Browser context was closed. Attempting to recreate...")
                        try:
                            # Close the old context if it exists
                            try:
                                await context.close()
                            except:
                                pass
                            # Recreate context
                            context = await p.chromium.launch_persistent_context(
                                user_data_dir,
                                executable_path=config["executable_path"],
                                headless=config["headless"],
                                args=["--remote-debugging-port=9222"]
                            )
                            print("Browser context recreated successfully. Please try adding the product again.")
                        except Exception as recreate_error:
                            print(f"Failed to recreate browser context: {recreate_error}")
                            print("Please restart the script.")
                            break
                    else:
                        print(f"Error adding product: {e}")
                        print("Please try again or type 'done' to finish.")

            # Save products to JSON
            if products:
                with open('products.json', 'w') as f:
                    json.dump(products, f)
                print(f"Products saved to products.json ({len(products)} products)")
            else:
                print("No products to save.")

            if not products:
                print("No products added.")
                await context.close()
                await p.stop()
                return

            await context.close()
            await p.stop()

    # Now start monitoring
    while True:
        try:
            async with async_playwright() as p:
                config = get_browser_config(is_setup=False)  # Headless mode for monitoring
                context = await p.chromium.launch_persistent_context(
                    user_data_dir,
                    channel=config.get("channel"),
                    headless=config["headless"],
                    args=["--remote-debugging-port=9222"]
                )

                print("Starting price monitoring...")

                # Send notification about products being tracked
                if products:
                    product_list = "\n".join([f"📦 {p['title'][:50]}... - 💰 {p['price']:,.0f}" for p in products])
                    message = f"🎯 Starting Deal Hunter Monitoring!\n\n📊 Tracking {len(products)} products:\n\n{product_list}\n\n⚡ Aggressive delay optimization active!"
                    await send_telegram(message)

                # Group products by domain
                domains = {}
                for product in products:
                    domain = urllib.parse.urlparse(product['url']).netloc
                    if domain not in domains:
                        domains[domain] = []
                    domains[domain].append(product)

                # Initialize delays for domains not in domain_delays
                for domain in domains:
                    if domain not in domain_delays:
                        domain_delays[domain] = 0.1  # Start very aggressively at 0.1s

                try:
                    while True:
                        # Check all domains concurrently for faster monitoring
                        domain_tasks = []
                        for domain, prods in domains.items():
                            task = asyncio.create_task(check_domain(domain, prods, context, domain_delays, products))
                            domain_tasks.append(task)
                        
                        await asyncio.gather(*domain_tasks)
                        
                        # Smart delay: use minimum delay across all domains for faster overall checking
                        min_delay = min(domain_delays.values()) if domain_delays else 0.1
                        await asyncio.sleep(min_delay)
                finally:
                    # Save domain delays
                    with open('domain_delays.json', 'w') as f:
                        json.dump(domain_delays, f)
                    await context.close()
                    await p.stop()
        except Exception as e:
            print(f"Browser crashed or closed, restarting in 10 seconds: {e}")
            await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
