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

try:
    import requests
except ImportError:
    requests = None

logging.basicConfig(filename='deal_hunter.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Telegram setup
telegram_enabled = False
if requests and 'TELEGRAM_BOT_TOKEN' in os.environ and 'TELEGRAM_CHAT_ID' in os.environ:
    telegram_enabled = True
    bot_token = os.environ['TELEGRAM_BOT_TOKEN']
    chat_id = os.environ['TELEGRAM_CHAT_ID']
    print("Telegram notifications enabled.")
else:
    print("Telegram notifications disabled (missing env vars or requests).")

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
    await asyncio.to_thread(input, f"Click on the {element_name} in the browser and press Enter")
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
        return await asyncio.to_thread(input, f"Could not detect selector for {element_name}, enter manually: ")

async def add_to_basket(url, selector, context):
    try:
        page = await context.new_page()
        await page.goto(url, timeout=30000)
        await page.wait_for_load_state('networkidle')
        await page.wait_for_timeout(3000)
        button = await page.query_selector(selector)
        if button:
            await button.click()
            print(f"Clicked add to basket for {url}")
        await page.close()
    except Exception as e:
        print(f"Error adding to basket for {url}: {e}")

async def main():
    print("Starting setup... (Microsoft Edge will open - ensuring no conflicts)")

    # Kill any running Edge processes to avoid conflicts
    try:
        subprocess.run(['taskkill', '/f', '/im', 'msedge.exe'], capture_output=True, check=False)
        print("Closed any running Edge instances.")
        await asyncio.sleep(2)  # wait for cleanup
    except Exception as e:
        print(f"Could not close Edge: {e}")

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
        # Setup new products
        async with async_playwright() as p:
            user_data_dir = f"C:\\Users\\{os.environ['USERNAME']}\\AppData\\Local\\Microsoft\\Edge\\User Data"
            context = await p.chromium.launch_persistent_context(
                user_data_dir,
                channel="msedge",
                headless=False,
                args=["--remote-debugging-port=9222"]
            )

            while True:
                url = input("Enter product URL (or 'done' to finish): ").strip()
                if url.lower() == 'done':
                    break
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
                message = f"Product No {product_index}\nName: {title}\nCurrent price: {current_price}\nLink: {url}"
                await send_telegram(message)

            # Save products to JSON
            with open('products.json', 'w') as f:
                json.dump(products, f)
            print("Products saved to products.json")

            if not products:
                print("No products added.")
                await context.close()
                await p.stop()
                return

            await context.close()
            await p.stop()

    # Now start monitoring
    async with async_playwright() as p:
        user_data_dir = f"C:\\Users\\{os.environ['USERNAME']}\\AppData\\Local\\Microsoft\\Edge\\User Data"
        context = await p.chromium.launch_persistent_context(
            user_data_dir,
            channel="msedge",
            headless=False,
            args=["--remote-debugging-port=9222"]
        )

        print("Starting price monitoring...")

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
                domain_delays[domain] = 5.0  # start with 5s

        try:
            while True:
                for domain, prods in domains.items():
                    count = len(prods)
                    delay = domain_delays[domain]
                    print(f"Checking {count} products on {domain}, current delay: {delay:.1f}s")
                    # Concurrently check all products in the domain
                    tasks = [get_price(product['url'], product['selector'], context) for product in prods]
                    new_prices = await asyncio.gather(*tasks)
                    failures = sum(1 for p in new_prices if p is None)
                    if failures == 0:
                        # All successful, decrease delay
                        domain_delays[domain] = max(2.0, delay * 0.9)
                    else:
                        # Some failures, increase delay
                        domain_delays[domain] = min(60.0, delay * 1.5)
                        print(f"Increased delay for {domain} due to {failures} failures")
                    for i, new_price in enumerate(new_prices):
                        product = prods[i]
                        if new_price and new_price != product['price']:
                            old_price = product['price']
                            price_change_percent = abs(new_price - old_price) / old_price * 100
                            
                            message = f"Price changed for {product['url']}: from {old_price} to {new_price} ({price_change_percent:.1f}% change)"
                            print(message)
                            logging.info(message)
                            
                            # Add to basket if price drops by 50% or more
                            if product['add_selector'] and not product['added'] and new_price <= old_price * 0.5:
                                await add_to_basket(product['url'], product['add_selector'], context)
                                product['added'] = True
                                added_text = "yes"
                                print(f"Added to basket due to ≥50% price drop for {product['url']}")
                            else:
                                added_text = "no"
                            
                            # Send Telegram notification only if price change is ≥40%
                            if price_change_percent >= 40:
                                message = f"Product No {product['index']}\nName: {product['title']}\nPrevious price: {old_price}\nNew price: {new_price}\nChange: {price_change_percent:.1f}%\nLink: {product['url']}\nAdded to basket? {added_text}"
                                await send_telegram(message)
                            
                            product['price'] = new_price
                    await asyncio.sleep(delay)
        finally:
            # Save domain delays
            with open('domain_delays.json', 'w') as f:
                json.dump(domain_delays, f)
            await context.close()
            await p.stop()

if __name__ == "__main__":
    asyncio.run(main())
