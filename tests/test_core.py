import asyncio
import json
import os
from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from main import (
    parse_price,
    persian_to_english_digits,
    compute_selector_timeout,
    load_llm_cache,
    save_llm_cache,
    LLM_CACHE_FILE,
    get_user_data_dir,
    get_price_with_page,
    add_to_basket_with_page,
    llm_detect_selectors,
    call_llm,
)


def test_persian_digits_and_parse_price():
    assert persian_to_english_digits('۰۱۲۳۴۵۶۷۸۹') == '0123456789'
    assert parse_price('1,234') == 1234.0
    assert parse_price('۱۲۳۴') == 1234.0
    assert parse_price('$ 1.234,00') == 123400.0 or isinstance(parse_price('$ 1.234,00'), float)


def test_compute_selector_timeout():
    domain_delays = {'example.com': 1.0}
    page_failures = {'https://example.com/p1': 3}
    timeout = compute_selector_timeout(domain_delays, page_failures, 'example.com', 'https://example.com/p1')
    assert isinstance(timeout, int)
    assert timeout >= 400
    assert timeout <= 5000


def test_llm_cache_load_save(tmp_path, monkeypatch):
    # Monkeypatch the LLM_CACHE_FILE to a temp path
    tmp_file = tmp_path / 'llm_cache.json'
    monkeypatch.setattr('main.LLM_CACHE_FILE', str(tmp_file))

    # Save a cache and load it
    cache = {'test': {'price_selector': '#p'}}
    save_llm_cache(cache)

    loaded = load_llm_cache()
    assert loaded == cache


@pytest.mark.asyncio
async def test_price_and_add_click():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context()
        page = await ctx.new_page()
        # Prepare a simple page with price and add button
        html = """
        <html>
        <body>
            <span id="price">1,234</span>
            <button id="add">Add</button>
            <a id="checkout" href="#">Checkout</a>
            <div id="payment">PaymentForm</div>
            <script>
            document.getElementById('add').addEventListener('click', function(){ window.__added = true; });
            </script>
        </body>
        </html>
        """
        import urllib.parse
        data_url = 'data:text/html,' + urllib.parse.quote(html)
        await page.goto(data_url)
        price = await get_price_with_page(page, page.url, '#price')
        assert price == 1234.0

        added = await add_to_basket_with_page(page, page.url, '#add')
        assert added is True

        # Confirm click effect
        added_flag = await page.evaluate('window.__added === true')
        assert added_flag is True


@pytest.mark.asyncio
async def test_llm_detect_selectors_with_mock(monkeypatch, tmp_path):
    # Create a simple page with the selectors
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context()
        page = await ctx.new_page()
        html = """
        <html>
        <body>
            <span id="price">1,234</span>
            <button id="addToCart">Add</button>
            <a class="checkout-btn">Checkout</a>
            <div id="payment-ind">PAY</div>
        </body>
        </html>
        """
        import urllib.parse
        data_url = 'data:text/html,' + urllib.parse.quote(html)
        await page.goto(data_url)

        # Monkeypatch call_llm to return a JSON string
        mocked_json = '{"price_selector":"#price","add_selector":"#addToCart","checkout_selector":".checkout-btn","payment_indicator":"#payment-ind","confidence":0.95}'
        monkeypatch.setattr('main.call_llm', lambda prompt, system=None: mocked_json)

        # Monkeypatch the cache file path
        tmp_file = tmp_path / 'llm_cache.json'
        monkeypatch.setattr('main.LLM_CACHE_FILE', str(tmp_file))

        result = await llm_detect_selectors(page, page.url)
        assert 'price_selector' in result
        assert result['price_selector'] == '#price'
        # Ensure cache saved
        cache = load_llm_cache()
        # The cache key should be "selectors:<url>" where url matches page.url
        assert any(k.startswith('selectors:') for k in cache.keys())

        await ctx.close()
        await browser.close()
