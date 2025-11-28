# Deal Hunter (Minimal)

Minimal headless product price monitor for Windows (works on a VPS).

Quick start

1. Create a virtualenv and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install -r requirements.txt
python -m playwright install chromium
```

2. (Optional) Local setup

```powershell
python main.py --setup
```

3. Headless (VPS)

```powershell
set VPS_MODE=true
python main.py
```

Minimal env variables
- GROQ_API_KEY (optional, recommended): the Groq API key for LLM detection.
- TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (optional): enable notifications
- VPS_MODE (optional) true to run headless

Safety and behavior
- The script attempts to avoid sending PII to LLMs by default (you can override this by setting the appropriate environment variables).
- Auto-place order behavior is disabled by default and requires a config change in `main.py` to enable.

Running tests
```powershell
python -m pip install -r requirements.txt
python -m pip install -r dev-requirements.txt || pip install pytest pytest-asyncio
python -m playwright install --with-deps
pytest -q
```

For advanced configuration, modify main.py directly; options are intentionally conservative in code.

