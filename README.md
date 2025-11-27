# Deal Hunter (Minimal)

Minimal headless product price monitor for Windows (works on a VPS).

Quick start

1. Create a virtualenv and install dependencies

`powershell
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install -r requirements.txt
python -m playwright install chromium
`

2. (Optional) Local setup

`powershell
python main.py --setup
`

3. Headless (VPS)

`powershell
set VPS_MODE=true
python main.py
`

Minimal env variables
- LLM_PROVIDER (optional): e.g., groq or openai (leave unset to disable LLM)
- LLM_API_KEY (required if LLM enabled)
- TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (optional): enable notifications
- VPS_MODE (optional) 	rue to run headless

Safety and behavior
- The script does not send checkout PII to LLMs by default.
- Auto-place order behavior is disabled by default.

For advanced configuration, modify main.py directly; options are intentionally conservative in code.

