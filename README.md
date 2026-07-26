# BCC AI Pricing Engine — Staff Portal

A proof-of-concept staff tool for Bombay Closet Cleanse. Upload a photo of a pre-loved clothing item and receive an AI-generated price recommendation, condition score, and market rationale — powered by Groq's free vision and text models.

---

## Quick start

### 1. Get a free Groq API key
Visit **[console.groq.com/keys](https://console.groq.com/keys)**, sign up for free, and generate an API key.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set the API key

**macOS / Linux**
```bash
export GROQ_API_KEY=gsk_your_key_here
```

**Windows (Command Prompt)**
```cmd
set GROQ_API_KEY=gsk_your_key_here
```

**Windows (PowerShell)**
```powershell
$env:GROQ_API_KEY="gsk_your_key_here"
```

Or copy `.env.example` to `.env`, fill in your key, and install `python-dotenv`:
```bash
pip install python-dotenv
# then edit .env: GROQ_API_KEY=gsk_your_key_here
```

### 4. Start the server
```bash
uvicorn main:app --reload --port 8000
```

### 5. Open in browser
Navigate to **[http://localhost:8000](http://localhost:8000)**

---

## How to use

1. **Upload** a JPEG or PNG photo of the clothing item (max 10 MB) — drag & drop or click the upload area
2. Click **Analyse Item** and wait ~5–10 seconds for the AI assessment
3. Review the **price band**, **condition score**, **demand signals**, and **market rationale**
4. Click **Accept** to log the recommended price, or click **Override price** to enter a custom amount
5. The **Session Intake Log** at the bottom of the page tracks all decisions for the current session (resets on server restart)

---

## Models used

| Call | Model | Purpose |
|------|-------|---------|
| Vision | `qwen/qwen3.6-27b` | Brand detection, condition scoring, price band |
| Text | `llama-3.3-70b-versatile` | Market rationale paragraph |

Both are free-tier models on Groq Cloud.

---

## Project structure

```
bcc-pricing-poc/
  main.py          — FastAPI backend, Groq API calls, session log
  index.html       — Self-contained frontend (no build step)
  requirements.txt — Python dependencies
  .env.example     — Environment variable template
  README.md        — This file
```
