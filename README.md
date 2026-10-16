# 🛡️ SachAI – India's Daily Misinformation & Cyber Scam Shield

**SachAI** is a real-time, multimodal fact-checking and anti-scam platform designed specifically for daily life in India. It combats viral WhatsApp rumours, suspicious SMS messages, phishing links, and fake government orders through modular AI verification, evidence provenance, and uncertainty-aware verdicts.

---

## 🏗️ Phase 1 Modular Architecture

The repository is organized with a decoupled, testable architecture:

```
sach-ai/
├── core/                       # Decoupled business logic & pipelines
│   ├── __init__.py             # Core package initialization
│   ├── models.py               # Validated Pydantic domain models
│   ├── pipeline.py             # Orchestrator with prompt injection defense
│   ├── provenance.py           # Authority tiering & source deduplication
│   ├── storage.py              # AbstractStorage & SQLite implementation
│   ├── tracing.py              # Structured telemetry, correlation IDs & error sanitization
│   └── evaluator.py            # Deterministic metrics (citation score, abstention, consistency)
├── tests/                      # Automated pytest suite (100% offline, 0 API key required)
│   ├── test_models.py          # Schema validation & boundary tests
│   ├── test_provenance.py      # Source authority tiering & deduplication tests
│   ├── test_storage.py         # SQLite persistence & retrieval tests
│   ├── test_pipeline_offline.py# Offline fact-checking & abstention tests
│   ├── test_scam_detection.py  # Scam SMS & APK malware heuristic tests
│   ├── test_prompt_injection.py# Injection defense & delimiter breakout tests
│   ├── test_truth_card.py      # PIL Truth Card image generation tests
│   └── test_evaluator.py       # Deterministic evaluation metrics tests
├── .github/workflows/ci.yml    # GitHub Actions automated CI testing
├── app.py                      # Clean Streamlit user interface
├── gemini_engine.py            # Free Gemini REST engine & verified PIB archive
└── requirements.txt            # Pinned runtime dependencies
```

---

## ⚡ Zero-Cost & 100% Free API Architecture

Unlike tools requiring expensive paid subscriptions, SachAI is built to run on **completely free APIs**:

1. **Google Gemini 1.5 / 2.0 Flash (Free API):**
   * Free tier (15 requests/min, zero credit card required) from [Google AI Studio](https://aistudio.google.com/app/apikey).
   * Real-time Google Search grounding.
   * Multimodal analysis for screenshots (OCR + forensic inspection) and voice notes.
2. **Built-in PIB & Cyber Threat Intelligence Archive (Zero Keys Needed):**
   * Instant verification for India's most common viral scams and rumours even with *no API key configured*.
3. **Free Text-To-Speech (gTTS):**
   * Generates natural Hindi audio debunks without any paid API keys.
4. **Optional Providers:** Supports Perplexity (Sonar) and OpenAI (GPT-4o) if keys are provided.

---

## 💾 Storage & Schema Strategy

* **Engine:** Local thread-safe SQLite database (`sachai_history.db`).
* **Table Schema (`verification_history`):**
  * `id`: Auto-incrementing primary key.
  * `correlation_id`: Unique trace identifier (`sach-xxxx`).
  * `claim`: Sanitized input text (up to 3,000 characters).
  * `verdict`: Structured verdict (`Likely True`, `Likely False`, `Cyber Fraud / Scam`, `Unclear`).
  * `confidence`: Calibrated integer score (0–100).
  * `category`: Categorization (e.g. `Government Scheme`, `Finance / Banking`, `Rumour / Panic`).
  * `risk_level`: Severity level (`Low`, `Moderate`, `High Viral Threat`, `Severe Scam`).
  * `engine_used`: Backend provider attribution.
  * `created_at`: UTC ISO timestamp.
  * `payload_json`: Full versioned result snapshot.
* **Privacy & DPDP Safety:** Raw image and audio binaries are **never stored** in the database; only extracted claims, metadata, and structured verdicts are retained.

---

## 🛡️ Security & Prompt Injection Defense

* **Untrusted Claim Delimitation:** User-submitted forwards are encapsulated inside `<UNTRUSTED_CLAIM_CONTENT>` delimiters before prompt assembly.
* **Adversarial Sanitization:** Explicit regex filters strip jailbreaks (`Ignore all previous instructions`, instruction override tokens, and XML tag injections).
* **Credential Masking:** Telemetry and error sanitization (`core/tracing.py`) mask Gemini, OpenAI, and Perplexity keys from error logs and client responses.
* **Separation of Concerns:** API failures and timeouts never result in factual verdicts (`True` or `False`); the system safely abstains with clear explanation.

---

## 📊 Deterministic Evaluation Framework

The evaluation engine (`core/evaluator.py`) provides verifiable, deterministic quality metrics:

* **Citation Support Score:** Weighted metric ($0.0 \to 1.0$) prioritizing Tier-1 official portals (`.gov.in`, `rbi.org.in`, `pib.gov.in`) and Tier-2 IFCN fact-checkers.
* **Verdict Consistency Check:** Enforces logical coherence between verdicts and risk ratings (e.g., prevents "Likely True" with "Severe Scam").
* **Abstention Rate:** Measures the fraction of queries where SachAI responsibly returns "Unclear" due to insufficient credible evidence, avoiding hallucinations.

---

## 🧪 Running Automated Tests

Run the complete offline test suite (runs in ~3 seconds with zero external network calls):

```bash
pytest -v tests/
```

Run test suite with coverage report:

```bash
pytest --cov=core tests/
```

---

## 🚀 Quickstart

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **(Optional) Configure your Free Gemini API Key:**
   * Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey).
   * Either paste it into the UI sidebar or set it in `.env`:
     ```env
     GEMINI_API_KEY=your_gemini_key_here
     ```
   * *Note: SachAI still functions out of the box with zero keys using its built-in PIB intelligence archive!*

3. **Launch the application:**
   ```bash
   streamlit run app.py
   ```
