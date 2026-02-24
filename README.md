🛡️ SachAI – India’s Real-Time Multilingual Fact-Check Shield

SachAI is a **multimodal misinformation detection platform** built for India.
It verifies **text claims, screenshots, and voice notes** in real-time using official government portals, PIB Fact Check, and credible news sources.

📌 Overview
Misinformation spreads rapidly across WhatsApp forwards, social media posts, and voice notes — especially in Tier-2 and Tier-3 cities.
SachAI provides:
* Real-time verification
* Hyperlocal reasoning
* Multilingual explanations (Hindi + English)
* Shareable debunk tools

 🧰 Tech Stack
* Streamlit – Frontend UI
* Perplexity API – Live RAG-based verification
* OpenAI (Vision + Whisper) – Screenshot claim extraction & voice transcription
* gTTS – Hindi audio generation
* PIL (Pillow) – Truth Card image generation
* Python – Backend logic

🚀 Features

 1. Text Claim Verification

* Paste WhatsApp forwards or social media posts
* Optional city/district input for hyperlocal context
* Simple Hindi + English explanations
* Confidence score
* Live citations

 2. Screenshot Verification
 
* Upload screenshots (.png / .jpg / .jpeg)
* Vision model extracts core claim
* Automatic verification
* Structured verdict response

 3. Voice Note Verification

* Upload voice notes (.ogg / .mp3 / .wav / .m4a)
* Whisper transcription
* Fact-checking of transcribed content
* Hindi debunk audio generation

4. Shareable Truth Tools

* WhatsApp-ready short debunk text (Hindi + English)
* Downloadable Truth Card (PNG)
* Downloadable Hindi debunk audio (MP3)

Designed to help users verify before forwarding.

 🏗 System Architecture


User Input
   ├── Text
   ├── Screenshot
   └── Voice Note
        ↓
Claim Extraction
   ├── GPT Vision
   └── Whisper
        ↓
Live Web Verification (Perplexity API)
        ↓
Structured JSON Verdict
        ↓
Outputs:
   ├── Verdict UI
   ├── Source Links
   ├── Truth Card Image
   └── Hindi Audio Debunk


# ⚙️ Installation

 1. Clone Repository

```bash
git clone https://github.com/your-username/sachai.git
cd sachai
```

2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

3. Install Dependencies

```bash
pip install -r requirements.txt
```

If no requirements file:

```bash
pip install streamlit openai requests pillow gtts
```

---

🔑 API Keys

SachAI supports per-user API key configuration via the sidebar.

You may:

Option A – Environment Variables

```bash
export OPENAI_API_KEY="your_key"
export PERPLEXITY_API_KEY="your_key"
```

Option B – Paste Keys in Sidebar

API keys are stored only in the active browser session.

 ▶️ Run the Application

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```



🧠 Verification Logic

The system:

1. Searches the real-time web
2. Prioritizes:

   * PIB Fact Check
   * Official .gov.in portals
   * WHO / UN
   * Major Indian news outlets
3. Classifies claims as:

   * Likely True
   * Likely False
   * Unclear
4. Returns structured JSON including:

   * Hindi explanation
   * English explanation
   * Confidence score
   * WhatsApp summaries
   * Source citations


# 🔐 Safety Principles

* Does not assume claims are false by default
* Uses "Unclear" when evidence is weak
* Encourages responsible sharing
* Provides multilingual explanations

# 📈 Future Improvements

* Regional misinformation heatmap (live clustering)
* Admin analytics dashboard
* WhatsApp bot integration
* Telegram bot integration
* Regional language expansion

