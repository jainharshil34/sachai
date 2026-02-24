import base64
import io
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS

try:
    from openai import OpenAI
except ImportError:  # Fallback if openai import style differs
    OpenAI = None  # type: ignore


# ---------- Logging configuration ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("SachAI")


# ---------- Environment / client helpers ----------
def get_openai_client() -> Optional[Any]:
    """
    Build an OpenAI client using either a UI-provided key (preferred) or env var.
    This lets each user bring their own key in the Streamlit interface.
    """
    ui_key = None
    # Use session_state if available (during a Streamlit run)
    try:
        ui_key = st.session_state.get("openai_api_key_ui")
    except Exception:
        ui_key = None

    api_key = ui_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY is not set (UI or env). Vision / Whisper features will be disabled.")
        return None
    if OpenAI is None:
        logger.error("openai package not available. Please install requirements.")
        return None
    return OpenAI(api_key=api_key)


def get_perplexity_api_key() -> Optional[str]:
    """
    Get Perplexity API key from UI (preferred) or env var.
    """
    ui_key = None
    try:
        ui_key = st.session_state.get("perplexity_api_key_ui")
    except Exception:
        ui_key = None

    key = ui_key or os.getenv("PERPLEXITY_API_KEY")
    if not key:
        logger.warning("PERPLEXITY_API_KEY is not set (UI or env). Verification will use mock fallback.")
    return key


# ---------- Core verification with Perplexity (RAG) ----------
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"


def build_perplexity_payload(
    claim: str,
    language_hint: str = "mixed",
    city: Optional[str] = None,
) -> Dict[str, Any]:
    system_prompt = (
        "You are SachAI, a multilingual fact-checking assistant focused on India. "
        "You MUST verify claims using recent, trustworthy sources such as PIB Fact Check, "
        "official Indian government portals (.gov.in), WHO/UN, and major Indian news outlets. "
        "You are neutral: do not assume rumours are fake by default. "
        "If the evidence is mixed or weak, clearly mark the claim as 'Unclear' instead of forcing true/false. "
        "Always think step-by-step but only return the final structured JSON.\n\n"
        "Return STRICT JSON with keys: "
        "verdict (Likely True / Likely False / Unclear), "
        "confidence (0-100), "
        "explanation_hindi, "
        "explanation_english, "
        "short_whatsapp_hindi, "
        "short_whatsapp_english, "
        "sources (list of {{title, url, source_type, snippet}}). "
        "Do not include any markdown, only raw JSON."
    )

    if city:
        system_prompt += f"\nHyperlocal context: The rumour is about {city} in India. "

    if language_hint.lower().startswith("hi"):
        system_prompt += "\nUser prefers Hindi, but also include English explanation."

    user_prompt = (
        f"Claim to verify:\n{claim}\n\n"
        "Steps you must follow (only in your hidden reasoning):\n"
        "1) Search the real-time web for this claim.\n"
        "2) Prioritize Indian government and credible fact-checking sources.\n"
        "3) Decide if the claim is likely true, likely false, or unclear.\n"
        "4) Explain in very simple Hindi and then in simple English.\n"
        "5) Craft short, forwardable WhatsApp-style summaries in Hindi and English.\n"
        "6) Provide at least 2-3 citations with URLs in the sources list."
    )

    return {
        "model": "sonar-pro",
        "temperature": 0.2,
        "max_tokens": 800,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }


def verify_claim_with_perplexity(
    claim: str,
    language_hint: str = "mixed",
    city: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call Perplexity to verify a claim. Returns a normalized dict even on failure.
    """
    logger.info("Starting verification with Perplexity...")
    api_key = get_perplexity_api_key()
    payload = build_perplexity_payload(claim, language_hint, city)

    if not api_key:
        # Fallback mock response for offline/dev demo
        logger.info("Using mock Perplexity response (no API key).")
        return {
            "verdict": "Unclear",
            "confidence": 55,
            "explanation_hindi": (
                "यह एक डेमो मोड उत्तर है: अभी लाइव सरकारी/समाचार स्रोतों से असली যাচाई नहीं हो रही है, "
                "इसलिए दावे को न पक्का सही कहा जा सकता है न पक्का गलत। "
                "इसे आगे भेजने से पहले भरोसेमंद स्रोतों से खुद भी जांच लें।"
            ),
            "explanation_english": (
                "This is a demo-mode answer: live checking against official/news sources is not active, "
                "so the claim cannot be clearly classified as true or false. "
                "Please double‑check with reliable sources before sharing."
            ),
            "short_whatsapp_hindi": (
                "ℹ️ यह SachAI का डेमो मोड उत्तर है – दावे की असली जांच अभी नहीं हो रही। "
                "बिना भरोसेमंद स्रोत देखे किसी भी संदेश को आगे न बढ़ाएँ।"
            ),
            "short_whatsapp_english": (
                "ℹ️ This is a demo SachAI response – the claim has not been fully verified. "
                "Avoid forwarding messages without checking credible sources."
            ),
            "sources": [
                {
                    "title": "PIB Fact Check (example listing)",
                    "url": "https://pib.gov.in/factcheck",
                    "source_type": "PIB Fact Check",
                    "snippet": "Official government fact checks on viral rumours.",
                }
            ],
        }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(PERPLEXITY_URL, headers=headers, json=payload, timeout=40)
        resp.raise_for_status()
        data = resp.json()
        logger.info("Perplexity raw response received.")

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        # Expecting strict JSON; if it fails, log and fall back
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from Perplexity response. Falling back to generic structure.")
            parsed = {}

        # Normalize fields
        verdict = parsed.get("verdict", "Unclear")
        confidence = int(parsed.get("confidence", 60))
        explanation_hindi = parsed.get(
            "explanation_hindi",
            "इस दावे पर स्पष्ट आधिकारिक जानकारी नहीं मिली, इसलिए इसे साझा करते समय सावधानी बरतें।",
        )
        explanation_english = parsed.get(
            "explanation_english",
            "There is no clear official information confirming this claim. Please be cautious before sharing.",
        )
        short_hi = parsed.get(
            "short_whatsapp_hindi",
            "ℹ️ इस दावे पर भरोसेमंद जानकारी नहीं मिली। कृपया बिना जांचे इसे आगे न बढ़ाएँ।",
        )
        short_en = parsed.get(
            "short_whatsapp_english",
            "ℹ️ This claim could not be confirmed from credible sources. Please avoid forwarding unchecked messages.",
        )
        sources = parsed.get("sources", [])
        if not isinstance(sources, list):
            sources = []

        return {
            "verdict": verdict,
            "confidence": confidence,
            "explanation_hindi": explanation_hindi,
            "explanation_english": explanation_english,
            "short_whatsapp_hindi": short_hi,
            "short_whatsapp_english": short_en,
            "sources": sources,
        }
    except Exception as e:
        logger.exception("Error while calling Perplexity API: %s", e)
        return {
            "verdict": "Unclear",
            "confidence": 50,
            "explanation_hindi": (
                "तकनीकी कारणों से अभी लाइव फैक्ट‑चेक पूरा नहीं हो सका। "
                "फिर भी, ऐसे संदेश को आगे भेजने से पहले विश्वसनीय स्रोतों से जांच लें।"
            ),
            "explanation_english": (
                "Due to a technical issue, live fact-checking could not be completed. "
                "Please verify from reliable sources before forwarding this message."
            ),
            "short_whatsapp_hindi": (
                "⚠️ अभी नेटवर्क समस्या के कारण फैक्ट‑चेक नहीं हो पाया। "
                "कृपया बिना जांचे संदेश आगे न बढ़ाएँ।"
            ),
            "short_whatsapp_english": (
                "⚠️ Fact-check could not be completed due to a technical issue. "
                "Please avoid forwarding unverified messages."
            ),
            "sources": [],
        }


# ---------- Vision: Extract claim from screenshot ----------
def extract_claim_from_image(img_bytes: bytes, client: Optional[Any]) -> str:
    if client is None:
        logger.info("OpenAI client unavailable, returning generic placeholder claim from image.")
        return "Image appears to contain a forwarded message whose exact text could not be extracted."

    logger.info("Extracting claim from image using GPT-4o vision...")
    encoded = base64.b64encode(img_bytes).decode("utf-8")
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a vision assistant. Your ONLY job is to read the image "
                        "and output ONE short sentence summarising the main factual claim or rumour. "
                        "No extra text, no explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Read this screenshot and summarise the core claim / rumour in one sentence.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                        },
                    ],
                },
            ],
        )
        content = resp.choices[0].message.content.strip()
        logger.info("Vision model extracted claim: %s", content)
        return content
    except Exception as e:
        logger.exception("Error during vision extraction: %s", e)
        return "Could not reliably read the screenshot, please type the main claim in your own words."


# ---------- Audio: Transcription and Hindi debunk ----------
def transcribe_audio(file_bytes: bytes, mime_type: str, client: Optional[Any]) -> str:
    if client is None:
        logger.info("OpenAI client unavailable, returning placeholder transcription.")
        return "Placeholder transcription: (OpenAI Whisper is not configured)."

    logger.info("Transcribing audio using Whisper...")
    try:
        # Streamlit file bytes -> in-memory file object
        audio_io = io.BytesIO(file_bytes)
        audio_io.name = f"audio.{mime_type.split('/')[-1]}"
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_io,
        )
        text = getattr(transcription, "text", "").strip()
        logger.info("Whisper transcription completed.")
        return text or "No speech detected in the audio."
    except Exception as e:
        logger.exception("Error during Whisper transcription: %s", e)
        return "Could not transcribe this audio. Please try again or type the claim manually."


def generate_hindi_debunk_audio(text_hindi: str) -> bytes:
    logger.info("Generating Hindi debunk audio using gTTS...")
    tts = gTTS(text=text_hindi, lang="hi")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# ---------- Truth Card image ----------
def generate_truth_card(
    claim: str,
    verdict: str,
    city: Optional[str],
    confidence: int,
) -> bytes:
    logger.info("Generating truth card image...")
    width, height = 800, 450
    bg_color = (249, 250, 251)  # gray-50

    if verdict.lower().startswith("likely true"):
        accent = (22, 163, 74)  # green-600
        verdict_label = "सही / TRUE"
    elif verdict.lower().startswith("likely false"):
        accent = (220, 38, 38)  # red-600
        verdict_label = "गलत / FALSE"
    else:
        accent = (245, 158, 11)  # amber-500
        verdict_label = "संदिग्ध / UNCLEAR"

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Border
    border_width = 12
    draw.rectangle(
        [border_width, border_width, width - border_width, height - border_width],
        outline=accent,
        width=border_width,
    )

    # Fonts (use default PIL fonts to avoid system dependencies)
    try:
        title_font = ImageFont.truetype("arial.ttf", 36)
        verdict_font = ImageFont.truetype("arialbd.ttf", 40)
        body_font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        title_font = ImageFont.load_default()
        verdict_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    padding = 40

    # Header text
    draw.text(
        (padding, padding),
        "SachAI Fact-Check Card",
        font=title_font,
        fill=(15, 23, 42),  # slate-900
    )

    # Verdict pill
    verdict_box_y = padding + 50
    pill_height = 50
    pill_width = 260
    pill_radius = 25
    x0, y0 = padding, verdict_box_y
    x1, y1 = x0 + pill_width, y0 + pill_height
    draw.rounded_rectangle([x0, y0, x1, y1], radius=pill_radius, fill=accent)
    draw.text(
        (x0 + 18, y0 + 12),
        verdict_label,
        font=verdict_font,
        fill=(255, 255, 255),
    )

    # Claim text (wrapped manually)
    claim_y = verdict_box_y + pill_height + 30
    max_width = width - 2 * padding

    def wrap_text(text: str, font_obj: ImageFont.ImageFont, max_w: int) -> List[str]:
        words = text.split()
        lines: List[str] = []
        current = ""
        for w in words:
            test = (current + " " + w).strip()
            if draw.textlength(test, font=font_obj) <= max_w:
                current = test
            else:
                lines.append(current)
                current = w
        if current:
            lines.append(current)
        return lines

    claim_lines = wrap_text(claim[:220], body_font, max_width)
    y = claim_y
    for line in claim_lines:
        draw.text((padding, y), line, font=body_font, fill=(30, 64, 175))  # blue-800
        y += 30

    # Footer meta
    footer_y = height - padding - 70
    now = datetime.now().strftime("%d %b %Y, %H:%M")
    city_str = city or "India"
    footer_text = f"{city_str} • {now} • Confidence: {confidence}%"
    draw.text(
        (padding, footer_y),
        footer_text,
        font=body_font,
        fill=(75, 85, 99),  # gray-600
    )
    draw.text(
        (padding, footer_y + 30),
        "Share this card on WhatsApp to stop the rumour.",
        font=body_font,
        fill=(55, 65, 81),  # gray-700
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# ---------- Streamlit UI ----------
def render_header():
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1rem 0;">
          <h2 style="margin-bottom: 0.25rem;">SachAI – भारत का भरोसेमंद Fact‑Check Shield</h2>
          <p style="color: #6B7280; margin-bottom: 0.5rem;">
            Text • Screenshots • Voice Notes – verified in real‑time against official Indian sources and credible news.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    st.sidebar.markdown("### B2B Analytics – Misinfo Heatmap (Simulated)")

    st.sidebar.markdown("**Tier 2/3 hotspots (last 24h)**")
    cities = ["Meerut", "Indore", "Nagpur", "Patna", "Jaipur", "Varanasi"]
    intensities = [78, 65, 59, 54, 48, 42]
    st.sidebar.bar_chart(
        data={"City": cities, "Rumour Index": intensities},
        x="City",
        y="Rumour Index",
    )

    st.sidebar.markdown("**Key metrics (demo)**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Rumours checked", "1,248", "+128")
    with col2:
        st.metric("Avg. response time", "4.2s", "-0.5s")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**API configuration (per user)**")
    st.sidebar.caption(
        "Paste your own API keys here. They are kept only in this browser session "
        "so each judge/user can use their own limits."
    )
    openai_key_input = st.sidebar.text_input(
        "OpenAI API key (optional)",
        value=st.session_state.get("openai_api_key_ui", ""),
        type="password",
    )
    perplexity_key_input = st.sidebar.text_input(
        "Perplexity API key (optional)",
        value=st.session_state.get("perplexity_api_key_ui", ""),
        type="password",
    )
    # Store back in session_state for use by get_openai_client/get_perplexity_api_key
    st.session_state["openai_api_key_ui"] = openai_key_input.strip() or st.session_state.get(
        "openai_api_key_ui", ""
    )
    st.session_state["perplexity_api_key_ui"] = perplexity_key_input.strip() or st.session_state.get(
        "perplexity_api_key_ui", ""
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Verification settings**")
    st.sidebar.toggle("Show detailed reasoning trail", value=True, key="show_trail")
    st.sidebar.selectbox(
        "Source preference",
        ["Official-first (Gov / PIB)", "Balanced (Gov + news)", "News-first"],
        index=0,
        key="source_pref",
    )
    st.sidebar.selectbox(
        "Language priority",
        ["Hindi-first", "English-first", "Mixed"],
        index=0,
        key="lang_priority",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Demo helper**")
    if st.sidebar.button("Load demo rumour: नमक की कमी, मेरठ", use_container_width=True):
        st.session_state["demo_claim"] = (
            "WhatsApp पर संदेश वायरल है कि मेरठ और आसपास के इलाकों में नमक की भारी कमी होने वाली है, "
            "इसलिए लोग बोरी‑बोरी नमक खरीद कर रख लें।"
        )
        st.session_state["demo_city"] = "Meerut"
        st.sidebar.success("Demo rumour loaded into Text tab.")


def render_proof_section(sources: List[Dict[str, Any]]):
    st.markdown("### Proof – Real-time web + official Indian sources")
    if not sources:
        st.info(
            "No explicit citations were returned. In a production deployment, this section "
            "would list PIB Fact Check links, government portals, and credible news articles."
        )
        return

    for src in sources:
        title = src.get("title", "Source")
        url = src.get("url", "")
        source_type = src.get("source_type", "Web")
        snippet = src.get("snippet", "")
        st.markdown(f"**{title}**  \n[{source_type}]  \n{snippet}")
        if url:
            st.markdown(f"[Open source link]({url})")
        st.markdown("---")


def render_shareable_section(
    claim: str,
    city: Optional[str],
    result: Dict[str, Any],
):
    st.markdown("### Shareable Truth Tools")
    col1, col2 = st.columns(2)

    short_hi = result.get("short_whatsapp_hindi", "")
    short_en = result.get("short_whatsapp_english", "")

    with col1:
        st.markdown("**WhatsApp text – Hindi**")
        st.code(short_hi, language="text")
    with col2:
        st.markdown("**WhatsApp text – English**")
        st.code(short_en, language="text")

    # Truth card image
    verdict = result.get("verdict", "Unclear")
    confidence = int(result.get("confidence", 60))
    img_bytes = generate_truth_card(claim=claim, verdict=verdict, city=city, confidence=confidence)
    st.markdown("**Truth Card (for WhatsApp)**")
    st.image(img_bytes, caption="SachAI Fact-Check Card", use_column_width=True)
    st.download_button(
        "Download Truth Card (PNG)",
        data=img_bytes,
        file_name="sachai_truth_card.png",
        mime="image/png",
    )

    # Hindi audio debunk
    st.markdown("**Audio Debunk (Hindi, ~15 sec)**")
    hindi_expl = result.get("explanation_hindi", "")
    audio_bytes = generate_hindi_debunk_audio(hindi_expl)
    st.audio(audio_bytes, format="audio/mp3")
    st.download_button(
        "Download Hindi Debunk Audio (MP3)",
        data=audio_bytes,
        file_name="sachai_debunk_hi.mp3",
        mime="audio/mpeg",
    )


def render_verdict_section(result: Dict[str, Any]):
    verdict = result.get("verdict", "Unclear")
    confidence = int(result.get("confidence", 60))
    exp_hi = result.get("explanation_hindi", "")
    exp_en = result.get("explanation_english", "")

    if verdict.lower().startswith("likely true"):
        color = "#16A34A"  # green-600
        label = "Likely True"
    elif verdict.lower().startswith("likely false"):
        color = "#DC2626"  # red-600
        label = "Likely False"
    else:
        color = "#F59E0B"  # amber-500
        label = "Unclear / Needs Caution"

    st.markdown(
        f"""
        <div style="
            border-radius: 0.75rem;
            border: 1px solid #E5E7EB;
            padding: 1.25rem;
            background-color: #FFFFFF;
        ">
          <div style="margin-bottom: 0.5rem;">
            <span style="
                background-color: {color}1A;
                color: {color};
                padding: 0.25rem 0.75rem;
                border-radius: 999px;
                font-size: 0.8rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            ">
              {label}
            </span>
          </div>
          <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
            <div style="flex: 1 1 240px;">
              <div style="font-size: 0.8rem; color: #6B7280; text-transform: uppercase; letter-spacing: 0.06em;">
                Verdict in simple Hindi
              </div>
              <p style="margin-top: 0.25rem; color: #111827;">{exp_hi}</p>
            </div>
            <div style="flex: 1 1 240px;">
              <div style="font-size: 0.8rem; color: #6B7280; text-transform: uppercase; letter-spacing: 0.06em;">
                Verdict in simple English
              </div>
              <p style="margin-top: 0.25rem; color: #111827;">{exp_en}</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Confidence**")
    st.progress(confidence / 100)
    st.caption(f"SachAI confidence: {confidence}% (demo scale)")


def handle_text_tab(openai_client: Optional[Any]):
    st.markdown("### Step 1 – Paste the claim or forwarded message")

    default_city = st.session_state.get("demo_city", "Meerut")
    col1, col2 = st.columns([2, 1])
    with col1:
        claim = st.text_area(
            "Claim text (Hindi / English / Mixed)",
            value=st.session_state.get("demo_claim", ""),
            height=140,
        )
    with col2:
        rumour_type = st.selectbox(
            "Rumour type",
            ["Forwarded WhatsApp message", "Social media post", "News headline"],
        )
        city = st.text_input("City / District (for hyperlocal reasoning)", value=default_city)

    lang_pref = st.selectbox("Language hint", ["Mixed", "Hindi", "English"])

    if st.button("Verify with SachAI 🔍", use_container_width=True, key="verify_text"):
        if not claim.strip():
            st.warning("Please paste a claim or message to verify.")
            return

        logger.info("Text tab: starting verification pipeline. Type=%s, City=%s", rumour_type, city)
        with st.spinner("SachAI is checking this claim against live sources..."):
            result = verify_claim_with_perplexity(
                claim,
                language_hint=lang_pref.lower(),
                city=city or None,
            )

        st.markdown("### What SachAI understood")
        st.write(claim.strip())

        render_verdict_section(result)
        render_proof_section(result.get("sources", []))

        render_shareable_section(claim=claim.strip(), city=city or None, result=result)

        if st.session_state.get("show_trail", True):
            with st.expander("Show SachAI's reasoning trail (demo logs)"):
                st.write(
                    "Open the terminal running `streamlit run app.py` to show live logs of "
                    "API calls and decision steps during your presentation."
                )


def handle_screenshot_tab(openai_client: Optional[Any]):
    st.markdown("### Step 1 – Upload the screenshot")
    img_file = st.file_uploader(
        "Upload a screenshot (WhatsApp, Twitter, etc.)",
        type=["png", "jpg", "jpeg"],
        key="screenshot_uploader",
    )
    from_whatsapp = st.checkbox("This is a forwarded WhatsApp screenshot", value=True)
    city = st.text_input("City / District (optional)", value="Meerut", key="city_screenshot")

    manual_claim = st.text_area(
        "Optional: If the screenshot text is unclear, type the main claim here",
        height=100,
    )

    if st.button("Verify screenshot with SachAI 🔍", use_container_width=True, key="verify_image"):
        if img_file is None and not manual_claim.strip():
            st.warning("Please upload a screenshot or type the main claim.")
            return

        img_bytes = img_file.read() if img_file else None
        logger.info("Screenshot tab: starting extraction + verification. WhatsApp=%s", from_whatsapp)

        with st.spinner("Reading screenshot and checking live sources..."):
            if manual_claim.strip():
                claim = manual_claim.strip()
            else:
                claim = extract_claim_from_image(img_bytes, openai_client) if img_bytes else ""
            result = verify_claim_with_perplexity(
                claim,
                language_hint="mixed",
                city=city or None,
            )

        col1, col2 = st.columns(2)
        with col1:
            if img_bytes:
                st.image(img_bytes, caption="Uploaded screenshot", use_column_width=True)
        with col2:
            st.markdown("### What SachAI understood")
            st.write(claim)

        render_verdict_section(result)
        render_proof_section(result.get("sources", []))
        render_shareable_section(claim=claim, city=city or None, result=result)


def handle_voice_tab(openai_client: Optional[Any]):
    st.markdown("### Step 1 – Upload the voice note")
    audio_file = st.file_uploader(
        "Upload a voice note (WhatsApp .ogg / .opus / .mp3 / .wav)",
        type=["ogg", "opus", "mp3", "wav", "m4a"],
        key="voice_uploader",
    )
    lang_hint = st.selectbox("Spoken language", ["Hindi", "Hinglish", "Other"])
    city = st.text_input("City / District (optional)", value="Meerut", key="city_voice")

    if audio_file is not None:
        st.audio(audio_file)

    if st.button("Transcribe & verify with SachAI 🔍", use_container_width=True, key="verify_voice"):
        if audio_file is None:
            st.warning("Please upload a voice note first.")
            return

        file_bytes = audio_file.read()
        mime_type = audio_file.type or "audio/ogg"
        logger.info("Voice tab: starting transcription + verification. Lang=%s, City=%s", lang_hint, city)

        with st.spinner("Transcribing voice note with Whisper..."):
            transcript = transcribe_audio(file_bytes, mime_type, openai_client)

        st.markdown("### Transcription (what SachAI heard)")
        st.write(transcript)

        with st.spinner("Checking this claim against live sources..."):
            result = verify_claim_with_perplexity(
                transcript,
                language_hint=lang_hint.lower(),
                city=city or None,
            )

        render_verdict_section(result)
        render_proof_section(result.get("sources", []))
        render_shareable_section(claim=transcript, city=city or None, result=result)


def main():
    st.set_page_config(
        page_title="SachAI – Fact‑Check Shield",
        page_icon="🛡️",
        layout="wide",
    )

    openai_client = get_openai_client()
    render_header()
    render_sidebar()

    tab1, tab2, tab3 = st.tabs(["Text Claim", "Screenshot", "Voice Note"])

    with tab1:
        handle_text_tab(openai_client)
    with tab2:
        handle_screenshot_tab(openai_client)
    with tab3:
        handle_voice_tab(openai_client)


if __name__ == "__main__":
    main()


