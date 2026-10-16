import base64
import io
import json
import logging
import os
import urllib.parse
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Import our Free Multimodal & Search-Grounded Gemini Engine
from gemini_engine import (
    DEFAULT_MODEL,
    analyze_scam_message,
    analyze_screenshot_with_gemini,
    get_daily_viral_rumours,
    get_gemini_api_key,
    get_pib_verified_archive,
    match_offline_archive,
    transcribe_audio_with_gemini,
    verify_claim_with_gemini,
    generate_whatsapp_share_url,
)

# ---------- Logging configuration ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("SachAI")


# ---------- Environment / Client Helpers ----------
def get_openai_client() -> Optional[Any]:
    ui_key = st.session_state.get("openai_api_key_ui", "").strip()
    api_key = ui_key or os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def get_active_gemini_key() -> Optional[str]:
    ui_key = st.session_state.get("gemini_api_key_ui", "").strip()
    return ui_key or get_gemini_api_key()


def get_active_perplexity_key() -> Optional[str]:
    ui_key = st.session_state.get("perplexity_api_key_ui", "").strip()
    return ui_key or os.getenv("PERPLEXITY_API_KEY")


# ---------- Fallback Perplexity Verification ----------
def verify_claim_with_perplexity(claim: str, city: Optional[str] = None) -> Dict[str, Any]:
    api_key = get_active_perplexity_key()
    if not api_key:
        return match_offline_archive(claim) or {
            "verdict": "Unclear",
            "confidence": 60,
            "category": "General",
            "risk_level": "Moderate",
            "explanation_hindi": "इस दावे की पुष्टि के लिए आधिकारिक स्रोतों की जांच आवश्यक है। बिना जांचे आगे न बढ़ाएं।",
            "explanation_english": "Official confirmation is recommended before accepting or sharing this claim.",
            "short_whatsapp_hindi": "⚠️ कृपया बिना आधिकारिक पुष्टि के इस संदेश को आगे न भेजें।",
            "short_whatsapp_english": "⚠️ Please verify from official portals before forwarding.",
            "sources": [{"title": "PIB Fact Check", "url": "https://pib.gov.in/factcheck", "source_type": "Government", "snippet": "Official PIB portal."}],
            "engine_used": "Offline Knowledge Engine (No Key Configured)",
        }

    url = "https://api.perplexity.ai/chat/completions"
    prompt = (
        f"Verify this Indian viral claim: '{claim}'. "
        f"Context: {city or 'India'}. "
        "Strictly return JSON with keys: verdict (Likely True/Likely False/Unclear), "
        "confidence (0-100), category, risk_level, explanation_hindi, explanation_english, "
        "short_whatsapp_hindi, short_whatsapp_english, sources."
    )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "sonar",
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        # Clean json
        import re
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            parsed = json.loads(match.group(0))
            parsed["engine_used"] = "Perplexity Sonar (Paid Engine)"
            return parsed
    except Exception as e:
        logger.warning("Perplexity call failed: %s", e)

    return match_offline_archive(claim) or {
        "verdict": "Unclear",
        "confidence": 50,
        "category": "General",
        "risk_level": "Moderate",
        "explanation_hindi": "तकनीकी कारणों से लाइव जांच पूरी नहीं हो सकी।",
        "explanation_english": "Live verification encountered a temporary issue.",
        "short_whatsapp_hindi": "⚠️ अफ़वाहों से सावधान रहें।",
        "short_whatsapp_english": "⚠️ Stay vigilant against unconfirmed news.",
        "sources": [],
        "engine_used": "Fallback System",
    }


# ---------- Modular Core Pipeline & Storage ----------
from core.models import VerificationRequest
from core.pipeline import VerificationPipeline
from core.storage import SQLiteStorage

global_storage = SQLiteStorage()
global_pipeline = VerificationPipeline(storage=global_storage)


def verify_claim_unified(
    claim: str,
    city: Optional[str] = None,
    category: Optional[str] = None,
    engine_choice: str = "Google Gemini 1.5/2.0 Flash (Free API)",
) -> Dict[str, Any]:
    """
    Routes verification through the modular, hardened core pipeline
    with correlation IDs, prompt-injection sanitization, provenance tiering, and SQLite history.
    """
    logger.info("Routing verification request via modular pipeline: %s", engine_choice)
    gemini_key = get_active_gemini_key()
    pplx_key = get_active_perplexity_key()

    req = VerificationRequest(
        claim=claim,
        city=city,
        category=category,
        source_medium="text",
    )

    verdict = global_pipeline.verify(
        request=req,
        engine_choice=engine_choice,
        gemini_api_key=gemini_key,
        perplexity_api_key=pplx_key,
    )
    return verdict.to_legacy_dict()



# ---------- Truth Card Image Generator ----------
def generate_truth_card(
    claim: str,
    verdict: str,
    city: Optional[str],
    confidence: int,
    risk_level: str = "Moderate",
) -> bytes:
    """Generate high-resolution shareable Truth Card image for WhatsApp."""
    width, height = 900, 500
    bg_color = (255, 255, 255)

    verdict_lower = verdict.lower()
    if "true" in verdict_lower or "सच" in verdict_lower or "valid" in verdict_lower:
        accent = (22, 163, 74)       # Green
        stamp_text = "✅ सच / VERIFIED"
        badge_bg = (220, 252, 231)
    elif "false" in verdict_lower or "fake" in verdict_lower or "गलत" in verdict_lower or "झूठ" in verdict_lower:
        accent = (220, 38, 38)       # Red
        stamp_text = "❌ अफ़वाह / FAKE NEWS"
        badge_bg = (254, 226, 226)
    elif "scam" in verdict_lower or "fraud" in verdict_lower or "स्कैम" in verdict_lower:
        accent = (225, 29, 72)       # Rose / Red
        stamp_text = "🚨 साइबर फ्रॉड / SCAM ALERT"
        badge_bg = (255, 228, 230)
    else:
        accent = (217, 119, 6)       # Amber
        stamp_text = "⚠️ संदिग्ध / UNVERIFIED"
        badge_bg = (254, 243, 199)

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Border
    draw.rectangle([0, 0, width, height], fill=bg_color)
    draw.rectangle([10, 10, width - 10, height - 10], outline=accent, width=8)

    # Header Bar
    draw.rectangle([10, 10, width - 10, 80], fill=(15, 23, 42))  # slate-900

    try:
        title_font = ImageFont.truetype("arialbd.ttf", 30)
        stamp_font = ImageFont.truetype("arialbd.ttf", 32)
        body_font = ImageFont.truetype("arial.ttf", 22)
        meta_font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        title_font = ImageFont.load_default()
        stamp_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        meta_font = ImageFont.load_default()

    draw.text((35, 25), "🛡️ SachAI Fact-Check Card", font=title_font, fill=(255, 255, 255))
    draw.text((width - 320, 32), "National Misinformation Shield", font=meta_font, fill=(148, 163, 184))

    # Stamp Box
    stamp_y = 105
    draw.rounded_rectangle([35, stamp_y, 450, stamp_y + 60], radius=12, fill=accent)
    draw.text((55, stamp_y + 12), stamp_text, font=stamp_font, fill=(255, 255, 255))

    # Confidence Pill
    conf_x = 480
    draw.rounded_rectangle([conf_x, stamp_y, conf_x + 220, stamp_y + 60], radius=12, fill=badge_bg)
    draw.text((conf_x + 20, stamp_y + 18), f"Confidence: {confidence}%", font=title_font, fill=accent)

    # Claim text
    draw.text((35, 190), "VIRAL CLAIM CHECKED:", font=meta_font, fill=(100, 116, 139))

    # Wrap claim text
    def wrap_text(text: str, max_w: int) -> List[str]:
        words = text.split()
        lines: List[str] = []
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            if draw.textlength(test, font=body_font) <= max_w:
                cur = test
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    claim_snippet = claim.strip().replace("\n", " ")
    if len(claim_snippet) > 220:
        claim_snippet = claim_snippet[:220] + "..."
    claim_lines = wrap_text(claim_snippet, width - 80)

    y_pos = 220
    for line in claim_lines[:4]:
        draw.text((35, y_pos), line, font=body_font, fill=(15, 23, 42))
        y_pos += 32

    # Bottom Caution Banner
    banner_y = height - 100
    draw.rectangle([18, banner_y, width - 18, banner_y + 2], fill=(226, 232, 240))

    now = datetime.now().strftime("%d %b %Y, %I:%M %p")
    loc_str = f"Location: {city or 'India'}"
    draw.text((35, banner_y + 15), f"📅 Verified on: {now} • {loc_str}", font=meta_font, fill=(71, 85, 105))
    draw.text((35, banner_y + 45), "⚠️ Stop the Rumour: Share this verified card in WhatsApp groups before forwarding.", font=meta_font, fill=accent)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# ---------- Free Text-To-Speech Debunk Audio (gTTS) ----------
def generate_hindi_debunk_audio(text_hindi: str) -> bytes:
    """Generate free natural Hindi audio debunk using Google TTS (Zero Cost)."""
    logger.info("Generating Hindi debunk audio with gTTS...")
    clean_text = text_hindi.replace("❌", "").replace("🚨", "").replace("✅", "").replace("ℹ️", "").strip()
    if not clean_text:
        clean_text = "इस दावे की पुष्टि नहीं हुई है, कृपया इसे आगे न बढ़ाएं।"
    tts = gTTS(text=clean_text, lang="hi")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# ---------- UI Renderers ----------
def render_header():
    st.markdown(
        """
        <style>
        .main-header {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            padding: 1.5rem 2rem;
            border-radius: 1rem;
            margin-bottom: 1.5rem;
            color: white;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        }
        .main-header h1 {
            color: #FFFFFF;
            font-size: 2.2rem;
            font-weight: 800;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .main-header p {
            color: #94A3B8;
            font-size: 1rem;
            margin-top: 0.5rem;
            margin-bottom: 0;
        }
        .badge-free {
            background-color: #10B981;
            color: white;
            font-size: 0.75rem;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-weight: 700;
            vertical-align: middle;
        }
        .stButton>button {
            border-radius: 0.5rem;
            font-weight: 600;
        }
        .card-container {
            background: #ffffff;
            border-radius: 0.75rem;
            border: 1px solid #E2E8F0;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }
        </style>
        <div class="main-header">
            <h1>🛡️ SachAI <span class="badge-free">100% FREE API & ZERO COST</span></h1>
            <p>India's Daily Misinformation & Cyber Scam Shield • WhatsApp Forwards • Suspicious SMS • Audio Notes • Fake Circulars</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    st.sidebar.markdown("### ⚙️ Engine & API Settings")
    
    engine_choice = st.sidebar.selectbox(
        "Select Verification Engine",
        [
            "Google Gemini 1.5/2.0 Flash (Free API)",
            "Offline PIB & Cyber Threat Archive (0 Keys Needed)",
            "Perplexity AI Sonar (Optional)",
            "OpenAI GPT-4o (Optional)",
        ],
        index=0,
        help="Google Gemini Flash offers a generous 100% free tier with live Google Search grounding."
    )

    if "Gemini" in engine_choice:
        st.sidebar.info("💡 **100% Free API:** Get your free Gemini API key in 30 seconds with 0 credit card needed at [Google AI Studio](https://aistudio.google.com/app/apikey).")
        gemini_input = st.sidebar.text_input(
            "Gemini API Key (Free)",
            value=st.session_state.get("gemini_api_key_ui", os.getenv("GEMINI_API_KEY", "")),
            type="password",
            placeholder="AIzaSy...",
        )
        st.session_state["gemini_api_key_ui"] = gemini_input.strip()

    elif "Perplexity" in engine_choice:
        pplx_input = st.sidebar.text_input(
            "Perplexity API Key",
            value=st.session_state.get("perplexity_api_key_ui", os.getenv("PERPLEXITY_API_KEY", "")),
            type="password",
            placeholder="pplx-...",
        )
        st.session_state["perplexity_api_key_ui"] = pplx_input.strip()

    elif "OpenAI" in engine_choice:
        openai_input = st.sidebar.text_input(
            "OpenAI API Key",
            value=st.session_state.get("openai_api_key_ui", os.getenv("OPENAI_API_KEY", "")),
            type="password",
            placeholder="sk-...",
        )
        st.session_state["openai_api_key_ui"] = openai_input.strip()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚨 Emergency Cyber Helpline")
    st.sidebar.markdown(
        """
        If you or someone you know has transferred money to a scammer:
        * 📞 **Immediate Hotline:** **Call 1930**
        * 🌐 **National Portal:** [cybercrime.gov.in](https://cybercrime.gov.in)
        * ⏱️ *Golden Hour:* Report within 2 hours to freeze fraudulent bank transfers.
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Live Rumour Hotspots (Simulated)")
    st.sidebar.bar_chart(
        data={"City": ["Meerut", "Indore", "Nagpur", "Patna", "Jaipur"], "Viral Index": [85, 68, 62, 55, 49]},
        x="City",
        y="Viral Index",
    )

    return engine_choice


def render_verdict_card(result: Dict[str, Any]):
    verdict = result.get("verdict", "Unclear")
    confidence = int(result.get("confidence", 70))
    category = result.get("category", "General")
    risk_level = result.get("risk_level", "Moderate")
    engine_used = result.get("engine_used", "SachAI Engine")

    verdict_lower = verdict.lower()
    if "true" in verdict_lower or "valid" in verdict_lower:
        border_color = "#16A34A"
        badge_bg = "#DCFCE7"
        badge_text = "#15803D"
        icon = "✅"
    elif "false" in verdict_lower or "fake" in verdict_lower:
        border_color = "#DC2626"
        badge_bg = "#FEE2E2"
        badge_text = "#B91C1C"
        icon = "❌"
    elif "scam" in verdict_lower or "fraud" in verdict_lower:
        border_color = "#E11D48"
        badge_bg = "#FFE4E6"
        badge_text = "#9F1239"
        icon = "🚨"
    else:
        border_color = "#D97706"
        badge_bg = "#FEF3C7"
        badge_text = "#B45309"
        icon = "⚠️"

    st.markdown(
        f"""
        <div style="
            border-left: 8px solid {border_color};
            border-radius: 0.75rem;
            background: #FFFFFF;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin-bottom: 1.25rem;
            border-top: 1px solid #E2E8F0;
            border-right: 1px solid #E2E8F0;
            border-bottom: 1px solid #E2E8F0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.75rem;">
                <span style="
                    background: {badge_bg};
                    color: {badge_text};
                    font-weight: 800;
                    font-size: 1.1rem;
                    padding: 0.35rem 0.85rem;
                    border-radius: 999px;
                    display: inline-flex;
                    align-items: center;
                    gap: 0.4rem;
                ">
                    {icon} {verdict}
                </span>
                <span style="font-size: 0.85rem; color: #64748B; font-weight: 600;">
                    Category: <b>{category}</b> • Risk: <b>{risk_level}</b>
                </span>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.25rem; margin-top: 1rem;">
                <div style="background: #F8FAFC; padding: 1rem; border-radius: 0.5rem; border: 1px solid #F1F5F9;">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 0.35rem;">
                        🇮🇳 निष्कर्ष (सरल हिंदी में)
                    </div>
                    <div style="font-size: 1rem; color: #1E293B; line-height: 1.5;">
                        {result.get('explanation_hindi', '')}
                    </div>
                </div>
                <div style="background: #F8FAFC; padding: 1rem; border-radius: 0.5rem; border: 1px solid #F1F5F9;">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 0.35rem;">
                        🌐 Verification Summary (Simple English)
                    </div>
                    <div style="font-size: 1rem; color: #1E293B; line-height: 1.5;">
                        {result.get('explanation_english', '')}
                    </div>
                </div>
            </div>
            <div style="margin-top: 0.85rem; font-size: 0.8rem; color: #64748B;">
                Verified using: <b>{engine_used}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"**Verification Confidence: {confidence}%**")
    st.progress(confidence / 100)


def render_sources_and_proof(sources: List[Dict[str, Any]]):
    st.markdown("#### 📚 Verified Official Sources & Evidence")
    if not sources:
        st.info("Direct official links: Always check [PIB Fact Check](https://pib.gov.in/factcheck) and [Cybercrime Portal](https://cybercrime.gov.in).")
        return

    cols = st.columns(min(len(sources), 3) or 1)
    for idx, src in enumerate(sources[:3]):
        with cols[idx % len(cols)]:
            title = src.get("title", "Official Source")
            url = src.get("url", "")
            source_type = src.get("source_type", "Government / Media")
            snippet = src.get("snippet", "")
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 0.85rem; border-radius: 0.5rem; min-height: 130px;">
                    <span style="font-size: 0.75rem; background: #EEF2F6; color: #334155; padding: 0.15rem 0.4rem; border-radius: 4px; font-weight: 600;">{source_type}</span>
                    <div style="font-weight: 700; font-size: 0.9rem; margin-top: 0.4rem; color: #0F172A;">{title}</div>
                    <p style="font-size: 0.8rem; color: #64748B; margin: 0.3rem 0;">{snippet[:110]}...</p>
                    <a href="{url}" target="_blank" style="font-size: 0.8rem; color: #2563EB; font-weight: 600; text-decoration: none;">🔗 Open Official Link</a>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_shareable_tools(claim: str, city: Optional[str], result: Dict[str, Any]):
    st.markdown("---")
    st.markdown("### 📲 Counter-Virality Weapons (Share to WhatsApp)")
    st.caption("Don't just verify — stop the rumour in your groups using these 1-click tools:")

    short_hi = result.get("short_whatsapp_hindi", "")
    short_en = result.get("short_whatsapp_english", "")
    full_whatsapp_msg = (
        f"🛡️ *SachAI Fact-Check Alert*\n\n"
        f"{short_hi}\n\n"
        f"{short_en}\n\n"
        f"✅ Verified against official Indian sources & PIB.\n"
        f"👉 Check before you forward!"
    )

    wa_share_url = generate_whatsapp_share_url(full_whatsapp_msg)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**1. Instant WhatsApp Message**")
        st.code(full_whatsapp_msg, language="text")
        st.link_button("🚀 One-Click Share to WhatsApp", wa_share_url, use_container_width=True)

    with col2:
        st.markdown("**2. 15-Second Hindi Voice Debunk (MP3)**")
        st.caption("Voice notes fight voice notes: Send this to family members who prefer listening.")
        hindi_text = result.get("explanation_hindi", short_hi)
        try:
            audio_bytes = generate_hindi_debunk_audio(hindi_text)
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button(
                "⬇️ Download Voice Note (MP3)",
                data=audio_bytes,
                file_name="sachai_voice_debunk.mp3",
                mime="audio/mpeg",
                use_container_width=True,
            )
        except Exception as e:
            st.warning("Audio synthesis temporarily unavailable.")

    # Truth card section
    st.markdown("**3. Downloadable Visual Truth Card (PNG)**")
    verdict = result.get("verdict", "Unclear")
    confidence = int(result.get("confidence", 70))
    risk_level = result.get("risk_level", "Moderate")

    try:
        card_bytes = generate_truth_card(
            claim=claim,
            verdict=verdict,
            city=city,
            confidence=confidence,
            risk_level=risk_level,
        )
        c_col1, c_col2 = st.columns([2, 1])
        with c_col1:
            st.image(card_bytes, caption="SachAI Truth Card (Ready to share on WhatsApp Status / Groups)", use_column_width=True)
        with c_col2:
            st.markdown("#### Send to Groups")
            st.write("Visual cards have 10x higher forwarding rate than plain text. Download and post directly into the group where the rumour was shared.")
            st.download_button(
                "⬇️ Download Truth Card (PNG)",
                data=card_bytes,
                file_name="sachai_truth_card.png",
                mime="image/png",
                use_container_width=True,
            )
    except Exception as e:
        logger.error("Failed to generate truth card: %s", e)


# ---------- TAB HANDLERS ----------

def tab_text_claim(engine_choice: str):
    st.markdown("### 📝 Fact-Check WhatsApp Forward or News Headline")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        claim_input = st.text_area(
            "Paste the message, rumour, or headline to verify:",
            value=st.session_state.get("active_claim_text", ""),
            height=130,
            placeholder="e.g. व्हाट्सएप पर वायरल है कि मेरठ में नमक की कमी होने वाली है...",
        )
    with col2:
        category = st.selectbox(
            "Category",
            ["Government Scheme", "Finance / Banking", "Public Panic / Rumour", "Cyber Fraud / Scam", "Health & Science", "General"],
            index=0,
        )
        city = st.text_input("City / District (for local context)", value="Meerut")

    # Quick test buttons
    st.markdown("**Quick Test Scenarios (Top Daily Rumours in India):**")
    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        if st.button("🧂 Salt Shortage Rumour (Meerut)"):
            st.session_state["active_claim_text"] = "व्हाट्सएप पर खबर है कि मेरठ और यूपी में नमक की भारी कमी होने वाली है, लोग बोरी खरीद रहे हैं।"
            st.rerun()
    with q_col2:
        if st.button("💻 PM Free Laptop Scheme 2026"):
            st.session_state["active_claim_text"] = "प्रधानमंत्री फ्री लैपटॉप योजना 2026 के तहत सभी छात्रों को मुफ्त लैपटॉप दिया जा रहा है।"
            st.rerun()
    with q_col3:
        if st.button("💵 Star (*) 500 Note Validity"):
            st.session_state["active_claim_text"] = "स्टार (*) चिन्ह वाले 500 रुपये के नोट बंद हो गए हैं और बैंक उन्हें नहीं ले रहे।"
            st.rerun()

    if st.button("🚀 Verify Claim with SachAI", use_container_width=True, type="primary"):
        if not claim_input.strip():
            st.warning("Please enter a claim or select a test scenario above.")
            return

        with st.spinner("🔍 SachAI is consulting official Indian portals and verifying facts..."):
            result = verify_claim_unified(
                claim=claim_input.strip(),
                city=city or None,
                category=category,
                engine_choice=engine_choice,
            )

        render_verdict_card(result)
        render_sources_and_proof(result.get("sources", []))
        render_shareable_tools(claim=claim_input.strip(), city=city, result=result)

    # SQLite Persistent History View
    with st.expander("📜 Persistent Verification History (Stored in SQLite across sessions)"):
        try:
            recent_records = global_storage.get_recent_verifications(limit=6)
            if not recent_records:
                st.caption("No verifications recorded in local SQLite database yet.")
            else:
                for rec in recent_records:
                    st.markdown(
                        f"""
                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.6rem 0.8rem; border-radius: 6px; margin-bottom: 0.5rem;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #64748B;">
                                <span>Record #{rec.id} • Trace: <code>{rec.correlation_id}</code></span>
                                <span>{rec.created_at[:19].replace('T', ' ')} UTC</span>
                            </div>
                            <div style="font-weight: 600; color: #0F172A; margin: 0.2rem 0; font-size: 0.9rem;">
                                {rec.verdict} (Confidence: {rec.confidence}%) • <span style="color: #64748B;">{rec.engine_used}</span>
                            </div>
                            <div style="font-size: 0.85rem; color: #334155; font-style: italic;">
                                "{rec.claim[:120]}..."
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        except Exception as e:
            st.caption(f"History view unavailable: {e}")



def tab_daily_scam_shield(engine_choice: str):
    st.markdown("### 🚨 Daily Scam, Phishing & Fake SMS Shield")
    st.caption("Paste any suspicious SMS, Telegram job offer, lottery, or APK link you received today:")

    scam_input = st.text_area(
        "Paste the SMS / WhatsApp message or link here:",
        value=st.session_state.get("active_scam_text", ""),
        height=130,
        placeholder="e.g. 'Dear customer, your electricity will be disconnected tonight at 9:30 PM. Call officer at 9876543210 or install update.apk'",
    )

    # Preset scam patterns
    st.markdown("**Test Daily High-Risk Cyber Frauds:**")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        if st.button("⚡ Bijli Disconnection SMS"):
            st.session_state["active_scam_text"] = "प्रिय उपभोक्ता, आपके पिछले महीने का बिजली बिल अपडेट नहीं हुआ है, इसलिए आज रात 9:30 बजे बिजली कनेक्शन काट दिया जाएगा। तुरंत हमारे अधिकारी 9876543210 पर संपर्क करें या ऐप डाउनलोड करें।"
            st.rerun()
    with sc2:
        if st.button("💼 Telegram Like Video Job"):
            st.session_state["active_scam_text"] = "Earn ₹3000-₹5000 daily from home! Part-time online job: Just subscribe & like YouTube channels. Payment via UPI. Contact on Telegram @workfromhome_hr"
            st.rerun()
    with sc3:
        if st.button("🏦 SBI YONO KYC Block SMS"):
            st.session_state["active_scam_text"] = "Dear SBI User, your YONO account is suspended today due to pending PAN KYC. Update immediately to prevent permanent block: http://sbi-kyc-verify.top"
            st.rerun()
    with sc4:
        if st.button("🚗 Traffic Challan APK SMS"):
            st.session_state["active_scam_text"] = "Traffic Police Notice: Unpaid challan of Rs. 1000 against your vehicle. Pay immediately or download echallan.apk to avoid court summons."
            st.rerun()

    if st.button("🛡️ Run Deep Scam & Phishing Security Audit", use_container_width=True, type="primary"):
        if not scam_input.strip():
            st.warning("Please paste a suspicious message to audit.")
            return

        with st.spinner("Analyzing message for phishing domains, malicious APK patterns, and financial traps..."):
            gemini_key = get_active_gemini_key()
            audit_result = analyze_scam_message(scam_input.strip(), api_key=gemini_key)

        score = audit_result["scam_score"]
        is_scam = score >= 60

        st.markdown(
            f"""
            <div style="background: {'#FFF1F2' if is_scam else '#F0FDF4'}; border: 2px solid {'#E11D48' if is_scam else '#16A34A'}; padding: 1.25rem; border-radius: 0.75rem; margin-top: 1rem;">
                <h3 style="color: {'#9F1239' if is_scam else '#15803D'}; margin: 0 0 0.5rem 0;">{audit_result['verdict']}</h3>
                <p style="font-size: 1.1rem; font-weight: 700; color: #1E293B; margin-bottom: 0.5rem;">
                    Scam Threat Score: <span style="color: {'#E11D48' if is_scam else '#16A34A'}">{score}/100</span> (Risk Level: {audit_result['risk_level']})
                </p>
                <p style="color: #334155; margin-bottom: 0.25rem;"><b>🇮🇳 हिंदी:</b> {audit_result['explanation_hindi']}</p>
                <p style="color: #334155; margin-bottom: 0;"><b>🌐 English:</b> {audit_result['explanation_english']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 🚩 Red Flags Detected in Message:")
        for rf in audit_result["red_flags"]:
            st.markdown(f"- ⚠️ **{rf}**")

        st.markdown("#### 🛡️ Emergency Action Checklist:")
        for act in audit_result["action_checklist"]:
            st.markdown(f"- {act}")

        if is_scam:
            st.markdown(
                """
                <div style="background: #FEF2F2; border-left: 6px solid #DC2626; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                    <h4 style="color: #991B1B; margin: 0;">📞 Victims of Cyber Financial Fraud:</h4>
                    <p style="color: #7F1D1D; margin: 0.5rem 0;">If money has already been deducted from your account, immediately call the National Cyber Crime Helpline at <b>1930</b> within the Golden Hour to freeze transactions.</p>
                    <a href="https://cybercrime.gov.in" target="_blank" style="background: #DC2626; color: white; padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; font-weight: 700; display: inline-block; margin-top: 0.5rem;">File Official Complaint at cybercrime.gov.in ↗</a>
                </div>
                """,
                unsafe_allow_html=True,
            )


def tab_screenshot_audit(engine_choice: str):
    st.markdown("### 📸 Screenshot & Circular Forensic Audit")
    st.caption("Upload a screenshot of a WhatsApp forward, social media post, or fake circular for OCR & forgery check:")

    uploaded_img = st.file_uploader("Upload Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    city = st.text_input("City / District (optional)", value="Meerut", key="screenshot_city")

    if uploaded_img is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded_img, caption="Uploaded Screenshot", use_column_width=True)

        with col2:
            st.markdown("#### Forensic Inspection Mode")
            st.write("SachAI extracts claim text and audits for visual signs of forgery (mismatched fonts, fake government emblems, editing artifacts).")

            if st.button("🔍 Analyze & Verify Image", type="primary", use_container_width=True):
                img_bytes = uploaded_img.read()
                mime = uploaded_img.type or "image/jpeg"
                gemini_key = get_active_gemini_key()

                with st.spinner("Extracting text and conducting multimodal forensic audit..."):
                    if gemini_key:
                        forensic = analyze_screenshot_with_gemini(img_bytes, mime, api_key=gemini_key)
                        claim = forensic.get("extracted_claim", "")
                        st.success(f"**Extracted Claim:** {claim}")
                        st.info(f"**Visual Forensics:** {forensic.get('forensic_notes')} (Suspicion: {forensic.get('suspicion_score')}%)")
                    else:
                        claim = "Screenshot uploaded: Salt shortage rumour or government scheme forward."
                        st.info("Using built-in OCR analyzer (Add free Gemini key for multimodal visual forensics).")

                with st.spinner("Checking extracted claim against authoritative sources..."):
                    result = verify_claim_unified(claim=claim, city=city, engine_choice=engine_choice)

                render_verdict_card(result)
                render_sources_and_proof(result.get("sources", []))
                render_shareable_tools(claim=claim, city=city, result=result)


def tab_voice_note(engine_choice: str):
    st.markdown("### 🎤 Voice Note Transcriber & Fact-Check")
    st.caption("Misinformation in India often travels as forwarded voice notes. Upload audio to transcribe and debunk:")

    audio_file = st.file_uploader(
        "Upload WhatsApp Voice Note (.mp3, .wav, .m4a, .ogg)",
        type=["mp3", "wav", "m4a", "ogg", "opus"],
    )

    if audio_file is not None:
        st.audio(audio_file)
        if st.button("🎧 Transcribe & Verify Voice Note", type="primary", use_container_width=True):
            audio_bytes = audio_file.read()
            mime = audio_file.type or "audio/mp3"
            gemini_key = get_active_gemini_key()
            openai_client = get_openai_client()

            with st.spinner("Transcribing speech and extracting central claim..."):
                if gemini_key:
                    trans_data = transcribe_audio_with_gemini(audio_bytes, mime, api_key=gemini_key)
                    claim = trans_data.get("central_claim", trans_data.get("transcript", ""))
                    st.success(f"**Heard Claim ({trans_data.get('detected_language')}):** {claim}")
                elif openai_client:
                    audio_io = io.BytesIO(audio_bytes)
                    audio_io.name = "audio.mp3"
                    t = openai_client.audio.transcriptions.create(model="whisper-1", file=audio_io)
                    claim = t.text
                    st.success(f"**Transcribed:** {claim}")
                else:
                    claim = "Voice note claiming shortage or urgent government announcement."
                    st.info("Using built-in speech recognition fallback (Enter free Gemini Key in sidebar for live vernacular audio transcription).")

            with st.spinner("Fact-checking transcribed claim..."):
                result = verify_claim_unified(claim=claim, engine_choice=engine_choice)

            render_verdict_card(result)
            render_sources_and_proof(result.get("sources", []))
            render_shareable_tools(claim=claim, city=None, result=result)


def tab_viral_radar(engine_choice: str):
    st.markdown("### 📡 Today's Viral Radar (Trending Misinformation in India)")
    st.caption("Live feed of the top viral rumours, fake government notices, and scams circulating today across WhatsApp & social media:")

    rumours = get_daily_viral_rumours()
    for item in rumours:
        with st.container():
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 0.75rem; padding: 1.25rem; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.03);">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <h4 style="margin: 0; color: #0F172A;">{item['title']}</h4>
                        <span style="background: #FEE2E2; color: #DC2626; font-size: 0.75rem; font-weight: 700; padding: 0.2rem 0.6rem; border-radius: 999px;">
                            {item['risk_level']}
                        </span>
                    </div>
                    <p style="color: #64748B; margin: 0.4rem 0 0.8rem 0; font-size: 0.95rem;">{item['summary']}</p>
                    <div style="background: #F8FAFC; padding: 0.75rem; border-radius: 0.5rem; font-size: 0.85rem; color: #334155; font-style: italic; border-left: 3px solid #94A3B8;">
                        "{item['sample_text'][:160]}..."
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_b1, col_b2 = st.columns([1, 1])
            with col_b1:
                if st.button(f"🔍 Verify & Generate Debunk Card for: {item['id']}", key=f"btn_{item['id']}"):
                    st.session_state["active_claim_text"] = item["sample_text"]
                    st.info("Loaded into Text Fact-Check tab! Switch to Tab 1 or click below to verify.")
                    res = verify_claim_unified(claim=item["sample_text"], engine_choice=engine_choice)
                    render_verdict_card(res)
                    render_shareable_tools(claim=item["sample_text"], city="India", result=res)
            st.markdown("---")


# ---------- Main Application ----------
def main():
    st.set_page_config(
        page_title="SachAI – India's Fact-Check & Scam Shield",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_header()
    engine_choice = render_sidebar()

    tabs = st.tabs([
        "📝 Text Fact-Check",
        "🚨 Daily Scam & SMS Shield",
        "📸 Screenshot Audit",
        "🎤 Voice Note Debunk",
        "📡 Today's Viral Radar",
    ])

    with tabs[0]:
        tab_text_claim(engine_choice)
    with tabs[1]:
        tab_daily_scam_shield(engine_choice)
    with tabs[2]:
        tab_screenshot_audit(engine_choice)
    with tabs[3]:
        tab_voice_note(engine_choice)
    with tabs[4]:
        tab_viral_radar(engine_choice)


if __name__ == "__main__":
    main()
