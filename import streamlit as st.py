import streamlit as st
import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Clients: Perplexity for real-time RAG; OpenAI for Vision and Whisper
pplx_client = OpenAI(api_key=os.getenv("PERPLEXITY_API_KEY"), base_url="https://api.perplexity.ai")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="SachAI - India's Fact-Check Shield", layout="wide")

st.title("🛡️ SachAI")
st.markdown("### India's Multilingual AI Misinformation Shield")

# --- Sidebar: Business Dashboard ---
st.sidebar.header("📊 Business Intel")
st.sidebar.metric("Viral Rumors Today", "1,284", "+12%")
st.sidebar.info("Focus: Tier 2/3 Cities (Meerut Hub)")

# --- Main Input Tabs ---
tabs = st.tabs(["📝 Text Claim", "📸 Screenshot", "🎤 Voice Note"])

claim_to_verify = ""

with tabs[0]:
    text_input = st.text_area("Paste a WhatsApp forward:", placeholder="e.g., Free laptops being distributed...")
    if text_input: claim_to_verify = text_input

with tabs[1]:
    uploaded_image = st.file_uploader("Upload WhatsApp screenshot", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        st.image(uploaded_image, width=300)
        if st.button("Extract Text"):
            img_bytes = uploaded_image.read()
            base64_image = base64.b64encode(img_bytes).decode('utf-8')
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": [{"type": "text", "text": "Extract the core news claim from this image."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]} ]
            )
            claim_to_verify = response.choices[0].message.content
            st.info(f"Extracted: {claim_to_verify}")

with tabs[2]:
    audio_file = st.file_uploader("Upload Voice Message", type=["mp3", "wav", "m4a", "ogg"])
    if audio_file:
        st.audio(audio_file)
        if st.button("Transcribe Voice"):
            transcript = openai_client.audio.transcriptions.create(model="whisper-1", file=audio_file)
            claim_to_verify = transcript.text
            st.info(f"Transcribed: {claim_to_verify}")

# --- Verification Engine (Real-time RAG) ---
if claim_to_verify and st.button("🚀 Verify Now"):
    with st.chat_message("assistant"):
        with st.spinner("Searching Official Indian Sources..."):
            response = pplx_client.chat.completions.create(
                model="sonar-pro",
                messages=[
                    {"role": "system", "content": "You are SachAI. Verify claims against PIB Fact Check and Indian news. Output: 1. Verdict (True/Fake/Partial) 2. Simple Explanation in Hindi & English 3. Citations."},
                    {"role": "user", "content": claim_to_verify}
                ]
            )
            verdict = response.choices[0].message.content
            st.markdown(verdict)
            
            # --- Business Value: Shareable Content ---
            st.subheader("📢 Shareable Correction")
            st.code(f"SachAI Verification: {verdict[:150]}...\nVerify more at SachAI.ai")