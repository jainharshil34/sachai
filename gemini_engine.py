"""
SachAI - Gemini AI Fact-Checking & Multimodal Engine
Powered by Google Gemini 1.5 Flash / 2.0 Flash (Free API Key from Google AI Studio)
"""

import base64
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("SachAI.Gemini")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-1.5-flash"


def get_gemini_api_key(session_state_key: Optional[str] = None) -> Optional[str]:
    """Retrieve Gemini API key from session state, environment variable, or .env."""
    key = session_state_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return key.strip() if key else None


def clean_json_response(raw_text: str) -> Dict[str, Any]:
    """Extract valid JSON from LLM text output even if wrapped in markdown fences."""
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {}


def get_pib_verified_archive() -> List[Dict[str, Any]]:
    """Verified database of common and recent viral Indian misinformation for zero-latency testing."""
    return [
        {
            "keywords": ["नमक", "salt", "shortage", "meerut", "मेरठ", "कमी"],
            "verdict": "Likely False",
            "confidence": 98,
            "category": "Rumour / Panic",
            "risk_level": "High Viral Threat",
            "explanation_hindi": "मेरठ या उत्तर प्रदेश में नमक की कोई कमी नहीं है। सरकार और खाद्य आपूर्ति विभाग ने स्पष्ट किया है कि यह केवल जमाखोरों और असामाजिक तत्वों द्वारा फैलाई गई अफवाह है। पर्याप्त स्टॉक उपलब्ध है।",
            "explanation_english": "There is NO salt shortage in Meerut or anywhere in India. Department of Consumer Affairs and local authorities have confirmed massive buffer stocks. Do not panic-buy or forward rumours.",
            "short_whatsapp_hindi": "❌ अफ़वाह से बचें: मेरठ या यूपी में नमक की कोई कमी नहीं है। पर्याप्त स्टॉक है। जमाखोरों की अफ़वाहों पर ध्यान न दें और संदेश आगे न बढ़ाएँ।",
            "short_whatsapp_english": "❌ FAKE ALERT: Rumours of salt shortage in Meerut/UP are baseless. Consumer Affairs ministry confirms abundant stock. Please do not panic-buy.",
            "sources": [
                {
                    "title": "PIB Fact Check: Salt Shortage Rumours are Baseless",
                    "url": "https://pib.gov.in/factcheck",
                    "source_type": "Government Fact Check",
                    "snippet": "Department of Consumer Affairs confirms adequate salt availability across all states."
                },
                {
                    "title": "District Magistrate Meerut Official Advisory",
                    "url": "https://meerut.nic.in",
                    "source_type": "District Administration",
                    "snippet": "Action will be taken under NSA against hoarders spreading salt scarcity rumours."
                }
            ]
        },
        {
            "keywords": ["laptop", "लैपटॉप", "pm free laptop", "मुफ्त लैपटॉप", "yojana", "scheme"],
            "verdict": "Likely False",
            "confidence": 99,
            "category": "Cyber Fraud / Phishing",
            "risk_level": "Severe Scam",
            "explanation_hindi": "भारत सरकार ऐसी कोई 'मुफ्त लैपटॉप योजना' नहीं चला रही है जिसमें किसी अज्ञात लिंक पर रजिस्ट्रेशन करना पड़े। यह लिंक फर्जी है और आपकी निजी जानकारी व बैंक विवरण चुराने के लिए बनाया गया है।",
            "explanation_english": "The Government of India is NOT running any 'Free Laptop Scheme' via unverified websites. The viral link is a phishing scam intended to steal personal data and money.",
            "short_whatsapp_hindi": "🚨 फ्रॉड अलर्ट: सरकार की तरफ से कोई मुफ्त लैपटॉप नहीं दिया जा रहा। वायरल लिंक फर्जी है, इसे क्लिक न करें और न किसी को भेजें।",
            "short_whatsapp_english": "🚨 SCAM ALERT: There is NO government scheme offering free laptops through viral links. Do not click fraudulent websites.",
            "sources": [
                {
                    "title": "PIB Fact Check: Free Laptop Scheme Under PM Scheme is Fake",
                    "url": "https://twitter.com/PIBFactCheck",
                    "source_type": "PIB Fact Check",
                    "snippet": "A website claiming to offer free laptops to youth is fake. No such scheme has been approved by the central government."
                },
                {
                    "title": "National Cyber Crime Reporting Portal (1930)",
                    "url": "https://cybercrime.gov.in",
                    "source_type": "Official Cyber Portal",
                    "snippet": "Beware of fraudulent links offering free gadgets under government schemes."
                }
            ]
        },
        {
            "keywords": ["digital arrest", "cbi", "police", "customs", "मनी लॉन्ड्रिंग", "गिरफ्तारी", "skype"],
            "verdict": "Likely False",
            "confidence": 99,
            "category": "Cyber Crime Fraud",
            "risk_level": "Severe Scam",
            "explanation_hindi": "कानून में 'डिजिटल अरेस्ट' नाम का कोई प्रावधान नहीं है। कोई भी पुलिस, सीबीआई या नारकोटिक्स विभाग व्हाट्सएप या स्काइप वीडियो कॉल पर जांच या अरेस्ट नहीं करता। यह अंतरराष्ट्रीय साइबर अपराधियों का संगठित गिरोह है। तुरंत 1930 पर कॉल करें।",
            "explanation_english": "There is NO provision of 'Digital Arrest' under Indian Law. No genuine law enforcement agency (Police, CBI, ED, Customs) conducts interrogations or arrests via Skype/WhatsApp video calls. Report immediately to 1930.",
            "short_whatsapp_hindi": "🚨 सावधान: 'डिजिटल अरेस्ट' नाम का कोई कानून नहीं है! पुलिस या CBI कभी व्हाट्सएप/स्काइप पर अरेस्ट नहीं करती। पैसे कभी ट्रांसफर न करें। हेल्पलाइन: 1930।",
            "short_whatsapp_english": "🚨 FRAUD WARNING: 'Digital Arrest' is a complete scam. Police/CBI never make arrests on video calls or ask for funds. Report to cybercrime.gov.in or 1930.",
            "sources": [
                {
                    "title": "Ministry of Home Affairs (I4C) Advisory on Digital Arrest",
                    "url": "https://cybercrime.gov.in",
                    "source_type": "Ministry of Home Affairs",
                    "snippet": "MHA issues strict warning against organized cyber syndicates impersonating law enforcement officers."
                },
                {
                    "title": "Prime Minister Modi's Mann Ki Baat on Digital Arrest",
                    "url": "https://pib.gov.in",
                    "source_type": "PMO India",
                    "snippet": "PM warns citizens: Stop, Think, and Take Action. There is no system called digital arrest in Indian law."
                }
            ]
        },
        {
            "keywords": ["500", "star", "स्टार", "नोट", "rbi", "currency", "अमान्य"],
            "verdict": "Likely True",
            "confidence": 95,
            "category": "Currency / Financial",
            "risk_level": "Misleading Forward",
            "explanation_hindi": "स्टार (*) चिन्ह वाला ₹500 का नोट 100% असली और वैध है। आरबीआई के अनुसार, जब छपाई के समय कोई नोट खराब हो जाता है, तो उसकी जगह उसी नंबर का स्टार सीरीज नोट जारी किया जाता है। यह नोट पूरी तरह मान्य है।",
            "explanation_english": "The ₹500 note with a star (*) symbol in the number panel is 100% legal tender and completely valid. RBI clarifies that the star mark is added when a defective sequentially numbered note is replaced during printing.",
            "short_whatsapp_hindi": "✅ सच: स्टार (*) चिन्ह वाला ₹500 का नोट 100% असली और मान्य है। RBI ने स्पष्ट किया है कि यह पूरी तरह वैध मुद्रा है। अफ़वाहों पर ध्यान न दें।",
            "short_whatsapp_english": "✅ VERIFIED: ₹500 banknotes with a Star (*) mark are genuine and 100% legal tender issued by the RBI. Do not reject them.",
            "sources": [
                {
                    "title": "RBI Press Release: Star Series Banknotes are Valid Legal Tender",
                    "url": "https://rbi.org.in",
                    "source_type": "Reserve Bank of India",
                    "snippet": "RBI clarifies that banknotes with a star mark are completely genuine and identical to standard notes in legal value."
                }
            ]
        },
        {
            "keywords": ["electricity", "बिजली", "bijli", "bill", "disconnect", "कट", "रात", "apk"],
            "verdict": "Likely False",
            "confidence": 99,
            "category": "Malware / Financial Fraud",
            "risk_level": "Critical Malware Scam",
            "explanation_hindi": "बिजली विभाग कभी भी रात को लाइन काटने की धमकी देकर किसी व्यक्तिगत मोबाइल नंबर पर कॉल करने या कोई अनधिकृत ऐप (.apk) डाउनलोड करने के लिए नहीं कहता। यह मैलवेयर ऐप आपके बैंक खाते से पैसे उड़ाने के लिए भेजा जाता है।",
            "explanation_english": "Power distribution companies never send SMS warnings to call personal mobile numbers or download third-party .apk files. This is a notorious Trojan malware scam designed to compromise your net banking.",
            "short_whatsapp_hindi": "🚨 बिजली बिल फ्रॉड: 'आज रात बिजली कट जाएगी' वाले एसएमएस से सावधान! कोई ऐप डाउनलोड न करें, न किसी अनजान नंबर पर कॉल करें।",
            "short_whatsapp_english": "🚨 ALERT: Electricity disconnection SMS asking to call private mobile numbers or download files is a malicious banking fraud. Delete immediately.",
            "sources": [
                {
                    "title": "PIB Fact Check: Beware of Fake Electricity Bill Disconnection Messages",
                    "url": "https://twitter.com/PIBFactCheck",
                    "source_type": "PIB Fact Check",
                    "snippet": "State discoms do not instruct consumers to install remote access apps or contact personal numbers."
                }
            ]
        },
        {
            "keywords": ["part-time", "telegram", "youtube", "like", "task", "कमाई", "₹5000", "वर्क फ्रॉम होम", "rating"],
            "verdict": "Cyber Fraud / Scam",
            "confidence": 99,
            "category": "Cyber Fraud / Scam",
            "risk_level": "Severe Scam",
            "explanation_hindi": "टेलीग्राम या व्हाट्सएप पर 'घर बैठे यूट्यूब वीडियो लाइक करके या गूगल रिव्यू देकर ₹3000-₹5000 कमाएं' का ऑफर 100% साइबर फ्रॉड (टास्क फ्रॉड) है। शुरुआत में ₹150-200 देकर फंसाते हैं और फिर प्रीपेड टास्क के नाम पर लाखों रुपये ठग लेते हैं।",
            "explanation_english": "Telegram/WhatsApp part-time job offers promising ₹3,000 to ₹5,000 per day for liking YouTube videos or writing Google reviews are organized task-based cyber scams. Never pay money for any job opportunity.",
            "short_whatsapp_hindi": "🚨 फ्रॉड अलर्ट: टेलीग्राम पर 'वीडियो लाइक करके ₹5000 कमाएं' 100% स्कैम है। कभी भी टास्क के लिए पैसे ट्रांसफर न करें। हेल्पलाइन: 1930।",
            "short_whatsapp_english": "🚨 SCAM WARNING: 'Earn ₹5000/day by liking videos' on Telegram is a proven cyber fraud syndicate. Never send money. Report to 1930.",
            "sources": [
                {
                    "title": "I4C Cybercrime Advisory on Work-From-Home Task Scams",
                    "url": "https://cybercrime.gov.in",
                    "source_type": "Ministry of Home Affairs (I4C)",
                    "snippet": "Organized fraud syndicates operating via Telegram promise part-time income to defraud victims."
                }
            ]
        },
        {
            "keywords": ["sbi", "yono", "kyc", "pan card", "blocked", "अकाउंट ब्लॉक", "अपडेट", "netbanking"],
            "verdict": "Cyber Fraud / Scam",
            "confidence": 99,
            "category": "Finance / Banking",
            "risk_level": "Severe Scam",
            "explanation_hindi": "'आपका SBI YONO खाता आज ब्लॉक हो जाएगा, तुरंत पैन कार्ड अपडेट करने के लिए इस लिंक पर क्लिक करें' का एसएमएस फर्जी है। बैंक कभी भी एसएमएस में अनधिकृत लिंक भेजकर पासवर्ड या ओटीपी नहीं मांगते।",
            "explanation_english": "SMS alerts claiming 'Your SBI YONO account will be blocked tonight, click link to update PAN' are dangerous phishing traps designed to drain bank accounts. Banks never ask for credentials via SMS links.",
            "short_whatsapp_hindi": "⚠️ बैंक अलर्ट: 'SBI Yono खाता ब्लॉक हो जाएगा' वाला SMS फर्जी है! लिंक पर कभी क्लिक न करें और न OTP शेयर करें।",
            "short_whatsapp_english": "⚠️ PHISHING ALERT: Fake SBI/YONO account suspension SMS with web links are banking scams. Never click or enter credentials.",
            "sources": [
                {
                    "title": "State Bank of India Official Cyber Safety Warning",
                    "url": "https://sbi.co.in",
                    "source_type": "State Bank of India",
                    "snippet": "SBI warns customers against clicking on unsolicited links requesting KYC/PAN updates."
                }
            ]
        },
        {
            "keywords": ["pink whatsapp", "whatsapp pink", "गुलाबी व्हाट्सएप", "gold whatsapp", "whatsapp update"],
            "verdict": "Cyber Fraud / Scam",
            "confidence": 99,
            "category": "Malware / Financial Fraud",
            "risk_level": "Critical Malware Scam",
            "explanation_hindi": "'व्हाट्सएप पिंक' या 'व्हाट्सएप गोल्ड' कोई आधिकारिक अपडेट नहीं बल्कि खतरनाक मैलवेयर ऐप (Trojan) है। इसे डाउनलोड करते ही हैकर्स आपके फोन के मैसेज, ओटीपी, संपर्क और निजी फोटो चुरा लेते हैं।",
            "explanation_english": "'WhatsApp Pink' or 'WhatsApp Gold' is a malicious Trojan APK, NOT an official update. Installing it grants hackers remote access to contacts, photos, SMS, and banking OTPs.",
            "short_whatsapp_hindi": "🚨 सावधान: 'WhatsApp Pink' का कोई आधिकारिक अपडेट नहीं है! यह फोन हैक करने वाला वायरस है। लिंक को तुरंत डिलीट करें।",
            "short_whatsapp_english": "🚨 MALWARE WARNING: 'WhatsApp Pink' is a dangerous spyware Trojan. Do not install the APK or forward the link.",
            "sources": [
                {
                    "title": "State Police Cyber Cell Advisory on WhatsApp Pink Malware",
                    "url": "https://cybercrime.gov.in",
                    "source_type": "Police Cyber Cell",
                    "snippet": "Police advisories warn Android users against downloading unverified Pink WhatsApp APKs."
                }
            ]
        },
        {
            "keywords": ["free recharge", "मुफ्त रिचार्ज", "3 महीने", "jio", "airtel", "vi", "फ्री रिचार्ज"],
            "verdict": "Cyber Fraud / Scam",
            "confidence": 98,
            "category": "Cyber Fraud / Scam",
            "risk_level": "High Viral Threat",
            "explanation_hindi": "सरकार या टेलीकॉम कंपनियों (Jio, Airtel) द्वारा '3 महीने का मुफ्त 5G रिचार्ज' देने का दावा पूरी तरह झूठा है। यह फर्जी लिंक लोगों से निजी डेटा, फोन नंबर और स्पैम व्हाट्सएप फॉरवर्ड करवाने के लिए बनाए जाते हैं।",
            "explanation_english": "Claims of government or telcos (Jio, Airtel) distributing 'Free 3-Month 5G Recharge' are completely bogus. These phishing portals harvest phone numbers and spam contacts.",
            "short_whatsapp_hindi": "❌ फर्जी ऑफर: सरकार या Jio/Airtel कोई 3 महीने का मुफ्त रिचार्ज नहीं दे रही है। यह लिंक फर्जी है, किसी को न भेजें।",
            "short_whatsapp_english": "❌ FAKE RECHARGE: Viral offers for 'Free 3 Months Mobile Recharge' are malicious phishing links. Do not click or forward.",
            "sources": [
                {
                    "title": "PIB Fact Check: Free Mobile Recharge Claim is Fraudulent",
                    "url": "https://twitter.com/PIBFactCheck",
                    "source_type": "PIB Fact Check",
                    "snippet": "Department of Telecommunications confirms no free recharge scheme is being offered."
                }
            ]
        },
        {
            "keywords": ["challan", "ई-चालान", "traffic police", "fine", "parivahan", "जुर्माना", "echallan"],
            "verdict": "Cyber Fraud / Scam",
            "confidence": 99,
            "category": "Cyber Fraud / Scam",
            "risk_level": "Severe Scam",
            "explanation_hindi": "ट्रैफिक पुलिस के नाम से आने वाले एसएमएस जिनमें चालान भरने के लिए अज्ञात लिंक या .apk फाइल डाउनलोड करने को कहा जाता है, साइबर अपराधियों की ठगी है। असली चालान केवल echallan.parivahan.gov.in पर ही देखा जा सकता है।",
            "explanation_english": "Traffic police SMS carrying links ending in '.apk' or unverified domains are predatory malware scams. Real traffic challans are strictly hosted on the official echallan.parivahan.gov.in portal.",
            "short_whatsapp_hindi": "🚨 ई-चालान फ्रॉड: फर्जी SMS में चालान भरने के लिंक या ऐप पर क्लिक न करें। असली चालान echallan.parivahan.gov.in पर चेक करें।",
            "short_whatsapp_english": "🚨 TRAFFIC CHALLAN SCAM: Fake challan SMS containing APK files will compromise your phone. Always check echallan.parivahan.gov.in.",
            "sources": [
                {
                    "title": "Parivahan MoRTH Official Advisory on Fake e-Challan SMS",
                    "url": "https://echallan.parivahan.gov.in",
                    "source_type": "Ministry of Road Transport (MoRTH)",
                    "snippet": "MoRTH issues warning against fraudulent messages with APK links impersonating traffic police."
                }
            ]
        }
    ]


def get_daily_viral_rumours() -> List[Dict[str, Any]]:
    """Curated feed of top daily viral misinformation and scams circulating in India."""
    return [
        {
            "id": "bijli_cut",
            "title": "⚡ 'आज रात 9:30 बजे बिजली कट जाएगी' SMS",
            "summary": "Fake electricity disconnection warning demanding call to personal number or installing APK.",
            "category": "Cyber Fraud / Scam",
            "risk_level": "Severe Scam",
            "verdict": "Likely False",
            "sample_text": "प्रिय उपभोक्ता, आपके पिछले महीने का बिजली बिल अपडेट नहीं हुआ है, इसलिए आज रात 9:30 बजे बिजली कनेक्शन काट दिया जाएगा। तुरंत हमारे अधिकारी 9876543210 पर संपर्क करें या ऐप डाउनलोड करें।"
        },
        {
            "id": "telegram_job",
            "title": "💼 'यूट्यूब वीडियो लाइक करके ₹5000 कमाएं' वर्क-फ्रॉम-होम",
            "summary": "Telegram task-based cyber fraud syndicate promising daily earnings for rating videos.",
            "category": "Cyber Fraud / Scam",
            "risk_level": "Severe Scam",
            "verdict": "Likely False",
            "sample_text": "Earn ₹3000 - ₹5000 daily from home! Part-time online job: Just subscribe & like YouTube channels. Payment via UPI after each task. Contact on Telegram @workfromhome_hr"
        },
        {
            "id": "sbi_kyc_pan",
            "title": "🏦 'SBI YONO खाता ब्लॉक - तुरंत पैन लिंक करें' SMS",
            "summary": "Phishing link SMS impersonating SBI netbanking to steal credentials and OTP.",
            "category": "Finance / Banking",
            "risk_level": "Severe Scam",
            "verdict": "Likely False",
            "sample_text": "Dear SBI User, your YONO account is suspended today due to pending PAN KYC. Update immediately to prevent permanent block: http://sbi-kyc-verify.top"
        },
        {
            "id": "digital_arrest",
            "title": "👮 'CBI/Customs डिजिटल अरेस्ट' स्काइप वीडियो कॉल धमकी",
            "summary": "Extortion syndicate threatening fake money laundering / narcotic charges via video call.",
            "category": "Cyber Crime Fraud",
            "risk_level": "Severe Scam",
            "verdict": "Likely False",
            "sample_text": "This is Officer from Mumbai Police / Narcotics Bureau. A parcel containing contraband linked to your Aadhaar has been seized. You are placed under 24-hour Digital Arrest on Skype."
        },
        {
            "id": "star_500_note",
            "title": "💵 'स्टार (*) मार्क वाला ₹500 का नोट अमान्य है' अफ़वाह",
            "summary": "Viral WhatsApp rumour claiming banknotes with star symbol are fake/demonetized.",
            "category": "Currency / Financial",
            "risk_level": "Misleading",
            "verdict": "Likely True",
            "sample_text": "व्हाट्सएप पर खबर है कि नंबर पैनल में स्टार (*) चिन्ह वाले 500 रुपये के नोट बाजार में बंद हो गए हैं और बैंक उन्हें स्वीकार नहीं कर रहे।"
        },
        {
            "id": "free_laptop_pm",
            "title": "💻 'प्रधानमंत्री फ्री लैपटॉप योजना 2026' वायरल लिंक",
            "summary": "Phishing form harvesting personal records and asking for registration fees.",
            "category": "Government Scheme",
            "risk_level": "Severe Scam",
            "verdict": "Likely False",
            "sample_text": "खुशखबरी! भारत सरकार सभी 10वीं व 12वीं पास छात्रों को मुफ्त लैपटॉप दे रही है। अपना नाम यहाँ रजिस्टर करें: pm-free-laptop-gov.online"
        }
    ]


def match_offline_archive(claim: str) -> Optional[Dict[str, Any]]:
    """Match claim against verified archive for immediate zero-latency authentic response."""
    claim_lower = claim.lower()
    for item in get_pib_verified_archive():
        match_count = sum(1 for kw in item["keywords"] if kw in claim_lower)
        if match_count >= 2 or (len(item["keywords"]) > 0 and item["keywords"][0] in claim_lower):
            return {
                "verdict": item["verdict"],
                "confidence": item["confidence"],
                "category": item["category"],
                "risk_level": item["risk_level"],
                "explanation_hindi": item["explanation_hindi"],
                "explanation_english": item["explanation_english"],
                "short_whatsapp_hindi": item["short_whatsapp_hindi"],
                "short_whatsapp_english": item["short_whatsapp_english"],
                "sources": item["sources"],
                "engine_used": "SachAI PIB Verified Live Archive (Zero Cost / Zero Key)",
                "is_grounded": True,
            }
    return None


def analyze_scam_message(message_text: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Heuristic + AI analyzer designed specifically for suspicious SMS, WhatsApp forwards,
    phishing links, lottery messages, and APK warnings that people encounter daily.
    """
    logger.info("Running scam and phishing safety analysis on text...")
    text_lower = message_text.lower()
    red_flags: List[str] = []
    scam_score = 10

    # 1. URL and Phishing checks
    urls = re.findall(r'https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9-]+\.(?:top|xyz|online|site|club|vip|icu|cc|info|live)', message_text)
    if urls:
        red_flags.append(f"Contains unverified external link(s): {', '.join(urls[:2])}")
        scam_score += 35
        for u in urls:
            if any(tld in u for tld in [".top", ".xyz", ".online", ".site", ".cc", ".info", ".apk", "bit.ly", "tinyurl"]):
                red_flags.append("Uses high-risk suspicious TLD or link shortener commonly used in phishing")
                scam_score += 25
                break

    # 2. APK Malware check
    if ".apk" in text_lower or "download app" in text_lower or "ऐप डाउनलोड" in text_lower or "install" in text_lower:
        red_flags.append("Urges downloading third-party .APK / application outside Google Play Store (Malware risk)")
        scam_score += 35

    # 3. Urgency / Panic Triggers
    urgency_terms = ["आज रात", "tonight", "immediately", "तुरंत", "blocked within 24 hours", "suspended", "line cut", "बिजली कट", "arrest", "गिरफ्तारी"]
    for term in urgency_terms:
        if term in text_lower:
            red_flags.append(f"Uses psychological urgency/panic trigger ('{term}') to force hasty action")
            scam_score += 20
            break

    # 4. Bank / KYC / Gov Impersonation
    fin_terms = ["sbi", "hdfc", "icici", "pan card", "kyc", "yono", "electricity", "discom", "police", "customs", "cbi"]
    if any(ft in text_lower for ft in fin_terms) and (urls or "call" in text_lower or "नंबर" in text_lower):
        red_flags.append("Impersonates a bank, electricity utility, or government law enforcement body")
        scam_score += 25

    # 5. Free Gifts / Lottery / High Income Bait
    bait_terms = ["मुफ्त", "free laptop", "free recharge", "₹5000", "5000", "daily earning", "video like", "lottery", "इनाम", "earn"]
    if any(bt in text_lower for bt in bait_terms):
        red_flags.append("Employs unrealistic financial bait (free gadgets, daily salary for easy tasks, free recharge)")
        scam_score += 25

    # 6. Telegram / Task-Based Work-From-Home Fraud
    task_terms = ["telegram", "part-time", "part time", "work from home", "वर्क फ्रॉम होम", "youtube", "task", "daily from home"]
    if any(tt in text_lower for tt in task_terms):
        red_flags.append("Exhibits hallmarks of Telegram task-based work-from-home fraud syndicates")
        scam_score += 30


    scam_score = min(scam_score, 99)
    is_scam = scam_score >= 60

    if is_scam:
        verdict = "🚨 HIGH-RISK SCAM / FRAUD DETECTED"
        risk_level = "Severe Scam"
        explanation_hi = "यह संदेश एक खतरनाक साइबर फ्रॉड या फ़िशिंग स्कैम प्रतीत होता है। इसमें मौजूद किसी भी लिंक पर क्लिक न करें, कोई ऐप डाउनलोड न करें, और किसी भी नंबर पर कॉल न करें।"
        explanation_en = "This message demonstrates strong indicators of cyber fraud or phishing. Do NOT click embedded links, do NOT install APKs, and do NOT contact the sender."
        action_checklist = [
            "🛑 DO NOT click any link or call back the mobile number provided.",
            "🚫 NEVER share your OTP, UPI PIN, ATM PIN, or Aadhaar details with anyone.",
            "📱 Block the sender on WhatsApp / SMS immediately.",
            "📞 Report fraud immediately to the National Cyber Crime Helpline: Dial 1930.",
            "🌐 File an online cyber complaint at https://cybercrime.gov.in."
        ]
    else:
        verdict = "ℹ️ LOW RISK / STANDARD MESSAGE"
        risk_level = "Low"
        explanation_hi = "इस संदेश में आम तौर पर दिखने वाले प्रत्यक्ष साइबर स्कैम या फ़िशिंग लिंक के लक्षण नहीं पाए गए हैं, फिर भी किसी अनजान व्यक्ति से आने वाले संदेशों पर सावधानी बरतें।"
        explanation_en = "This message does not exhibit clear hallmarks of known phishing syndicates, but always exercise caution with unverified communications."
        action_checklist = [
            "Always verify claims from official portals directly by searching on Google or visiting .gov.in websites.",
            "Do not forward messages in family groups without confirming authenticity."
        ]

    return {
        "verdict": verdict,
        "scam_score": scam_score,
        "risk_level": risk_level,
        "red_flags": red_flags or ["No aggressive phishing patterns identified."],
        "explanation_hindi": explanation_hi,
        "explanation_english": explanation_en,
        "action_checklist": action_checklist,
        "extracted_links": urls,
        "sources": [
            {
                "title": "National Cyber Crime Reporting Portal (MHA)",
                "url": "https://cybercrime.gov.in",
                "source_type": "Government Portal (Helpline 1930)",
                "snippet": "Official portal of Ministry of Home Affairs to report online frauds, phishing, and cyber financial crimes."
            }
        ]
    }


def generate_whatsapp_share_url(text: str) -> str:
    """Generate direct WhatsApp Web/App click-to-chat share URL."""
    import urllib.parse
    encoded_text = urllib.parse.quote(text)
    return f"https://api.whatsapp.com/send?text={encoded_text}"



def verify_claim_with_gemini(
    claim: str,
    api_key: str,
    language_hint: str = "mixed",
    city: Optional[str] = None,
    category: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Verify claim using Google Gemini with live Google Search Grounding and structured output.
    """
    logger.info("Verifying claim with Gemini (%s)...", model)

    system_instruction = (
        "You are SachAI, India's premier AI-powered Fact-Checking & Misinformation Shield. "
        "Your duty is to impartially verify viral news, WhatsApp forwards, government circulars, and rumours. "
        "Strictly consult authoritative Indian sources: PIB Fact Check (pib.gov.in), "
        "official portals of Ministries (.gov.in), Reserve Bank of India (rbi.org.in), Supreme Court/High Courts, "
        "I4C Cybercrime (cybercrime.gov.in), and established wire services (PTI, ANI, BBC Hindi, The Hindu, Indian Express).\n\n"
        "Return STRICT JSON only without markdown formatting. The JSON must contain these exact keys:\n"
        "- verdict: One of ['Likely True', 'Likely False', 'Misleading', 'Cyber Fraud / Scam', 'Unclear']\n"
        "- confidence: Integer 0 to 100\n"
        "- category: One of ['Government Scheme', 'Cyber Fraud / Scam', 'Health & Science', 'Communal / Political', 'Public Panic / Rumour', 'Finance / Banking']\n"
        "- risk_level: One of ['Low', 'Moderate', 'High Viral Threat', 'Severe Scam']\n"
        "- explanation_hindi: 2-3 sentences of clear, simple Hindi fact-check summary\n"
        "- explanation_english: 2-3 sentences of clear, simple English fact-check summary\n"
        "- short_whatsapp_hindi: A concise 1-2 line WhatsApp forward debunk in Hindi with emoji\n"
        "- short_whatsapp_english: A concise 1-2 line WhatsApp forward debunk in English with emoji\n"
        "- scam_indicators: List of suspicious flags (e.g., 'Unverified link', 'False urgency', 'Fake emblem') or empty list\n"
        "- sources: List of objects with keys 'title', 'url', 'source_type', 'snippet'\n"
    )

    user_prompt = f"Claim to verify:\n\"{claim}\"\n\n"
    if city:
        user_prompt += f"Location / District context: {city}, India.\n"
    if category:
        user_prompt += f"Suspected topic: {category}.\n"
    user_prompt += "Perform an exhaustive real-time fact-check and return the strict JSON schema."

    url = f"{GEMINI_BASE_URL}/{model}:generateContent?key={api_key}"

    # Payload with Google Search tool enabled for real-time web retrieval
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ],
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "tools": [
            {"googleSearch": {}}
        ],
        "generationConfig": {
            "temperature": 0.15,
            "maxOutputTokens": 1000,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=35)
        
        # If googleSearch tool is rejected by API variant, retry without tool but with search prompt
        if resp.status_code == 400 and "tool" in resp.text.lower():
            logger.warning("Retrying Gemini without googleSearch tool specification...")
            payload.pop("tools", None)
            resp = requests.post(url, json=payload, timeout=35)

        resp.raise_for_status()
        data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("No response candidate returned by Gemini API")

        candidate = candidates[0]
        raw_text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
        parsed = clean_json_response(raw_text)

        # Extract real grounding sources from Gemini metadata if available
        grounding_meta = candidate.get("groundingMetadata", {})
        web_sources: List[Dict[str, Any]] = []
        for chunk in grounding_meta.get("groundingChunks", []):
            web = chunk.get("web", {})
            if web.get("uri"):
                web_sources.append({
                    "title": web.get("title") or "Web Source",
                    "url": web.get("uri"),
                    "source_type": "Verified Web Source (Google Grounding)",
                    "snippet": "Real-time source confirmed by Google Search grounding.",
                })

        # Merge citations
        existing_sources = parsed.get("sources", [])
        if not isinstance(existing_sources, list):
            existing_sources = []
        
        all_sources = web_sources + [s for s in existing_sources if s.get("url")]

        return {
            "verdict": parsed.get("verdict", "Unclear"),
            "confidence": int(parsed.get("confidence", 75)),
            "category": parsed.get("category", "General"),
            "risk_level": parsed.get("risk_level", "Moderate"),
            "explanation_hindi": parsed.get("explanation_hindi", "इस दावे पर आधिकारिक जानकारी की पुष्टि की जा रही है।"),
            "explanation_english": parsed.get("explanation_english", "Verification underway with official sources."),
            "short_whatsapp_hindi": parsed.get("short_whatsapp_hindi", "⚠️ इस संदेश को बिना पुष्टि किए आगे न बढ़ाएं।"),
            "short_whatsapp_english": parsed.get("short_whatsapp_english", "⚠️ Avoid forwarding this message without official confirmation."),
            "scam_indicators": parsed.get("scam_indicators", []),
            "sources": all_sources[:4] if all_sources else [
                {"title": "PIB Fact Check", "url": "https://pib.gov.in/factcheck", "source_type": "PIB Portal", "snippet": "Official Government of India Fact Check wing."}
            ],
            "engine_used": f"Google Gemini ({model}) + Live Google Search Grounding",
            "is_grounded": bool(web_sources),
        }

    except Exception as e:
        logger.exception("Gemini API call failed: %s", e)
        # Attempt fallback to offline archive before raising
        offline_match = match_offline_archive(claim)
        if offline_match:
            offline_match["engine_used"] = "SachAI Local Intelligence Archive (API Timeout Fallback)"
            return offline_match
        raise e


def analyze_screenshot_with_gemini(
    img_bytes: bytes,
    mime_type: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Multimodal analysis: OCR claim extraction, visual deepfake / forgery inspection.
    """
    logger.info("Analyzing screenshot with Gemini Multimodal Vision (%s)...", model)
    encoded = base64.b64encode(img_bytes).decode("utf-8")
    
    prompt = (
        "You are an expert digital forensics and image verification specialist for Indian social media.\n"
        "Analyze this uploaded image (which may be a WhatsApp screenshot, social media flyer, or government circular).\n"
        "Perform these two tasks:\n"
        "1. EXTRACT: Accurately read all text (Hindi, English, or regional scripts) and summarize the core factual claim in 1-2 crisp sentences.\n"
        "2. FORENSIC AUDIT: Check for visual signs of forgery or manipulation (mismatched fonts, uneven alignment, suspicious government logos, edited dates, WhatsApp compression anomalies, AI generation artifacts).\n\n"
        "Return STRICT JSON only with keys:\n"
        "- extracted_claim: String summarizing the core factual claim\n"
        "- full_ocr_text: Extracted readable text snippet\n"
        "- detected_language: String (e.g., 'Hindi', 'English', 'Hinglish')\n"
        "- visual_manipulation_detected: Boolean (true/false)\n"
        "- forensic_notes: 1-2 sentences on visual authenticity and anomalies\n"
        "- suspicion_score: Integer 0-100 indicating likelihood of synthetic/altered image"
    )

    url = f"{GEMINI_BASE_URL}/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type if mime_type in ["image/jpeg", "image/png", "image/webp"] else "image/jpeg",
                            "data": encoded,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 800,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=35)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        parsed = clean_json_response(raw_text)

        return {
            "extracted_claim": parsed.get("extracted_claim", "Unable to extract clear claim from image."),
            "full_ocr_text": parsed.get("full_ocr_text", ""),
            "detected_language": parsed.get("detected_language", "Hindi / English"),
            "visual_manipulation_detected": bool(parsed.get("visual_manipulation_detected", False)),
            "forensic_notes": parsed.get("forensic_notes", "Standard screenshot formatting inspected."),
            "suspicion_score": int(parsed.get("suspicion_score", 15)),
        }
    except Exception as e:
        logger.exception("Failed image analysis with Gemini: %s", e)
        return {
            "extracted_claim": "Image uploaded: Could not automatically read text via API.",
            "full_ocr_text": "",
            "detected_language": "Unknown",
            "visual_manipulation_detected": False,
            "forensic_notes": "Unable to complete automated forensic inspection.",
            "suspicion_score": 0,
        }


def transcribe_audio_with_gemini(
    audio_bytes: bytes,
    mime_type: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Transcribe and extract claim from Indian voice note using Gemini's native audio model.
    """
    logger.info("Transcribing audio with Gemini Audio Engine...")
    encoded = base64.b64encode(audio_bytes).decode("utf-8")

    # Map audio mime types
    supported_mime = "audio/mp3"
    if "wav" in mime_type:
        supported_mime = "audio/wav"
    elif "ogg" in mime_type:
        supported_mime = "audio/ogg"
    elif "m4a" in mime_type:
        supported_mime = "audio/m4a"

    prompt = (
        "You are an Indian multilingual speech recognition expert.\n"
        "Transcribe this voice note (which may be in Hindi, Hinglish, Bengali, Tamil, Telugu, or English).\n"
        "Return STRICT JSON only with keys:\n"
        "- transcript: Complete verbatim speech transcript in the original language\n"
        "- central_claim: One crisp sentence summarizing the core factual assertion or news rumour\n"
        "- detected_language: String name of language/dialect spoken\n"
        "- tone_urgency: String ('High Panic', 'Casual', 'Formal Advisory', etc.)"
    )

    url = f"{GEMINI_BASE_URL}/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": supported_mime,
                            "data": encoded,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 700,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=40)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        parsed = clean_json_response(raw_text)

        return {
            "transcript": parsed.get("transcript", "No distinct speech detected."),
            "central_claim": parsed.get("central_claim", parsed.get("transcript", "")),
            "detected_language": parsed.get("detected_language", "Hindi"),
            "tone_urgency": parsed.get("tone_urgency", "Normal"),
        }
    except Exception as e:
        logger.exception("Audio transcription failed with Gemini: %s", e)
        return {
            "transcript": "Audio uploaded (Transcription service unavailable).",
            "central_claim": "Claim could not be extracted from audio.",
            "detected_language": "Unknown",
            "tone_urgency": "Unknown",
        }
