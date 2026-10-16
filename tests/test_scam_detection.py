"""
Tests for gemini_engine.py analyze_scam_message
Validates detection of phishing URLs, suspicious TLDs, APK malware, and urgency triggers.
"""

from gemini_engine import analyze_scam_message


def test_electricity_bill_scam_detection():
    msg = "Dear customer, your electricity will be disconnected tonight at 9:30 PM. Call officer at 9876543210 or install update.apk"
    result = analyze_scam_message(msg)

    assert result["scam_score"] >= 60
    assert "SCAM" in result["verdict"] or "FRAUD" in result["verdict"]
    assert result["risk_level"] == "Severe Scam"
    # Verify APK and urgency flags were detected
    red_flags_str = " ".join(result["red_flags"]).lower()
    assert "apk" in red_flags_str or "malware" in red_flags_str
    assert "urgency" in red_flags_str or "panic" in red_flags_str


def test_telegram_part_time_job_scam():
    msg = "Earn ₹5000 daily from home! Part-time job liking YouTube videos. Payment via UPI. Contact Telegram @hr_jobs"
    result = analyze_scam_message(msg)

    assert result["scam_score"] >= 60
    assert result["risk_level"] == "Severe Scam"
    red_flags_str = " ".join(result["red_flags"]).lower()
    assert "financial bait" in red_flags_str or "₹5000" in red_flags_str or "unrealistic" in red_flags_str


def test_phishing_link_suspicious_tld():
    msg = "Update your SBI Yono PAN KYC immediately to avoid block: http://sbi-kyc-update.top"
    result = analyze_scam_message(msg)

    assert result["scam_score"] >= 60
    red_flags_str = " ".join(result["red_flags"]).lower()
    assert "tld" in red_flags_str or "phishing" in red_flags_str
    assert len(result["extracted_links"]) >= 1


def test_benign_message_low_score():
    msg = "Hello, can we meet tomorrow at the office for lunch?"
    result = analyze_scam_message(msg)

    assert result["scam_score"] < 60
    assert "LOW RISK" in result["verdict"]
    assert result["risk_level"] == "Low"
