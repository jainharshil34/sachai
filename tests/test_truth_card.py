"""
Tests for app.py generate_truth_card
Validates PNG header, PIL rendering dimensions, and binary validity.
"""

from app import generate_truth_card
from PIL import Image
import io


def test_generate_truth_card_valid_png():
    card_bytes = generate_truth_card(
        claim="मेरठ में नमक की कमी",
        verdict="Likely False",
        city="Meerut",
        confidence=98,
        risk_level="High Viral Threat",
    )

    assert isinstance(card_bytes, bytes)
    assert len(card_bytes) > 5000

    # Verify PNG magic bytes
    assert card_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    # Verify PIL can open and dimensions match 900x500
    img = Image.open(io.BytesIO(card_bytes))
    assert img.size == (900, 500)
    assert img.format == "PNG"


def test_generate_truth_card_true_verdict():
    card_bytes = generate_truth_card(
        claim="₹500 note star mark valid",
        verdict="Likely True",
        city=None,
        confidence=95,
        risk_level="Low",
    )
    img = Image.open(io.BytesIO(card_bytes))
    assert img.size == (900, 500)
