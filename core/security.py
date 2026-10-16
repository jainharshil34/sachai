"""
SachAI Security & SSRF Protection
Provides URL safety validation, private IP/metadata blocklists, and untrusted passage sanitization.
"""

import ipaddress
import logging
import re
import socket
import urllib.parse
from typing import Tuple

logger = logging.getLogger("SachAI.Security")

BLOCKED_HOSTNAMES = {
    "localhost",
    "metadata.google.internal",
    "169.254.169.254",
    "0.0.0.0",
}


def validate_url_ssrf_safe(url: str) -> Tuple[bool, str]:
    """
    Validate that a URL is safe for network requests (SSRF protection).
    Rejects private network ranges, loopbacks, link-local metadata endpoints, and non-HTTP protocols.
    Returns: (is_safe: bool, reason: str)
    """
    if not url or not isinstance(url, str):
        return False, "Empty or invalid URL"

    try:
        parsed = urllib.parse.urlparse(url.strip())
    except Exception as e:
        return False, f"Malformed URL: {e}"

    if parsed.scheme.lower() not in ["http", "https"]:
        return False, f"Prohibited scheme '{parsed.scheme}'. Only HTTP and HTTPS are permitted."

    hostname = (parsed.hostname or "").lower().strip()
    if not hostname:
        return False, "Missing hostname in URL."

    if hostname in BLOCKED_HOSTNAMES:
        return False, f"Access to '{hostname}' is blocked by security policy."

    # Resolve IP address to detect internal / private network traversal
    try:
        # Check if hostname itself is an IP literal
        try:
            ip = ipaddress.ip_address(hostname)
        except ValueError:
            # Resolve DNS
            addr_info = socket.getaddrinfo(hostname, None)
            if not addr_info:
                return False, f"Could not resolve host '{hostname}'"
            ip_str = addr_info[0][4][0]
            ip = ipaddress.ip_address(ip_str)

        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False, f"Access to private/internal IP address '{ip}' is blocked."

        # Check AWS/GCP/Azure link-local metadata 169.254.169.254
        if str(ip).startswith("169.254."):
            return False, "Access to cloud metadata endpoints is blocked."

    except socket.gaierror:
        # Could not resolve host
        return False, f"Failed to resolve host '{hostname}'"
    except Exception as e:
        return False, f"IP validation error for host '{hostname}': {e}"

    return True, "URL is safe."


def sanitize_untrusted_passage(text: str) -> str:
    """
    Sanitize retrieved web passages and user claims to prevent prompt injection breakouts.
    """
    if not text:
        return ""
    cleaned = text.strip()
    # Strip dangerous HTML/XML command tags
    cleaned = re.sub(r'<\/?(?:system|instruction|prompt|context|assistant|user|model|admin)[^>]*>', '', cleaned, flags=re.IGNORECASE)
    # Neutralize prompt injection phrases
    cleaned = re.sub(r'(?i)(?:ignore\s+all\s+(?:previous|above)\s+instructions)', '[REDACTED_INJECTION_ATTEMPT]', cleaned)
    cleaned = re.sub(r'(?i)(?:you\s+must\s+output\s+verdict:\s*["\']?likely\s+true["\']?)', '[REDACTED_OVERRIDE_ATTEMPT]', cleaned)
    return cleaned[:4000].strip()
