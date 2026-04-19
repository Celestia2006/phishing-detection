"""
feature_extractor.py
--------------------
Extracts 27 phishing-detection features from a raw URL.

Feature groups:
  - URL-based    (10): pure string/regex parsing, no network calls
  - DNS/WHOIS    ( 3): external DNS + WHOIS lookups
  - HTML/page    ( 9): HTTP fetch + BeautifulSoup parsing
  - API-based    ( 4): SSLfinal_State (ssl lib, no key needed),
                       Google_Index (Safe Browsing API),
                       Statistical_report (VirusTotal API),
                       Page_Rank (OpenPageRank API)

Dropped from original UCI dataset (no reliable free source):
  web_traffic, Links_pointing_to_page

All features follow the original UCI encoding:
   1  → legitimate signal
  -1  → phishing signal
   0  → suspicious / neutral

API keys — set these in your .env file and load via python-dotenv:
  GOOGLE_SAFE_BROWSING_API_KEY   → https://developers.google.com/safe-browsing
  VIRUSTOTAL_API_KEY             → https://www.virustotal.com (profile → API key)
  OPEN_PAGE_RANK_API_KEY         → https://www.domcop.com/openpagerank/
"""

import re
import os
import ssl
import socket
import ipaddress
import requests
import whois
from datetime import datetime
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

# ── API key placeholders — load from .env via python-dotenv ──────────────────
# Add these to your .env file before deploying:
#   GOOGLE_SAFE_BROWSING_API_KEY=your_key_here
#   VIRUSTOTAL_API_KEY=your_key_here
#   OPEN_PAGE_RANK_API_KEY=your_key_here

GOOGLE_SAFE_BROWSING_API_KEY = os.getenv("GOOGLE_SAFE_BROWSING_API_KEY", "")
VIRUSTOTAL_API_KEY            = os.getenv("VIRUSTOTAL_API_KEY", "")
OPEN_PAGE_RANK_API_KEY        = os.getenv("OPEN_PAGE_RANK_API_KEY", "")

# ── constants ────────────────────────────────────────────────────────────────

SHORTENING_SERVICES = re.compile(
    r"bit\.ly|goo\.gl|tinyurl\.com|ow\.ly|t\.co|is\.gd|cli\.gs|"
    r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|"
    r"su\.pr|twurl\.nl|snipurl\.com|short\.to|BudURL\.com|ping\.fm|"
    r"post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"
    r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|"
    r"to\.ly|bit\.do|t2mio\.com|lnkd\.in|db\.tt|qr\.ae|"
    r"adf\.ly|bitly\.com|cur\.lv|tinyurl\.com|ity\.im|"
    r"q\.gs|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|"
    r"buzurl\.com|cutt\.us|u\.bb|yourls\.org|prettylinkpro\.com|"
    r"scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|"
    r"tweez\.me|v\.gd|tr\.im|link\.zip\.net",
    re.IGNORECASE,
)

REQUEST_TIMEOUT = 6  # seconds for HTTP fetch


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_domain(parsed):
    """Return the bare domain (no www.) from a ParseResult."""
    return parsed.netloc.replace("www.", "").split(":")[0]


def _fetch_page(url: str):
    """
    Attempt to fetch the URL and return a BeautifulSoup object.
    Returns None on any error (timeout, SSL, etc.).
    """
    try:
        resp = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"},
            allow_redirects=True,
            verify=False,          # phishing sites often have bad certs
        )
        return BeautifulSoup(resp.text, "html.parser"), resp
    except Exception:
        return None, None


def _get_whois(domain: str):
    """Return whois data or None on failure."""
    try:
        return whois.whois(domain)
    except Exception:
        return None


def _ratio(count, total):
    """Safe ratio — returns 0 if total is 0."""
    return count / total if total else 0


# ── URL-based features (10) ───────────────────────────────────────────────────

def having_IP_Address(parsed) -> int:
    """
    -1 if the host is a raw IP address (phishing signal).
     1 otherwise.
    """
    host = parsed.netloc.split(":")[0]
    try:
        ipaddress.ip_address(host)
        return -1
    except ValueError:
        return 1


def url_length(url: str) -> int:
    """
     1 if length < 54
     0 if 54 <= length <= 75
    -1 if length > 75
    """
    n = len(url)
    if n < 54:
        return 1
    if n <= 75:
        return 0
    return -1


def shortening_service(url: str) -> int:
    """-1 if a known URL shortener is present, else 1."""
    return -1 if SHORTENING_SERVICES.search(url) else 1


def having_at_symbol(url: str) -> int:
    """-1 if '@' is in the URL (browser ignores everything before it)."""
    return -1 if "@" in url else 1


def double_slash_redirecting(url: str) -> int:
    """
    -1 if '//' appears after position 7 (beyond the 'https://').
    Indicates a redirect trick.
    """
    return -1 if url.rfind("//") > 7 else 1


def prefix_suffix(parsed) -> int:
    """-1 if '-' appears in the domain (e.g. paypal-secure.com)."""
    return -1 if "-" in parsed.netloc else 1


def having_sub_domain(parsed) -> int:
    """
    Counts dots in the domain to infer subdomain depth.
     1 → 1 dot  (e.g. example.com)
     0 → 2 dots (e.g. login.example.com)
    -1 → 3+ dots (e.g. login.secure.example.com)
    """
    host = _get_domain(parsed)
    dots = host.count(".")
    if dots == 1:
        return 1
    if dots == 2:
        return 0
    return -1


def https_token(parsed) -> int:
    """-1 if 'https' appears literally inside the domain (deceptive trick)."""
    return -1 if "https" in parsed.netloc.lower() else 1


def redirect(url: str) -> int:
    """
    Counts '//' occurrences excluding the protocol.
     0 if redirects <= 1
    -1 if redirects > 1
    """
    stripped = url[7:]  # remove 'http://' or 'https:/'
    return 0 if stripped.count("//") <= 1 else -1


def port(parsed) -> int:
    """
    -1 if a non-standard port is explicitly set in the URL.
     1 if no port or port is 80/443.
    """
    standard = {80, 443, ""}
    p = parsed.port
    if p is None or p in standard:
        return 1
    return -1


# ── DNS / WHOIS features (3) ──────────────────────────────────────────────────

def dns_record(domain: str) -> int:
    """-1 if no DNS A-record exists for the domain."""
    try:
        socket.gethostbyname(domain)
        return 1
    except socket.gaierror:
        return -1


def age_of_domain(w) -> int:
    """
    Uses WHOIS creation date.
    -1 if domain is younger than 6 months (suspicious).
     1 if older than 6 months.
     -1 if WHOIS lookup fails.
    """
    if w is None:
        return -1
    try:
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if creation is None:
            return -1
        age_months = (datetime.now() - creation).days / 30
        return 1 if age_months >= 6 else -1
    except Exception:
        return -1


def domain_registration_length(w) -> int:
    """
    Uses WHOIS expiry date.
    -1 if registration expires within 1 year (phishing domains are short-lived).
     1 if registration extends beyond 1 year.
    -1 if WHOIS data is unavailable.
    """
    if w is None:
        return -1
    try:
        expiry = w.expiration_date
        if isinstance(expiry, list):
            expiry = expiry[0]
        if expiry is None:
            return -1
        days_left = (expiry - datetime.now()).days
        return 1 if days_left > 365 else -1
    except Exception:
        return -1


# ── HTML / page features (9) ─────────────────────────────────────────────────

def favicon(soup, base_url: str, domain: str) -> int:
    """
    -1 if the favicon is loaded from a different domain.
     1 if same domain or no favicon found.
    """
    if soup is None:
        return -1
    link = soup.find("link", rel=lambda r: r and "icon" in r)
    if not link or not link.get("href"):
        return 1
    href = urljoin(base_url, link["href"])
    favicon_domain = _get_domain(urlparse(href))
    return 1 if domain in favicon_domain or favicon_domain in domain else -1


def request_url(soup, domain: str) -> int:
    """
    Checks what % of embedded resources (img, audio, video, embed)
    are loaded from external domains.
     1 if < 22 % external
     0 if 22–61 %
    -1 if > 61 %
    """
    if soup is None:
        return -1
    tags = soup.find_all(["img", "audio", "video", "embed"])
    total = len(tags)
    if total == 0:
        return 1
    external = sum(
        1 for t in tags
        if t.get("src") and domain not in t["src"] and t["src"].startswith("http")
    )
    ratio = _ratio(external, total)
    if ratio < 0.22:
        return 1
    if ratio < 0.61:
        return 0
    return -1


def url_of_anchor(soup, domain: str) -> int:
    """
    Checks what % of <a> hrefs point outside the domain or use '#'/'javascript:'.
     1 if < 31 % suspicious
     0 if 31–67 %
    -1 if > 67 %
    """
    if soup is None:
        return -1
    anchors = soup.find_all("a", href=True)
    total = len(anchors)
    if total == 0:
        return 1
    suspicious = sum(
        1 for a in anchors
        if a["href"].startswith("#")
        or a["href"].startswith("javascript")
        or (a["href"].startswith("http") and domain not in a["href"])
    )
    ratio = _ratio(suspicious, total)
    if ratio < 0.31:
        return 1
    if ratio < 0.67:
        return 0
    return -1


def links_in_tags(soup, domain: str) -> int:
    """
    Checks <meta>, <link>, <script> tags for external domain references.
     1 if < 17 % external
     0 if 17–81 %
    -1 if > 81 %
    """
    if soup is None:
        return -1
    tags = soup.find_all(["meta", "link", "script"])
    total = len(tags)
    if total == 0:
        return 1
    external = sum(
        1 for t in tags
        if t.get("src", t.get("href", "")).startswith("http")
        and domain not in t.get("src", t.get("href", ""))
    )
    ratio = _ratio(external, total)
    if ratio < 0.17:
        return 1
    if ratio < 0.81:
        return 0
    return -1


def sfh(soup, domain: str) -> int:
    """
    Server Form Handler — checks where <form> actions point.
    -1 if action is blank, 'about:blank', or an external domain.
     0 if action is empty string ('')
     1 if action points to same domain.
    """
    if soup is None:
        return -1
    forms = soup.find_all("form", action=True)
    if not forms:
        return 1
    for form in forms:
        action = form["action"].strip()
        if action in ("", "about:blank"):
            return -1
        if action.startswith("http") and domain not in action:
            return -1
    return 1


def submitting_to_email(soup) -> int:
    """-1 if any form uses 'mailto:' action (submits data to an email)."""
    if soup is None:
        return -1
    forms = soup.find_all("form", action=True)
    for form in forms:
        if "mailto:" in form["action"].lower():
            return -1
    return 1


def abnormal_url(w, domain: str) -> int:
    """
    -1 if the WHOIS registered domain doesn't match the URL domain.
     1 if they match.
    """
    if w is None:
        return -1
    try:
        whois_domain = w.domain_name
        if isinstance(whois_domain, list):
            whois_domain = whois_domain[0]
        if whois_domain is None:
            return -1
        return 1 if whois_domain.lower() in domain.lower() else -1
    except Exception:
        return -1


def on_mouseover(soup) -> int:
    """-1 if any onmouseover handler modifies window.status (status bar spoofing)."""
    if soup is None:
        return -1
    for tag in soup.find_all(True):
        handler = tag.get("onmouseover", "")
        if "window.status" in handler:
            return -1
    return 1


def right_click(soup) -> int:
    """-1 if right-click is disabled via event.button==2 check in script."""
    if soup is None:
        return -1
    scripts = " ".join(s.get_text() for s in soup.find_all("script"))
    return -1 if "event.button==2" in scripts else 1


def popup_window(soup) -> int:
    """-1 if any popup (window.open / alert / prompt / confirm) collects input."""
    if soup is None:
        return -1
    scripts = " ".join(s.get_text() for s in soup.find_all("script"))
    for fn in ("window.open", "prompt(", "confirm("):
        if fn in scripts:
            return -1
    return 1


def iframe(soup) -> int:
    """-1 if any <iframe> lacks a border (invisible iframe — common in phishing)."""
    if soup is None:
        return -1
    for frame in soup.find_all("iframe"):
        if frame.get("frameborder") == "0" or frame.get("border") == "0":
            return -1
    return 1


# ── API-based features (4) ────────────────────────────────────────────────────

def ssl_final_state(domain: str) -> int:
    """
    Checks SSL certificate validity using Python's ssl library — no API key needed.
     1 if a valid, non-expired certificate exists.
    -1 if SSL is missing, expired, or self-signed.
     0 if the check cannot be completed.
    """
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(
            socket.create_connection((domain, 443), timeout=5),
            server_hostname=domain,
        ) as s:
            cert = s.getpeercert()
            # Check expiry
            expiry_str = cert.get("notAfter", "")
            if expiry_str:
                expiry = datetime.strptime(expiry_str, "%b %d %H:%M:%S %Y %Z")
                if expiry < datetime.now():
                    return -1   # expired cert
            return 1
    except ssl.SSLCertVerificationError:
        return -1   # invalid / self-signed
    except (socket.timeout, ConnectionRefusedError, OSError):
        return 0    # couldn't reach port 443 — inconclusive
    except Exception:
        return -1


def google_index(url: str) -> int:
    """
    Uses Google Safe Browsing API to check if the URL is flagged as unsafe.
     1 if Google considers the URL safe (or no API key set).
    -1 if flagged as phishing, malware, or unwanted software.
     0 if the API call fails.

    Requires: GOOGLE_SAFE_BROWSING_API_KEY in .env
    Docs: https://developers.google.com/safe-browsing/v4/lookup-api
    """
    if not GOOGLE_SAFE_BROWSING_API_KEY:
        # No key — return neutral rather than a false phishing signal
        return 0

    endpoint = (
        f"https://safebrowsing.googleapis.com/v4/threatMatches:find"
        f"?key={GOOGLE_SAFE_BROWSING_API_KEY}"
    )
    payload = {
        "client": {"clientId": "phishguard", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }
    try:
        resp = requests.post(endpoint, json=payload, timeout=5)
        data = resp.json()
        # If 'matches' key exists and is non-empty, URL is flagged
        return -1 if data.get("matches") else 1
    except Exception:
        return 0


def statistical_report(url: str) -> int:
    """
    Uses VirusTotal API to check if the URL is flagged by any security vendors.
     1 if no vendors flag it (or no API key set).
    -1 if 2 or more vendors flag it as malicious or phishing.
     0 if fewer than 2 vendors flag it (inconclusive) or API call fails.

    VirusTotal aggregates 70+ security vendors — a URL flagged by 2+
    is a strong and reliable phishing/malware signal.

    Requires: VIRUSTOTAL_API_KEY in .env
    Free tier: 500 requests/day, 4 requests/minute
    Docs: https://developers.virustotal.com/reference/scan-url
    """
    if not VIRUSTOTAL_API_KEY:
        return 0

    import base64

    # VirusTotal v3 URL scan — encode URL to base64 for the endpoint ID
    url_id   = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
    endpoint = f"https://www.virustotal.com/api/v3/urls/{url_id}"
    headers  = {"x-apikey": VIRUSTOTAL_API_KEY}

    try:
        resp = requests.get(endpoint, headers=headers, timeout=8)

        # 404 means URL not in VirusTotal database yet — submit it first
        if resp.status_code == 404:
            submit_resp = requests.post(
                "https://www.virustotal.com/api/v3/urls",
                headers={"x-apikey": VIRUSTOTAL_API_KEY},
                data={"url": url},
                timeout=8,
            )
            if submit_resp.status_code != 200:
                return 0
            # After submission, re-fetch the analysis
            resp = requests.get(endpoint, headers=headers, timeout=8)
            if resp.status_code != 200:
                return 0

        data       = resp.json()
        stats      = (
            data.get("data", {})
                .get("attributes", {})
                .get("last_analysis_stats", {})
        )
        malicious  = stats.get("malicious", 0)
        suspicious = stats.get("suspicious", 0)
        flagged    = malicious + suspicious

        if flagged >= 2:
            return -1   # clearly flagged — strong phishing signal
        if flagged == 1:
            return 0    # one vendor flagged — inconclusive
        return 1        # clean

    except Exception:
        return 0


def page_rank(domain: str) -> int:
    """
    Uses OpenPageRank API to get the domain's PageRank score.
     1 if PageRank >= 4  (established, trustworthy domain)
     0 if PageRank 1–3   (low but existing presence)
    -1 if PageRank == 0 or domain not found (no web presence — suspicious)

    Requires: OPEN_PAGE_RANK_API_KEY in .env
    Docs: https://www.domcop.com/openpagerank/documentation
    Free tier: 10,000 requests/hour
    """
    if not OPEN_PAGE_RANK_API_KEY:
        return 0

    endpoint = "https://openpagerank.com/api/v1.0/getPageRank"
    headers  = {"API-OPR": OPEN_PAGE_RANK_API_KEY}
    params   = {"domains[]": domain}
    try:
        resp = requests.get(endpoint, headers=headers, params=params, timeout=5)
        data = resp.json()
        rank = data.get("response", [{}])[0].get("page_rank_integer", 0)
        if rank >= 4:
            return 1
        if rank >= 1:
            return 0
        return -1
    except Exception:
        return 0

FEATURE_ORDER = [
    "having_IP_Address",
    "URL_Length",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",
    "SSLfinal_State",
    "Domain_registeration_length",
    "Favicon",
    "port",
    "HTTPS_token",
    "Request_URL",
    "URL_of_Anchor",
    "Links_in_tags",
    "SFH",
    "Submitting_to_email",
    "Abnormal_URL",
    "Redirect",
    "on_mouseover",
    "RightClick",
    "popUpWidnow",
    "Iframe",
    "age_of_domain",
    "DNSRecord",
    "Page_Rank",
    "Google_Index",
    "Statistical_report",
]


def extract_features(url: str) -> dict:
    """
    Main entry point.

    Parameters
    ----------
    url : str
        Raw URL to analyse (e.g. 'http://paypal-secure.xyz/login').

    Returns
    -------
    dict with keys matching FEATURE_ORDER, all values in {-1, 0, 1}.
    Raises ValueError if the URL cannot be parsed.
    """
    parsed = urlparse(url)
    if not parsed.netloc:
        raise ValueError(f"Could not parse URL: {url!r}")

    domain = _get_domain(parsed)

    # Network calls (done once, shared across features)
    soup, resp = _fetch_page(url)
    w = _get_whois(domain)

    features = {
        # URL-based
        "having_IP_Address":       having_IP_Address(parsed),
        "URL_Length":              url_length(url),
        "Shortining_Service":      shortening_service(url),
        "having_At_Symbol":        having_at_symbol(url),
        "double_slash_redirecting": double_slash_redirecting(url),
        "Prefix_Suffix":           prefix_suffix(parsed),
        "having_Sub_Domain":       having_sub_domain(parsed),
        "HTTPS_token":             https_token(parsed),
        "Redirect":                redirect(url),
        "port":                    port(parsed),

        # DNS / WHOIS
        "DNSRecord":               dns_record(domain),
        "age_of_domain":           age_of_domain(w),
        "Domain_registeration_length": domain_registration_length(w),

        # HTML / page
        "Favicon":                 favicon(soup, url, domain),
        "Request_URL":             request_url(soup, domain),
        "URL_of_Anchor":           url_of_anchor(soup, domain),
        "Links_in_tags":           links_in_tags(soup, domain),
        "SFH":                     sfh(soup, domain),
        "Submitting_to_email":     submitting_to_email(soup),
        "Abnormal_URL":            abnormal_url(w, domain),
        "on_mouseover":            on_mouseover(soup),
        "RightClick":              right_click(soup),
        "popUpWidnow":             popup_window(soup),
        "Iframe":                  iframe(soup),

        # API-based
        "SSLfinal_State":          ssl_final_state(domain),
        "Google_Index":            google_index(url),
        "Statistical_report":      statistical_report(url),
        "Page_Rank":               page_rank(domain),
    }

    # Return in the fixed order the model expects
    return {k: features[k] for k in FEATURE_ORDER}