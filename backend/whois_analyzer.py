"""
whois_analyzer.py
-----------------
Performs comprehensive WHOIS-based domain analysis and returns structured
data for the frontend WHOISPanel.jsx component.

Fields returned:
  - domain_name         : registered domain name
  - registrar           : registrar organisation name
  - creation_date       : when the domain was first registered (ISO string)
  - expiry_date         : when the registration expires (ISO string)
  - last_updated        : when the WHOIS record was last modified (ISO string)
  - country             : registrant country (if not privacy-masked)
  - name_servers        : list of authoritative name servers
  - privacy_protected   : whether registrant details are hidden
  - domain_age_days     : age of the domain in days (derived)
  - days_until_expiry   : days until registration expires (derived)
  - risk_flags          : list of RiskFlag objects for the frontend panel
  - risk_level          : overall risk level — "low" | "medium" | "high"
  - lookup_success      : bool — False if WHOIS query failed entirely

Usage (called from main.py)
---------------------------
    from whois_analyzer import analyze_whois, WHOISResult

    result = analyze_whois(domain)
    # result.domain_name, result.risk_flags, result.risk_level → ready for frontend
"""

import re
import whois
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional


# ── thresholds ────────────────────────────────────────────────────────────────

YOUNG_DOMAIN_DAYS     = 180    # domains younger than this are suspicious
SHORT_EXPIRY_DAYS     = 365    # domains expiring within this window are suspicious
RECENT_UPDATE_DAYS    = 30     # domains updated very recently may be repurposed
EXPIRY_CRITICAL_DAYS  = 30     # domains expiring this soon are a strong signal

# Name server providers associated with free/low-cost hosting (common in phishing)
SUSPICIOUS_NS_KEYWORDS = [
    "freenom", "afraid.org", "duckdns", "no-ip", "changeip",
    "dnsdynamic", "dynu", "freedns", "cloudns", "he.net",
]

# Known legitimate, reputable name server providers
REPUTABLE_NS_KEYWORDS = [
    "cloudflare", "awsdns", "google", "azure", "digitalocean",
    "namecheap", "godaddy", "rackspace", "akamai", "fastly",
]

# Keywords that indicate WHOIS privacy protection is active
PRIVACY_KEYWORDS = [
    "privacy", "proxy", "whoisguard", "withheld", "redacted",
    "protected", "private", "anonymize", "anonymous", "contact privacy",
]


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class RiskFlag:
    """
    A single risk signal found during WHOIS analysis.
    Rendered as a warning row in the frontend WHOISPanel.jsx.
    """
    severity : str    # "high" | "medium" | "low"
    title    : str    # short label (e.g. "Young Domain")
    detail   : str    # human-readable explanation for the user
    icon     : str    # emoji icon for the frontend


@dataclass
class WHOISResult:
    """
    Full WHOIS analysis result returned to the FastAPI /whois endpoint.
    All optional fields are None when WHOIS data is unavailable.
    """
    # Core fields
    domain_name       : Optional[str]
    registrar         : Optional[str]
    creation_date     : Optional[str]       # ISO 8601 string for JSON serialisation
    expiry_date       : Optional[str]
    last_updated      : Optional[str]
    country           : Optional[str]
    name_servers      : list[str]           = field(default_factory=list)

    # Derived fields
    privacy_protected : bool                = False
    domain_age_days   : Optional[int]       = None
    days_until_expiry : Optional[int]       = None
    days_since_update : Optional[int]       = None

    # Risk analysis
    risk_flags        : list[RiskFlag]      = field(default_factory=list)
    risk_level        : str                 = "low"   # "low" | "medium" | "high"

    # Meta
    lookup_success    : bool                = True
    lookup_error      : Optional[str]       = None
    queried_at        : Optional[str]       = None    # ISO timestamp of this lookup


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_date(value) -> Optional[datetime]:
    """
    Normalise a WHOIS date field — can be a datetime, a list of datetimes,
    a string, or None. Always returns a single datetime or None.
    """
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0]
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d-%b-%Y"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return None


def _safe_str(value) -> Optional[str]:
    """Normalise a WHOIS string field — handle lists and strip whitespace."""
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, str):
        return value.strip() or None
    return str(value).strip() or None


def _safe_list(value) -> list[str]:
    """Normalise a WHOIS list field (e.g. name_servers)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip().lower()]
    if isinstance(value, list):
        return [str(v).strip().lower() for v in value if v]
    return []


def _to_iso(dt: Optional[datetime]) -> Optional[str]:
    """Convert datetime to ISO 8601 string for JSON serialisation."""
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _is_privacy_protected(registrar: Optional[str], registrant: Optional[str]) -> bool:
    """
    Detect privacy protection by scanning registrar name and
    registrant organisation for known privacy proxy keywords.
    """
    combined = " ".join(filter(None, [registrar, registrant])).lower()
    return any(kw in combined for kw in PRIVACY_KEYWORDS)


def _ns_reputation(name_servers: list[str]) -> tuple[bool, bool]:
    """
    Check name servers against known suspicious and reputable providers.
    Returns (is_suspicious, is_reputable) booleans.
    """
    ns_string = " ".join(name_servers).lower()
    suspicious = any(kw in ns_string for kw in SUSPICIOUS_NS_KEYWORDS)
    reputable  = any(kw in ns_string for kw in REPUTABLE_NS_KEYWORDS)
    return suspicious, reputable


def _age_label(days: int) -> str:
    """Convert domain age in days to a human-readable string."""
    if days < 30:
        return f"{days} day{'s' if days != 1 else ''}"
    if days < 365:
        months = days // 30
        return f"{months} month{'s' if months != 1 else ''}"
    years  = days // 365
    months = (days % 365) // 30
    if months:
        return f"{years} year{'s' if years != 1 else ''}, {months} month{'s' if months != 1 else ''}"
    return f"{years} year{'s' if years != 1 else ''}"


# ── risk analysis ─────────────────────────────────────────────────────────────

def _analyse_risks(
    domain_age_days   : Optional[int],
    days_until_expiry : Optional[int],
    days_since_update : Optional[int],
    privacy_protected : bool,
    name_servers      : list[str],
    registrar         : Optional[str],
    country           : Optional[str],
) -> tuple[list[RiskFlag], str]:
    """
    Evaluate all WHOIS fields and produce a list of RiskFlags and
    an overall risk level string ("low" | "medium" | "high").
    """
    flags  = []
    score  = 0   # accumulate risk points to determine overall level

    # ── Domain age ────────────────────────────────────────────────────────────
    if domain_age_days is None:
        flags.append(RiskFlag(
            severity = "high",
            title    = "Domain Age Unknown",
            detail   = "Could not determine when this domain was registered. "
                       "This may indicate a very new or privacy-shielded domain.",
            icon     = "⚠️",
        ))
        score += 2

    elif domain_age_days < 30:
        flags.append(RiskFlag(
            severity = "high",
            title    = "Extremely New Domain",
            detail   = f"This domain was registered only {_age_label(domain_age_days)} ago. "
                       "Phishing sites are almost always newly created to avoid blacklists.",
            icon     = "🚨",
        ))
        score += 3

    elif domain_age_days < YOUNG_DOMAIN_DAYS:
        flags.append(RiskFlag(
            severity = "medium",
            title    = "Recently Registered Domain",
            detail   = f"This domain is only {_age_label(domain_age_days)} old. "
                       "Legitimate businesses typically have domains older than 6 months.",
            icon     = "⚠️",
        ))
        score += 2

    else:
        flags.append(RiskFlag(
            severity = "low",
            title    = "Established Domain",
            detail   = f"This domain has been registered for {_age_label(domain_age_days)}, "
                       "which is a positive trust signal.",
            icon     = "✅",
        ))

    # ── Expiry date ───────────────────────────────────────────────────────────
    if days_until_expiry is None:
        flags.append(RiskFlag(
            severity = "medium",
            title    = "Expiry Date Unknown",
            detail   = "Could not determine when this domain expires. "
                       "Legitimate domains typically have multi-year registrations.",
            icon     = "⚠️",
        ))
        score += 1

    elif days_until_expiry < 0:
        flags.append(RiskFlag(
            severity = "high",
            title    = "Domain Already Expired",
            detail   = "This domain's registration has expired. "
                       "Active phishing sites on expired domains are a major red flag.",
            icon     = "🚨",
        ))
        score += 3

    elif days_until_expiry <= EXPIRY_CRITICAL_DAYS:
        flags.append(RiskFlag(
            severity = "high",
            title    = "Domain Expiring Very Soon",
            detail   = f"This domain expires in just {days_until_expiry} day{'s' if days_until_expiry != 1 else ''}. "
                       "Phishing domains are typically short-lived and disposable.",
            icon     = "🚨",
        ))
        score += 2

    elif days_until_expiry <= SHORT_EXPIRY_DAYS:
        flags.append(RiskFlag(
            severity = "medium",
            title    = "Short Registration Period",
            detail   = f"This domain expires in {_age_label(days_until_expiry)}. "
                       "Established businesses typically renew registrations years in advance.",
            icon     = "⚠️",
        ))
        score += 1

    else:
        flags.append(RiskFlag(
            severity = "low",
            title    = "Long Registration Period",
            detail   = f"Domain is registered for another {_age_label(days_until_expiry)}, "
                       "which is consistent with a legitimate, long-term business.",
            icon     = "✅",
        ))

    # ── Last updated ──────────────────────────────────────────────────────────
    if days_since_update is not None and days_since_update <= RECENT_UPDATE_DAYS:
        flags.append(RiskFlag(
            severity = "medium",
            title    = "WHOIS Record Recently Modified",
            detail   = f"This domain's registration details were updated {days_since_update} "
                       f"day{'s' if days_since_update != 1 else ''} ago. "
                       "A recently modified old domain may indicate a hijacked or repurposed site.",
            icon     = "⚠️",
        ))
        score += 1

    # ── Privacy protection ────────────────────────────────────────────────────
    if privacy_protected:
        flags.append(RiskFlag(
            severity = "medium",
            title    = "Registrant Identity Hidden",
            detail   = "The domain owner's details are hidden behind a WHOIS privacy service. "
                       "While common, privacy protection prevents verifying who owns this domain.",
            icon     = "⚠️",
        ))
        score += 1
    else:
        flags.append(RiskFlag(
            severity = "low",
            title    = "Registrant Identity Visible",
            detail   = "The domain owner's details are publicly available in WHOIS records, "
                       "which is a positive transparency signal.",
            icon     = "✅",
        ))

    # ── Name servers ──────────────────────────────────────────────────────────
    if not name_servers:
        flags.append(RiskFlag(
            severity = "high",
            title    = "No Name Servers Found",
            detail   = "No authoritative name servers are registered for this domain. "
                       "This is highly unusual for any legitimate website.",
            icon     = "🚨",
        ))
        score += 2

    else:
        ns_suspicious, ns_reputable = _ns_reputation(name_servers)
        if ns_suspicious:
            flags.append(RiskFlag(
                severity = "high",
                title    = "Suspicious Name Server Provider",
                detail   = f"This domain uses a free or low-cost DNS provider "
                           f"({', '.join(name_servers[:2])}), which is common among "
                           "phishing and throwaway domains.",
                icon     = "🚨",
            ))
            score += 2
        elif ns_reputable:
            flags.append(RiskFlag(
                severity = "low",
                title    = "Reputable Name Server Provider",
                detail   = f"This domain uses a well-known DNS provider "
                           f"({', '.join(name_servers[:2])}), consistent with a "
                           "professionally managed website.",
                icon     = "✅",
            ))
        else:
            flags.append(RiskFlag(
                severity = "low",
                title    = "Name Servers Present",
                detail   = f"Name servers found: {', '.join(name_servers[:2])}. "
                           "No known suspicious DNS providers detected.",
                icon     = "ℹ️",
            ))

    # ── Country ───────────────────────────────────────────────────────────────
    if country:
        # High-risk country TLDs/registrations are outside scope here,
        # but we surface the country as an informational flag
        flags.append(RiskFlag(
            severity = "low",
            title    = "Registrant Country",
            detail   = f"Domain is registered in: {country}.",
            icon     = "ℹ️",
        ))

    # ── Determine overall risk level ──────────────────────────────────────────
    if score >= 5:
        risk_level = "high"
    elif score >= 2:
        risk_level = "medium"
    else:
        risk_level = "low"

    return flags, risk_level


# ── main public function ───────────────────────────────────────────────────────

def analyze_whois(domain: str) -> WHOISResult:
    """
    Perform a full WHOIS lookup and risk analysis on a domain.

    Parameters
    ----------
    domain : str
        Bare domain name, no protocol or path (e.g. "paypal-secure.xyz").
        Typically extracted via _get_domain() in feature_extractor.py.

    Returns
    -------
    WHOISResult with all fields populated where available.
    lookup_success = False if the WHOIS query fails entirely.
    """
    now = datetime.now(timezone.utc)

    try:
        w = whois.whois(domain)
    except Exception as e:
        # WHOIS lookup failed entirely — return a minimal failed result
        return WHOISResult(
            domain_name     = domain,
            registrar       = None,
            creation_date   = None,
            expiry_date     = None,
            last_updated    = None,
            country         = None,
            name_servers    = [],
            privacy_protected = False,
            domain_age_days = None,
            days_until_expiry = None,
            risk_flags      = [RiskFlag(
                severity = "high",
                title    = "WHOIS Lookup Failed",
                detail   = f"Could not retrieve WHOIS data for '{domain}'. "
                           "This may indicate the domain is very new, uses a private registry, "
                           "or is actively blocking WHOIS queries — all common in phishing.",
                icon     = "🚨",
            )],
            risk_level      = "high",
            lookup_success  = False,
            lookup_error    = str(e),
            queried_at      = _to_iso(now),
        )

    # ── Parse raw WHOIS fields ────────────────────────────────────────────────
    domain_name  = _safe_str(w.domain_name) or domain
    registrar    = _safe_str(w.registrar)
    country      = _safe_str(w.country)
    name_servers = _safe_list(w.name_servers)

    # Attempt to get registrant org for privacy detection
    registrant_org = _safe_str(getattr(w, "org", None)) or \
                     _safe_str(getattr(w, "registrant_org", None))

    creation_dt  = _safe_date(w.creation_date)
    expiry_dt    = _safe_date(w.expiration_date)
    updated_dt   = _safe_date(w.updated_date)

    # ── Derived fields ────────────────────────────────────────────────────────
    domain_age_days    = (now - creation_dt).days if creation_dt else None
    days_until_expiry  = (expiry_dt - now).days   if expiry_dt   else None
    days_since_update  = (now - updated_dt).days  if updated_dt  else None

    privacy_protected  = _is_privacy_protected(registrar, registrant_org)

    # ── Risk analysis ─────────────────────────────────────────────────────────
    risk_flags, risk_level = _analyse_risks(
        domain_age_days   = domain_age_days,
        days_until_expiry = days_until_expiry,
        days_since_update = days_since_update,
        privacy_protected = privacy_protected,
        name_servers      = name_servers,
        registrar         = registrar,
        country           = country,
    )

    return WHOISResult(
        domain_name       = domain_name.lower(),
        registrar         = registrar,
        creation_date     = _to_iso(creation_dt),
        expiry_date       = _to_iso(expiry_dt),
        last_updated      = _to_iso(updated_dt),
        country           = country,
        name_servers      = name_servers,
        privacy_protected = privacy_protected,
        domain_age_days   = domain_age_days,
        days_until_expiry = days_until_expiry,
        days_since_update = days_since_update,
        risk_flags        = risk_flags,
        risk_level        = risk_level,
        lookup_success    = True,
        lookup_error      = None,
        queried_at        = _to_iso(now),
    )
