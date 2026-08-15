"""The controlled vocabulary that makes contradiction detection deterministic.

Two agents disagreeing about payment volume is only mechanically detectable if
both of them filed their number under the *same* predicate, in the same unit,
for the same period. That is the entire job of this module:

* :class:`Predicate` — the closed set of things a quantitative claim can be about.
* :data:`PREDICATE_ALIASES` — the many ways a source phrases each one.
* :func:`parse_value` — "$1.9T", "1.9 trillion", "1,900,000,000,000" → 1.9e12.
* :data:`HALF_LIFE_DAYS` — how fast each predicate goes stale.

Predicates deliberately cover only what can be pinned to a number. Qualitative
findings ("compliance challenges with KYC") are still captured as claims, but
they take the qualitative path and are never groupby-compared.
"""

import re
from enum import Enum

# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


class Unit(str, Enum):
    USD = "usd"
    PERCENT = "percent"
    COUNT = "count"
    MONTHS = "months"
    RATING = "rating"          # e.g. Glassdoor 3.4/5
    NONE = "none"


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


class Predicate(str, Enum):
    # Financial
    REVENUE = "revenue"
    ARR = "arr"
    REVENUE_GROWTH_YOY = "revenue_growth_yoy"
    GROSS_MARGIN = "gross_margin"
    OPERATING_MARGIN = "operating_margin"
    NET_INCOME = "net_income"
    OPERATING_CASH_FLOW = "operating_cash_flow"
    BURN_RATE = "burn_rate"
    RUNWAY = "runway"
    MARKET_CAP = "market_cap"
    VALUATION = "valuation"
    PAYMENT_VOLUME = "payment_volume"
    FUNDING_TOTAL = "funding_total"
    LAST_ROUND_AMOUNT = "last_round_amount"

    # Market
    TAM = "tam"
    SAM = "sam"
    SOM = "som"
    MARKET_SHARE = "market_share"
    CUSTOMER_COUNT = "customer_count"

    # Company
    EMPLOYEE_COUNT = "employee_count"

    # Sentiment
    GLASSDOOR_RATING = "glassdoor_rating"


#: Canonical unit for each predicate. Values are normalised into this unit,
#: so a claim's number is always directly comparable to its peers.
PREDICATE_UNITS: dict[Predicate, Unit] = {
    Predicate.REVENUE: Unit.USD,
    Predicate.ARR: Unit.USD,
    Predicate.REVENUE_GROWTH_YOY: Unit.PERCENT,
    Predicate.GROSS_MARGIN: Unit.PERCENT,
    Predicate.OPERATING_MARGIN: Unit.PERCENT,
    Predicate.NET_INCOME: Unit.USD,
    Predicate.OPERATING_CASH_FLOW: Unit.USD,
    Predicate.BURN_RATE: Unit.USD,
    Predicate.RUNWAY: Unit.MONTHS,
    Predicate.MARKET_CAP: Unit.USD,
    Predicate.VALUATION: Unit.USD,
    Predicate.PAYMENT_VOLUME: Unit.USD,
    Predicate.FUNDING_TOTAL: Unit.USD,
    Predicate.LAST_ROUND_AMOUNT: Unit.USD,
    Predicate.TAM: Unit.USD,
    Predicate.SAM: Unit.USD,
    Predicate.SOM: Unit.USD,
    Predicate.MARKET_SHARE: Unit.PERCENT,
    Predicate.CUSTOMER_COUNT: Unit.COUNT,
    Predicate.EMPLOYEE_COUNT: Unit.COUNT,
    Predicate.GLASSDOOR_RATING: Unit.RATING,
}

#: Surface forms → predicate. Lowercased, punctuation-stripped on lookup.
#: This is the layer that collapses "TPV" / "payment volume" / "total payments
#: processed" onto one predicate so their values can actually be compared.
PREDICATE_ALIASES: dict[str, Predicate] = {
    # Revenue
    "revenue": Predicate.REVENUE,
    "total revenue": Predicate.REVENUE,
    "net revenue": Predicate.REVENUE,
    "annual revenue": Predicate.REVENUE,
    "sales": Predicate.REVENUE,
    "top line": Predicate.REVENUE,
    # ARR
    "arr": Predicate.ARR,
    "annual recurring revenue": Predicate.ARR,
    "annualized recurring revenue": Predicate.ARR,
    "run rate revenue": Predicate.ARR,
    # Growth
    "revenue growth": Predicate.REVENUE_GROWTH_YOY,
    "yoy revenue growth": Predicate.REVENUE_GROWTH_YOY,
    "yoy growth": Predicate.REVENUE_GROWTH_YOY,
    "year over year growth": Predicate.REVENUE_GROWTH_YOY,
    "growth rate": Predicate.REVENUE_GROWTH_YOY,
    # Margins
    "gross margin": Predicate.GROSS_MARGIN,
    "gross profit margin": Predicate.GROSS_MARGIN,
    "operating margin": Predicate.OPERATING_MARGIN,
    "ebit margin": Predicate.OPERATING_MARGIN,
    # Profit / cash
    "net income": Predicate.NET_INCOME,
    "net profit": Predicate.NET_INCOME,
    "bottom line": Predicate.NET_INCOME,
    "earnings": Predicate.NET_INCOME,
    "operating cash flow": Predicate.OPERATING_CASH_FLOW,
    "cash flow from operations": Predicate.OPERATING_CASH_FLOW,
    "ocf": Predicate.OPERATING_CASH_FLOW,
    "burn rate": Predicate.BURN_RATE,
    "cash burn": Predicate.BURN_RATE,
    "monthly burn": Predicate.BURN_RATE,
    "runway": Predicate.RUNWAY,
    "cash runway": Predicate.RUNWAY,
    # Valuation
    "market cap": Predicate.MARKET_CAP,
    "market capitalization": Predicate.MARKET_CAP,
    "market capitalisation": Predicate.MARKET_CAP,
    "valuation": Predicate.VALUATION,
    "post money valuation": Predicate.VALUATION,
    "pre money valuation": Predicate.VALUATION,
    "company valuation": Predicate.VALUATION,
    # Payments
    "payment volume": Predicate.PAYMENT_VOLUME,
    "total payment volume": Predicate.PAYMENT_VOLUME,
    "tpv": Predicate.PAYMENT_VOLUME,
    "total payments processed": Predicate.PAYMENT_VOLUME,
    "payments processed": Predicate.PAYMENT_VOLUME,
    "processing volume": Predicate.PAYMENT_VOLUME,
    "gross payment volume": Predicate.PAYMENT_VOLUME,
    "gpv": Predicate.PAYMENT_VOLUME,
    # Funding
    "total funding": Predicate.FUNDING_TOTAL,
    "funding raised": Predicate.FUNDING_TOTAL,
    "capital raised": Predicate.FUNDING_TOTAL,
    "last round": Predicate.LAST_ROUND_AMOUNT,
    "latest round": Predicate.LAST_ROUND_AMOUNT,
    "round size": Predicate.LAST_ROUND_AMOUNT,
    # Market
    "tam": Predicate.TAM,
    "total addressable market": Predicate.TAM,
    "sam": Predicate.SAM,
    "serviceable addressable market": Predicate.SAM,
    "som": Predicate.SOM,
    "serviceable obtainable market": Predicate.SOM,
    "market share": Predicate.MARKET_SHARE,
    "share of market": Predicate.MARKET_SHARE,
    # Counts
    "customer count": Predicate.CUSTOMER_COUNT,
    "customers": Predicate.CUSTOMER_COUNT,
    "number of customers": Predicate.CUSTOMER_COUNT,
    "employee count": Predicate.EMPLOYEE_COUNT,
    "employees": Predicate.EMPLOYEE_COUNT,
    "headcount": Predicate.EMPLOYEE_COUNT,
    # Sentiment
    "glassdoor rating": Predicate.GLASSDOOR_RATING,
    "glassdoor score": Predicate.GLASSDOOR_RATING,
    "employee rating": Predicate.GLASSDOOR_RATING,
}

#: How long until a claim is worth half as much. Market cap moves daily;
#: an addressable market estimate stays useful for a year.
HALF_LIFE_DAYS: dict[Predicate, float] = {
    Predicate.MARKET_CAP: 30,
    Predicate.PAYMENT_VOLUME: 180,
    Predicate.VALUATION: 270,
    Predicate.REVENUE: 365,
    Predicate.ARR: 180,
    Predicate.REVENUE_GROWTH_YOY: 270,
    Predicate.GROSS_MARGIN: 365,
    Predicate.OPERATING_MARGIN: 365,
    Predicate.NET_INCOME: 365,
    Predicate.OPERATING_CASH_FLOW: 365,
    Predicate.BURN_RATE: 120,
    Predicate.RUNWAY: 90,
    Predicate.FUNDING_TOTAL: 365,
    Predicate.LAST_ROUND_AMOUNT: 365,
    Predicate.TAM: 365,
    Predicate.SAM: 365,
    Predicate.SOM: 365,
    Predicate.MARKET_SHARE: 270,
    Predicate.CUSTOMER_COUNT: 180,
    Predicate.EMPLOYEE_COUNT: 180,
    Predicate.GLASSDOOR_RATING: 365,
}

DEFAULT_HALF_LIFE_DAYS = 270.0

#: Relative difference above which two values for the same
#: (subject, predicate, period) are treated as contradicting.
DEFAULT_TOLERANCE = 0.05

#: Predicates where sources legitimately vary more (estimates, not filings).
TOLERANCE_OVERRIDES: dict[Predicate, float] = {
    Predicate.TAM: 0.25,
    Predicate.SAM: 0.25,
    Predicate.SOM: 0.25,
    Predicate.MARKET_SHARE: 0.15,
    Predicate.VALUATION: 0.10,
    Predicate.EMPLOYEE_COUNT: 0.10,
    Predicate.CUSTOMER_COUNT: 0.10,
}


#: Plausible magnitude range per predicate, as a sanity check on extraction.
#: A model asked for "the valuation" will sometimes hand back a share price, a
#: ranking, or a stray number from the same sentence. A company valued at $49
#: is an extraction error, not evidence, and must not reach the claim graph.
PLAUSIBLE_RANGE: dict[Predicate, tuple[float, float]] = {
    # Company-scale USD amounts — anything under $100k is a parsing artefact.
    Predicate.REVENUE: (1e5, 1e14),
    Predicate.ARR: (1e5, 1e14),
    Predicate.NET_INCOME: (-1e13, 1e13),
    Predicate.OPERATING_CASH_FLOW: (-1e13, 1e13),
    Predicate.BURN_RATE: (0.0, 1e12),
    Predicate.MARKET_CAP: (1e5, 1e14),
    Predicate.VALUATION: (1e5, 1e14),
    Predicate.PAYMENT_VOLUME: (1e5, 1e15),
    Predicate.FUNDING_TOTAL: (1e4, 1e13),
    Predicate.LAST_ROUND_AMOUNT: (1e4, 1e12),
    Predicate.TAM: (1e6, 1e15),
    Predicate.SAM: (1e5, 1e15),
    Predicate.SOM: (1e4, 1e15),
    # Percentages — growth can exceed 100%, margins can go deeply negative.
    Predicate.REVENUE_GROWTH_YOY: (-100.0, 1e4),
    Predicate.GROSS_MARGIN: (-1e3, 100.0),
    Predicate.OPERATING_MARGIN: (-1e4, 100.0),
    Predicate.MARKET_SHARE: (0.0, 100.0),
    # Counts and scales.
    Predicate.RUNWAY: (0.0, 600.0),
    Predicate.CUSTOMER_COUNT: (1.0, 1e10),
    Predicate.EMPLOYEE_COUNT: (1.0, 1e7),
    Predicate.GLASSDOOR_RATING: (1.0, 5.0),
}


def is_plausible(predicate: Predicate, value: float) -> bool:
    """Whether *value* is a believable magnitude for *predicate*."""
    bounds = PLAUSIBLE_RANGE.get(predicate)
    if bounds is None:
        return True
    low, high = bounds
    return low <= value <= high


def half_life_days(predicate: Predicate) -> float:
    return HALF_LIFE_DAYS.get(predicate, DEFAULT_HALF_LIFE_DAYS)


def tolerance(predicate: Predicate) -> float:
    return TOLERANCE_OVERRIDES.get(predicate, DEFAULT_TOLERANCE)


_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_WS_RE = re.compile(r"\s+")


def normalize_label(label: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", label.lower())).strip()


def resolve_predicate(label: str) -> Predicate | None:
    """Map a free-text metric name onto a Predicate, or None if unknown.

    Returning None is meaningful: an unrecognised metric becomes a qualitative
    claim rather than being forced into a predicate it does not belong to.
    """
    cleaned = normalize_label(label)
    if not cleaned:
        return None

    if cleaned in PREDICATE_ALIASES:
        return PREDICATE_ALIASES[cleaned]

    try:
        return Predicate(cleaned.replace(" ", "_"))
    except ValueError:
        pass

    # Longest-alias containment, so "total payment volume in 2025" still lands.
    for alias in sorted(PREDICATE_ALIASES, key=len, reverse=True):
        if alias in cleaned:
            return PREDICATE_ALIASES[alias]
    return None


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------

_SCALES: dict[str, float] = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mm": 1e6, "million": 1e6, "millions": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9, "billions": 1e9,
    "t": 1e12, "tn": 1e12, "trillion": 1e12, "trillions": 1e12,
}

_NUMBER_RE = re.compile(
    r"(?P<sign>-|\(|minus\s+)?\s*\$?\s*"
    r"(?P<num>\d[\d,]*\.?\d*)\s*"
    r"(?P<scale>k|mm|m|bn|b|tn|t|thousand|millions?|billions?|trillions?)?",
    re.IGNORECASE,
)


def parse_value(raw: str, unit: Unit) -> float | None:
    """Parse a human-written magnitude into a canonical number.

    ``"$1.9T"``, ``"1.9 trillion"`` and ``"1,900,000,000,000"`` all become
    ``1.9e12``, so sources phrased differently still compare equal.

    Percentages are returned on a 0–100 scale; a bare ratio like ``0.34`` for a
    percent-valued predicate is scaled up, since sources report margins both ways.
    """
    if raw is None:
        return None

    text = str(raw).strip()
    if not text:
        return None

    match = _NUMBER_RE.search(text)
    if not match:
        return None

    try:
        value = float(match.group("num").replace(",", ""))
    except ValueError:
        return None

    scale = match.group("scale")
    if scale:
        value *= _SCALES[scale.lower()]

    sign = match.group("sign")
    negative = bool(sign) or ("(" in text and ")" in text)
    if negative:
        value = -value

    if unit is Unit.PERCENT:
        # "0.34" and "34%" both mean 34% — disambiguate on the % sign.
        if "%" not in text and abs(value) <= 1.0:
            value *= 100.0

    return value


# ---------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------

_PERIOD_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:fy|fiscal(?:\s+year)?)\s*'?(\d{2,4})\b", re.I), "FY{y}"),
    (re.compile(r"\bq([1-4])\s*'?(?:fy)?\s*(\d{2,4})\b", re.I), "Q{q}-{y}"),
    (re.compile(r"\b(\d{4})\s*q([1-4])\b", re.I), "Q{q}-{y}"),
    (re.compile(r"\b(20\d{2}|19\d{2})\b"), "FY{y}"),
]


def normalize_period(raw: str | None) -> str:
    """Normalise a period label so equivalent phrasings group together.

    ``"FY2025"``, ``"fiscal 2025"`` and ``"2025"`` all become ``"FY2025"``.
    Anything without a recognisable period becomes ``"UNKNOWN"`` — which never
    groups with a dated claim, so undated numbers cannot manufacture a conflict.
    """
    if not raw:
        return "UNKNOWN"

    text = str(raw).strip()
    if not text:
        return "UNKNOWN"

    if re.fullmatch(r"ttm|trailing twelve months", text, re.I):
        return "TTM"

    for pattern, template in _PERIOD_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        groups = m.groups()
        if template == "FY{y}":
            return "FY" + _four_digit_year(groups[0])
        if pattern.pattern.startswith(r"\bq"):
            return f"Q{groups[0]}-{_four_digit_year(groups[1])}"
        return f"Q{groups[1]}-{_four_digit_year(groups[0])}"

    return "UNKNOWN"


def _four_digit_year(year: str) -> str:
    if len(year) == 4:
        return year
    n = int(year)
    return str(2000 + n if n < 70 else 1900 + n)
