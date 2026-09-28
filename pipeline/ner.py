from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Ticker detection
# ---------------------------------------------------------------------------

# Matches bare tickers like AAPL, TSLA, NKE (1-5 uppercase letters)
# Also matches $-prefixed tickers like $AAPL (common on social media)
_CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")
_BARE_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")

# Common English words that look like tickers — exclude to reduce false positives
_TICKER_STOPWORDS = frozenset(
    [
        "A", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "IF", "IN",
        "IS", "IT", "ME", "MY", "NO", "OF", "OK", "ON", "OR", "SO", "TO",
        "UP", "US", "WE", "AND", "ARE", "BUT", "CAN", "DID", "FOR", "GET",
        "GOT", "HAD", "HAS", "HER", "HIM", "HIS", "HOW", "ITS", "LET",
        "MAY", "NOT", "NOW", "OFF", "OUR", "OUT", "OWN", "PUT", "SAY",
        "SHE", "THE", "TOO", "TWO", "USE", "WAS", "WAY", "WHO", "WHY",
        "WITH", "YOU", "ALL", "ANY", "ARE", "ASK", "BAD", "BIG", "BUY",
        "DAY", "DID", "DIE", "DON", "EAT", "END", "FAR", "FEW", "FIT",
        "FUN", "GOD", "GOT", "GUY", "HIT", "HOT", "JOB", "LET", "LOT",
        "LOW", "MAN", "MEN", "MOM", "MUY", "NEW", "OLD", "ONE", "PAY",
        "RUN", "SAY", "SEE", "SET", "SIT", "SIX", "TEN", "TRY", "TWO",
        "WIN", "WON", "YES", "YET", # two-letter extras
        "AI", "AR", "CA", "CO", "DC", "FL", "ID", "IL", "LA", "MD",
        "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NJ", "NM",
        "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
        "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY",
        # finance/business abbreviations that aren't tickers
        "LLC", "LTD", "INC", "CEO", "CFO", "COO", "CTO", "IPO", "ETF",
        "IRA", "GDP", "CPI", "PPI", "FED", "USD", "EUR", "GBP", "JPY",
        "APR", "APY", "ATM", "PIN", "PDF", "URL", "FAQ", "DIY", "TBD",
        "FYI", "IMO", "TBH", "SMH", "LOL", "OMG", "DM", "PM",
    ]
)


# ---------------------------------------------------------------------------
# Known company-name → ticker dictionary (expandable)
# Covers the most common brands mentioned on TikTok consumer content
# ---------------------------------------------------------------------------
_COMPANY_NAME_TO_TICKER: dict[str, str] = {
    # Consumer / Retail
    "apple": "AAPL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "instagram": "META",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "netflix": "NFLX",
    "disney": "DIS",
    "nike": "NKE",
    "adidas": "ADDYY",
    "lululemon": "LULU",
    "under armour": "UA",
    "gap": "GPS",
    "h&m": "HNNMY",
    "zara": "ITX",
    "shein": None,  # private
    "temu": None,  # private (PDD parent is NASDAQ: PDD)
    "pdd": "PDD",
    "target": "TGT",
    "walmart": "WMT",
    "costco": "COST",
    "sephora": None,  # subsidiary of LVMH
    "lvmh": "LVMUY",
    "ulta": "ULTA",
    "ulta beauty": "ULTA",
    "best buy": "BBY",
    # Food & Beverage
    "starbucks": "SBUX",
    "mcdonald's": "MCD",
    "mcdonalds": "MCD",
    "chipotle": "CMG",
    "domino's": "DPZ",
    "dominos": "DPZ",
    "yum brands": "YUM",
    "taco bell": "YUM",
    "kfc": "YUM",
    "pizza hut": "YUM",
    "coca cola": "KO",
    "coke": "KO",
    "pepsi": "PEP",
    "pepsico": "PEP",
    "monster": "MNST",
    "celsius": "CELH",
    "celsius holdings": "CELH",
    "red bull": None,  # private
    "gatorade": "PEP",
    "prime": None,  # private
    "olipop": None,  # private
    "liquid iv": None,  # subsidiary of Unilever
    "unilever": "UL",
    "kraft heinz": "KHC",
    "general mills": "GIS",
    "kellogg": "K",
    "kellanova": "K",
    "campbell": "CPB",
    "hershey": "HSY",
    "mondelez": "MDLZ",
    # Skincare / Beauty
    "estee lauder": "EL",
    "lauder": "EL",
    "l'oreal": "LRLCY",
    "loreal": "LRLCY",
    "procter gamble": "PG",
    "p&g": "PG",
    "johnson johnson": "JNJ",
    "j&j": "JNJ",
    "colgate": "CL",
    "crest": "PG",
    "olay": "PG",
    "cerave": "L",  # L'Oreal
    "neutrogena": "JNJ",
    "aveeno": "JNJ",
    "e.l.f.": "ELF",
    "elf cosmetics": "ELF",
    "elf": "ELF",
    "revlon": None,  # delisted/bankrupt
    "maybelline": "LRLCY",
    "covergirl": None,  # Coty
    "coty": "COTY",
    "beiersdorf": "BDRFY",
    "nivea": "BDRFY",
    # Footwear
    "deckers": "DECK",
    "ugg": "DECK",
    "hoka": "DECK",
    "skechers": "SKX",
    "steve madden": "SHOO",
    "steven madden": "SHOO",
    "crocs": "CROX",
    "vans": "VFC",
    "timberland": "VFC",
    "north face": "VFC",
    "vf corp": "VFC",
    "birkenstock": "BIRK",
    "new balance": None,  # private
    "asics": None,  # private (Japan, OTC)
    "on running": "ONON",
    "on cloud": "ONON",
    # Tech / Electronics
    "samsung": "SSNLF",
    "sony": "SONY",
    "lg": None,  # private
    "bose": None,  # private
    "jbl": None,  # subsidiary Harman/Samsung
    "harman": "SSNLF",
    "dyson": None,  # private
    "instant pot": None,  # private / Instant Brands bankrupt
    "ninja": None,  # private (SharkNinja → SN)
    "sharkninja": "SN",
    "shark ninja": "SN",
    "kitchenaid": "WHR",
    "whirlpool": "WHR",
    # Supplements / Health
    "gnc": None,  # private post-bankruptcy
    "vitamin shoppe": None,  # Franchise Group
    "optimum nutrition": None,  # Glanbia subsidiary
    "glanbia": None,  # Dublin listed
    "garden of life": None,  # private
    "athletic greens": None,  # private (AG1)
    "ag1": None,
    # E-commerce / Platform
    "shopify": "SHOP",
    "etsy": "ETSY",
    "ebay": "EBAY",
    "mercari": None,  # Japan listed
    "poshmark": None,  # acquired by Naver/delisted
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class NERResult:
    cashtag_tickers: list[str] = field(default_factory=list)
    bare_tickers: list[str] = field(default_factory=list)
    company_names: list[str] = field(default_factory=list)
    mapped_tickers: list[str] = field(default_factory=list)  # from company-name lookup


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_ner(text: str) -> NERResult:
    """Extract ticker symbols and company names from *text*.

    Returns an NERResult with:
    - cashtag_tickers: tickers preceded by '$'
    - bare_tickers: uppercase sequences that survive stopword filter
    - company_names: known company names found in text
    - mapped_tickers: ticker symbols resolved from company_names
    """
    result = NERResult()
    if not text:
        return result

    blob = text  # preserve case for ticker scan
    blob_lower = text.lower()

    # 1. $CASHTAG detection (highest confidence)
    result.cashtag_tickers = sorted({m.group(1) for m in _CASHTAG_RE.finditer(blob)})

    # 2. Bare uppercase ticker detection (lower confidence, filtered by stopwords)
    result.bare_tickers = sorted(
        {
            m.group(1)
            for m in _BARE_TICKER_RE.finditer(blob)
            if m.group(1) not in _TICKER_STOPWORDS and len(m.group(1)) >= 2
        }
    )

    # 3. Company-name lookup (case-insensitive, longest match first)
    found_names: list[str] = []
    found_tickers: list[str] = []
    # sort by length descending so "estee lauder" matches before "lauder"
    for name in sorted(_COMPANY_NAME_TO_TICKER, key=len, reverse=True):
        if name in blob_lower:
            found_names.append(name)
            ticker = _COMPANY_NAME_TO_TICKER[name]
            if ticker:
                found_tickers.append(ticker)

    result.company_names = found_names
    result.mapped_tickers = sorted(set(found_tickers))

    return result


def all_tickers(ner: NERResult) -> list[str]:
    """Deduplicated union of all detected tickers, cashtags first."""
    seen: set[str] = set()
    out: list[str] = []
    for t in ner.cashtag_tickers + ner.mapped_tickers + ner.bare_tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
