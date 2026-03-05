HIGH_PURCHASE_INTENT_PHRASES = [
    "where can i buy this",
    "is there a link",
    "drop the link",
    "how much is this",
    "is this on amazon",
    "need this now",
    "shut up and take my money",
    "add to cart",
    "just ordered",
    "sold out everywhere",
]

NEGATIVE_SENTIMENT_PHRASES = [
    "is this a scam",
    "don't buy this",
    "waste of money",
    "looks cheap",
    "took forever to ship",
    "never arrived",
    "broke after one use",
    "not worth it",
    "return this",
]

EXCLUDED_FINANCE_TERMS = [
    "#stocktok",
    "#fintok",
    "$",
    "ticker",
    "nasdaq",
    "nyse",
    "investing",
    "stocks",
]

PRODUCT_CATEGORIES = [
    "skincare",
    "fitness",
    "kitchen",
    "tech",
    "supplements",
    "shoes",
    "snacks",
]

TIKTOKMADEMEBUYIT_CATEGORY_QUERIES = [
    f"#tiktokmademebuyit {category}" for category in PRODUCT_CATEGORIES
]

RETAILER_KEYWORDS = [
    "amazon",
    "target",
    "walmart",
    "sephora",
    "ulta",
    "costco",
    "best buy",
    "nike",
    "temu",
    "shein",
]
