# Financial TikTok Keyword Universe

Search keywords and filter terms for the Bright Data TikTok collection pipeline.
Machine-readable version: `pipeline/tiktok_collection/keywords_finance.json`

Override with `TICKER_UNIVERSE_PATH` env var pointing to a JSON file with the same schema.

---

## Tier 1 — Cashtags / Ticker Universe (~50 retail stocks)

Queries are submitted as `$TICKER` and `TICKER stock` to Bright Data.

| Symbol | Company |
|--------|---------|
| AAPL | Apple |
| TSLA | Tesla |
| NVDA | Nvidia |
| AMZN | Amazon |
| META | Meta Platforms |
| MSFT | Microsoft |
| GOOGL / GOOG | Alphabet |
| AMD | Advanced Micro Devices |
| COIN | Coinbase |
| SOFI | SoFi Technologies |
| NIO | NIO Inc. |
| RIVN | Rivian |
| HOOD | Robinhood |
| PLTR | Palantir |
| GME | GameStop |
| AMC | AMC Entertainment |
| SPY | S&P 500 ETF |
| QQQ | Nasdaq 100 ETF |
| BABA | Alibaba |
| NFLX | Netflix |
| DIS | Walt Disney |
| UBER | Uber |
| LYFT | Lyft |
| SNAP | Snap |
| RBLX | Roblox |
| BB | BlackBerry |
| LCID | Lucid Motors |
| NKLA | Nikola |
| SPCE | Virgin Galactic |
| SQ | Block (formerly Square) |
| PYPL | PayPal |
| V | Visa |
| MA | Mastercard |
| JPM | JPMorgan Chase |
| BAC | Bank of America |
| WFC | Wells Fargo |
| GS | Goldman Sachs |
| MS | Morgan Stanley |
| C | Citigroup |
| XOM | ExxonMobil |
| CVX | Chevron |
| BA | Boeing |
| F | Ford |
| GM | General Motors |
| T | AT&T |
| VZ | Verizon |
| INTC | Intel |
| QCOM | Qualcomm |
| MU | Micron |
| SMCI | Super Micro Computer |
| ARM | Arm Holdings |
| MSTR | MicroStrategy |

To expand to the full CRSP-mappable universe, provide a JSON file at `TICKER_UNIVERSE_PATH`
with the same schema as `keywords_finance.json` (keys: `tickers`, `company_names`, `hashtags`, `market_terms`).

---

## Tier 2 — Community / Genre Hashtags

Used in Pass 1 Rule C (genre + market term required together).

```
#stocktok          #fintok           #stockmarket
#investing         #stockstobuy      #daytrading
#optionstrading    #dividends        #earnings
#stockanalysis     #wallstreet       #financetiktok
#investingforbeginners               #stocks
#wallstreetbets    #robinhoodstocks  #stockpicks
#bullmarket        #bearmarket       #options
#swingtrading      #pennystocks      #etf
#memestock         #shortselling
```

---

## Tier 3 — Market Condition Terms

Also used in Pass 1 Rule C (must appear with a genre hashtag).

```
stock market today     market crash          market rally
fed rate               federal reserve       interest rate
earnings report        earnings call         short squeeze
buy the dip            sell off              market correction
bull run               bear market           stock split
ipo                    initial public offering
earnings per share     price target
analyst upgrade        analyst downgrade
options expiry         gamma squeeze         short interest
put options            call options          401k
portfolio update       stock picks
```

---

## Pass 1 Filter Rules

A video passes if **any** of these rules fire:

| Rule | Condition |
|------|-----------|
| A — Cashtag | `$[A-Z]{1,5}` pattern found in description or transcript |
| B — Ticker/company | Known ticker or company name (from this list) found in text |
| C — Genre + market | Finance hashtag (Tier 2) **AND** market term (Tier 3) both present |

Dollar amounts like `$5` and `$100` are NOT matched by Rule A (digits ≠ `[A-Z]`).

---

## Pass 2 LLM Filter (Anthropic)

Videos surviving Pass 1 are sent to Claude (`FIN_LLM_MODEL`, default `claude-haiku-4-5-20251001`).

**YES** (keep): Stock picks, earnings discussion, price predictions, buy/sell opinions,
portfolio reveals mentioning specific stocks, market outlook.

**NO** (drop): Generic personal finance, crypto-only, real estate, forex, sports betting,
get-rich-quick promos with no specific stock, platform tutorials.
