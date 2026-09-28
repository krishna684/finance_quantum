# Data Collection Protocol

## Research Objective

This project constructs a return-aligned financial media dataset to test whether institutional
and retail media sentiment predicts stock and market returns.

The dataset supports construction of:

1. Daily aggregate (market-level) sentiment indices
2. Daily firm-level cross-sectional sentiment measures
3. Attention-weighted and velocity-adjusted sentiment signals
4. Image-based vs text-based sentiment comparisons
5. Institutional vs retail sentiment divergence measures
6. Clean merges with CRSP daily returns, TAQ intraday data, Compustat fundamentals

**Core principle:** Every field collected must directly support causal mapping from media
exposure to subsequent returns.

---

## Data Sources

| Source | Channel | Coverage |
|--------|---------|---------|
| Wall Street Journal | Institutional media | Articles + images |
| New York Times | Institutional media | Articles + images |
| TikTok Research API | Retail consumer demand | Videos + comments |

---

## WSJ and NYT Collection Protocol

**Access:** Via MU Library licensed academic databases permitting Text and Data Mining (TDM).
No website scraping.

**Time window:** January 1, 2021 – present.

### Article-level fields (`news_articles`)

| Field | Format | Purpose |
|-------|--------|---------|
| `article_id` | UUID PK | Unique identifier |
| `source` | WSJ or NYT | Source discrimination |
| `headline` | Full text | Sentiment extraction |
| `full_article_body` | Complete text | Sentiment extraction |
| `publication_date` | YYYY-MM-DD | Return alignment |
| `publication_time` | HH:MM:SS | Intraday alignment |
| `timezone` | ISO 8601 UTC | Bias prevention |
| `section_category` | Text | Isolate market-relevant content |
| `front_page_flag` | Boolean | Attention proxy |
| `author_names` | Text | Journalist fixed effects |
| `word_count` | Integer | Salience proxy |
| `database_source_id` | Text | Audit trail |

### Image-level fields (`news_images`)

| Field | Format | Purpose |
|-------|--------|---------|
| `image_id` | UUID PK | Unique identifier |
| `article_id` | FK | Relational link |
| `image_binary` | JPEG/PNG | Visual sentiment extraction |
| `image_caption` | Full text | Caption sentiment |
| `placement` | Lead/inline/thumbnail | Attention weighting |
| `image_order` | Integer | Position within article |
| `resolution` | Width × height px | Quality filtering |
| `sha256_hash` | Text | Deduplication |

---

## TikTok Consumer Demand Signal Protocol

### Objective

Capture product-level consumer demand and negative sentiment as leading indicators of publicly
traded firm revenue. TikTok users discuss products and brands, not tickers — this product-level
buzz is a leading indicator of firm revenue before it shows up in earnings.

Two signal types:
1. **High purchase intent** — rising consumer demand before official sales numbers
2. **Negative consumer sentiment** — quality problems or brand damage that may foreshadow revenue misses

Explicitly excluded: direct financial content (#stocktok, #fintok, ticker symbols, investing hashtags).

### Search queries

**High purchase intent:**
- "where can I buy this", "is there a link", "drop the link", "how much is this"
- "is this on amazon", "need this now", "shut up and take my money"
- "add to cart", "just ordered", "sold out everywhere"
- `#tiktokmademebuyit` × {skincare, fitness, kitchen, tech, supplements, shoes, snacks}

**Negative consumer sentiment:**
- "is this a scam", "don't buy this", "waste of money", "looks cheap"
- "took forever to ship", "never arrived", "broke after one use", "not worth it", "return this"

### Sampling rules (fixed ex ante)

- Top 50–100 videos per query sorted by engagement (views + likes + comments)
- Plus random low-engagement control sample (same queries, `is_random=true`)
- Start with past 12 months, extend backward
- Save everything on first contact — TikTok content disappears fast

### Video-level fields (`tiktok_videos`)

| Field | Format | Purpose |
|-------|--------|---------|
| `video_id` | PK | Unique identifier |
| `creator_username` | Text | Creator tracking |
| `post_datetime` | ISO 8601 UTC | Return alignment |
| `full_transcript` | Text | Sentiment extraction |
| `hashtags` | Array | Topic clustering |
| `views` / `likes` / `shares` / `total_comments` | Integer (snapshot) | Engagement proxies |
| `collection_timestamp` | ISO 8601 UTC | Snapshot time audit |

### Comment-level fields (`tiktok_comments`)

| Field | Format | Purpose |
|-------|--------|---------|
| `comment_id` | PK | Unique identifier |
| `video_id` | FK | Relational link |
| `comment_text` | Full text | Sentiment extraction |
| `comment_timestamp` | ISO 8601 UTC | Temporal analysis |
| `comment_likes` | Integer | Comment salience |
| `reply_indicator` | Boolean | Thread structure |
| `parent_comment_id` | FK | Thread structure |

### Entity extraction (required at collection time)

For every video and comment, extract and record:
- Brand names, product names, retailers, product categories

These map TikTok content to publicly traded companies via CRSP PERMNO.

---

## Brand-to-Firm Mapping

Maps every brand and product mention to a CRSP PERMNO via:

1. **SEC EDGAR 10-K filings** — LLM extraction from Item 1 and Item 7
2. **Compustat segment data** — revenue by segment/product line
3. **Wikipedia / Wikidata** — structured brand ownership data

### Mapping table schema

| Column | Description |
|--------|-------------|
| `parent_company` | Legal entity name |
| `ticker` | Exchange ticker |
| `permno` | CRSP PERMNO |
| `brand_name` | Consumer-facing brand |
| `product_line` | Specific product line |
| `product_category` | Category (skincare, tech, etc.) |

Fuzzy matching required for colloquial brand references (e.g., "Stanley cup" → Stanley 1913).

---

## Time Alignment (Look-Ahead Bias Prevention)

All timestamps stored in UTC. Each media event classified as:

| Window | Definition | Return assignment |
|--------|-----------|------------------|
| Pre-market | 4:00 PM prev day – 9:30 AM ET | Open-to-close return day t |
| Market hours | 9:30 AM – 4:00 PM ET | Remaining intraday or t+1 |
| Post-market | 4:00 PM – midnight ET | Next trading day |
| Weekend / holiday | — | Next CRSP trading day |

---

## Storage Architecture

- Primary: **Parquet**; secondary: **CSV**
- All timestamps: ISO 8601, timezone-aware, UTC
- Linked by: `article_id`, `image_id`, `video_id`, brand/product name keys
- Designed for merge with CRSP/TAQ/Compustat via PERMNO

---

## Derived Attention Metrics

**Comment velocity:**

$$V_i = \frac{C_i(0, 24h)}{24}$$

**Engagement acceleration:**

$$A_i = \frac{d^2 E_i}{dt^2}$$

where $E_i$ = cumulative engagement (likes + comments + shares).

---

## Firm Identifier Mapping

All detected tickers, firm names, brands, and products mapped to CRSP PERMNO, NCUSIP,
and Compustat gvkey. Accounts for ticker changes, mergers, delistings, multiple share classes,
effective date ranges, and brand ownership changes over time.
