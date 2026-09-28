import json
import pytest
from pipeline.news.models import NewsArticle, NewsImage
from pipeline.news.storage import NewsStorage


@pytest.fixture
def ns(tmp_path):
    s = NewsStorage(tmp_path / "news_test.db")
    s.init_schema()
    yield s
    s.close()


def _article(article_id="a001", source="WSJ"):
    return NewsArticle(
        article_id=article_id,
        source=source,
        headline="Apple Reports Record Earnings",
        full_body="Apple beat estimates driven by iPhone sales.",
        publication_date="2024-01-16",
        publication_time="09:30:00",
        timezone="America/New_York",
        section="Markets",
        front_page=True,
        authors=["Jane Smith"],
        word_count=8,
        database_source_id=f"DB-{article_id}",
        url=None,
    )


def test_upsert_article(ns):
    art = _article()
    enrichment = {
        "sentiment_score": 0.5,
        "sentiment_label": "positive",
        "mapped_tickers": ["AAPL"],
        "company_names": ["apple"],
        "cashtag_tickers": [],
        "extracted_brands": [],
        "extracted_products": [],
        "extracted_retailers": [],
        "extracted_categories": ["tech"],
    }
    ns.upsert_article(art, enrichment)
    ns.commit()
    row = ns.conn.execute("SELECT * FROM news_articles WHERE article_id='a001'").fetchone()
    assert row is not None
    assert row["source"] == "WSJ"
    assert row["front_page"] == 1
    assert json.loads(row["mapped_tickers"]) == ["AAPL"]
    assert row["sentiment_label"] == "positive"


def test_article_exists(ns):
    ns.upsert_article(_article())
    ns.commit()
    assert ns.article_exists("DB-a001")
    assert not ns.article_exists("nonexistent")


def test_upsert_image_dedup(ns):
    ns.upsert_article(_article())
    ns.commit()
    img = NewsImage(
        image_id="img001",
        article_id="a001",
        image_binary=b"fake_image_bytes",
        caption="Apple campus",
        placement="lead",
        image_order=1,
        width_px=800,
        height_px=600,
        sha256_hash=NewsImage.compute_hash(b"fake_image_bytes"),
        file_extension="jpg",
    )
    ns.upsert_image(img)
    ns.upsert_image(img)  # duplicate — INSERT OR IGNORE
    ns.commit()
    count = ns.conn.execute("SELECT COUNT(*) FROM news_images").fetchone()[0]
    assert count == 1


def test_image_hash_exists(ns):
    ns.upsert_article(_article())
    img = NewsImage(
        image_id="img002",
        article_id="a001",
        image_binary=b"data",
        caption=None,
        placement=None,
        image_order=0,
        width_px=None,
        height_px=None,
        sha256_hash=NewsImage.compute_hash(b"data"),
        file_extension="png",
    )
    ns.upsert_image(img)
    ns.commit()
    assert ns.image_hash_exists(NewsImage.compute_hash(b"data"))
    assert not ns.image_hash_exists("00" * 32)
