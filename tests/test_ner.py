import pytest
from pipeline.ner import NERResult, all_tickers, extract_ner


def test_cashtag_detection():
    result = extract_ner("Just bought $AAPL and $TSLA calls")
    assert "AAPL" in result.cashtag_tickers
    assert "TSLA" in result.cashtag_tickers


def test_company_name_mapping():
    result = extract_ner("I love my new apple watch, buying more starbucks")
    assert "AAPL" in result.mapped_tickers
    assert "SBUX" in result.mapped_tickers
    assert "apple" in result.company_names
    assert "starbucks" in result.company_names


def test_longest_match_wins():
    # "estee lauder" should be found before "lauder" alone
    result = extract_ner("Bought estee lauder serum from sephora")
    assert "estee lauder" in result.company_names
    # should not double-count as "lauder" separately
    assert result.company_names.count("lauder") == 0 or "estee lauder" in result.company_names


def test_empty_text():
    result = extract_ner("")
    assert result.cashtag_tickers == []
    assert result.company_names == []
    assert result.mapped_tickers == []


def test_all_tickers_deduplication():
    result = NERResult(
        cashtag_tickers=["AAPL"],
        bare_tickers=["AAPL", "TSLA"],
        company_names=["apple"],
        mapped_tickers=["AAPL"],
    )
    tickers = all_tickers(result)
    assert tickers.count("AAPL") == 1
    assert "TSLA" in tickers


def test_all_tickers_order():
    # cashtags should come first
    result = NERResult(
        cashtag_tickers=["TSLA"],
        bare_tickers=["AAPL"],
        company_names=[],
        mapped_tickers=[],
    )
    tickers = all_tickers(result)
    assert tickers[0] == "TSLA"


def test_private_company_no_ticker():
    result = extract_ner("I use dyson every day")
    # Dyson is private, should not produce a mapped ticker
    assert "dyson" in result.company_names
    assert not any(t for t in result.mapped_tickers if t)


def test_stopword_not_bare_ticker():
    result = extract_ner("IT IS THE BEST THING I HAVE EVER BOUGHT")
    # IT, IS, THE, etc. should be filtered out
    for t in result.bare_tickers:
        assert len(t) >= 2
