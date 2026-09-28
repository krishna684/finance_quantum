import pytest
from pipeline.crsp import CRSPMapper, LookupResult


STUB = {"AAPL": "14593", "TSLA": "93436", "NKE": "57665"}


def test_stub_lookup():
    mapper = CRSPMapper.from_stub(STUB)
    result = mapper.lookup("AAPL")
    assert result is not None
    assert result.permno == "14593"
    assert result.ticker == "AAPL"


def test_lookup_case_insensitive():
    mapper = CRSPMapper.from_stub(STUB)
    assert mapper.lookup("aapl") is not None
    assert mapper.lookup("Tsla") is not None


def test_missing_ticker():
    mapper = CRSPMapper.from_stub(STUB)
    assert mapper.lookup("ZZZZ") is None


def test_enrich_tickers():
    mapper = CRSPMapper.from_stub(STUB)
    results = mapper.enrich_tickers(["AAPL", "TSLA", "UNKNOWN"])
    tickers = {r["ticker"] for r in results}
    assert "AAPL" in tickers
    assert "TSLA" in tickers
    assert "UNKNOWN" not in tickers


def test_enrich_deduplication():
    mapper = CRSPMapper.from_stub(STUB)
    results = mapper.enrich_tickers(["AAPL", "AAPL", "AAPL"])
    assert len(results) == 1


def test_ticker_count():
    mapper = CRSPMapper.from_stub(STUB)
    assert mapper.ticker_count == len(STUB)


def test_from_csv(tmp_path):
    csv_content = "permno,ticker,comnam,ncusip,gvkey,namedt,nameendt\n14593,AAPL,APPLE INC,037833100,1690,2000-01-01,\n93436,TSLA,TESLA INC,,315189,2010-06-29,\n"
    p = tmp_path / "ids.csv"
    p.write_text(csv_content)
    mapper = CRSPMapper.from_csv(p)
    result = mapper.lookup("AAPL")
    assert result.permno == "14593"
    assert result.company_name == "APPLE INC"
    assert result.ncusip == "037833100"


def test_date_range_lookup(tmp_path):
    csv_content = (
        "permno,ticker,comnam,namedt,nameendt\n"
        "99999,OLD,OLD CORP,2000-01-01,2020-12-31\n"
        "99999,NEW,NEW CORP,2021-01-01,\n"
    )
    p = tmp_path / "ids.csv"
    p.write_text(csv_content)
    mapper = CRSPMapper.from_csv(p)
    old = mapper.lookup("OLD", as_of="2015-06-01")
    assert old is not None
    assert old.ticker == "OLD"
    new = mapper.lookup("NEW", as_of="2022-01-01")
    assert new is not None
    assert new.ticker == "NEW"
