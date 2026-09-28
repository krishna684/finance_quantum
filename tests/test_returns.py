import pytest
from datetime import date
from pipeline.returns import align_timestamp, CRSPReturns, AlignedEvent


# ---------------------------------------------------------------------------
# align_timestamp tests
# ---------------------------------------------------------------------------

def test_pre_market():
    # 7 AM ET on a Tuesday
    event = align_timestamp("2024-01-16T07:00:00-05:00")
    assert event.window == "pre_market"
    assert event.trading_day == date(2024, 1, 16)


def test_market_hours():
    # 10:30 AM ET Tuesday
    event = align_timestamp("2024-01-16T10:30:00-05:00")
    assert event.window == "market"
    assert event.trading_day == date(2024, 1, 16)


def test_post_market():
    # 5 PM ET Tuesday → next trading day (Wednesday)
    event = align_timestamp("2024-01-16T17:00:00-05:00")
    assert event.window == "post_market"
    assert event.trading_day == date(2024, 1, 17)


def test_weekend_rolls_forward():
    # Saturday → next Monday
    event = align_timestamp("2024-01-13T12:00:00-05:00")
    assert event.window == "weekend"
    assert event.trading_day == date(2024, 1, 15)  # Monday Jan 15


def test_friday_post_market_goes_to_monday():
    # Friday 6 PM → Monday
    event = align_timestamp("2024-01-12T18:00:00-05:00")
    assert event.window == "post_market"
    assert event.trading_day == date(2024, 1, 15)


def test_utc_timestamp():
    # UTC timestamp → should be converted to ET
    event = align_timestamp("2024-01-16T15:00:00+00:00")  # 10 AM ET
    assert event.window == "market"


# ---------------------------------------------------------------------------
# CRSPReturns tests
# ---------------------------------------------------------------------------

def _build_returns(tmp_path):
    content = (
        "permno,date,ret,retx,prc,vol,shrout\n"
        "14593,2024-01-15,0.01,0.01,185.50,50000000,15000000\n"
        "14593,2024-01-16,0.02,0.02,189.21,45000000,15000000\n"
        "14593,2024-01-17,-0.005,-0.005,188.26,42000000,15000000\n"
        "93436,2024-01-15,0.03,0.03,250.00,20000000,3000000\n"
        "93436,2024-01-16,-0.01,-0.01,247.50,18000000,3000000\n"
    )
    p = tmp_path / "dsf.csv"
    p.write_text(content)
    return CRSPReturns.from_csv(p)


def test_get_return(tmp_path):
    ret = _build_returns(tmp_path)
    r = ret.get_return("14593", "2024-01-15")
    assert abs(r - 0.01) < 1e-6


def test_get_return_missing(tmp_path):
    ret = _build_returns(tmp_path)
    assert ret.get_return("14593", "2024-01-20") is None
    assert ret.get_return("ZZZZ", "2024-01-15") is None


def test_window_return(tmp_path):
    ret = _build_returns(tmp_path)
    # 2-day return: (1.01)(1.02) - 1
    r = ret.window_return("14593", "2024-01-15", "2024-01-16")
    assert abs(r - (1.01 * 1.02 - 1)) < 1e-6


def test_permno_count(tmp_path):
    ret = _build_returns(tmp_path)
    assert ret.permno_count == 2


def test_date_range(tmp_path):
    ret = _build_returns(tmp_path)
    start, end = ret.date_range
    assert start == date(2024, 1, 15)
    assert end == date(2024, 1, 17)


def test_crsp_missing_return_codes(tmp_path):
    content = "permno,date,ret,retx,prc,vol,shrout\n14593,2024-01-15,C,C,185.50,.,.\n"
    p = tmp_path / "dsf.csv"
    p.write_text(content)
    ret = CRSPReturns.from_csv(p)
    assert ret.get_return("14593", "2024-01-15") is None
