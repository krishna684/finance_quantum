import pytest
from pipeline.sentiment import score_text


def test_positive_text():
    score, label = score_text("I just ordered this and it is amazing, love it so much!")
    assert label == "positive"
    assert score > 0.05


def test_negative_text():
    score, label = score_text("is this a scam? waste of money, broke after one use, do not buy this")
    assert label == "negative"
    assert score < -0.05


def test_neutral_empty():
    score, label = score_text("")
    assert label == "neutral"
    assert score == 0.0


def test_neutral_whitespace():
    score, label = score_text("   ")
    assert label == "neutral"
    assert score == 0.0


def test_score_range():
    for text in ["great product", "terrible quality", "ok I guess"]:
        score, _ = score_text(text)
        assert -1.0 <= score <= 1.0
