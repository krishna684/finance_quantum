from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

# Thresholds from VADER documentation
_POS_THRESHOLD = 0.05
_NEG_THRESHOLD = -0.05


def score_text(text: str) -> tuple[float, str]:
    """Return (compound_score, label) for *text*.

    compound_score: float in [-1, 1]
    label: 'positive' | 'negative' | 'neutral'
    """
    if not text or not text.strip():
        return 0.0, "neutral"

    scores = _analyzer.polarity_scores(text)
    compound = round(scores["compound"], 4)

    if compound >= _POS_THRESHOLD:
        label = "positive"
    elif compound <= _NEG_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"

    return compound, label
