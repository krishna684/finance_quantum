import pytest
from pipeline.entities import ExtractionResult, extract_entities


def test_retailer_detected():
    result = extract_entities("I found this on amazon and also saw it at walmart")
    assert "amazon" in result.retailers
    assert "walmart" in result.retailers


def test_category_detected():
    result = extract_entities("great skincare routine and fitness tips")
    assert "skincare" in result.categories
    assert "fitness" in result.categories


def test_empty_text():
    result = extract_entities("")
    assert result.brands == []
    assert result.products == []
    assert result.retailers == []
    assert result.categories == []


def test_finance_terms_not_in_entities():
    # Finance exclusion is done in collector, not entities — but verify no crash
    result = extract_entities("#stocktok investing stocks nasdaq")
    assert isinstance(result, ExtractionResult)


def test_product_phrase_length_limit():
    result = extract_entities("this is a very long text " * 50)
    assert len(result.products) <= 20
    assert len(result.brands) <= 20


def test_case_insensitive_retailer():
    result = extract_entities("AMAZON is sold out, try TARGET")
    assert "amazon" in result.retailers
    assert "target" in result.retailers
