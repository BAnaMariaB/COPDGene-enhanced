"""Sanity checks on the source configuration (no network, no Airflow runtime)."""

import copd_ingestion as ci


def test_core_sources_shape():
    assert set(ci.SOURCES) == {"demographics", "imaging", "spirometry"}
    for name, cfg in ci.SOURCES.items():
        assert cfg["url"].startswith("https://"), f"{name} url must be https"
        assert cfg["ext"] in {"csv", "json", "html"}, f"{name} has unexpected ext"


def test_context_sources_shape():
    assert set(ci.CONTEXT_SOURCES) == {"cdc_copd_prevalence", "smoking_prevalence"}
    for name, cfg in ci.CONTEXT_SOURCES.items():
        assert cfg["url"].startswith("https://"), f"{name} url must be https"
        assert cfg["kind"] in {"api", "web_scrape"}, f"{name} has unexpected kind"


def test_cdc_source_is_copd_api():
    cdc = ci.CONTEXT_SOURCES["cdc_copd_prevalence"]
    assert cdc["kind"] == "api"
    assert cdc["ext"] == "json"
    assert "data.cdc.gov" in cdc["url"]
    assert "topicid=COPD" in cdc["url"]  # filtered to COPD, not the whole dataset


def test_wikipedia_source_is_scrape():
    wiki = ci.CONTEXT_SOURCES["smoking_prevalence"]
    assert wiki["kind"] == "web_scrape"
    assert wiki["ext"] == "html"
    assert "wikipedia.org" in wiki["url"]
