"""Unit tests for the ingest_source task body.

The network is mocked, so these are fast and offline. They verify the two things
ingestion must guarantee: the payload is landed byte-for-byte, and an accurate
provenance sidecar is written next to it.
"""

import hashlib
import json
import os

import pytest

import copd_ingestion as ci
from conftest import underlying


class _FakeResponse:
    def __init__(self, content: bytes, status: int = 200):
        self.content = content
        self.status_code = status
        self.headers = {
            "Content-Type": "text/plain",
            "Content-Length": str(len(content)),
        }

    def raise_for_status(self):
        if self.status_code >= 400:
            raise ci.requests.HTTPError(f"status {self.status_code}")


def _patch_get(monkeypatch, response_or_exc):
    """Patch requests.Session.get to return a fake response (or raise)."""

    def fake_get(self, url, timeout=None):
        if isinstance(response_or_exc, Exception):
            raise response_or_exc
        return response_or_exc

    monkeypatch.setattr(ci.requests.Session, "get", fake_get)


def test_lands_bytes_and_writes_sidecar(tmp_path, monkeypatch):
    payload = b"sid,age\n1001,64\n1002,71\n"
    monkeypatch.setattr(ci, "RAW_ROOT", str(tmp_path))
    _patch_get(monkeypatch, _FakeResponse(payload))

    ingest = underlying(ci.ingest_source)
    dest = ingest(
        source_name="demographics",
        url="https://example.org/demo.csv",
        ext="csv",
        source_kind="static_file",
        ds="2026-07-15",
        run_id="manual__run",
    )

    # File landed, byte-for-byte identical to what was "downloaded".
    assert os.path.isfile(dest)
    with open(dest, "rb") as fh:
        assert fh.read() == payload

    # Partitioned by source name and run date.
    assert dest.endswith(os.path.join("demographics", "2026-07-15", "demographics.csv"))

    # Provenance sidecar is accurate.
    with open(dest + ".meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["sha256"] == hashlib.sha256(payload).hexdigest()
    assert meta["bytes_written"] == len(payload)
    assert meta["ingestion_kind"] == "static_file"
    assert meta["source_url"] == "https://example.org/demo.csv"
    assert meta["http_status"] == 200
    assert meta["partition_ds"] == "2026-07-15"


def test_records_ingestion_kind_for_context_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "RAW_ROOT", str(tmp_path))
    _patch_get(monkeypatch, _FakeResponse(b'[{"topicid":"COPD"}]'))

    ingest = underlying(ci.ingest_source)
    dest = ingest(
        source_name="cdc_copd_prevalence",
        url="https://data.cdc.gov/resource/hksd-2xuw.json?topicid=COPD",
        ext="json",
        source_kind="api",
        ds="2026-07-15",
        run_id="manual__run",
    )
    with open(dest + ".meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["ingestion_kind"] == "api"


def test_missing_ds_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "RAW_ROOT", str(tmp_path))
    ingest = underlying(ci.ingest_source)
    with pytest.raises(ValueError):
        ingest(source_name="demographics", url="https://x/y.csv", ext="csv")


def test_download_error_propagates(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "RAW_ROOT", str(tmp_path))
    _patch_get(monkeypatch, ci.requests.ConnectionError("network down"))

    ingest = underlying(ci.ingest_source)
    with pytest.raises(ci.requests.RequestException):
        ingest(
            source_name="demographics",
            url="https://x/y.csv",
            ext="csv",
            ds="2026-07-15",
        )
    # The partition dir is created before the download, but no data file or
    # sidecar should be written when the download fails.
    landed = os.path.join(str(tmp_path), "demographics", "2026-07-15", "demographics.csv")
    assert not os.path.exists(landed)
    assert not os.path.exists(landed + ".meta.json")
