"""Optional live smoke test: every configured URL is actually reachable.

Deselected by default (marked `live`) because it hits the network. Run with:
    pytest -m live
Useful to confirm the CDC API and Wikipedia page still respond before a demo.
"""

import pytest
import requests

import copd_ingestion as ci

ALL_SOURCES = {**ci.SOURCES, **ci.CONTEXT_SOURCES}


@pytest.mark.live
@pytest.mark.parametrize("name", sorted(ALL_SOURCES))
def test_source_url_is_reachable(name):
    url = ALL_SOURCES[name]["url"]
    # Use the same User-Agent the DAG sends, so this mirrors real ingestion
    # (e.g. Wikipedia returns 403 to the default python-requests UA).
    resp = requests.get(url, headers={"User-Agent": ci.USER_AGENT}, timeout=60)
    assert resp.status_code == 200, f"{name}: HTTP {resp.status_code}"
    assert len(resp.content) > 0, f"{name}: empty response"
