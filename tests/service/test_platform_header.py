from __future__ import annotations

from pathlib import Path


STATIC = Path(__file__).parents[2] / "src" / "autoalpha" / "service" / "static"
PLATFORM_PAGES = (
    "index.html",
    "research_tasks.html",
    "autocombine.html",
    "llm_team.html",
    "factors.html",
    "screener.html",
    "paper_trading.html",
    "backtest.html",
    "batch_backtest.html",
    "data_center.html",
    "settings.html",
)


def test_every_platform_page_loads_the_shared_header() -> None:
    for filename in PLATFORM_PAGES:
        markup = (STATIC / filename).read_text(encoding="utf-8")
        assert "/static/platform_header.css" in markup, filename
        assert "/static/platform_header.js" in markup, filename


def test_shared_header_is_the_single_canonical_navigation_source() -> None:
    script = (STATIC / "platform_header.js").read_text(encoding="utf-8")
    expected_routes = (
        "research",
        "tasks",
        "combine",
        "strategies",
        "llm",
        "factors",
        "screener",
        "paper",
        "backtest",
        "data",
        "settings",
    )

    assert script.count("<a data-platform-route=") == 1
    assert "header.replaceChildren(identity, navSlot, tools)" in script
    assert '.id = "autoResearchNav"' in script
    for route in expected_routes:
        assert f'["{route}",' in script


def test_shared_header_uses_stable_centered_grid_tracks() -> None:
    stylesheet = (STATIC / "platform_header.css").read_text(encoding="utf-8")

    assert "grid-template-areas: \"identity navigation tools\"" in stylesheet
    assert "grid-template-columns: minmax(190px, 1fr) auto minmax(190px, 1fr)" in stylesheet
    assert "grid-area: navigation" in stylesheet
