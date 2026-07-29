from __future__ import annotations

from pathlib import Path

STATIC = Path(__file__).parents[2] / "src" / "autoalpha" / "service" / "static"
PLATFORM_PAGES = (
    "index.html",
    "research_tasks.html",
    "autocombine.html",
    "quantcombine.html",
    "llm_team.html",
    "factors.html",
    "screener.html",
    "paper_trading.html",
    "formal_strategies.html",
    "backtest.html",
    "batch_backtest.html",
    "data_center.html",
    "settings.html",
    "system_guide.html",
)


def test_every_platform_page_loads_the_shared_header() -> None:
    for filename in PLATFORM_PAGES:
        markup = (STATIC / filename).read_text(encoding="utf-8")
        assert "/static/platform_header.css" in markup, filename
        assert "/static/platform_header.js" in markup, filename
        assert markup.count("/static/platform_header.js") == 1, filename
        header_end = markup.index("</header>")
        shared_script = markup.index("/static/platform_header.js")
        main_start = markup.index("<main")
        assert header_end < shared_script < main_start, filename


def test_shared_header_is_the_single_canonical_navigation_source() -> None:
    script = (STATIC / "platform_header.js").read_text(encoding="utf-8")
    expected_routes = (
        "research",
        "tasks",
        "combine",
        "quantcombine",
        "strategies",
        "llm",
        "factors",
        "screener",
        "paper",
        "backtest",
        "data",
        "guide",
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


def test_system_guide_covers_the_full_research_control_plane() -> None:
    markup = (STATIC / "system_guide.html").read_text(encoding="utf-8")
    application = (STATIC.parent / "app.py").read_text(encoding="utf-8")
    expected_sections = (
        "position",
        "architecture",
        "lifecycle",
        "autoalpha",
        "llm-team",
        "evaluation",
        "data",
        "factor-assets",
        "autocombine",
        "execution",
        "surfaces",
        "governance",
        "boundaries",
    )

    assert markup.count('data-stage="') == 12
    assert "DETERMINISTIC" in markup
    assert "EOD_T__OPEN_T1_TO_OPEN_T2" in markup
    assert "@app.get(\"/guide\"" in application
    for section in expected_sections:
        assert f'id="{section}"' in markup


def test_factor_library_api_uses_materialized_cache_contract() -> None:
    application = (STATIC.parent / "app.py").read_text(encoding="utf-8")

    assert "async def factor_library(response: Response, refresh: bool = False)" in application
    assert "FactorLibraryRefreshRequest" in application
    assert 'store.materialized_snapshot("factor_library")' in application
    assert '"factor_library_refresh"' in application
    assert "Factor library refresh queued in Job Center." in application
    assert 'store.upsert_materialized_snapshot(\n        "factor_library",' in application
    assert "MATERIALIZED_FACTOR_LIBRARY_API_V1" in application
    assert "AUTOALPHA_FACTOR_KNOWLEDGE_INTEGRITY_V1" in application
    assert "factor_knowledge_missing_count" in application
