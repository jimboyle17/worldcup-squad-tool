"""World Cup 2026 Squad Assessment Tool — Streamlit Web App."""

import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

from src.analysis.composition import compare_teams, squad_summary
from src.analysis.manager_assessment import ManagerAssessment
from src.analysis.stats import (
    comparison_dataframe,
    managers_dataframe,
    players_dataframe,
)
from src.analysis.team_rating import calculate_overall_rating
from src.models.manager import Manager
from src.models.team import Team
from src.scraper.cache import ScraperCache
from src.scraper.transfermarkt import TransfermarktScraper
from src.scraper.wikipedia import WikipediaScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="World Cup 2026 Squad Tool",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_value(v: float) -> str:
    if v >= 1_000_000_000:
        return f"\u20ac{v / 1_000_000_000:.2f}bn"
    elif v >= 1_000_000:
        return f"\u20ac{v / 1_000_000:.1f}m"
    elif v >= 1_000:
        return f"\u20ac{v / 1_000:.0f}k"
    return f"\u20ac{v:.0f}"


# ── cached resource initialisation ───────────────────────────────────────────


@st.cache_resource
def _get_scrapers():
    """Create long-lived scraper / cache objects (shared across sessions)."""
    config_path = BASE_DIR / "config.json"
    if config_path.exists():
        config = _load_json(str(config_path))
    else:
        config = {"cache_expiry_hours": 24, "scrape_delay_seconds": 2}

    db_path = str(BASE_DIR / "data" / "cache.db")
    cache = ScraperCache(db_path, expiry_hours=config.get("cache_expiry_hours", 24))
    tm_scraper = TransfermarktScraper(cache, delay=config.get("scrape_delay_seconds", 2))
    wiki_scraper = WikipediaScraper(cache, delay=1.0)
    return cache, tm_scraper, wiki_scraper


def _load_all_teams(tm_scraper, wiki_scraper) -> List[Team]:
    """Load all teams via 3-pass pipeline.

    Results are stored in ``st.session_state["teams"]`` so they survive
    across page navigations without re-scraping.
    """
    if "teams" in st.session_state:
        return st.session_state["teams"]

    teams_path = BASE_DIR / "data" / "teams_2026.json"
    teams_meta = _load_json(str(teams_path)).get("teams", [])

    managers_path = BASE_DIR / "data" / "managers_2026.json"
    managers_dict: Dict[str, Manager] = {}
    if managers_path.exists():
        for m in _load_json(str(managers_path)).get("managers", []):
            try:
                mgr = Manager.from_dict(m)
                managers_dict[mgr.team_name] = mgr
            except Exception as e:
                logger.error(f"Failed to parse manager {m.get('team_name', '?')}: {e}")

    total = len(teams_meta)
    progress = st.progress(0, text="Loading squads...")
    teams: List[Team] = []

    # Pass 1 — squads from Transfermarkt
    for i, meta in enumerate(teams_meta):
        try:
            team = tm_scraper.scrape_team(meta)
        except Exception as e:
            logger.error(f"Failed to load {meta['name']}: {e}")
            team = Team(
                name=meta["name"],
                country_code=meta.get("country_code", ""),
                confederation=meta.get("confederation", ""),
                fifa_ranking=meta.get("fifa_ranking", 0),
                transfermarkt_id=meta["transfermarkt_id"],
            )
        teams.append(team)
        progress.progress((i + 1) / total / 2, text=f"Squads... {i + 1}/{total}")

    # Pass 2 — caps / goals from Wikipedia
    for i, team in enumerate(teams):
        try:
            wiki_scraper.update_team_players(team)
        except Exception as e:
            logger.error(f"Wikipedia data failed for {team.name}: {e}")
        progress.progress(0.5 + (i + 1) / total / 2, text=f"Caps/goals... {i + 1}/{total}")

    # Pass 3 — attach managers
    for team in teams:
        mgr = managers_dict.get(team.name)
        if mgr:
            team.manager = mgr

    progress.empty()
    teams.sort(key=lambda t: t.fifa_ranking)
    st.session_state["teams"] = teams
    return teams


def _compute_assessments(teams: List[Team]) -> Dict[str, ManagerAssessment]:
    """Compute manager assessments (cached in session_state)."""
    if "assessments" in st.session_state:
        return st.session_state["assessments"]

    assessments: Dict[str, ManagerAssessment] = {}
    for team in teams:
        if team.manager:
            assessments[team.name] = ManagerAssessment(
                manager=team.manager,
                current_fifa_ranking=team.fifa_ranking,
            )
    st.session_state["assessments"] = assessments
    return assessments


# ── page renderers ───────────────────────────────────────────────────────────


def _page_teams(teams: List[Team], assessments: Dict[str, ManagerAssessment]):
    st.header("World Cup 2026 Teams")

    rows = []
    for t in teams:
        a = assessments.get(t.name)
        rows.append({
            "Team": t.name,
            "Confederation": t.confederation,
            "FIFA Ranking": t.fifa_ranking,
            "Squad Size": t.squad_size,
            "Total Value": t.total_value_display,
            "Avg Age": round(t.average_age, 1),
            "Mgr Score": a.composite_score if a else None,
            "Overall Rating": calculate_overall_rating(t, teams, a),
        })

    df = pd.DataFrame(rows)

    event = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    # Handle row selection → navigate to squad detail
    if event and event.selection and event.selection.rows:
        row_idx = event.selection.rows[0]
        selected_name = df.iloc[row_idx]["Team"]
        st.session_state["selected_team"] = selected_name
        st.session_state["page"] = "Squad Detail"
        st.rerun()


def _page_squad_detail(teams: List[Team], assessments: Dict[str, ManagerAssessment]):
    st.header("Squad Detail")

    team_names = [t.name for t in teams]
    teams_dict = {t.name: t for t in teams}

    default_idx = 0
    if "selected_team" in st.session_state and st.session_state["selected_team"] in team_names:
        default_idx = team_names.index(st.session_state["selected_team"])

    selected = st.selectbox("Select team", team_names, index=default_idx)
    team = teams_dict[selected]
    assessment = assessments.get(team.name)
    summary = squad_summary(team)

    # ── stat metrics row ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Squad Size", summary["squad_size"])
    c2.metric("Avg Age", summary["average_age"])
    c3.metric("Total Value", _fmt_value(summary["total_value"]))
    c4.metric("Avg Caps", summary["average_caps"])
    c5.metric("Most Valuable", summary["most_valuable"])

    # ── composition summary ──
    pos = summary["position_breakdown"]
    age = summary["age_distribution"]
    st.caption(
        f"**Positions:** GK {pos.get('GK', 0)} | DF {pos.get('DF', 0)} | "
        f"MF {pos.get('MF', 0)} | FW {pos.get('FW', 0)}  ·  "
        f"**Ages:** U21 {age.get('U21', 0)} | 21-25 {age.get('21-25', 0)} | "
        f"26-29 {age.get('26-29', 0)} | 30+ {age.get('30+', 0)}"
    )

    # ── manager card ──
    if assessment:
        st.subheader(f"Manager: {assessment.manager.name}")
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)

        mc1.metric("Experience", round(assessment.experience_score, 1))
        mc2.metric("Honours", round(assessment.honours_score, 1))
        mc3.metric("Club Achievement", round(assessment.club_achievement_score, 1))
        mc4.metric("Tenure", round(assessment.tenure_score, 1))
        mc5.metric(
            "Composite",
            assessment.composite_score,
            delta=assessment.rating_impact_pct,
        )
        st.divider()

    # ── position filter ──
    pos_filter = st.selectbox("Filter by position", ["All", "GK", "DF", "MF", "FW"])
    players = team.squad
    if pos_filter != "All":
        players = [p for p in players if p.position == pos_filter]

    # ── player table ──
    if players:
        pdf = players_dataframe(players)
        display_cols = ["Name", "Position", "Detail", "Age", "Club", "Market Value", "Caps", "Goals"]
        available = [c for c in display_cols if c in pdf.columns]
        st.dataframe(pdf[available], use_container_width=True, hide_index=True)
        st.caption(f"{len(players)} players shown")

        # download button
        csv = pdf[available].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download CSV",
            csv,
            file_name=f"{team.name.lower().replace(' ', '_')}_squad.csv",
            mime="text/csv",
        )
    else:
        st.info("No players to display.")


def _page_managers(teams: List[Team], assessments: Dict[str, ManagerAssessment]):
    st.header("Manager Assessments")

    if not assessments:
        st.info("No manager data available.")
        return

    df = managers_dataframe(assessments)
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download CSV",
        csv,
        file_name="manager_assessments.csv",
        mime="text/csv",
    )


def _page_compare(teams: List[Team], assessments: Dict[str, ManagerAssessment]):
    st.header("Compare Teams")

    team_names = [t.name for t in teams]
    teams_dict = {t.name: t for t in teams}

    col_a, col_b = st.columns(2)
    with col_a:
        name_a = st.selectbox("Team A", team_names, index=0)
    with col_b:
        default_b = min(1, len(team_names) - 1)
        name_b = st.selectbox("Team B", team_names, index=default_b)

    team_a = teams_dict[name_a]
    team_b = teams_dict[name_b]
    assessment_a = assessments.get(name_a)
    assessment_b = assessments.get(name_b)

    rows = compare_teams(team_a, team_b, assessment_a, assessment_b)
    df = comparison_dataframe(rows, name_a, name_b)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    # sidebar nav
    st.sidebar.title("WC 2026")
    page = st.sidebar.radio(
        "Navigate",
        ["Teams", "Squad Detail", "Managers", "Compare"],
        index=["Teams", "Squad Detail", "Managers", "Compare"].index(
            st.session_state.get("page", "Teams")
        ),
        key="nav_radio",
    )
    st.session_state["page"] = page

    # load data (cached after first run)
    cache, tm_scraper, wiki_scraper = _get_scrapers()
    teams = _load_all_teams(tm_scraper, wiki_scraper)
    assessments = _compute_assessments(teams)

    # render selected page
    if page == "Teams":
        _page_teams(teams, assessments)
    elif page == "Squad Detail":
        _page_squad_detail(teams, assessments)
    elif page == "Managers":
        _page_managers(teams, assessments)
    elif page == "Compare":
        _page_compare(teams, assessments)


if __name__ == "__main__":
    main()
