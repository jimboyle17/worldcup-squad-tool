"""World Cup 2026 Squad Assessment & Betting Edge Tool — Streamlit Web App."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.analysis.composition import compare_teams, squad_summary
from src.analysis.home_advantage import HomeAdvantageConfig, home_advantage_info
from src.analysis.manager_assessment import ManagerAssessment
from src.analysis.match_probability import (
    MatchPrediction,
    TeamStrength,
    compute_team_strengths,
    predict_match,
    predict_knockout_match,
)
from src.analysis.stats import (
    comparison_dataframe,
    managers_dataframe,
    players_dataframe,
)
from src.analysis.team_rating import calculate_base_team_rating, calculate_overall_rating
from src.analysis.tournament_sim import (
    GroupDraw,
    SimulationResult,
    load_group_draw,
    run_tournament_simulation,
)
from src.analysis.value_finder import (
    MarketOdds,
    ValueBet,
    calculate_overround,
    find_value_bets,
    generate_odds_template,
    parse_odds_csv,
)
from src.analysis.pl_simulator import (
    DEFAULT_PL_RATINGS,
    DEFAULT_PL_TEAMS,
    PL_AVG_GOALS_PER_TEAM,
    PL_HOME_XG_BOOST,
    PLSimulationResult,
    generate_pl_ratings_template,
    get_pl_match_predictions,
    parse_pl_ratings_csv,
    run_pl_simulation,
)
from src.models.manager import Manager
from src.models.squad import Squad
from src.models.team import Team
from src.analysis.power_ranking import compute_power_rankings
from src.scraper.cache import ScraperCache
from src.scraper.oddsportal import OddsPortalScraper
from src.scraper.transfermarkt import TransfermarktScraper
from src.scraper.wikipedia import WikipediaScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="World Cup 2026 Betting Edge Tool",
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


def _fmt_pct(v: float) -> str:
    """Format a probability as a percentage string."""
    return f"{v * 100:.1f}%"


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
    """Load all teams via 3-pass pipeline."""
    if "teams" in st.session_state:
        return st.session_state["teams"]

    teams_path = BASE_DIR / "data" / "teams_2026.json"
    teams_meta = _load_json(str(teams_path)).get("teams", [])

    # Filter out TBD/playoff teams (no Transfermarkt ID)
    teams_meta = [m for m in teams_meta if m.get("transfermarkt_id")]

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

    # Check if squad data loaded
    teams_with_squads = sum(1 for t in teams if t.squad_size > 0)
    if teams_with_squads == 0:
        st.session_state["squad_data_missing"] = True
    elif teams_with_squads < len(teams):
        st.session_state["squad_data_partial"] = True

    teams.sort(key=lambda t: t.fifa_ranking)
    st.session_state["teams"] = teams
    return teams


def _load_player_games(teams: List[Team], tm_scraper, force_refresh: bool = False):
    """Load recent game counts for all players (opt-in, slow first load)."""
    if not force_refresh and "player_games_loaded" in st.session_state:
        return

    all_players = [(t, p) for t in teams for p in t.squad if p.transfermarkt_id]
    total = len(all_players)
    if total == 0:
        st.session_state["player_games_loaded"] = True
        return

    progress = st.progress(0, text="Player games... 0/{0}".format(total))
    for i, (team, player) in enumerate(all_players):
        try:
            g30, g60 = tm_scraper.scrape_player_recent_games(
                player.transfermarkt_id, force_refresh=force_refresh,
            )
            player.games_last_30 = g30
            player.games_last_60 = g60
        except Exception as e:
            logger.error(f"Failed to load games for {player.name}: {e}")
        progress.progress((i + 1) / total, text=f"Player games... {i + 1}/{total}")

    progress.empty()
    st.session_state["player_games_loaded"] = True


def _load_power_rankings_from_file() -> Dict[str, float]:
    """Load pre-computed power rankings from JSON file."""
    path = BASE_DIR / "data" / "power_rankings.json"
    if not path.exists():
        return {}
    data = _load_json(str(path))
    return {name: info["rating"] for name, info in data.items()}


def _load_power_rankings(cache: ScraperCache, try_live: bool = False) -> Dict[str, float]:
    """Load power rankings — from file by default, optionally try live scrape."""
    if "power_ratings" in st.session_state:
        return st.session_state["power_ratings"]

    if try_live:
        odds_scraper = OddsPortalScraper(cache, delay=3.0)
        try:
            progress = st.progress(0, text="Launching browser & scraping odds data...")
            all_matches = odds_scraper.scrape_all_competitions()
            progress.progress(0.7, text="Fitting Bradley-Terry model...")

            if all_matches:
                wc_names = None
                if "teams" in st.session_state:
                    wc_names = {t.name for t in st.session_state["teams"]}

                rankings = compute_power_rankings(all_matches, wc_names)
                power_ratings = {name: pr.rating for name, pr in rankings.items()}

                progress.empty()
                st.session_state.pop("power_error", None)
                st.session_state["power_ratings"] = power_ratings
                return power_ratings

            progress.empty()
        except Exception as e:
            logger.error(f"Live power rankings scrape failed: {e}")
        finally:
            odds_scraper.close()

    # Load from pre-computed JSON file (default)
    power_ratings = _load_power_rankings_from_file()
    if power_ratings:
        logger.info(f"Loaded {len(power_ratings)} power ratings from file")
        st.session_state.pop("power_error", None)
        st.session_state["power_ratings"] = power_ratings
        return power_ratings

    st.session_state["power_error"] = "No pre-computed power rankings file found"
    return {}


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


def _compute_all_team_ratings(
    teams: List[Team],
    assessments: Dict[str, ManagerAssessment],
    home_config: Optional[HomeAdvantageConfig] = None,
    power_ratings: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute overall ratings for all teams. Cached in session_state."""
    cache_key = "all_team_ratings"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    ratings = {}
    for t in teams:
        a = assessments.get(t.name)
        pr = power_ratings.get(t.name) if power_ratings else None
        ratings[t.name] = calculate_overall_rating(t, teams, a, home_config, pr)

    st.session_state[cache_key] = ratings
    return ratings


# ── page renderers ───────────────────────────────────────────────────────────


def _page_teams(teams: List[Team], assessments: Dict[str, ManagerAssessment],
                home_config: HomeAdvantageConfig = None,
                power_ratings: Dict[str, float] = None):
    st.header("World Cup 2026 Teams")

    rows = []
    for t in teams:
        a = assessments.get(t.name)
        sq = Squad(t.squad)
        pr = power_ratings.get(t.name) if power_ratings else None
        base_r = calculate_base_team_rating(t, teams, pr)
        info = home_advantage_info(t.name, base_r, home_config)
        rows.append({
            "Team": t.name,
            "Confederation": t.confederation,
            "FIFA Ranking": t.fifa_ranking,
            "Squad Size": t.squad_size,
            "Total Value": t.total_value_display,
            "Best XI Value": _fmt_value(sq.best_xi_value()),
            "Best XVIII Value": _fmt_value(sq.best_xviii_value()),
            "Avg Age": round(t.average_age, 1),
            "Mgr Score": a.composite_score if a else None,
            "Power Rating": round(pr, 1) if pr is not None else None,
            "Home Boost": info["boost_label"],
            "Overall Rating": calculate_overall_rating(t, teams, a, home_config, pr),
        })

    df = pd.DataFrame(rows)

    event = st.dataframe(
        df,
        width="stretch",
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
        display_cols = ["Name", "Position", "Detail", "Age", "Club", "Market Value", "Caps", "Goals", "Last 30d", "Last 60d"]
        available = [c for c in display_cols if c in pdf.columns]
        st.dataframe(pdf[available], width="stretch", hide_index=True)
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
    st.dataframe(df, width="stretch", hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download CSV",
        csv,
        file_name="manager_assessments.csv",
        mime="text/csv",
    )


def _page_compare(teams: List[Team], assessments: Dict[str, ManagerAssessment],
                  home_config: HomeAdvantageConfig = None,
                  power_ratings: Dict[str, float] = None):
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

    rows = compare_teams(team_a, team_b, assessment_a, assessment_b, home_config, power_ratings)
    df = comparison_dataframe(rows, name_a, name_b)
    st.dataframe(df, width="stretch", hide_index=True)


def _page_match_predictor(
    teams: List[Team],
    team_ratings: Dict[str, float],
):
    """Match Predictor page: head-to-head probability predictions."""
    st.header("Match Predictor")
    st.caption("Poisson goal model predictions based on team overall ratings")

    team_names = [t.name for t in teams]
    strengths = compute_team_strengths(team_ratings)

    col_a, col_b = st.columns(2)
    with col_a:
        default_a = 0
        if "predictor_team_a" in st.session_state and st.session_state["predictor_team_a"] in team_names:
            default_a = team_names.index(st.session_state["predictor_team_a"])
        name_a = st.selectbox("Team A", team_names, index=default_a, key="match_team_a")
    with col_b:
        default_b = min(1, len(team_names) - 1)
        if "predictor_team_b" in st.session_state and st.session_state["predictor_team_b"] in team_names:
            default_b = team_names.index(st.session_state["predictor_team_b"])
        name_b = st.selectbox("Team B", team_names, index=default_b, key="match_team_b")

    if name_a == name_b:
        st.warning("Please select two different teams.")
        return

    pred = predict_match(name_a, name_b, strengths)

    # ── Win/Draw/Loss bar ──
    st.subheader("Match Outcome Probabilities")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{name_a} Win", _fmt_pct(pred.win_a))
    c2.metric("Draw", _fmt_pct(pred.draw))
    c3.metric(f"{name_b} Win", _fmt_pct(pred.win_b))

    # Visual probability bar
    bar_data = pd.DataFrame({
        "Outcome": [f"{name_a} Win", "Draw", f"{name_b} Win"],
        "Probability": [pred.win_a * 100, pred.draw * 100, pred.win_b * 100],
    })
    st.bar_chart(bar_data, x="Outcome", y="Probability", horizontal=True)

    # ── Expected Goals ──
    st.subheader("Expected Goals")
    xg1, xg2 = st.columns(2)
    xg1.metric(name_a, f"{pred.xg_a:.2f} xG")
    xg2.metric(name_b, f"{pred.xg_b:.2f} xG")

    # ── Knockout advancement ──
    p_adv_a, p_adv_b = predict_knockout_match(name_a, name_b, strengths)
    st.subheader("Knockout Match (with penalties)")
    ko1, ko2 = st.columns(2)
    ko1.metric(f"{name_a} Advances", _fmt_pct(p_adv_a))
    ko2.metric(f"{name_b} Advances", _fmt_pct(p_adv_b))

    # ── Derived Markets ──
    st.subheader("Betting Markets (Fair Odds)")

    markets_rows = []
    for label, vals in pred.over_under.items():
        markets_rows.append({
            "Market": f"Over {label}",
            "Probability": _fmt_pct(vals["over"]),
            "Fair Odds": f"{1.0 / max(vals['over'], 0.001):.2f}",
        })
        markets_rows.append({
            "Market": f"Under {label}",
            "Probability": _fmt_pct(vals["under"]),
            "Fair Odds": f"{1.0 / max(vals['under'], 0.001):.2f}",
        })

    markets_rows.append({
        "Market": "BTTS Yes",
        "Probability": _fmt_pct(pred.btts["yes"]),
        "Fair Odds": f"{1.0 / max(pred.btts['yes'], 0.001):.2f}",
    })
    markets_rows.append({
        "Market": "BTTS No",
        "Probability": _fmt_pct(pred.btts["no"]),
        "Fair Odds": f"{1.0 / max(pred.btts['no'], 0.001):.2f}",
    })

    st.dataframe(pd.DataFrame(markets_rows), width="stretch", hide_index=True)

    # ── Top Scorelines ──
    st.subheader("Most Likely Scorelines")

    score_rows = []
    for goals_a, goals_b, prob in pred.top_scorelines[:8]:
        score_rows.append({
            "Scoreline": f"{goals_a}-{goals_b}",
            "Probability": _fmt_pct(prob),
            "Fair Odds": f"{1.0 / max(prob, 0.001):.2f}",
        })
    st.dataframe(pd.DataFrame(score_rows), width="stretch", hide_index=True)

    # ── Scoreline Heatmap ──
    try:
        import plotly.graph_objects as go

        st.subheader("Scoreline Probability Heatmap")

        max_display = 6  # Show 0-5 goals
        matrix = pred.scoreline_matrix[:max_display, :max_display]
        labels = [str(i) for i in range(max_display)]

        fig = go.Figure(data=go.Heatmap(
            z=matrix * 100,
            x=[f"{name_b} {g}" for g in labels],
            y=[f"{name_a} {g}" for g in labels],
            colorscale="YlOrRd",
            text=[[f"{matrix[i][j]*100:.1f}%" for j in range(max_display)] for i in range(max_display)],
            texttemplate="%{text}",
            hovertemplate=f"{name_a} %{{y}} - {name_b} %{{x}}<br>Probability: %{{z:.1f}}%<extra></extra>",
        ))
        fig.update_layout(
            height=400,
            xaxis_title=f"{name_b} Goals",
            yaxis_title=f"{name_a} Goals",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, width="stretch")
    except ImportError:
        st.info("Install plotly for scoreline heatmap visualisation.")


def _page_tournament_sim(
    teams: List[Team],
    team_ratings: Dict[str, float],
):
    """Tournament Simulator page: Monte Carlo simulation of the full tournament."""
    st.header("Tournament Simulator")
    st.caption("Monte Carlo simulation of the 2026 FIFA World Cup")

    # Load group draw
    draw_path = BASE_DIR / "data" / "group_draw.json"
    if not draw_path.exists():
        st.error("Group draw file not found. Please ensure data/group_draw.json exists.")
        return

    draw = load_group_draw(str(draw_path))

    # Display groups
    st.subheader("Group Draw")
    group_cols = st.columns(4)
    for idx, (group_name, group_teams) in enumerate(sorted(draw.groups.items())):
        col = group_cols[idx % 4]
        with col:
            team_list = "\n".join(f"- {t}" for t in group_teams)
            st.markdown(f"**Group {group_name}**\n{team_list}")

    st.divider()

    # Simulation controls
    st.subheader("Simulation Settings")
    sim_options = {
        "Quick (10K)": 10000,
        "Standard (100K)": 100000,
        "Thorough (500K)": 500000,
    }
    sim_choice = st.select_slider(
        "Simulation count",
        options=list(sim_options.keys()),
        value="Standard (100K)",
    )
    n_sims = sim_options[sim_choice]

    # Check which teams in the draw have ratings
    draw_teams = draw.all_teams()
    missing_ratings = [t for t in draw_teams if t not in team_ratings]
    if missing_ratings:
        st.warning(
            f"{len(missing_ratings)} teams in the draw don't have ratings "
            f"(e.g. TBD playoff slots). They'll use default strength."
        )

    # Run simulation
    if st.button("Run Simulation", type="primary"):
        strengths = compute_team_strengths(team_ratings)

        # Add default strengths for teams not in our ratings (TBD slots)
        for team in draw_teams:
            if team not in strengths:
                strengths[team] = TeamStrength(name=team, rating=35.0, strength=0.5)

        progress_bar = st.progress(0, text="Starting simulation...")

        def _progress_cb(frac: float, text: str):
            progress_bar.progress(frac, text=text)

        result = run_tournament_simulation(
            draw, strengths, n_simulations=n_sims,
            progress_callback=_progress_cb,
        )
        progress_bar.empty()

        st.session_state["sim_result"] = result
        st.success(f"Completed {n_sims:,} simulations!")

    # Display results
    if "sim_result" in st.session_state:
        result: SimulationResult = st.session_state["sim_result"]

        st.subheader("Tournament Winner Probabilities")

        sorted_teams = result.get_sorted_by_win_prob()
        display_stages = ["Winner", "Runner-up", "SF", "QF", "R16", "R32", "Group Stage"]

        rows = []
        for team, probs in sorted_teams:
            # Skip TBD teams
            if team.startswith("TBD"):
                continue
            row = {"Team": team}
            for stage in display_stages:
                row[stage] = _fmt_pct(probs.get(stage, 0))
            # Add group qualification rate
            qual_rate = result.group_qual_counts.get(team, 0) / result.n_simulations
            row["Qualify"] = _fmt_pct(qual_rate)
            rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", hide_index=True)

        # Top 10 bar chart
        st.subheader("Top 10 Teams by Win Probability")
        top_10 = sorted_teams[:10]
        chart_data = pd.DataFrame({
            "Team": [t for t, _ in top_10 if not t.startswith("TBD")],
            "Win %": [p.get("Winner", 0) * 100 for t, p in top_10 if not t.startswith("TBD")],
        })
        st.bar_chart(chart_data, x="Team", y="Win %")

        # Group breakdown
        with st.expander("Group-by-Group Breakdown"):
            for g_name in sorted(draw.groups.keys()):
                st.markdown(f"**Group {g_name}**")
                g_rows = []
                for team in draw.groups[g_name]:
                    qual_rate = result.group_qual_counts.get(team, 0) / result.n_simulations
                    win_rate = result.get_probabilities().get(team, {}).get("Winner", 0)
                    g_rows.append({
                        "Team": team,
                        "Qualify": _fmt_pct(qual_rate),
                        "Win Tournament": _fmt_pct(win_rate),
                    })
                st.dataframe(pd.DataFrame(g_rows), width="stretch", hide_index=True)

        # CSV download
        csv_rows = []
        for team, probs in sorted_teams:
            if team.startswith("TBD"):
                continue
            row = {"Team": team}
            for stage in display_stages:
                row[stage] = round(probs.get(stage, 0), 4)
            csv_rows.append(row)
        csv_df = pd.DataFrame(csv_rows)
        csv_data = csv_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download Results CSV",
            csv_data,
            file_name="tournament_sim_results.csv",
            mime="text/csv",
        )


def _page_value_finder(
    teams: List[Team],
    team_ratings: Dict[str, float],
):
    """Value Finder page: compare model probabilities to bookmaker odds."""
    st.header("Value Finder")
    st.caption(
        "Compare our model's win probabilities to bookmaker odds to find value bets. "
        "Positive edge = our model thinks the team is more likely to win than the odds imply."
    )

    # First, we need simulation results for outright winner probabilities
    if "sim_result" not in st.session_state:
        st.info(
            "Run a tournament simulation first (Tournament Sim page) to generate "
            "outright winner probabilities, or the tool will use ratings-based estimates."
        )
        # Generate quick estimates from ratings
        total_rating = sum(team_ratings.values())
        model_probs = {t: r / total_rating for t, r in team_ratings.items()} if total_rating > 0 else {}
    else:
        result: SimulationResult = st.session_state["sim_result"]
        probs = result.get_probabilities()
        model_probs = {t: p.get("Winner", 0) for t, p in probs.items() if not t.startswith("TBD")}

    # Show our model probabilities
    with st.expander("Our Model's Win Probabilities"):
        model_rows = sorted(model_probs.items(), key=lambda x: x[1], reverse=True)
        prob_df = pd.DataFrame([
            {"Team": t, "Win Probability": _fmt_pct(p), "Implied Fair Odds": f"{1.0/max(p, 0.001):.1f}"}
            for t, p in model_rows if p > 0
        ])
        st.dataframe(prob_df, width="stretch", hide_index=True)

    st.divider()

    # ── Odds Input ──
    st.subheader("Input Bookmaker Odds")

    input_method = st.radio(
        "Input method",
        ["Upload CSV", "Manual Entry"],
        horizontal=True,
    )

    market_odds: List[MarketOdds] = []

    if input_method == "Upload CSV":
        st.caption("Upload a CSV with columns: team, decimal_odds, market_type (optional), bookmaker (optional)")

        # Template download
        team_names = [t.name for t in teams if not t.name.startswith("TBD")]
        template = generate_odds_template(team_names)
        st.download_button(
            "Download CSV Template",
            template.encode("utf-8"),
            file_name="odds_template.csv",
            mime="text/csv",
        )

        uploaded = st.file_uploader("Upload odds CSV", type=["csv"])
        if uploaded:
            csv_content = uploaded.read().decode("utf-8")
            market_odds = parse_odds_csv(csv_content)
            st.success(f"Loaded {len(market_odds)} odds entries")

    else:  # Manual Entry
        st.caption("Enter decimal odds for teams (leave blank to skip)")

        team_names = sorted([t.name for t in teams if not t.name.startswith("TBD")])

        # Use data_editor for manual input
        manual_data = pd.DataFrame({
            "Team": team_names,
            "Decimal Odds": [None] * len(team_names),
            "Bookmaker": [""] * len(team_names),
        })

        edited = st.data_editor(
            manual_data,
            width="stretch",
            hide_index=True,
            num_rows="fixed",
        )

        for _, row in edited.iterrows():
            odds_val = row.get("Decimal Odds")
            if odds_val is not None and odds_val and float(odds_val) > 1.0:
                market_odds.append(MarketOdds(
                    team=row["Team"],
                    decimal_odds=float(odds_val),
                    bookmaker=row.get("Bookmaker", ""),
                ))

    if not market_odds:
        return

    st.divider()

    # ── Controls ──
    st.subheader("Analysis Settings")
    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        min_edge = st.slider("Minimum edge (%)", min_value=-10.0, max_value=20.0, value=0.0, step=0.5)
    with ctrl2:
        kelly_options = {"Quarter Kelly (Conservative)": 0.25, "Half Kelly": 0.50, "Full Kelly (Aggressive)": 1.0}
        kelly_label = st.selectbox("Kelly fraction", list(kelly_options.keys()))
        kelly_frac = kelly_options[kelly_label]

    # ── Find value bets ──
    value_bets = find_value_bets(model_probs, market_odds, min_edge=min_edge, kelly_fraction=kelly_frac)

    # ── Summary stats ──
    all_odds = [mo.decimal_odds for mo in market_odds]
    overround = calculate_overround(all_odds)
    positive_edge = [v for v in value_bets if v.edge_pct > 0]

    s1, s2, s3 = st.columns(3)
    s1.metric("Market Overround", f"{overround:.1f}%")
    s2.metric("Value Bets Found", len(positive_edge))
    if positive_edge:
        best = positive_edge[0]
        s3.metric("Best Value", f"{best.team} ({best.edge_pct:+.1f}%)")
    else:
        s3.metric("Best Value", "None found")

    # ── Value bets table ──
    st.subheader("Value Analysis")

    if value_bets:
        vb_rows = []
        for vb in value_bets:
            vb_rows.append({
                "Team": vb.team,
                "Our Prob": _fmt_pct(vb.model_prob),
                "Market Odds": f"{vb.decimal_odds:.2f}",
                "Implied Prob": _fmt_pct(vb.implied_prob),
                "Edge": f"{vb.edge_pct:+.1f}%",
                "Value Ratio": f"{vb.value_ratio:.2f}x",
                "Kelly Stake": f"{vb.kelly_stake_pct:.1f}%" if vb.kelly_stake_pct > 0 else "-",
                "Bookmaker": vb.bookmaker or "-",
            })

        vb_df = pd.DataFrame(vb_rows)

        # Color-code by edge
        def _highlight_edge(row):
            edge_str = row["Edge"]
            edge_val = float(edge_str.rstrip("%").replace("+", ""))
            if edge_val > 5:
                return ["background-color: #c6efce"] * len(row)
            elif edge_val > 0:
                return ["background-color: #e2efda"] * len(row)
            elif edge_val > -5:
                return ["background-color: #fff2cc"] * len(row)
            else:
                return ["background-color: #fce4ec"] * len(row)

        styled = vb_df.style.apply(_highlight_edge, axis=1)
        st.dataframe(styled, width="stretch", hide_index=True)

        # CSV download
        csv_data = vb_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download Value Analysis CSV",
            csv_data,
            file_name="value_analysis.csv",
            mime="text/csv",
        )
    else:
        st.info("No bets match the current filter criteria.")


# ── PL Simulator page ─────────────────────────────────────────────────────────


def _page_pl_simulator():
    """Premier League Season Simulator page."""
    st.header("Premier League Season Simulator")
    st.caption(
        "Monte Carlo simulation of a full 38-game PL season. "
        "Input team ratings (0-100) and simulate thousands of seasons "
        "to get league position probabilities and match-by-match predictions."
    )

    # ── Section 1: Team Ratings Input ─────────────────────────────────────
    st.subheader("1. Team Ratings")

    input_method = st.radio(
        "Input method",
        ["Manual Entry", "Upload CSV"],
        horizontal=True,
        key="pl_input_method",
    )

    team_ratings: Optional[Dict[str, float]] = None

    if input_method == "Upload CSV":
        template_csv = generate_pl_ratings_template()
        st.download_button(
            "Download CSV Template",
            template_csv.encode("utf-8"),
            file_name="pl_ratings_template.csv",
            mime="text/csv",
        )

        uploaded = st.file_uploader("Upload team ratings CSV", type=["csv"], key="pl_csv")
        if uploaded:
            try:
                csv_content = uploaded.read().decode("utf-8")
                team_ratings = parse_pl_ratings_csv(csv_content)
                st.success(f"Loaded {len(team_ratings)} teams from CSV")
            except ValueError as e:
                st.error(f"CSV error: {e}")
            except Exception as e:
                st.error(f"Failed to parse CSV: {e}")
    else:
        # Manual entry with pre-populated defaults
        if "pl_manual_ratings" not in st.session_state:
            st.session_state["pl_manual_ratings"] = pd.DataFrame({
                "Team": DEFAULT_PL_TEAMS,
                "Rating (0-100)": [float(DEFAULT_PL_RATINGS[t]) for t in DEFAULT_PL_TEAMS],
            })

        edited_df = st.data_editor(
            st.session_state["pl_manual_ratings"],
            width="stretch",
            hide_index=True,
            num_rows="fixed",
            column_config={
                "Team": st.column_config.TextColumn("Team", width="medium"),
                "Rating (0-100)": st.column_config.NumberColumn(
                    "Rating (0-100)", min_value=0, max_value=100, step=1, format="%d",
                ),
            },
            key="pl_ratings_editor",
        )

        ratings_dict = {}
        valid = True
        for _, row in edited_df.iterrows():
            name = str(row["Team"]).strip()
            rating = row["Rating (0-100)"]
            if not name or pd.isna(rating):
                valid = False
                break
            ratings_dict[name] = float(rating)

        if valid and len(ratings_dict) >= 2:
            team_ratings = ratings_dict
        else:
            st.warning("Please fill in all team names and ratings.")

    if team_ratings is None:
        return

    st.divider()

    # ── Section 2: Simulation Settings ────────────────────────────────────
    st.subheader("2. Simulation Settings")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        sim_options = {
            "Quick (10K)": 10_000,
            "Standard (100K)": 100_000,
            "Thorough (500K)": 500_000,
        }
        sim_choice = st.select_slider(
            "Simulation count",
            options=list(sim_options.keys()),
            value="Standard (100K)",
            key="pl_sim_count",
        )
        n_sims = sim_options[sim_choice]

    with col_s2:
        home_boost = st.slider(
            "Home xG boost",
            min_value=0.0,
            max_value=0.50,
            value=PL_HOME_XG_BOOST,
            step=0.05,
            help="Expected goals boost for home team (0.25 is historical PL average)",
            key="pl_home_boost",
        )

    with col_s3:
        avg_goals = st.slider(
            "Avg goals per team",
            min_value=0.8,
            max_value=2.0,
            value=PL_AVG_GOALS_PER_TEAM,
            step=0.1,
            help="Base expected goals per team per match (PL average ~1.4)",
            key="pl_avg_goals",
        )

    # ── Section 3: Run Simulation ─────────────────────────────────────────
    if st.button("Run PL Simulation", type="primary", key="pl_run_btn"):
        progress_bar = st.progress(0, text="Starting PL simulation...")

        def _progress_cb(frac: float, text: str):
            progress_bar.progress(frac, text=text)

        result = run_pl_simulation(
            team_ratings=team_ratings,
            n_simulations=n_sims,
            home_xg_boost=home_boost,
            avg_goals=avg_goals,
            progress_callback=_progress_cb,
        )
        progress_bar.empty()

        match_preds = get_pl_match_predictions(team_ratings, home_boost, avg_goals)

        st.session_state["pl_sim_result"] = result
        st.session_state["pl_match_preds"] = match_preds
        st.success(f"Completed {n_sims:,} season simulations!")

    # ── Section 4: Results Display ────────────────────────────────────────
    if "pl_sim_result" not in st.session_state:
        return

    result: PLSimulationResult = st.session_state["pl_sim_result"]

    st.divider()
    st.subheader("3. Results")

    tab1, tab2, tab3 = st.tabs([
        "League Table", "Full Position Probabilities", "Match Predictions",
    ])

    # ── Tab 1: League Table Summary ───────────────────────────────────────
    with tab1:
        summary_df = result.summary_table()
        st.dataframe(
            summary_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Win League %": st.column_config.ProgressColumn(
                    "Win League %", format="%.1f%%", min_value=0, max_value=100,
                ),
                "Top 4 %": st.column_config.ProgressColumn(
                    "Top 4 %", format="%.1f%%", min_value=0, max_value=100,
                ),
                "Top 6 %": st.column_config.ProgressColumn(
                    "Top 6 %", format="%.1f%%", min_value=0, max_value=100,
                ),
                "Bottom 3 %": st.column_config.ProgressColumn(
                    "Bottom 3 %", format="%.1f%%", min_value=0, max_value=100,
                ),
            },
        )

        # Title race bar chart
        st.subheader("Title Race")
        top_contenders = summary_df.nlargest(8, "Win League %")
        st.bar_chart(
            top_contenders.set_index("Team")[["Win League %"]],
        )

        csv_data = summary_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download League Table CSV", csv_data,
            file_name="pl_league_table.csv", mime="text/csv", key="dl_pl_table",
        )

    # ── Tab 2: Full Position Probabilities ────────────────────────────────
    with tab2:
        pos_df = result.position_probabilities()

        try:
            import plotly.graph_objects as go

            z_data = pos_df.values * 100
            fig = go.Figure(data=go.Heatmap(
                z=z_data,
                x=[str(i + 1) for i in range(len(pos_df.columns))],
                y=pos_df.index.tolist(),
                colorscale="YlOrRd",
                text=[[f"{v:.1f}%" for v in row] for row in z_data],
                texttemplate="%{text}",
                hovertemplate=(
                    "Team: %{y}<br>Position: %{x}<br>"
                    "Probability: %{z:.1f}%<extra></extra>"
                ),
                colorbar=dict(title="Prob %"),
            ))
            fig.update_layout(
                height=max(600, len(pos_df) * 32),
                xaxis_title="League Position",
                yaxis_title="Team",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, width="stretch")
        except ImportError:
            st.info("Install plotly for the position probability heatmap.")

        # Raw table
        display_df = pos_df.map(lambda x: f"{x * 100:.1f}%")
        display_df = display_df.reset_index().rename(columns={"index": "Team"})
        st.dataframe(display_df, width="stretch", hide_index=True)

        csv_pos = pos_df.reset_index().rename(columns={"index": "Team"})
        csv_data = csv_pos.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download Position Probabilities CSV", csv_data,
            file_name="pl_position_probabilities.csv", mime="text/csv",
            key="dl_pl_positions",
        )

    # ── Tab 3: Match Predictions ──────────────────────────────────────────
    with tab3:
        if "pl_match_preds" in st.session_state:
            preds = st.session_state["pl_match_preds"]

            all_teams = sorted(set(p.home_team for p in preds))
            filter_team = st.selectbox(
                "Filter by team (or show all)",
                ["All Teams"] + all_teams,
                key="pl_match_filter",
            )

            if filter_team != "All Teams":
                filtered = [
                    p for p in preds
                    if p.home_team == filter_team or p.away_team == filter_team
                ]
            else:
                filtered = preds

            def _prob_with_odds(prob: float) -> str:
                """Format as 'XX.X% (Y.YY)' with fair decimal odds in brackets."""
                pct = round(prob * 100, 1)
                odds = round(1.0 / prob, 2) if prob > 0.001 else 0
                return f"{pct}% ({odds:.2f})"

            match_rows = []
            for p in filtered:
                match_rows.append({
                    "Home": p.home_team,
                    "Away": p.away_team,
                    "Home xG": p.home_xg,
                    "Away xG": p.away_xg,
                    "Home Win": _prob_with_odds(p.home_win),
                    "Draw": _prob_with_odds(p.draw),
                    "Away Win": _prob_with_odds(p.away_win),
                    "Over 2.5": _prob_with_odds(p.over_2_5),
                    "BTTS": _prob_with_odds(p.btts_yes),
                })

            match_df = pd.DataFrame(match_rows)
            st.dataframe(match_df, width="stretch", hide_index=True)
            st.caption(f"Showing {len(filtered)} of {len(preds)} fixtures")

            csv_data = match_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download Match Predictions CSV", csv_data,
                file_name="pl_match_predictions.csv", mime="text/csv",
                key="dl_pl_matches",
            )
        else:
            st.info("Run a simulation to see match predictions.")


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    # sidebar nav
    st.sidebar.title("WC 2026")
    pages = ["Teams", "Squad Detail", "Managers", "Compare",
             "Match Predictor", "Tournament Sim", "Value Finder", "PL Simulator"]
    page = st.sidebar.radio(
        "Navigate",
        pages,
        index=pages.index(st.session_state.get("page", "Teams")),
        key="nav_radio",
    )
    st.session_state["page"] = page

    # sidebar options
    st.sidebar.divider()
    st.sidebar.subheader("Options")
    apply_home = st.sidebar.checkbox("Apply Home Advantage", value=True)
    home_config = HomeAdvantageConfig(enabled=apply_home)

    load_games = st.sidebar.checkbox("Load Player Games (slow)", value=False)

    # Power rankings: always load from file, optionally refresh live
    _playwright_available = False
    try:
        import playwright.sync_api
        _playwright_available = True
    except ImportError:
        pass

    if _playwright_available:
        refresh_power = st.sidebar.checkbox("Refresh Power Rankings (live)", value=False)
    else:
        refresh_power = False

    if st.sidebar.button("Force Refresh All Data"):
        st.session_state["force_refresh"] = True
        for key in ["teams", "assessments", "player_games_loaded", "power_ratings", "all_team_ratings", "sim_result"]:
            st.session_state.pop(key, None)
        st.rerun()

    force_refresh = st.session_state.pop("force_refresh", False)

    # load data (cached after first run)
    cache, tm_scraper, wiki_scraper = _get_scrapers()
    teams = _load_all_teams(tm_scraper, wiki_scraper)
    assessments = _compute_assessments(teams)

    # optional: player recent games
    if load_games:
        _load_player_games(teams, tm_scraper, force_refresh=force_refresh)

    # Power rankings: always load from file, optionally refresh live
    power_ratings = _load_power_rankings(cache, try_live=refresh_power)

    # Show persistent power ranking error if any
    if "power_error" in st.session_state:
        st.sidebar.error(f"Power Rankings: {st.session_state['power_error']}")

    # Show warning if squad data is missing
    if st.session_state.get("squad_data_missing"):
        st.warning(
            "Squad data unavailable (Transfermarkt returned 403). "
            "Team ratings are based on power rankings and manager data. "
            "Match Predictor, Tournament Sim, and Value Finder still work fully."
        )

    # Compute all team ratings (shared across new pages)
    team_ratings = _compute_all_team_ratings(teams, assessments, home_config, power_ratings)

    # render selected page
    if page == "Teams":
        _page_teams(teams, assessments, home_config, power_ratings)
    elif page == "Squad Detail":
        _page_squad_detail(teams, assessments)
    elif page == "Managers":
        _page_managers(teams, assessments)
    elif page == "Compare":
        _page_compare(teams, assessments, home_config, power_ratings)
    elif page == "Match Predictor":
        _page_match_predictor(teams, team_ratings)
    elif page == "Tournament Sim":
        _page_tournament_sim(teams, team_ratings)
    elif page == "Value Finder":
        _page_value_finder(teams, team_ratings)
    elif page == "PL Simulator":
        _page_pl_simulator()


if __name__ == "__main__":
    main()
