"""Bradley-Terry power ranking model from betting odds.

Pipeline:
1. Convert decimal odds to implied probabilities (remove bookmaker margin)
2. Adjust for home advantage to get neutral-venue strengths
3. Fit Bradley-Terry model via iterative maximum likelihood
4. Map fitted parameters to 0-100 power ratings
"""

import math
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# Competition weights — major tournaments count more
COMPETITION_WEIGHTS: Dict[str, float] = {
    "world_cup_2022": 1.3,
    "nations_league": 1.2,
    "copa_america_2024": 1.3,
    "afcon_2024": 1.1,
    "afcon_2023": 1.1,
    "asian_cup": 1.0,
}

HOME_ADVANTAGE_PROB = 0.07  # ~7% home advantage in international football


@dataclass
class PowerRanking:
    team: str
    rating: float  # 0-100
    lambda_param: float  # Raw Bradley-Terry parameter
    matches_used: int


def decimal_odds_to_implied_probs(
    home_odds: float, draw_odds: float, away_odds: float
) -> Tuple[float, float, float]:
    """Convert decimal odds to normalized implied probabilities.

    Removes bookmaker overround (~5-8%) by normalizing.
    """
    raw_home = 1.0 / home_odds
    raw_draw = 1.0 / draw_odds
    raw_away = 1.0 / away_odds
    total = raw_home + raw_draw + raw_away

    return raw_home / total, raw_draw / total, raw_away / total


def adjust_for_home_advantage(
    home_prob: float, away_prob: float, draw_prob: float
) -> Tuple[float, float]:
    """Remove home advantage to get neutral-venue 2-way probabilities.

    Returns (team_a_win_prob, team_b_win_prob) on neutral venue.
    """
    # Subtract home advantage from home team's probability
    neutral_home = max(0.05, home_prob - HOME_ADVANTAGE_PROB)
    neutral_away = away_prob + HOME_ADVANTAGE_PROB * 0.5
    # Distribute remaining draw probability proportionally
    neutral_draw = draw_prob + HOME_ADVANTAGE_PROB * 0.5

    # Convert to 2-way (win/loss) probabilities
    total_decisive = neutral_home + neutral_away
    if total_decisive <= 0:
        return 0.5, 0.5

    return neutral_home / total_decisive, neutral_away / total_decisive


def _find_connected_component(
    teams: Set[str], matchups: Dict[str, Set[str]]
) -> Set[str]:
    """Find the largest connected component via BFS."""
    if not teams:
        return set()

    visited: Set[str] = set()
    components: List[Set[str]] = []

    for start in teams:
        if start in visited:
            continue
        component: Set[str] = set()
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for neighbor in matchups.get(node, set()):
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(component)

    return max(components, key=len) if components else set()


def fit_bradley_terry(
    matches: List[Dict],
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> Dict[str, float]:
    """Fit a Bradley-Terry model to pairwise match data.

    Each match dict should have: team_a, team_b, prob_a (win probability for team_a),
    and weight (competition importance).

    Returns dict mapping team name to lambda parameter.
    """
    # Collect all teams and build adjacency
    teams: Set[str] = set()
    matchups: Dict[str, Set[str]] = defaultdict(set)

    for m in matches:
        teams.add(m["team_a"])
        teams.add(m["team_b"])
        matchups[m["team_a"]].add(m["team_b"])
        matchups[m["team_b"]].add(m["team_a"])

    # Find largest connected component
    connected = _find_connected_component(teams, matchups)
    disconnected = teams - connected

    if disconnected:
        logger.info(f"{len(disconnected)} teams not in main component, will get default rating")

    # Initialize lambda parameters
    lam: Dict[str, float] = {t: 1.0 for t in connected}

    # Iterative maximum likelihood estimation
    for iteration in range(max_iterations):
        new_lam: Dict[str, float] = {}
        max_change = 0.0

        for team in connected:
            numerator = 0.0
            denominator = 0.0

            for m in matches:
                if m["team_a"] != team and m["team_b"] != team:
                    continue
                if m["team_a"] not in connected or m["team_b"] not in connected:
                    continue

                weight = m.get("weight", 1.0)

                if m["team_a"] == team:
                    prob_win = m["prob_a"]
                    opponent = m["team_b"]
                else:
                    prob_win = 1.0 - m["prob_a"]
                    opponent = m["team_a"]

                numerator += prob_win * weight
                denominator += weight * lam[opponent] / (lam[team] + lam[opponent])

            if denominator > 0:
                new_lam[team] = numerator / denominator
            else:
                new_lam[team] = lam[team]

            change = abs(new_lam[team] - lam[team])
            if change > max_change:
                max_change = change

        # Normalize so geometric mean = 1
        log_mean = sum(math.log(max(v, 1e-10)) for v in new_lam.values()) / len(new_lam)
        norm_factor = math.exp(log_mean)
        lam = {t: v / norm_factor for t, v in new_lam.items()}

        if max_change < tolerance:
            logger.info(f"Bradley-Terry converged after {iteration + 1} iterations")
            break

    # Add disconnected teams with default
    for t in disconnected:
        lam[t] = 0.1  # Low default

    return lam


def _lambda_to_rating(lam: Dict[str, float]) -> Dict[str, float]:
    """Convert Bradley-Terry lambda parameters to 0-100 ratings."""
    if not lam:
        return {}

    # Log-transform for better spread
    log_lam = {t: math.log(max(v, 1e-10)) for t, v in lam.items()}

    min_log = min(log_lam.values())
    max_log = max(log_lam.values())
    spread = max_log - min_log

    if spread <= 0:
        return {t: 50.0 for t in lam}

    # Map to 10-95 range
    ratings = {}
    for t, log_v in log_lam.items():
        normalized = (log_v - min_log) / spread
        rating = 10 + normalized * 85  # 10-95 range
        ratings[t] = round(rating, 1)

    return ratings


def compute_power_rankings(
    match_odds_list: List,  # List of MatchOdds from oddsportal scraper
    wc_team_names: Optional[Set[str]] = None,
) -> Dict[str, PowerRanking]:
    """Full pipeline: odds → probabilities → Bradley-Terry → power rankings.

    Args:
        match_odds_list: List of MatchOdds dataclass instances
        wc_team_names: If provided, only include these teams in output

    Returns:
        Dict mapping team name to PowerRanking
    """
    # Step 1: Convert odds to pairwise match data
    bt_matches: List[Dict] = []
    team_match_counts: Dict[str, int] = defaultdict(int)

    for match in match_odds_list:
        if match.home_odds <= 1.0 or match.draw_odds <= 1.0 or match.away_odds <= 1.0:
            continue

        # Odds → probabilities
        home_prob, draw_prob, away_prob = decimal_odds_to_implied_probs(
            match.home_odds, match.draw_odds, match.away_odds
        )

        # Remove home advantage → neutral-venue 2-way
        prob_a, prob_b = adjust_for_home_advantage(home_prob, away_prob, draw_prob)

        weight = COMPETITION_WEIGHTS.get(match.competition, 0.9)

        bt_matches.append({
            "team_a": match.home_team,
            "team_b": match.away_team,
            "prob_a": prob_a,
            "weight": weight,
        })

        team_match_counts[match.home_team] += 1
        team_match_counts[match.away_team] += 1

    if not bt_matches:
        logger.warning("No valid matches for Bradley-Terry model")
        return {}

    logger.info(f"Fitting Bradley-Terry model with {len(bt_matches)} matches, "
                f"{len(team_match_counts)} teams")

    # Step 2: Fit Bradley-Terry
    lam = fit_bradley_terry(bt_matches)

    # Step 3: Convert to ratings
    ratings = _lambda_to_rating(lam)

    # Step 4: Build PowerRanking objects
    result: Dict[str, PowerRanking] = {}
    for team, rating in ratings.items():
        if wc_team_names and team not in wc_team_names:
            continue
        result[team] = PowerRanking(
            team=team,
            rating=rating,
            lambda_param=lam.get(team, 0),
            matches_used=team_match_counts.get(team, 0),
        )

    # Assign default rating (25) to WC teams with no data
    if wc_team_names:
        for team in wc_team_names:
            if team not in result:
                result[team] = PowerRanking(
                    team=team, rating=25.0, lambda_param=0, matches_used=0
                )

    return result
