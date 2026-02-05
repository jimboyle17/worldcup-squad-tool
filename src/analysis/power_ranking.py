"""Power ranking model combining betting odds with actual match results.

Pipeline:
1. Convert decimal odds to implied probabilities (remove bookmaker margin)
2. Extract actual match outcomes from scores (W/D/L)
3. Combine odds + results into effective observations per match
4. Fit regularized Bradley-Terry model via iterative maximum likelihood
5. Map fitted parameters to 0-100 power ratings

The model uses Bayesian regularization (prior toward average strength)
to prevent divergence for teams with few connections in the match graph.
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

# How much weight to give actual results vs odds (0=pure odds, 1=pure results)
RESULT_WEIGHT = 0.6


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


def _score_to_result(home_score: Optional[int], away_score: Optional[int]) -> Optional[float]:
    """Convert a match score to a result for the home team.

    Returns 1.0 for home win, 0.0 for away win, 0.5 for draw.
    Returns None if scores are not available.
    """
    if home_score is None or away_score is None:
        return None
    if home_score > away_score:
        return 1.0
    elif home_score < away_score:
        return 0.0
    return 0.5


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
    max_iterations: int = 500,
    tolerance: float = 1e-4,
    prior_strength: float = 2.0,
    damping: float = 0.5,
) -> Dict[str, float]:
    """Fit a regularized Bradley-Terry model to pairwise match data.

    Each match dict should have:
        team_a, team_b: team names
        prob_a: effective win probability for team_a (combining odds + result)
        weight: competition importance weight

    The prior_strength parameter adds pseudo-observations at 50/50 against
    an average opponent, preventing divergence for thinly-connected teams.

    The damping parameter (0-1) blends old and new lambda values each
    iteration to improve convergence stability.

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

    # Pre-compute per-team match lists for efficiency
    team_matches: Dict[str, List[Dict]] = defaultdict(list)
    for m in matches:
        if m["team_a"] in connected and m["team_b"] in connected:
            team_matches[m["team_a"]].append(m)
            team_matches[m["team_b"]].append(m)

    # Iterative maximum likelihood estimation with regularization
    prev_lam = dict(lam)
    for iteration in range(max_iterations):
        new_lam: Dict[str, float] = {}

        for team in connected:
            # Numerator: sum of (effective wins * weight) + prior
            # Denominator: sum of (weight * opponent_lam / (team_lam + opponent_lam)) + prior
            numerator = prior_strength * 0.5  # Prior: 50% wins
            denominator = prior_strength * 0.5  # Prior: against average (lam=1)

            for m in team_matches[team]:
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
                raw = numerator / denominator
                # Damped update: blend old and new to improve convergence
                new_lam[team] = damping * raw + (1 - damping) * lam[team]
            else:
                new_lam[team] = lam[team]

        # Normalize so geometric mean = 1 (prevents parameter drift)
        log_mean = sum(math.log(max(v, 1e-10)) for v in new_lam.values()) / len(new_lam)
        norm_factor = math.exp(log_mean)
        if norm_factor > 0:
            lam = {t: v / norm_factor for t, v in new_lam.items()}
        else:
            lam = new_lam

        # Measure convergence AFTER normalization to avoid residual oscillation
        max_change = max(
            abs(lam[t] - prev_lam[t]) / max(prev_lam[t], 1e-10)
            for t in connected
        )
        prev_lam = dict(lam)

        if max_change < tolerance:
            logger.info(f"Bradley-Terry converged after {iteration + 1} iterations")
            break
    else:
        logger.warning(f"Bradley-Terry did not converge after {max_iterations} iterations "
                       f"(max_change={max_change:.4g})")

    # Add disconnected teams with default
    for t in disconnected:
        lam[t] = 0.1  # Low default

    return lam


def _lambda_to_rating(lam: Dict[str, float]) -> Dict[str, float]:
    """Convert Bradley-Terry lambda parameters to 0-100 ratings.

    Uses log-transform and maps to 10-95 range.
    """
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
    """Full pipeline: odds + results -> probabilities -> Bradley-Terry -> power rankings.

    For each match, the effective observation is a blend of:
    - Actual result (W=1, D=0.5, L=0) when score is available
    - Odds-implied win probability (always available)

    This gives richer signal than either source alone: odds capture pre-match
    market assessment, while results capture what actually happened on the pitch.

    Args:
        match_odds_list: List of MatchOdds dataclass instances
        wc_team_names: If provided, only include these teams in output

    Returns:
        Dict mapping team name to PowerRanking
    """
    # Step 1: Convert odds + results to pairwise match data
    bt_matches: List[Dict] = []
    team_match_counts: Dict[str, int] = defaultdict(int)
    matches_with_results = 0

    for match in match_odds_list:
        if match.home_odds <= 1.0 or match.draw_odds <= 1.0 or match.away_odds <= 1.0:
            continue

        # Odds -> probabilities
        home_prob, draw_prob, away_prob = decimal_odds_to_implied_probs(
            match.home_odds, match.draw_odds, match.away_odds
        )

        # Remove home advantage -> neutral-venue 2-way
        odds_prob_a, odds_prob_b = adjust_for_home_advantage(home_prob, away_prob, draw_prob)

        # Get actual result if scores are available
        home_score = getattr(match, 'home_score', None)
        away_score = getattr(match, 'away_score', None)
        result_a = _score_to_result(home_score, away_score)

        # Combine odds and result into effective probability
        if result_a is not None:
            # Blend actual result with odds-implied probability
            # Result gets RESULT_WEIGHT, odds get (1 - RESULT_WEIGHT)
            effective_prob_a = RESULT_WEIGHT * result_a + (1 - RESULT_WEIGHT) * odds_prob_a
            matches_with_results += 1
        else:
            # No result available — use odds only
            effective_prob_a = odds_prob_a

        # Clamp to avoid extreme values (no team has 0% or 100% true strength)
        effective_prob_a = max(0.05, min(0.95, effective_prob_a))

        weight = COMPETITION_WEIGHTS.get(match.competition, 0.9)

        bt_matches.append({
            "team_a": match.home_team,
            "team_b": match.away_team,
            "prob_a": effective_prob_a,
            "weight": weight,
        })

        team_match_counts[match.home_team] += 1
        team_match_counts[match.away_team] += 1

    if not bt_matches:
        logger.warning("No valid matches for Bradley-Terry model")
        return {}

    logger.info(f"Fitting Bradley-Terry model with {len(bt_matches)} matches "
                f"({matches_with_results} with results), {len(team_match_counts)} teams")

    # Step 2: Fit regularized Bradley-Terry
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
