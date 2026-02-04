"""Overall team rating combining squad metrics with manager multiplier.

Base rating (0-100) = FIFA ranking (40%) + market value (35%)
                     + average caps (15%) + squad balance (10%).

Overall rating = base rating * manager multiplier (0.85 – 1.15).
"""

from typing import List, Optional

from ..models.squad import Squad
from ..models.team import Team
from .manager_assessment import ManagerAssessment


def _ranking_score(fifa_ranking: int, total_teams: int = 48) -> float:
    """Convert FIFA ranking to 0-100 score (rank 1 = 100, last = ~10)."""
    if fifa_ranking <= 0:
        return 50.0
    # Linear scale: rank 1 → 100, rank 48 → 10
    return max(0, 100 - (fifa_ranking - 1) * (90 / max(1, total_teams - 1)))


def _value_score(team_value: float, all_values: List[float]) -> float:
    """Convert market value to 0-100 score relative to tournament field."""
    if not all_values:
        return 50.0
    max_val = max(all_values)
    if max_val <= 0:
        return 50.0
    # Logarithmic scale to compress the huge range
    import math
    log_val = math.log1p(team_value)
    log_max = math.log1p(max_val)
    if log_max <= 0:
        return 50.0
    return round((log_val / log_max) * 100, 1)


def _caps_score(avg_caps: float) -> float:
    """Convert average caps to 0-100 score. 40+ caps = ~100."""
    return min(100, avg_caps * 2.5)


def _balance_score(squad: Squad) -> float:
    """Score squad balance (0-100) based on position distribution.

    Ideal: ~3 GK, ~8 DF, ~8 MF, ~7 FW out of 26 players.
    Penalizes heavy imbalance.
    """
    breakdown = squad.position_breakdown()
    size = squad.size
    if size == 0:
        return 50.0

    gk = breakdown.get("GK", 0)
    df = breakdown.get("DF", 0)
    mf = breakdown.get("MF", 0)
    fw = breakdown.get("FW", 0)

    # Ideal ratios for a 26-player squad
    ideal = {"GK": 3, "DF": 8, "MF": 8, "FW": 7}
    scale = size / 26.0 if size > 0 else 1.0

    penalty = 0.0
    for pos, actual in [("GK", gk), ("DF", df), ("MF", mf), ("FW", fw)]:
        expected = ideal[pos] * scale
        deviation = abs(actual - expected) / max(1, expected)
        penalty += deviation * 25  # 25 points per 100% deviation

    return max(0, min(100, 100 - penalty))


def calculate_base_team_rating(team: Team, all_teams: List[Team]) -> float:
    """Compute the base team rating (0-100) from squad and ranking data.

    Weights: FIFA ranking 40%, market value 35%, avg caps 15%, balance 10%.
    """
    squad = Squad(team.squad)
    all_values = [t.total_market_value for t in all_teams]

    r_score = _ranking_score(team.fifa_ranking, len(all_teams))
    v_score = _value_score(team.total_market_value, all_values)
    c_score = _caps_score(squad.average_caps())
    b_score = _balance_score(squad)

    base = (
        r_score * 0.40
        + v_score * 0.35
        + c_score * 0.15
        + b_score * 0.10
    )
    return round(max(0, min(100, base)), 1)


def calculate_overall_rating(
    team: Team,
    all_teams: List[Team],
    assessment: Optional[ManagerAssessment] = None,
) -> float:
    """Overall rating = base rating * manager multiplier."""
    base = calculate_base_team_rating(team, all_teams)
    if assessment:
        return round(min(100, base * assessment.rating_multiplier), 1)
    return base
