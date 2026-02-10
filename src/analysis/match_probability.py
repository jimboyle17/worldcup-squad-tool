"""Match probability engine using a Poisson goal model.

Converts team overall ratings (0-100) into match predictions:
win/draw/loss probabilities, expected goals, scoreline matrix,
derived betting markets (over/under, BTTS), and fair decimal odds.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── constants ────────────────────────────────────────────────────────────────

AVG_GOALS_PER_TEAM = 1.25  # World Cup historical average (~2.5 per match)
MAX_GOALS = 8              # Scoreline matrix dimension (0..8)
PENALTY_FAVOURITE_RATE = 0.55  # Historical favourite win rate in shootouts


# ── data structures ──────────────────────────────────────────────────────────

@dataclass
class TeamStrength:
    """Attack/defence strength derived from overall rating."""
    name: str
    rating: float
    strength: float  # multiplicative factor (mean ≈ 1.0)


@dataclass
class MatchPrediction:
    """Full prediction output for a single match."""
    team_a: str
    team_b: str
    xg_a: float
    xg_b: float
    win_a: float
    draw: float
    win_b: float
    scoreline_matrix: np.ndarray  # shape (MAX_GOALS+1, MAX_GOALS+1)
    top_scorelines: List[Tuple[int, int, float]] = field(default_factory=list)
    over_under: Dict[str, Dict[str, float]] = field(default_factory=dict)
    btts: Dict[str, float] = field(default_factory=dict)
    fair_odds: Dict[str, float] = field(default_factory=dict)


# ── core functions ───────────────────────────────────────────────────────────

def rating_to_strength(rating: float, all_ratings: List[float]) -> float:
    """Convert a 0-100 rating to a multiplicative strength factor.

    Uses an exponential mapping so that:
    - The mean strength across all teams ≈ 1.0
    - A top team (~95 rating) gets ~2.0x
    - A bottom team (~25 rating) gets ~0.4x

    The mapping is: strength = exp(k * (rating - mean_rating))
    where k is chosen so the spread is sensible.
    """
    if not all_ratings:
        return 1.0
    mean_r = sum(all_ratings) / len(all_ratings)
    k = 0.035  # calibrated for good spread across 25-95 range
    return math.exp(k * (rating - mean_r))


def compute_team_strengths(
    team_ratings: Dict[str, float],
) -> Dict[str, TeamStrength]:
    """Batch convert all team ratings into strength factors."""
    all_ratings = list(team_ratings.values())
    return {
        name: TeamStrength(
            name=name,
            rating=rating,
            strength=rating_to_strength(rating, all_ratings),
        )
        for name, rating in team_ratings.items()
    }


def _poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function: P(X=k) given rate lambda."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def _build_scoreline_matrix(xg_a: float, xg_b: float) -> np.ndarray:
    """Build a (MAX_GOALS+1) x (MAX_GOALS+1) scoreline probability matrix.

    Assumes independence between team goal counts (standard Poisson model).
    """
    n = MAX_GOALS + 1
    prob_a = np.array([_poisson_pmf(i, xg_a) for i in range(n)])
    prob_b = np.array([_poisson_pmf(j, xg_b) for j in range(n)])
    return np.outer(prob_a, prob_b)


def predict_match(
    team_a: str,
    team_b: str,
    strengths: Dict[str, TeamStrength],
    avg_goals: float = AVG_GOALS_PER_TEAM,
) -> MatchPrediction:
    """Produce a full Poisson model prediction for team_a vs team_b.

    Expected goals: xG_A = avg_goals * strength_A / strength_B
                    xG_B = avg_goals * strength_B / strength_A

    This formulation ensures that two equal teams each get avg_goals,
    while mismatches shift the balance.
    """
    s_a = strengths.get(team_a)
    s_b = strengths.get(team_b)

    if s_a is None or s_b is None:
        # Fallback for unknown teams
        str_a = 1.0
        str_b = 1.0
    else:
        str_a = s_a.strength
        str_b = s_b.strength

    # Compute expected goals
    ratio = str_a / max(str_b, 0.01)
    xg_a = avg_goals * math.sqrt(ratio)
    xg_b = avg_goals * math.sqrt(1.0 / ratio)

    # Clamp to sensible range
    xg_a = max(0.2, min(4.0, xg_a))
    xg_b = max(0.2, min(4.0, xg_b))

    # Build scoreline matrix
    matrix = _build_scoreline_matrix(xg_a, xg_b)

    # Win/Draw/Loss from matrix
    n = MAX_GOALS + 1
    win_a = sum(matrix[i][j] for i in range(n) for j in range(n) if i > j)
    draw = sum(matrix[i][i] for i in range(n))
    win_b = sum(matrix[i][j] for i in range(n) for j in range(n) if i < j)

    # Normalise to handle floating point
    total = win_a + draw + win_b
    if total > 0:
        win_a /= total
        draw /= total
        win_b /= total

    # Top scorelines
    scorelines = []
    for i in range(n):
        for j in range(n):
            scorelines.append((i, j, float(matrix[i][j])))
    scorelines.sort(key=lambda x: x[2], reverse=True)
    top_scorelines = scorelines[:10]

    # Over/Under markets
    over_under = {}
    for threshold in [1.5, 2.5, 3.5]:
        over = sum(
            matrix[i][j]
            for i in range(n)
            for j in range(n)
            if (i + j) > threshold
        )
        under = 1.0 - over
        label = f"{threshold:.1f}"
        over_under[label] = {"over": float(over), "under": float(under)}

    # BTTS (Both Teams to Score)
    btts_yes = sum(
        matrix[i][j] for i in range(1, n) for j in range(1, n)
    )
    btts = {"yes": float(btts_yes), "no": float(1.0 - btts_yes)}

    # Fair decimal odds (no margin)
    def _fair_odds(prob: float) -> float:
        return round(1.0 / max(prob, 0.001), 2)

    fair_odds = {
        f"{team_a} Win": _fair_odds(win_a),
        "Draw": _fair_odds(draw),
        f"{team_b} Win": _fair_odds(win_b),
    }
    for label, vals in over_under.items():
        fair_odds[f"Over {label}"] = _fair_odds(vals["over"])
        fair_odds[f"Under {label}"] = _fair_odds(vals["under"])
    fair_odds["BTTS Yes"] = _fair_odds(btts["yes"])
    fair_odds["BTTS No"] = _fair_odds(btts["no"])

    return MatchPrediction(
        team_a=team_a,
        team_b=team_b,
        xg_a=round(xg_a, 2),
        xg_b=round(xg_b, 2),
        win_a=round(win_a, 4),
        draw=round(draw, 4),
        win_b=round(win_b, 4),
        scoreline_matrix=matrix,
        top_scorelines=top_scorelines,
        over_under=over_under,
        btts=btts,
        fair_odds=fair_odds,
    )


def predict_knockout_match(
    team_a: str,
    team_b: str,
    strengths: Dict[str, TeamStrength],
) -> Tuple[float, float]:
    """Predict knockout match with penalty shootout resolution for draws.

    Returns (p_advance_a, p_advance_b) summing to 1.0.
    """
    pred = predict_match(team_a, team_b, strengths)

    # In a draw, favourite wins shootout ~55% of the time
    s_a = strengths.get(team_a)
    s_b = strengths.get(team_b)
    if s_a and s_b and s_a.strength >= s_b.strength:
        pen_a = PENALTY_FAVOURITE_RATE
    else:
        pen_a = 1.0 - PENALTY_FAVOURITE_RATE

    p_advance_a = pred.win_a + pred.draw * pen_a
    p_advance_b = pred.win_b + pred.draw * (1.0 - pen_a)

    total = p_advance_a + p_advance_b
    return (p_advance_a / total, p_advance_b / total)


# ── batch simulation helpers (for Monte Carlo) ──────────────────────────────

def simulate_matches_batch(
    xg_a_array: np.ndarray,
    xg_b_array: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised batch goal simulation using NumPy Poisson sampling.

    Args:
        xg_a_array: expected goals for team A (shape: N,)
        xg_b_array: expected goals for team B (shape: N,)
        rng: NumPy random generator

    Returns:
        (goals_a, goals_b) arrays of shape (N,)
    """
    goals_a = rng.poisson(xg_a_array)
    goals_b = rng.poisson(xg_b_array)
    return goals_a, goals_b


def compute_xg_for_match(
    strength_a: float,
    strength_b: float,
    avg_goals: float = AVG_GOALS_PER_TEAM,
) -> Tuple[float, float]:
    """Compute expected goals for a single match given strength factors.

    Returns (xg_a, xg_b).
    """
    ratio = strength_a / max(strength_b, 0.01)
    xg_a = avg_goals * math.sqrt(ratio)
    xg_b = avg_goals * math.sqrt(1.0 / ratio)
    return (
        max(0.2, min(4.0, xg_a)),
        max(0.2, min(4.0, xg_b)),
    )


# Public alias for use by pl_simulator
build_scoreline_matrix = _build_scoreline_matrix
