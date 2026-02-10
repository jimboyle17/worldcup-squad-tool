"""Premier League season simulator using Monte Carlo + Poisson goal model.

Simulates a full 38-game PL season N times, producing per-team
probabilities of finishing in each league position (1st-20th).

Reuses the Poisson match engine from match_probability.py but adds
PL-specific home advantage and league-format aggregation.
"""

import csv
import io
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .match_probability import (
    MAX_GOALS,
    TeamStrength,
    build_scoreline_matrix,
    compute_team_strengths,
    compute_xg_for_match,
)

# ── constants ────────────────────────────────────────────────────────────────

PL_AVG_GOALS_PER_TEAM = 1.4   # PL averages ~2.8 goals/match total
PL_HOME_XG_BOOST = 0.25       # Additive xG boost for home team
BATCH_SIZE = 1000              # Seasons processed per batch

DEFAULT_PL_TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich Town",
    "Leicester City", "Liverpool", "Manchester City", "Manchester Utd",
    "Newcastle Utd", "Nottm Forest", "Southampton", "Tottenham",
    "West Ham", "Wolves",
]

DEFAULT_PL_RATINGS: Dict[str, float] = {
    "Arsenal": 85, "Aston Villa": 70, "Bournemouth": 58, "Brentford": 60,
    "Brighton": 65, "Chelsea": 72, "Crystal Palace": 57, "Everton": 50,
    "Fulham": 60, "Ipswich Town": 42, "Leicester City": 48, "Liverpool": 88,
    "Manchester City": 90, "Manchester Utd": 72, "Newcastle Utd": 72,
    "Nottm Forest": 62, "Southampton": 40, "Tottenham": 73,
    "West Ham": 60, "Wolves": 52,
}


# ── data structures ──────────────────────────────────────────────────────────

@dataclass
class PLMatchPrediction:
    """Analytical prediction for a single PL fixture."""
    home_team: str
    away_team: str
    home_xg: float
    away_xg: float
    home_win: float
    draw: float
    away_win: float
    over_2_5: float
    btts_yes: float


@dataclass
class PLSimulationResult:
    """Aggregated results across all PL season simulations."""
    n_simulations: int
    team_names: List[str]
    # shape (n_teams, n_teams): position_counts[i][j] = times team i finished position j+1
    position_counts: np.ndarray
    total_points: np.ndarray   # shape (n_teams,) accumulated across sims
    total_gd: np.ndarray       # shape (n_teams,)
    total_gf: np.ndarray       # shape (n_teams,)

    def position_probabilities(self) -> pd.DataFrame:
        """Team x position probability matrix."""
        probs = self.position_counts / self.n_simulations
        cols = [str(i + 1) for i in range(len(self.team_names))]
        df = pd.DataFrame(probs, index=self.team_names, columns=cols)
        df = df.sort_values("1", ascending=False)
        return df

    def summary_table(self) -> pd.DataFrame:
        """Summary with key threshold probabilities."""
        probs = self.position_counts / self.n_simulations
        rows = []
        for i, name in enumerate(self.team_names):
            win_pct = probs[i, 0]
            top4_pct = float(probs[i, :4].sum())
            top6_pct = float(probs[i, :6].sum())
            bottom3_pct = float(probs[i, -3:].sum())
            most_likely = int(np.argmax(probs[i]) + 1)
            avg_pts = self.total_points[i] / self.n_simulations
            avg_gd = self.total_gd[i] / self.n_simulations

            rows.append({
                "Team": name,
                "Avg Pts": round(float(avg_pts), 1),
                "Avg GD": round(float(avg_gd), 1),
                "Win League %": round(win_pct * 100, 1),
                "Top 4 %": round(top4_pct * 100, 1),
                "Top 6 %": round(top6_pct * 100, 1),
                "Bottom 3 %": round(bottom3_pct * 100, 1),
                "Most Likely Pos": most_likely,
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("Avg Pts", ascending=False).reset_index(drop=True)
        return df


# ── fixture generation ────────────────────────────────────────────────────────

def generate_pl_fixtures(
    team_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate all home/away fixtures as index arrays.

    Returns (home_indices, away_indices), each shape (n*(n-1),).
    Every team plays every other team once at home and once away.
    """
    n = len(team_names)
    home_idx = []
    away_idx = []
    for i in range(n):
        for j in range(n):
            if i != j:
                home_idx.append(i)
                away_idx.append(j)
    return np.array(home_idx, dtype=np.int32), np.array(away_idx, dtype=np.int32)


# ── xG computation ────────────────────────────────────────────────────────────

def compute_pl_xg_arrays(
    strengths: Dict[str, TeamStrength],
    team_names: List[str],
    home_indices: np.ndarray,
    away_indices: np.ndarray,
    avg_goals: float = PL_AVG_GOALS_PER_TEAM,
    home_xg_boost: float = PL_HOME_XG_BOOST,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute xG arrays for all fixtures with home advantage.

    Returns (xg_home, xg_away) arrays, each shape (n_fixtures,).
    """
    strength_values = np.array(
        [strengths[team_names[i]].strength for i in range(len(team_names))]
    )

    home_str = strength_values[home_indices]
    away_str = strength_values[away_indices]

    ratio = home_str / np.maximum(away_str, 0.01)
    xg_home = avg_goals * np.sqrt(ratio) + home_xg_boost
    xg_away = avg_goals * np.sqrt(1.0 / ratio)

    xg_home = np.clip(xg_home, 0.2, 4.0)
    xg_away = np.clip(xg_away, 0.2, 4.0)

    return xg_home, xg_away


# ── simulation engine ─────────────────────────────────────────────────────────

def run_pl_simulation(
    team_ratings: Dict[str, float],
    n_simulations: int = 100_000,
    home_xg_boost: float = PL_HOME_XG_BOOST,
    avg_goals: float = PL_AVG_GOALS_PER_TEAM,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    seed: Optional[int] = None,
) -> PLSimulationResult:
    """Run Monte Carlo PL season simulation.

    Each simulation plays all 380 matches, computes standings,
    and records each team's finishing position.
    """
    rng = np.random.default_rng(seed)
    team_names = sorted(team_ratings.keys())
    n_teams = len(team_names)

    # Pre-compute
    strengths = compute_team_strengths(team_ratings)
    home_indices, away_indices = generate_pl_fixtures(team_names)
    n_fixtures = len(home_indices)
    xg_home, xg_away = compute_pl_xg_arrays(
        strengths, team_names, home_indices, away_indices, avg_goals, home_xg_boost,
    )

    # Accumulators
    position_counts = np.zeros((n_teams, n_teams), dtype=np.int64)
    total_points = np.zeros(n_teams, dtype=np.int64)
    total_gd = np.zeros(n_teams, dtype=np.int64)
    total_gf = np.zeros(n_teams, dtype=np.int64)

    n_done = 0
    while n_done < n_simulations:
        batch = min(BATCH_SIZE, n_simulations - n_done)

        # Sample goals: shape (batch, n_fixtures)
        goals_home = rng.poisson(np.tile(xg_home, (batch, 1)))
        goals_away = rng.poisson(np.tile(xg_away, (batch, 1)))

        for sim in range(batch):
            gh = goals_home[sim]
            ga = goals_away[sim]

            pts = np.zeros(n_teams, dtype=np.int32)
            gd = np.zeros(n_teams, dtype=np.int32)
            gf = np.zeros(n_teams, dtype=np.int32)

            # Goals for
            np.add.at(gf, home_indices, gh)
            np.add.at(gf, away_indices, ga)

            # Goal difference
            diff = gh - ga
            np.add.at(gd, home_indices, diff)
            np.add.at(gd, away_indices, -diff)

            # Points
            home_wins = gh > ga
            away_wins = ga > gh
            draws = gh == ga

            np.add.at(pts, home_indices[home_wins], 3)
            np.add.at(pts, home_indices[draws], 1)
            np.add.at(pts, away_indices[away_wins], 3)
            np.add.at(pts, away_indices[draws], 1)

            # Sort: points desc, GD desc, GF desc, random tiebreak
            tiebreak = rng.random(n_teams) * 0.001
            sort_key = pts * 1_000_000 + gd * 1_000 + gf + tiebreak
            ranking = np.argsort(-sort_key)

            for pos, team_idx in enumerate(ranking):
                position_counts[team_idx, pos] += 1

            total_points += pts
            total_gd += gd
            total_gf += gf

        n_done += batch
        if progress_callback:
            progress_callback(
                n_done / n_simulations,
                f"Simulated {n_done:,} / {n_simulations:,} seasons",
            )

    return PLSimulationResult(
        n_simulations=n_simulations,
        team_names=team_names,
        position_counts=position_counts,
        total_points=total_points,
        total_gd=total_gd,
        total_gf=total_gf,
    )


# ── analytical match predictions ──────────────────────────────────────────────

def get_pl_match_predictions(
    team_ratings: Dict[str, float],
    home_xg_boost: float = PL_HOME_XG_BOOST,
    avg_goals: float = PL_AVG_GOALS_PER_TEAM,
) -> List[PLMatchPrediction]:
    """Analytical W/D/L predictions for all 380 PL fixtures."""
    strengths = compute_team_strengths(team_ratings)
    team_names = sorted(team_ratings.keys())
    n = MAX_GOALS + 1
    predictions = []

    for home_team in team_names:
        for away_team in team_names:
            if home_team == away_team:
                continue

            s_h = strengths[home_team].strength
            s_a = strengths[away_team].strength
            base_xg_h, base_xg_a = compute_xg_for_match(s_h, s_a, avg_goals)

            xg_h = min(4.0, base_xg_h + home_xg_boost)
            xg_a = base_xg_a

            matrix = build_scoreline_matrix(xg_h, xg_a)

            win_h = sum(matrix[i][j] for i in range(n) for j in range(n) if i > j)
            draw = sum(matrix[i][i] for i in range(n))
            win_a = sum(matrix[i][j] for i in range(n) for j in range(n) if i < j)
            total = win_h + draw + win_a
            if total > 0:
                win_h /= total
                draw /= total
                win_a /= total

            over_2_5 = sum(
                matrix[i][j] for i in range(n) for j in range(n) if (i + j) > 2
            )
            btts_yes = sum(
                matrix[i][j] for i in range(1, n) for j in range(1, n)
            )

            predictions.append(PLMatchPrediction(
                home_team=home_team,
                away_team=away_team,
                home_xg=round(xg_h, 2),
                away_xg=round(xg_a, 2),
                home_win=round(win_h, 4),
                draw=round(draw, 4),
                away_win=round(win_a, 4),
                over_2_5=round(over_2_5, 4),
                btts_yes=round(btts_yes, 4),
            ))

    return predictions


# ── CSV helpers ───────────────────────────────────────────────────────────────

def generate_pl_ratings_template(
    team_names: Optional[List[str]] = None,
    default_ratings: Optional[Dict[str, float]] = None,
) -> str:
    """Generate CSV template with team names and default ratings."""
    if team_names is None:
        team_names = DEFAULT_PL_TEAMS
    if default_ratings is None:
        default_ratings = DEFAULT_PL_RATINGS

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["team", "rating"])
    for name in team_names:
        writer.writerow([name, default_ratings.get(name, 50)])
    return output.getvalue()


def parse_pl_ratings_csv(csv_content: str) -> Dict[str, float]:
    """Parse uploaded CSV into team ratings dict.

    Expected columns: team, rating
    Validates: ratings in 0-100 range.
    """
    reader = csv.DictReader(io.StringIO(csv_content))
    ratings: Dict[str, float] = {}

    for row in reader:
        team = row.get("team", "").strip()
        rating_str = row.get("rating", "").strip()
        if not team or not rating_str:
            continue
        try:
            rating = float(rating_str)
        except ValueError:
            raise ValueError(f"Invalid rating for {team}: {rating_str}")
        if not (0 <= rating <= 100):
            raise ValueError(f"Rating for {team} must be 0-100, got {rating}")
        ratings[team] = rating

    if len(ratings) < 2:
        raise ValueError(f"Need at least 2 teams, got {len(ratings)}")

    return ratings
