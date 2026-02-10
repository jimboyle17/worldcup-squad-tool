"""Monte Carlo tournament simulator for the 2026 FIFA World Cup.

Simulates the entire tournament N times using the Poisson goal model,
producing per-team probabilities of reaching each stage.

Format: 48 teams, 12 groups of 4, top 2 + 8 best 3rd = 32 advance,
then R32 → R16 → QF → SF → Final.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .match_probability import (
    TeamStrength,
    compute_xg_for_match,
    simulate_matches_batch,
    PENALTY_FAVOURITE_RATE,
)

# ── constants ────────────────────────────────────────────────────────────────

STAGES = ["Group Stage", "R32", "R16", "QF", "SF", "Runner-up", "Winner"]
BATCH_SIZE = 1000


# ── data structures ──────────────────────────────────────────────────────────

@dataclass
class GroupDraw:
    """The 12 groups of 4 teams."""
    groups: Dict[str, List[str]]  # {"A": ["Team1", ...], ...}

    def all_teams(self) -> List[str]:
        teams = []
        for g in sorted(self.groups.keys()):
            teams.extend(self.groups[g])
        return teams

    def validate(self) -> bool:
        if len(self.groups) != 12:
            return False
        for g, teams in self.groups.items():
            if len(teams) != 4:
                return False
        return True


@dataclass
class SimulationResult:
    """Aggregated results across all simulations."""
    n_simulations: int
    # team -> stage -> count
    stage_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # team -> group -> qualification count
    group_qual_counts: Dict[str, int] = field(default_factory=dict)

    def get_probabilities(self) -> Dict[str, Dict[str, float]]:
        """Convert counts to probabilities."""
        result = {}
        for team, stages in self.stage_counts.items():
            result[team] = {
                stage: count / self.n_simulations
                for stage, count in stages.items()
            }
        return result

    def get_sorted_by_win_prob(self) -> List[Tuple[str, Dict[str, float]]]:
        """Return teams sorted by win probability (descending)."""
        probs = self.get_probabilities()
        items = list(probs.items())
        items.sort(key=lambda x: x[1].get("Winner", 0), reverse=True)
        return items


def load_group_draw(path: str) -> GroupDraw:
    """Load group draw from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return GroupDraw(groups=data["groups"])


# ── simulation engine ────────────────────────────────────────────────────────

def _simulate_group_stage(
    draw: GroupDraw,
    strengths: Dict[str, TeamStrength],
    rng: np.random.Generator,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Simulate all group matches and return standings.

    Returns:
        (group_standings, third_placed)
        group_standings: {"A": [1st, 2nd, 3rd, 4th], ...}
        third_placed: list of (team, points, gd, gf) for all 3rd-placed teams
    """
    group_standings = {}
    all_third = []

    for group_name in sorted(draw.groups.keys()):
        teams = draw.groups[group_name]

        # Each team plays 3 matches: 0v1, 0v2, 0v3, 1v2, 1v3, 2v3
        matches = [
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3), (2, 3),
        ]

        # Track points, goal difference, goals for
        points = {t: 0 for t in teams}
        gd = {t: 0 for t in teams}
        gf = {t: 0 for t in teams}

        # Compute xG for all 6 matches
        xg_a_list = []
        xg_b_list = []
        match_teams = []

        for i, j in matches:
            t_a, t_b = teams[i], teams[j]
            s_a = strengths.get(t_a)
            s_b = strengths.get(t_b)
            str_a = s_a.strength if s_a else 1.0
            str_b = s_b.strength if s_b else 1.0
            xga, xgb = compute_xg_for_match(str_a, str_b)
            xg_a_list.append(xga)
            xg_b_list.append(xgb)
            match_teams.append((t_a, t_b))

        # Simulate all 6 matches at once
        goals_a, goals_b = simulate_matches_batch(
            np.array(xg_a_list), np.array(xg_b_list), rng
        )

        for idx, (t_a, t_b) in enumerate(match_teams):
            ga, gb = int(goals_a[idx]), int(goals_b[idx])
            gf[t_a] += ga
            gf[t_b] += gb
            gd[t_a] += ga - gb
            gd[t_b] += gb - ga

            if ga > gb:
                points[t_a] += 3
            elif ga == gb:
                points[t_a] += 1
                points[t_b] += 1
            else:
                points[t_b] += 3

        # Sort: points desc, then GD desc, then GF desc, then random
        standing = sorted(
            teams,
            key=lambda t: (points[t], gd[t], gf[t], rng.random()),
            reverse=True,
        )
        group_standings[group_name] = standing

        # Track third place for best-3rd comparison
        third = standing[2]
        all_third.append((third, points[third], gd[third], gf[third]))

    return group_standings, all_third


def _select_best_thirds(
    all_third: List[Tuple[str, int, int, int]],
    rng: np.random.Generator,
) -> List[str]:
    """Select the 8 best third-placed teams from 12 groups.

    Sorted by points, then GD, then GF, then random tiebreak.
    """
    sorted_thirds = sorted(
        all_third,
        key=lambda x: (x[1], x[2], x[3], rng.random()),
        reverse=True,
    )
    return [t[0] for t in sorted_thirds[:8]]


def _simulate_knockout_match(
    team_a: str,
    team_b: str,
    strengths: Dict[str, TeamStrength],
    rng: np.random.Generator,
) -> str:
    """Simulate a single knockout match. Returns the winner."""
    s_a = strengths.get(team_a)
    s_b = strengths.get(team_b)
    str_a = s_a.strength if s_a else 1.0
    str_b = s_b.strength if s_b else 1.0

    xga, xgb = compute_xg_for_match(str_a, str_b)
    goals_a = rng.poisson(xga)
    goals_b = rng.poisson(xgb)

    if goals_a > goals_b:
        return team_a
    elif goals_b > goals_a:
        return team_b
    else:
        # Penalty shootout
        if str_a >= str_b:
            pen_fav_rate = PENALTY_FAVOURITE_RATE
            return team_a if rng.random() < pen_fav_rate else team_b
        else:
            pen_fav_rate = PENALTY_FAVOURITE_RATE
            return team_b if rng.random() < pen_fav_rate else team_a


def _simulate_knockout_bracket(
    r32_teams: List[str],
    strengths: Dict[str, TeamStrength],
    rng: np.random.Generator,
    stage_tracker: Dict[str, str],
) -> str:
    """Simulate the full knockout bracket from R32 to Final.

    Args:
        r32_teams: 32 teams entering the knockout (paired for R32)
        strengths: team strength data
        rng: random generator
        stage_tracker: dict to record each team's furthest stage

    Returns:
        Tournament winner
    """
    current_round = r32_teams.copy()

    rounds = ["R32", "R16", "QF", "SF", "Final"]

    for round_idx, round_name in enumerate(rounds):
        next_round = []
        for i in range(0, len(current_round), 2):
            if i + 1 >= len(current_round):
                next_round.append(current_round[i])
                continue

            team_a = current_round[i]
            team_b = current_round[i + 1]
            winner = _simulate_knockout_match(team_a, team_b, strengths, rng)
            loser = team_b if winner == team_a else team_a

            # Track stage reached
            if round_name == "Final":
                stage_tracker[winner] = "Winner"
                stage_tracker[loser] = "Runner-up"
            elif round_name == "SF":
                # Losers of SF reach SF stage
                stage_tracker[loser] = "SF"
            else:
                stage_tracker[loser] = round_name

            next_round.append(winner)

        current_round = next_round

    return current_round[0] if current_round else ""


def run_tournament_simulation(
    draw: GroupDraw,
    strengths: Dict[str, TeamStrength],
    n_simulations: int = 100000,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    seed: Optional[int] = None,
) -> SimulationResult:
    """Run the full Monte Carlo tournament simulation.

    Args:
        draw: group stage draw
        strengths: team strength factors
        n_simulations: number of tournament simulations
        progress_callback: optional callback(progress_fraction, status_text)
        seed: optional random seed for reproducibility

    Returns:
        SimulationResult with per-team stage counts
    """
    rng = np.random.default_rng(seed)
    all_teams = draw.all_teams()

    # Initialise result
    result = SimulationResult(n_simulations=n_simulations)
    for team in all_teams:
        result.stage_counts[team] = {stage: 0 for stage in STAGES}
        result.group_qual_counts[team] = 0

    for sim in range(n_simulations):
        if progress_callback and sim % max(1, n_simulations // 100) == 0:
            progress_callback(
                sim / n_simulations,
                f"Simulation {sim:,}/{n_simulations:,}",
            )

        # 1. Group stage
        group_standings, all_third = _simulate_group_stage(draw, strengths, rng)

        # 2. Determine qualifiers: top 2 per group + 8 best thirds
        qualifiers = []
        group_winners = {}
        group_runners = {}

        for g_name in sorted(group_standings.keys()):
            standing = group_standings[g_name]
            group_winners[g_name] = standing[0]
            group_runners[g_name] = standing[1]
            qualifiers.extend(standing[:2])

            # Record group stage exit for 4th place
            result.stage_counts[standing[3]]["Group Stage"] += 1

        best_thirds = _select_best_thirds(all_third, rng)
        qualifiers.extend(best_thirds)

        # Track qualification
        for team in qualifiers:
            result.group_qual_counts[team] = result.group_qual_counts.get(team, 0) + 1

        # Record group stage exit for non-qualifying 3rd-placed teams
        for team_info in all_third:
            if team_info[0] not in best_thirds:
                result.stage_counts[team_info[0]]["Group Stage"] += 1

        # 3. Build R32 bracket
        # Simplified bracket: group winners vs runners from other groups
        # Pair: 1A v best3rd, 2A v 1_other, etc.
        # For simplicity, use a standard bracket pairing
        groups_sorted = sorted(group_standings.keys())
        r32_teams = []

        # Pair group winners with runners-up from other groups
        # Standard FIFA bracket for 48 teams:
        # 1A vs 2C, 1C vs 2A, 1B vs 2D, 1D vs 2B, ...
        pairings_winners_runners = [
            ("A", "C"), ("C", "A"),
            ("B", "D"), ("D", "B"),
            ("E", "G"), ("G", "E"),
            ("F", "H"), ("H", "F"),
            ("I", "K"), ("K", "I"),
            ("J", "L"), ("L", "J"),
        ]

        for w_group, r_group in pairings_winners_runners:
            r32_teams.append(group_winners.get(w_group, ""))
            r32_teams.append(group_runners.get(r_group, ""))

        # Add best 3rd-placed teams (shuffled into remaining bracket slots)
        rng.shuffle(best_thirds)
        for i in range(0, len(best_thirds), 2):
            if i + 1 < len(best_thirds):
                r32_teams.append(best_thirds[i])
                r32_teams.append(best_thirds[i + 1])
            else:
                r32_teams.append(best_thirds[i])

        # 4. Simulate knockout bracket
        stage_tracker: Dict[str, str] = {}
        _simulate_knockout_bracket(r32_teams, strengths, rng, stage_tracker)

        # 5. Record results
        for team, stage in stage_tracker.items():
            if team in result.stage_counts:
                result.stage_counts[team][stage] += 1

        # Teams that were in R32 but not tracked (shouldn't happen, but safety)
        for team in qualifiers:
            if team not in stage_tracker:
                result.stage_counts[team]["R32"] += 1

    if progress_callback:
        progress_callback(1.0, "Complete!")

    return result
