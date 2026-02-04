"""Manager scoring engine.

Produces a composite 0-100 score from five sub-scores, then maps it
to a team-rating multiplier between 0.85 (-15%) and 1.15 (+15%).
"""

import math
from dataclasses import dataclass, field
from typing import Dict

from ..models.manager import Manager


# Points per honour level (with diminishing returns for duplicates)
HONOUR_POINTS = {
    "world_cup": 25,
    "continental": 15,
    "champions_league": 18,
    "league": 10,
    "cup": 5,
}

# Career level weights for experience scoring
LEVEL_WEIGHT = {
    "international_senior": 3.0,
    "club_top": 2.5,
    "club_mid": 1.5,
    "club_lower": 1.0,
    "youth": 0.8,
}

# Tournament result scores for achievement delta
RESULT_SCORES = {
    "Winner": 10,
    "Runner-up": 7,
    "Third place": 6,
    "Semi-finals": 5,
    "Quarter-finals": 3,
    "Round of 16": 1,
    "Group Stage": -2,
}


@dataclass
class ManagerAssessment:
    """Computes five sub-scores and a composite for a national-team manager."""

    manager: Manager
    current_fifa_ranking: int

    # Cached sub-scores (computed lazily)
    _experience_score: float = field(default=None, init=False, repr=False)
    _honours_score: float = field(default=None, init=False, repr=False)
    _club_achievement_score: float = field(default=None, init=False, repr=False)
    _tenure_score: float = field(default=None, init=False, repr=False)
    _achievement_delta_score: float = field(default=None, init=False, repr=False)

    # --- Sub-score weights ---
    WEIGHTS: Dict[str, float] = field(
        default_factory=lambda: {
            "experience": 0.20,
            "honours": 0.25,
            "club_achievement": 0.15,
            "tenure": 0.10,
            "achievement_delta": 0.30,
        },
        init=False,
        repr=False,
    )

    # ------------------------------------------------------------------
    # 1. Experience score (0-100)
    # ------------------------------------------------------------------
    @property
    def experience_score(self) -> float:
        if self._experience_score is not None:
            return self._experience_score

        years = self.manager.total_years_managing
        roles = len(self.manager.career_history)

        # Years component: 0-50 points, saturates around 20 years
        years_pts = min(50, years * 2.5)

        # Roles component: 0-25 points, saturates around 8 roles
        roles_pts = min(25, roles * 3.125)

        # Caliber component: 0-25 points based on weighted level of positions
        caliber_pts = 0.0
        for entry in self.manager.career_history:
            weight = LEVEL_WEIGHT.get(entry.level, 1.0)
            duration = max(1, (entry.end_year or 2026) - entry.start_year)
            caliber_pts += weight * min(duration, 5)
        caliber_pts = min(25, caliber_pts * 1.5)

        self._experience_score = min(100, years_pts + roles_pts + caliber_pts)
        return self._experience_score

    # ------------------------------------------------------------------
    # 2. Honours score (0-100)
    # ------------------------------------------------------------------
    @property
    def honours_score(self) -> float:
        if self._honours_score is not None:
            return self._honours_score

        if not self.manager.honours:
            self._honours_score = 0.0
            return 0.0

        # Count honours per level for diminishing returns
        level_counts: Dict[str, int] = {}
        raw_pts = 0.0
        for h in self.manager.honours:
            level_counts[h.level] = level_counts.get(h.level, 0) + 1
            count = level_counts[h.level]
            base = HONOUR_POINTS.get(h.level, 5)
            # Diminishing returns: each duplicate worth 60% of previous
            raw_pts += base * (0.6 ** (count - 1))

        # Map raw points to 0-100 (raw ~60+ is world-class)
        self._honours_score = min(100, raw_pts * (100 / 60))
        return self._honours_score

    # ------------------------------------------------------------------
    # 3. Club Achievement score (0-100)
    # ------------------------------------------------------------------
    @property
    def club_achievement_score(self) -> float:
        if self._club_achievement_score is not None:
            return self._club_achievement_score

        if not self.manager.career_history:
            self._club_achievement_score = 0.0
            return 0.0

        # Highest level managed
        level_ranks = {
            "club_top": 40,
            "international_senior": 35,
            "club_mid": 20,
            "club_lower": 10,
            "youth": 5,
        }
        highest = max(
            (level_ranks.get(e.level, 0) for e in self.manager.career_history),
            default=0,
        )

        # Breadth: unique teams at club level
        unique_clubs = self.manager.clubs_managed_count
        breadth_pts = min(30, unique_clubs * 6)

        # International experience bonus
        intl_roles = sum(
            1 for e in self.manager.career_history if e.level == "international_senior"
        )
        intl_pts = min(30, intl_roles * 15)

        self._club_achievement_score = min(100, highest + breadth_pts + intl_pts)
        return self._club_achievement_score

    # ------------------------------------------------------------------
    # 4. Tenure score (0-100) — bell curve, sweet spot 2-6 years
    # ------------------------------------------------------------------
    @property
    def tenure_score(self) -> float:
        if self._tenure_score is not None:
            return self._tenure_score

        years = self.manager.tenure_years

        # Bell curve centered at 4 years, sigma ~2.5
        # Peak=100 at 4 years, ~60 at 1 year, ~60 at 8 years
        self._tenure_score = 100 * math.exp(-((years - 4) ** 2) / (2 * 2.5 ** 2))
        return self._tenure_score

    # ------------------------------------------------------------------
    # 5. Achievement Delta score (-50 to +50)
    # ------------------------------------------------------------------
    @property
    def achievement_delta_score(self) -> float:
        if self._achievement_delta_score is not None:
            return self._achievement_delta_score

        # Ranking improvement: positive = improved
        ranking_at_start = self.manager.fifa_ranking_at_appointment
        ranking_now = self.current_fifa_ranking
        ranking_delta = ranking_at_start - ranking_now  # positive = improved

        # Scale: each rank improvement ~1.5 pts, capped at +/- 30
        ranking_pts = max(-30, min(30, ranking_delta * 1.5))

        # Tournament results vs expectations
        tournament_pts = 0.0
        for result in self.manager.recent_tournament_results:
            score = RESULT_SCORES.get(result.result, 0)
            # Recent tournaments weighted more
            recency = max(0.5, 1.0 - (2026 - result.year) * 0.15)
            tournament_pts += score * recency

        tournament_pts = max(-20, min(20, tournament_pts))

        self._achievement_delta_score = max(-50, min(50, ranking_pts + tournament_pts))
        return self._achievement_delta_score

    # ------------------------------------------------------------------
    # Composite score and multiplier
    # ------------------------------------------------------------------
    @property
    def composite_score(self) -> float:
        """Weighted composite score (0-100).

        Achievement delta is -50..+50, so we remap it to 0..100 before
        weighting, then combine with the other 0-100 sub-scores.
        """
        # Remap achievement delta from [-50,+50] to [0,100]
        delta_normalized = (self.achievement_delta_score + 50)

        raw = (
            self.experience_score * self.WEIGHTS["experience"]
            + self.honours_score * self.WEIGHTS["honours"]
            + self.club_achievement_score * self.WEIGHTS["club_achievement"]
            + self.tenure_score * self.WEIGHTS["tenure"]
            + delta_normalized * self.WEIGHTS["achievement_delta"]
        )
        return round(max(0, min(100, raw)), 1)

    @property
    def rating_multiplier(self) -> float:
        """Map composite score to 0.85 – 1.15 multiplier.

        Score 0  → 0.85  (-15%)
        Score 50 → 1.00  (neutral)
        Score 100→ 1.15  (+15%)
        """
        return round(0.85 + (self.composite_score / 100) * 0.30, 3)

    @property
    def rating_impact_pct(self) -> str:
        """Human-readable impact string like '+8.2%' or '-3.1%'."""
        pct = (self.rating_multiplier - 1.0) * 100
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.1f}%"

    def to_dict(self) -> Dict:
        """Full breakdown dict for display / CSV export."""
        return {
            "Team": self.manager.team_name,
            "Manager": self.manager.name,
            "Nationality": self.manager.nationality,
            "Age": self.manager.age,
            "Tenure (years)": self.manager.tenure_years,
            "Honours": len(self.manager.honours),
            "Experience": round(self.experience_score, 1),
            "Honours Score": round(self.honours_score, 1),
            "Club Achievement": round(self.club_achievement_score, 1),
            "Tenure Score": round(self.tenure_score, 1),
            "Achievement Delta": round(self.achievement_delta_score, 1),
            "Composite Score": self.composite_score,
            "Rating Impact": self.rating_impact_pct,
            "Multiplier": self.rating_multiplier,
        }
