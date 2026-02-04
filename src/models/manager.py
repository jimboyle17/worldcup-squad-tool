from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional


@dataclass
class CareerEntry:
    team: str
    role: str  # "Head Coach", "Assistant Coach", etc.
    start_year: int
    end_year: Optional[int]  # None = current
    level: str  # "international_senior", "club_top", "club_mid", "club_lower", "youth"

    @classmethod
    def from_dict(cls, d: dict) -> "CareerEntry":
        return cls(
            team=d["team"],
            role=d.get("role", "Head Coach"),
            start_year=d["start_year"],
            end_year=d.get("end_year"),
            level=d.get("level", "club_mid"),
        )


@dataclass
class Honour:
    title: str
    year: int
    level: str  # "world_cup", "continental", "champions_league", "league", "cup"
    with_team: str

    @classmethod
    def from_dict(cls, d: dict) -> "Honour":
        return cls(
            title=d["title"],
            year=d["year"],
            level=d.get("level", "cup"),
            with_team=d.get("with_team", ""),
        )


@dataclass
class TournamentResult:
    tournament: str
    result: str  # "Winner", "Runner-up", "Semi-finals", "Quarter-finals", "Group Stage", etc.
    year: int

    @classmethod
    def from_dict(cls, d: dict) -> "TournamentResult":
        return cls(
            tournament=d["tournament"],
            result=d["result"],
            year=d["year"],
        )


@dataclass
class Manager:
    team_name: str
    name: str
    nationality: str
    date_of_birth: str  # ISO format YYYY-MM-DD
    tenure_start: str  # ISO format YYYY-MM-DD
    fifa_ranking_at_appointment: int
    career_history: List[CareerEntry] = field(default_factory=list)
    honours: List[Honour] = field(default_factory=list)
    recent_tournament_results: List[TournamentResult] = field(default_factory=list)

    @property
    def age(self) -> int:
        try:
            dob = date.fromisoformat(self.date_of_birth)
            today = date.today()
            return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        except (ValueError, TypeError):
            return 0

    @property
    def tenure_years(self) -> float:
        try:
            start = date.fromisoformat(self.tenure_start)
            today = date.today()
            return round((today - start).days / 365.25, 1)
        except (ValueError, TypeError):
            return 0.0

    @property
    def total_years_managing(self) -> float:
        if not self.career_history:
            return 0.0
        current_year = date.today().year
        total = 0.0
        for entry in self.career_history:
            end = entry.end_year if entry.end_year else current_year
            total += max(0, end - entry.start_year)
        return total

    @property
    def clubs_managed_count(self) -> int:
        return len(set(
            e.team for e in self.career_history
            if e.role == "Head Coach" and e.level.startswith("club")
        ))

    @classmethod
    def from_dict(cls, d: dict) -> "Manager":
        return cls(
            team_name=d["team_name"],
            name=d["name"],
            nationality=d.get("nationality", ""),
            date_of_birth=d.get("date_of_birth", "1970-01-01"),
            tenure_start=d.get("tenure_start", "2024-01-01"),
            fifa_ranking_at_appointment=d.get("fifa_ranking_at_appointment", 50),
            career_history=[CareerEntry.from_dict(c) for c in d.get("career_history", [])],
            honours=[Honour.from_dict(h) for h in d.get("honours", [])],
            recent_tournament_results=[
                TournamentResult.from_dict(t) for t in d.get("recent_tournament_results", [])
            ],
        )

    def to_summary_dict(self) -> dict:
        return {
            "Team": self.team_name,
            "Manager": self.name,
            "Nationality": self.nationality,
            "Age": self.age,
            "Tenure (years)": self.tenure_years,
            "Honours": len(self.honours),
            "Career Roles": len(self.career_history),
            "Years Managing": self.total_years_managing,
        }
