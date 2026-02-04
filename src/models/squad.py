from collections import Counter
from typing import Dict, List, Optional

from .player import Player


class Squad:
    """Provides composition analysis for a list of players."""

    def __init__(self, players: List[Player]):
        self.players = players

    @property
    def size(self) -> int:
        return len(self.players)

    def position_breakdown(self) -> Dict[str, int]:
        counts = Counter(p.position for p in self.players)
        return {pos: counts.get(pos, 0) for pos in ["GK", "DF", "MF", "FW"]}

    def position_detail_breakdown(self) -> Dict[str, int]:
        return dict(Counter(p.position_detail for p in self.players))

    def average_age(self) -> float:
        if not self.players:
            return 0.0
        return sum(p.age for p in self.players) / len(self.players)

    def age_distribution(self) -> Dict[str, int]:
        buckets = {"U21": 0, "21-25": 0, "26-29": 0, "30+": 0}
        for p in self.players:
            if p.age < 21:
                buckets["U21"] += 1
            elif p.age <= 25:
                buckets["21-25"] += 1
            elif p.age <= 29:
                buckets["26-29"] += 1
            else:
                buckets["30+"] += 1
        return buckets

    def youngest(self) -> Optional[Player]:
        return min(self.players, key=lambda p: p.age) if self.players else None

    def oldest(self) -> Optional[Player]:
        return max(self.players, key=lambda p: p.age) if self.players else None

    def total_value(self) -> float:
        return sum(p.market_value for p in self.players)

    def average_value(self) -> float:
        if not self.players:
            return 0.0
        return self.total_value() / len(self.players)

    def value_by_position(self) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for p in self.players:
            values[p.position] = values.get(p.position, 0) + p.market_value
        return values

    def most_valuable(self) -> Optional[Player]:
        return max(self.players, key=lambda p: p.market_value) if self.players else None

    def club_diversity(self) -> Dict[str, int]:
        return dict(Counter(p.club for p in self.players).most_common())

    def unique_clubs_count(self) -> int:
        return len(set(p.club for p in self.players))

    def average_caps(self) -> float:
        if not self.players:
            return 0.0
        return sum(p.appearances for p in self.players) / len(self.players)

    def most_experienced(self) -> Optional[Player]:
        return max(self.players, key=lambda p: p.appearances) if self.players else None

    def least_experienced(self) -> Optional[Player]:
        return min(self.players, key=lambda p: p.appearances) if self.players else None

    def players_by_position(self, position: str) -> List[Player]:
        return [p for p in self.players if p.position == position]
