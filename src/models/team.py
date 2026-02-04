from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from .player import Player

if TYPE_CHECKING:
    from .manager import Manager


@dataclass
class Team:
    name: str
    country_code: str  # ISO 3166-1 alpha-3
    confederation: str  # UEFA, CONMEBOL, etc.
    fifa_ranking: int
    transfermarkt_id: str
    squad: List[Player] = field(default_factory=list)
    manager: Optional["Manager"] = None

    @property
    def total_market_value(self) -> float:
        return sum(p.market_value for p in self.squad)

    @property
    def total_value_display(self) -> str:
        val = self.total_market_value
        if val >= 1_000_000_000:
            return f"€{val / 1_000_000_000:.2f}bn"
        elif val >= 1_000_000:
            return f"€{val / 1_000_000:.2f}m"
        else:
            return f"€{val:.0f}"

    @property
    def average_age(self) -> float:
        if not self.squad:
            return 0.0
        return sum(p.age for p in self.squad) / len(self.squad)

    @property
    def squad_size(self) -> int:
        return len(self.squad)

    def to_summary_dict(self) -> dict:
        return {
            "Team": self.name,
            "Confederation": self.confederation,
            "FIFA Ranking": self.fifa_ranking,
            "Squad Size": self.squad_size,
            "Total Value (€)": self.total_market_value,
            "Total Value": self.total_value_display,
            "Avg Age": round(self.average_age, 1),
        }
