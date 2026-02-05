from dataclasses import dataclass, field


@dataclass
class Player:
    name: str
    position: str  # GK, DF, MF, FW
    position_detail: str  # CB, LB, CDM, etc.
    age: int
    club: str
    nationality: str
    market_value: float  # In EUR
    appearances: int = 0  # National team caps
    goals: int = 0
    assists: int = 0
    transfermarkt_id: str = ""
    games_last_30: int = 0
    games_last_60: int = 0

    @property
    def market_value_display(self) -> str:
        """Format market value for display (e.g., '€80.00m', '€500.00k')."""
        if self.market_value >= 1_000_000:
            return f"€{self.market_value / 1_000_000:.2f}m"
        elif self.market_value >= 1_000:
            return f"€{self.market_value / 1_000:.0f}k"
        else:
            return f"€{self.market_value:.0f}"

    def to_dict(self) -> dict:
        return {
            "Name": self.name,
            "Position": self.position,
            "Detail": self.position_detail,
            "Age": self.age,
            "Club": self.club,
            "Nationality": self.nationality,
            "Market Value (€)": self.market_value,
            "Market Value": self.market_value_display,
            "Caps": self.appearances,
            "Goals": self.goals,
            "Last 30d": self.games_last_30,
            "Last 60d": self.games_last_60,
        }
