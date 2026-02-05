"""Home advantage calculation for WC 2026 host nations (USA, Canada, Mexico).

Historical single-host advantage is ~8%. With 3 co-hosts, each gets ~55%
of full advantage → 4.4% base boost. Stronger hosts benefit more.
"""

from dataclasses import dataclass
from typing import Dict, Optional


HOST_NATIONS = {"United States", "Canada", "Mexico"}


@dataclass
class HomeAdvantageConfig:
    base_advantage: float = 0.044  # 4.4% base boost for co-hosts
    enabled: bool = True


def calculate_home_advantage_multiplier(
    team_name: str,
    base_rating: float,
    config: Optional[HomeAdvantageConfig] = None,
) -> float:
    """Return the home advantage multiplier for a team.

    Non-host teams always return 1.0. Host teams get a boost scaled by
    their base rating (stronger teams benefit more from home crowd).
    """
    if config is None or not config.enabled:
        return 1.0

    if team_name not in HOST_NATIONS:
        return 1.0

    # Strength scaling: stronger hosts benefit more
    if base_rating >= 70:
        scale = 1.0
    elif base_rating >= 55:
        # Linear from 0.7 at 55 to 1.0 at 70
        scale = 0.7 + (base_rating - 55) * (0.3 / 15)
    elif base_rating >= 40:
        # Linear from 0.4 at 40 to 0.7 at 55
        scale = 0.4 + (base_rating - 40) * (0.3 / 15)
    else:
        scale = 0.3

    return 1.0 + config.base_advantage * scale


def home_advantage_info(
    team_name: str,
    base_rating: float,
    config: Optional[HomeAdvantageConfig] = None,
) -> Dict:
    """Return a dict with home advantage details for display."""
    if config is None or not config.enabled or team_name not in HOST_NATIONS:
        return {
            "is_host": team_name in HOST_NATIONS,
            "boost_pct": 0.0,
            "boost_label": "N/A",
            "multiplier": 1.0,
        }

    multiplier = calculate_home_advantage_multiplier(team_name, base_rating, config)
    boost_pct = round((multiplier - 1.0) * 100, 1)

    return {
        "is_host": True,
        "boost_pct": boost_pct,
        "boost_label": f"+{boost_pct}%",
        "multiplier": multiplier,
    }
