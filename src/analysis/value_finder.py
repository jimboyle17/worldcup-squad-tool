"""Value finder: compare model probabilities to bookmaker odds.

Identifies value bets where our model probability exceeds the
bookmaker's implied probability, and calculates Kelly criterion
stake sizing.
"""

import csv
import io
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MarketOdds:
    """A single bookmaker price for a team/market."""
    team: str
    decimal_odds: float
    market_type: str = "outright_winner"
    bookmaker: str = ""


@dataclass
class ValueBet:
    """A potential value bet identified by the model."""
    team: str
    model_prob: float
    decimal_odds: float
    implied_prob: float
    edge_pct: float
    value_ratio: float
    kelly_fraction: float
    kelly_stake_pct: float
    market_type: str = "outright_winner"
    bookmaker: str = ""


def calculate_edge(
    model_prob: float,
    decimal_odds: float,
) -> tuple[float, float]:
    """Calculate the edge between model probability and market odds.

    Returns:
        (edge_pct, value_ratio)
        - edge_pct: percentage edge (positive = value, negative = avoid)
        - value_ratio: model_prob / implied_prob (>1.0 = value)
    """
    if decimal_odds <= 1.0:
        return (0.0, 0.0)
    implied_prob = 1.0 / decimal_odds
    edge_pct = (model_prob - implied_prob) * 100
    value_ratio = model_prob / max(implied_prob, 0.001)
    return (round(edge_pct, 2), round(value_ratio, 3))


def kelly_criterion(
    model_prob: float,
    decimal_odds: float,
    fraction: float = 0.25,
) -> float:
    """Calculate Kelly criterion stake as fraction of bankroll.

    Uses fractional Kelly (default quarter-Kelly) for conservative sizing.

    Kelly formula: f* = (bp - q) / b
    where b = decimal_odds - 1, p = model_prob, q = 1 - p

    Args:
        model_prob: our estimated probability of winning
        decimal_odds: bookmaker decimal odds
        fraction: Kelly fraction (0.25 = quarter-Kelly, conservative)

    Returns:
        Recommended stake as fraction of bankroll (0 if no edge).
    """
    if decimal_odds <= 1.0 or model_prob <= 0 or model_prob >= 1.0:
        return 0.0

    b = decimal_odds - 1.0
    q = 1.0 - model_prob
    kelly = (b * model_prob - q) / b

    if kelly <= 0:
        return 0.0

    return round(kelly * fraction, 4)


def remove_overround(odds_list: List[float]) -> List[float]:
    """Strip bookmaker margin from a set of odds to get fair probabilities.

    The overround is the sum of implied probabilities minus 1.0.
    We distribute the margin proportionally.

    Args:
        odds_list: list of decimal odds for a complete market

    Returns:
        List of fair probabilities summing to ~1.0
    """
    if not odds_list or any(o <= 1.0 for o in odds_list):
        return [1.0 / len(odds_list)] * len(odds_list) if odds_list else []

    implied = [1.0 / o for o in odds_list]
    total = sum(implied)
    if total <= 0:
        return [1.0 / len(odds_list)] * len(odds_list)

    return [p / total for p in implied]


def calculate_overround(odds_list: List[float]) -> float:
    """Calculate the bookmaker's overround (margin) from a set of odds.

    Returns:
        Overround as a percentage (e.g., 10.5 means 110.5% total implied).
    """
    if not odds_list:
        return 0.0
    implied_sum = sum(1.0 / o for o in odds_list if o > 0)
    return round((implied_sum - 1.0) * 100, 2)


def find_value_bets(
    model_probs: Dict[str, float],
    market_odds: List[MarketOdds],
    min_edge: float = 0.0,
    kelly_fraction: float = 0.25,
) -> List[ValueBet]:
    """Compare model probabilities to market odds and find value bets.

    Args:
        model_probs: dict of team_name -> model probability
        market_odds: list of MarketOdds from bookmaker
        min_edge: minimum edge percentage to include (default 0 = all)
        kelly_fraction: Kelly fraction for stake sizing

    Returns:
        List of ValueBet sorted by edge (highest first)
    """
    results = []

    for mo in market_odds:
        if mo.decimal_odds <= 1.0:
            continue

        model_p = model_probs.get(mo.team, 0.0)
        if model_p <= 0:
            continue

        implied_p = 1.0 / mo.decimal_odds
        edge_pct, value_ratio = calculate_edge(model_p, mo.decimal_odds)
        kelly_stake = kelly_criterion(model_p, mo.decimal_odds, kelly_fraction)

        if edge_pct >= min_edge:
            results.append(ValueBet(
                team=mo.team,
                model_prob=round(model_p, 4),
                decimal_odds=mo.decimal_odds,
                implied_prob=round(implied_p, 4),
                edge_pct=edge_pct,
                value_ratio=value_ratio,
                kelly_fraction=kelly_fraction,
                kelly_stake_pct=round(kelly_stake * 100, 2),
                market_type=mo.market_type,
                bookmaker=mo.bookmaker,
            ))

    results.sort(key=lambda v: v.edge_pct, reverse=True)
    return results


def parse_odds_csv(csv_content: str) -> List[MarketOdds]:
    """Parse a CSV string into a list of MarketOdds.

    Expected columns: team, decimal_odds, market_type (optional),
                      bookmaker (optional)
    """
    results = []
    reader = csv.DictReader(io.StringIO(csv_content))

    for row in reader:
        team = row.get("team", "").strip()
        odds_str = row.get("decimal_odds", "").strip()

        if not team or not odds_str:
            continue

        try:
            odds = float(odds_str)
        except ValueError:
            continue

        if odds <= 1.0:
            continue

        results.append(MarketOdds(
            team=team,
            decimal_odds=odds,
            market_type=row.get("market_type", "outright_winner").strip() or "outright_winner",
            bookmaker=row.get("bookmaker", "").strip(),
        ))

    return results


def generate_odds_template(team_names: List[str]) -> str:
    """Generate a CSV template string with all team names pre-populated."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["team", "decimal_odds", "market_type", "bookmaker"])
    for name in sorted(team_names):
        writer.writerow([name, "", "outright_winner", ""])
    return output.getvalue()
