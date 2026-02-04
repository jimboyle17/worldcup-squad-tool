from typing import TYPE_CHECKING, Dict, List

import pandas as pd

from ..models.player import Player
from ..models.team import Team

if TYPE_CHECKING:
    from .manager_assessment import ManagerAssessment


def players_dataframe(players: List[Player]) -> pd.DataFrame:
    """Convert a list of players to a pandas DataFrame."""
    if not players:
        return pd.DataFrame()
    return pd.DataFrame([p.to_dict() for p in players])


def teams_summary_dataframe(teams: List[Team]) -> pd.DataFrame:
    """Convert a list of teams to a summary DataFrame."""
    if not teams:
        return pd.DataFrame()
    return pd.DataFrame([t.to_summary_dict() for t in teams])


def comparison_dataframe(rows: List[Dict], team_a_name: str, team_b_name: str) -> pd.DataFrame:
    """Convert comparison rows to a DataFrame with proper column names."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.rename(columns={"Value A": team_a_name, "Value B": team_b_name})
    return df


def managers_dataframe(assessments: Dict[str, "ManagerAssessment"]) -> pd.DataFrame:
    """Convert manager assessments to a DataFrame for CSV export."""
    if not assessments:
        return pd.DataFrame()
    return pd.DataFrame([a.to_dict() for a in assessments.values()])


def export_to_csv(df: pd.DataFrame, filepath: str):
    """Export a DataFrame to CSV."""
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
