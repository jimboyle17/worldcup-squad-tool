from typing import TYPE_CHECKING, Dict, List, Optional

from ..models.player import Player
from ..models.squad import Squad
from ..models.team import Team

if TYPE_CHECKING:
    from .manager_assessment import ManagerAssessment


def squad_summary(team: Team) -> Dict:
    """Generate a full composition summary for a team."""
    squad = Squad(team.squad)

    youngest = squad.youngest()
    oldest = squad.oldest()
    most_valuable = squad.most_valuable()
    most_exp = squad.most_experienced()

    return {
        "team": team.name,
        "squad_size": squad.size,
        "position_breakdown": squad.position_breakdown(),
        "position_details": squad.position_detail_breakdown(),
        "average_age": round(squad.average_age(), 1),
        "age_distribution": squad.age_distribution(),
        "youngest": f"{youngest.name} ({youngest.age})" if youngest else "N/A",
        "oldest": f"{oldest.name} ({oldest.age})" if oldest else "N/A",
        "total_value": squad.total_value(),
        "average_value": squad.average_value(),
        "value_by_position": squad.value_by_position(),
        "most_valuable": (
            f"{most_valuable.name} ({most_valuable.market_value_display})"
            if most_valuable
            else "N/A"
        ),
        "unique_clubs": squad.unique_clubs_count(),
        "club_breakdown": squad.club_diversity(),
        "average_caps": round(squad.average_caps(), 1),
        "most_experienced": (
            f"{most_exp.name} ({most_exp.appearances} caps)" if most_exp else "N/A"
        ),
    }


def compare_teams(
    team_a: Team,
    team_b: Team,
    assessment_a: Optional["ManagerAssessment"] = None,
    assessment_b: Optional["ManagerAssessment"] = None,
) -> List[Dict]:
    """Compare two teams across multiple dimensions. Returns list of comparison rows."""
    sa = Squad(team_a.squad)
    sb = Squad(team_b.squad)

    def fmt_val(v: float) -> str:
        if v >= 1_000_000_000:
            return f"€{v / 1_000_000_000:.2f}bn"
        elif v >= 1_000_000:
            return f"€{v / 1_000_000:.2f}m"
        elif v >= 1_000:
            return f"€{v / 1_000:.0f}k"
        return f"€{v:.0f}"

    pos_a = sa.position_breakdown()
    pos_b = sb.position_breakdown()

    rows = [
        {"Metric": "FIFA Ranking", "Value A": team_a.fifa_ranking, "Value B": team_b.fifa_ranking},
        {"Metric": "Squad Size", "Value A": sa.size, "Value B": sb.size},
        {"Metric": "Average Age", "Value A": round(sa.average_age(), 1), "Value B": round(sb.average_age(), 1)},
        {"Metric": "Total Value", "Value A": fmt_val(sa.total_value()), "Value B": fmt_val(sb.total_value())},
        {"Metric": "Avg Value/Player", "Value A": fmt_val(sa.average_value()), "Value B": fmt_val(sb.average_value())},
        {"Metric": "Goalkeepers", "Value A": pos_a.get("GK", 0), "Value B": pos_b.get("GK", 0)},
        {"Metric": "Defenders", "Value A": pos_a.get("DF", 0), "Value B": pos_b.get("DF", 0)},
        {"Metric": "Midfielders", "Value A": pos_a.get("MF", 0), "Value B": pos_b.get("MF", 0)},
        {"Metric": "Forwards", "Value A": pos_a.get("FW", 0), "Value B": pos_b.get("FW", 0)},
        {"Metric": "Unique Clubs", "Value A": sa.unique_clubs_count(), "Value B": sb.unique_clubs_count()},
        {"Metric": "Avg Caps", "Value A": round(sa.average_caps(), 1), "Value B": round(sb.average_caps(), 1)},
    ]

    # Manager comparison rows
    mgr_a = team_a.manager
    mgr_b = team_b.manager
    rows.append({
        "Metric": "Manager",
        "Value A": mgr_a.name if mgr_a else "-",
        "Value B": mgr_b.name if mgr_b else "-",
    })
    rows.append({
        "Metric": "Mgr Experience",
        "Value A": round(assessment_a.experience_score, 1) if assessment_a else "-",
        "Value B": round(assessment_b.experience_score, 1) if assessment_b else "-",
    })
    rows.append({
        "Metric": "Mgr Honours",
        "Value A": round(assessment_a.honours_score, 1) if assessment_a else "-",
        "Value B": round(assessment_b.honours_score, 1) if assessment_b else "-",
    })
    rows.append({
        "Metric": "Mgr Composite",
        "Value A": assessment_a.composite_score if assessment_a else "-",
        "Value B": assessment_b.composite_score if assessment_b else "-",
    })
    rows.append({
        "Metric": "Mgr Rating Impact",
        "Value A": assessment_a.rating_impact_pct if assessment_a else "-",
        "Value B": assessment_b.rating_impact_pct if assessment_b else "-",
    })

    # Overall rating row (import here to avoid circular)
    from .team_rating import calculate_overall_rating
    rows.append({
        "Metric": "Overall Rating",
        "Value A": calculate_overall_rating(team_a, [team_a, team_b], assessment_a),
        "Value B": calculate_overall_rating(team_b, [team_a, team_b], assessment_b),
    })

    return rows
