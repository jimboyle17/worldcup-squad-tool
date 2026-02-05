from typing import TYPE_CHECKING, Dict, List, Optional

from ..models.player import Player
from ..models.squad import Squad
from ..models.team import Team
from .home_advantage import HomeAdvantageConfig, home_advantage_info

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
    home_config: Optional[HomeAdvantageConfig] = None,
    power_ratings: Optional[Dict[str, float]] = None,
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

    # Power Rating row
    pr_a = power_ratings.get(team_a.name) if power_ratings else None
    pr_b = power_ratings.get(team_b.name) if power_ratings else None
    rows.append({
        "Metric": "Power Rating",
        "Value A": round(pr_a, 1) if pr_a is not None else "-",
        "Value B": round(pr_b, 1) if pr_b is not None else "-",
    })

    # Home Boost row
    # Compute base ratings for info display
    from .team_rating import calculate_base_team_rating
    base_a = calculate_base_team_rating(team_a, [team_a, team_b], pr_a)
    base_b = calculate_base_team_rating(team_b, [team_a, team_b], pr_b)
    info_a = home_advantage_info(team_a.name, base_a, home_config)
    info_b = home_advantage_info(team_b.name, base_b, home_config)
    rows.append({
        "Metric": "Home Boost",
        "Value A": info_a["boost_label"],
        "Value B": info_b["boost_label"],
    })

    # Overall rating row
    from .team_rating import calculate_overall_rating
    rows.append({
        "Metric": "Overall Rating",
        "Value A": calculate_overall_rating(team_a, [team_a, team_b], assessment_a, home_config, pr_a),
        "Value B": calculate_overall_rating(team_b, [team_a, team_b], assessment_b, home_config, pr_b),
    })

    return rows
