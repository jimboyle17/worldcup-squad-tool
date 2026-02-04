from typing import Optional

import customtkinter as ctk

from ...analysis.manager_assessment import ManagerAssessment


def _score_color(score: float) -> str:
    """Return a hex color: green for high scores, red for low."""
    if score >= 70:
        return "#2ecc71"
    elif score >= 50:
        return "#f39c12"
    elif score >= 30:
        return "#e67e22"
    else:
        return "#e74c3c"


class ManagerCard(ctk.CTkFrame):
    """Compact card showing manager info and assessment breakdown.

    Displayed inside SquadView below the stat cards row.
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(corner_radius=10, fg_color=("gray85", "gray20"))

        # Top row: name + nationality + tenure
        self.top_row = ctk.CTkFrame(self, fg_color="transparent")
        self.top_row.pack(fill="x", padx=14, pady=(10, 2))

        self.name_label = ctk.CTkLabel(
            self.top_row,
            text="Manager: -",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.name_label.pack(side="left")

        self.tenure_label = ctk.CTkLabel(
            self.top_row,
            text="",
            font=ctk.CTkFont(size=14),
            text_color=("gray40", "gray60"),
        )
        self.tenure_label.pack(side="right")

        self.nationality_label = ctk.CTkLabel(
            self.top_row,
            text="",
            font=ctk.CTkFont(size=14),
            text_color=("gray40", "gray60"),
        )
        self.nationality_label.pack(side="right", padx=(0, 16))

        # Scores row
        self.scores_row = ctk.CTkFrame(self, fg_color="transparent")
        self.scores_row.pack(fill="x", padx=14, pady=(4, 10))

        self._score_labels = {}
        for label_text in [
            "Experience", "Honours", "Club Ach.", "Tenure", "Ach. Delta",
            "Composite", "Rating Impact",
        ]:
            frame = ctk.CTkFrame(self.scores_row, fg_color="transparent")
            frame.pack(side="left", padx=(0, 12))

            title = ctk.CTkLabel(
                frame,
                text=label_text,
                font=ctk.CTkFont(size=12),
                text_color=("gray40", "gray60"),
            )
            title.pack()

            value = ctk.CTkLabel(
                frame,
                text="-",
                font=ctk.CTkFont(size=15, weight="bold"),
            )
            value.pack()
            self._score_labels[label_text] = value

    def set_assessment(self, assessment: Optional[ManagerAssessment]):
        """Update the card with a ManagerAssessment, or clear if None."""
        if assessment is None:
            self.name_label.configure(text="Manager: -")
            self.nationality_label.configure(text="")
            self.tenure_label.configure(text="")
            for lbl in self._score_labels.values():
                lbl.configure(text="-", text_color=("gray70", "gray50"))
            return

        mgr = assessment.manager
        self.name_label.configure(text=f"Manager: {mgr.name}")
        self.nationality_label.configure(text=mgr.nationality)
        self.tenure_label.configure(text=f"Tenure: {mgr.tenure_years}yr")

        scores = {
            "Experience": assessment.experience_score,
            "Honours": assessment.honours_score,
            "Club Ach.": assessment.club_achievement_score,
            "Tenure": assessment.tenure_score,
            "Ach. Delta": assessment.achievement_delta_score,
            "Composite": assessment.composite_score,
        }

        for name, score in scores.items():
            lbl = self._score_labels.get(name)
            if lbl:
                display = f"{score:.0f}"
                color = _score_color(score if name != "Ach. Delta" else score + 50)
                lbl.configure(text=display, text_color=color)

        impact_lbl = self._score_labels.get("Rating Impact")
        if impact_lbl:
            pct = (assessment.rating_multiplier - 1.0) * 100
            color = _score_color(assessment.composite_score)
            impact_lbl.configure(text=assessment.rating_impact_pct, text_color=color)
