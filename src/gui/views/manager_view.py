from tkinter import filedialog
from typing import Callable, Dict, List, Optional

import customtkinter as ctk
from tksheet import Sheet

from ...analysis.manager_assessment import ManagerAssessment
from ...analysis.stats import export_to_csv, managers_dataframe
from ...models.team import Team


class ManagerView(ctk.CTkFrame):
    """Sortable table view displaying manager assessments for all teams."""

    COLUMNS = [
        "Team", "Manager", "Nationality", "Age", "Tenure",
        "Honours", "Exp Score", "Composite", "Rating Impact",
    ]

    def __init__(
        self,
        master,
        on_team_select: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.configure(fg_color="transparent")
        self._teams: List[Team] = []
        self._assessments: Dict[str, ManagerAssessment] = {}
        self._on_team_select = on_team_select
        self._sort_col = None
        self._sort_asc = True

        # Header bar
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=8, pady=(8, 4))

        ctk.CTkLabel(
            header,
            text="Manager Assessments",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).pack(side="left")

        self.export_btn = ctk.CTkButton(
            header,
            text="Export CSV",
            font=ctk.CTkFont(size=14),
            width=120,
            height=36,
            command=self._export_csv,
        )
        self.export_btn.pack(side="right")

        # Table
        self.sheet = Sheet(
            self,
            headers=self.COLUMNS,
            show_x_scrollbar=True,
            show_y_scrollbar=True,
            height=600,
            font=("Calibri", 20, "normal"),
            header_font=("Calibri", 21, "bold"),
        )
        self.sheet.enable_bindings(
            "single_select", "column_select", "row_select", "arrowkeys", "copy",
        )
        self.sheet.extra_bindings("column_select", self._on_column_click)
        self.sheet.extra_bindings("cell_select", self._on_row_click)
        self.sheet.pack(fill="both", expand=True, padx=8, pady=4)

        # Status bar
        self.status_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=13))
        self.status_label.pack(anchor="w", padx=12, pady=(0, 8))

    def set_data(
        self,
        teams: List[Team],
        assessments: Dict[str, ManagerAssessment],
    ):
        self._teams = list(teams)
        self._assessments = assessments
        self._refresh_data()
        self.status_label.configure(text=f"{len(teams)} managers loaded")

    def _refresh_data(self):
        data = []
        for t in self._teams:
            a = self._assessments.get(t.name)
            mgr = t.manager
            if mgr and a:
                data.append([
                    t.name,
                    mgr.name,
                    mgr.nationality,
                    mgr.age,
                    mgr.tenure_years,
                    len(mgr.honours),
                    round(a.experience_score, 1),
                    a.composite_score,
                    a.rating_impact_pct,
                ])
            else:
                data.append([
                    t.name, "-", "-", "-", "-", "-", "-", "-", "-",
                ])
        self.sheet.set_sheet_data(data)
        self.sheet.set_all_column_widths()

    def _on_column_click(self, event):
        selected = (
            event.get("selected") if hasattr(event, "get")
            else getattr(event, "selected", None)
        )
        col = getattr(selected, "column", None) if selected else None
        if col is None:
            col = getattr(event, "column", None)
        if col is None or col < 0 or col >= len(self.COLUMNS):
            return

        if self._sort_col == col:
            self._sort_asc = not self._sort_asc
        else:
            self._sort_col = col
            self._sort_asc = True

        col_name = self.COLUMNS[col]

        def _sort_key(t: Team):
            a = self._assessments.get(t.name)
            mgr = t.manager
            if col_name == "Team":
                return t.name.lower()
            if not mgr or not a:
                return ""
            if col_name == "Manager":
                return mgr.name.lower()
            if col_name == "Nationality":
                return mgr.nationality.lower()
            if col_name == "Age":
                return mgr.age
            if col_name == "Tenure":
                return mgr.tenure_years
            if col_name == "Honours":
                return len(mgr.honours)
            if col_name == "Exp Score":
                return a.experience_score
            if col_name == "Composite":
                return a.composite_score
            if col_name == "Rating Impact":
                return a.rating_multiplier
            return ""

        self._teams.sort(key=_sort_key, reverse=not self._sort_asc)
        self._refresh_data()

    def _on_row_click(self, event):
        selected = (
            event.get("selected") if hasattr(event, "get") else None
        )
        row = getattr(selected, "row", None) if selected else None
        if (
            row is not None
            and 0 <= row < len(self._teams)
            and self._on_team_select
        ):
            self._on_team_select(self._teams[row])

    def _export_csv(self):
        if not self._assessments:
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Export Manager Assessments",
            initialfile="worldcup_2026_managers.csv",
        )
        if filepath:
            df = managers_dataframe(self._assessments)
            export_to_csv(df, filepath)
            self.status_label.configure(text=f"Exported to {filepath}")
