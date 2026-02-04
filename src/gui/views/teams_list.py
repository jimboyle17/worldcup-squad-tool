import tkinter as tk
from tkinter import filedialog
from typing import Callable, Dict, List, Optional

import customtkinter as ctk
from tksheet import Sheet

from ...analysis.manager_assessment import ManagerAssessment
from ...analysis.stats import export_to_csv, teams_summary_dataframe
from ...analysis.team_rating import calculate_overall_rating
from ...models.team import Team


class TeamsListView(ctk.CTkFrame):
    """View showing all 48 teams in a sortable table."""

    COLUMNS = [
        "Team", "Confederation", "FIFA Ranking", "Squad Size",
        "Total Value", "Avg Age", "Mgr Score", "Overall Rating",
    ]

    def __init__(self, master, on_team_select: Optional[Callable] = None, **kwargs):
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
            text="World Cup 2026 Teams",
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
            "single_select",
            "column_select",
            "row_select",
            "arrowkeys",
            "copy",
        )
        self.sheet.extra_bindings("column_select", self._on_column_click)
        self.sheet.extra_bindings("cell_select", self._on_row_click)
        self.sheet.pack(fill="both", expand=True, padx=8, pady=4)

        # Status bar
        self.status_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=13))
        self.status_label.pack(anchor="w", padx=12, pady=(0, 8))

    def set_teams(
        self,
        teams: List[Team],
        assessments: Optional[Dict[str, ManagerAssessment]] = None,
    ):
        self._teams = list(teams)
        self._assessments = assessments or {}
        self._refresh_data()
        self.status_label.configure(text=f"{len(teams)} teams loaded")

    def _refresh_data(self):
        data = []
        for t in self._teams:
            a = self._assessments.get(t.name)
            mgr_score = a.composite_score if a else "-"
            overall = (
                calculate_overall_rating(t, self._teams, a)
                if a else calculate_overall_rating(t, self._teams)
            )
            data.append([
                t.name,
                t.confederation,
                t.fifa_ranking,
                t.squad_size,
                t.total_value_display,
                round(t.average_age, 1),
                mgr_score,
                overall,
            ])
        self.sheet.set_sheet_data(data)
        self.sheet.set_all_column_widths()

    def _on_column_click(self, event):
        selected = event.get("selected") if hasattr(event, "get") else getattr(event, "selected", None)
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
        key_map = {
            "Team": lambda t: t.name.lower(),
            "Confederation": lambda t: t.confederation,
            "FIFA Ranking": lambda t: t.fifa_ranking,
            "Squad Size": lambda t: t.squad_size,
            "Total Value": lambda t: t.total_market_value,
            "Avg Age": lambda t: t.average_age,
            "Mgr Score": lambda t: (
                self._assessments[t.name].composite_score
                if t.name in self._assessments else -1
            ),
            "Overall Rating": lambda t: calculate_overall_rating(
                t, self._teams, self._assessments.get(t.name)
            ),
        }
        key_fn = key_map.get(col_name)
        if key_fn:
            self._teams.sort(key=key_fn, reverse=not self._sort_asc)
            self._refresh_data()

    def _on_row_click(self, event):
        selected = event.get("selected") if hasattr(event, "get") else None
        row = getattr(selected, "row", None) if selected else None
        if row is not None and 0 <= row < len(self._teams) and self._on_team_select:
            self._on_team_select(self._teams[row])

    def _export_csv(self):
        if not self._teams:
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Export Teams List",
            initialfile="worldcup_2026_teams.csv",
        )
        if filepath:
            df = teams_summary_dataframe(self._teams)
            export_to_csv(df, filepath)
            self.status_label.configure(text=f"Exported to {filepath}")
