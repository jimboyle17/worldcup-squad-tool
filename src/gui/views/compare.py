from tkinter import filedialog
from typing import Dict, List, Optional

import customtkinter as ctk
from tksheet import Sheet

from ...analysis.composition import compare_teams
from ...analysis.manager_assessment import ManagerAssessment
from ...analysis.stats import comparison_dataframe, export_to_csv
from ...models.team import Team


class CompareView(ctk.CTkFrame):
    """Side-by-side comparison of two teams."""

    def __init__(self, master, teams: Optional[List[Team]] = None, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="transparent")
        self._teams: List[Team] = teams or []
        self._assessments: Dict[str, ManagerAssessment] = {}
        self._team_a: Optional[Team] = None
        self._team_b: Optional[Team] = None

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=8, pady=(8, 4))

        ctk.CTkLabel(
            header,
            text="Team Comparison",
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

        # Selection row
        sel_frame = ctk.CTkFrame(self, fg_color="transparent")
        sel_frame.pack(fill="x", padx=8, pady=8)

        ctk.CTkLabel(sel_frame, text="Team A:", font=ctk.CTkFont(size=14)).pack(side="left")
        self.combo_a = ctk.CTkComboBox(
            sel_frame,
            values=[],
            command=self._on_select_a,
            font=ctk.CTkFont(size=14),
            width=220,
        )
        self.combo_a.pack(side="left", padx=(4, 16))

        ctk.CTkLabel(sel_frame, text="Team B:", font=ctk.CTkFont(size=14)).pack(side="left")
        self.combo_b = ctk.CTkComboBox(
            sel_frame,
            values=[],
            command=self._on_select_b,
            font=ctk.CTkFont(size=14),
            width=220,
        )
        self.combo_b.pack(side="left", padx=4)

        self.compare_btn = ctk.CTkButton(
            sel_frame,
            text="Compare",
            font=ctk.CTkFont(size=14),
            width=120,
            height=36,
            command=self._run_comparison,
        )
        self.compare_btn.pack(side="left", padx=16)

        # Comparison table
        self.sheet = Sheet(
            self,
            headers=["Metric", "Team A", "Team B"],
            show_x_scrollbar=True,
            show_y_scrollbar=True,
            height=450,
            font=("Calibri", 20, "normal"),
            header_font=("Calibri", 21, "bold"),
        )
        self.sheet.enable_bindings("single_select", "arrowkeys", "copy")
        self.sheet.pack(fill="both", expand=True, padx=8, pady=4)

        # Status
        self.status_label = ctk.CTkLabel(self, text="Select two teams to compare", font=ctk.CTkFont(size=13))
        self.status_label.pack(anchor="w", padx=12, pady=(0, 8))

    def set_teams(
        self,
        teams: List[Team],
        assessments: Optional[Dict[str, ManagerAssessment]] = None,
    ):
        self._teams = list(teams)
        self._assessments = assessments or {}
        names = [t.name for t in teams]
        self.combo_a.configure(values=names)
        self.combo_b.configure(values=names)
        if len(names) >= 2:
            self.combo_a.set(names[0])
            self.combo_b.set(names[1])

    def _find_team(self, name: str) -> Optional[Team]:
        for t in self._teams:
            if t.name == name:
                return t
        return None

    def _on_select_a(self, value: str):
        self._team_a = self._find_team(value)

    def _on_select_b(self, value: str):
        self._team_b = self._find_team(value)

    def _run_comparison(self):
        name_a = self.combo_a.get()
        name_b = self.combo_b.get()
        self._team_a = self._find_team(name_a)
        self._team_b = self._find_team(name_b)

        if not self._team_a or not self._team_b:
            self.status_label.configure(text="Please select two valid teams")
            return

        if self._team_a.name == self._team_b.name:
            self.status_label.configure(text="Please select two different teams")
            return

        rows = compare_teams(
            self._team_a,
            self._team_b,
            self._assessments.get(self._team_a.name),
            self._assessments.get(self._team_b.name),
        )
        self.sheet.headers([f"Metric", self._team_a.name, self._team_b.name])

        data = [[r["Metric"], str(r["Value A"]), str(r["Value B"])] for r in rows]
        self.sheet.set_sheet_data(data)
        self.sheet.set_all_column_widths()
        self.status_label.configure(text=f"Comparing {self._team_a.name} vs {self._team_b.name}")

    def _export_csv(self):
        if not self._team_a or not self._team_b:
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Export Comparison",
            initialfile=f"compare_{self._team_a.name}_{self._team_b.name}.csv".lower().replace(" ", "_"),
        )
        if filepath:
            rows = compare_teams(
                self._team_a,
                self._team_b,
                self._assessments.get(self._team_a.name),
                self._assessments.get(self._team_b.name),
            )
            df = comparison_dataframe(rows, self._team_a.name, self._team_b.name)
            export_to_csv(df, filepath)
            self.status_label.configure(text=f"Exported to {filepath}")
