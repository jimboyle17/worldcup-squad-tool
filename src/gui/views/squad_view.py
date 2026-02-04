from tkinter import filedialog
from typing import Callable, List, Optional

import customtkinter as ctk

from ...analysis.composition import squad_summary
from ...analysis.manager_assessment import ManagerAssessment
from ...analysis.stats import export_to_csv, players_dataframe
from ...models.player import Player
from ...models.team import Team
from ..components.manager_card import ManagerCard
from ..components.player_table import PlayerTable
from ..components.stat_cards import StatCardRow


class SquadView(ctk.CTkFrame):
    """View showing detailed squad information for a single team."""

    def __init__(self, master, on_back: Optional[Callable] = None, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="transparent")
        self._team: Optional[Team] = None
        self._all_players: List[Player] = []
        self._on_back = on_back

        # Header bar
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=8, pady=(8, 4))

        self.back_btn = ctk.CTkButton(
            header,
            text="< Back",
            font=ctk.CTkFont(size=14),
            width=80,
            height=36,
            command=self._go_back,
        )
        self.back_btn.pack(side="left")

        self.team_label = ctk.CTkLabel(
            header,
            text="",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.team_label.pack(side="left", padx=16)

        self.export_btn = ctk.CTkButton(
            header,
            text="Export CSV",
            font=ctk.CTkFont(size=14),
            width=120,
            height=36,
            command=self._export_csv,
        )
        self.export_btn.pack(side="right")

        # Stat cards
        self.cards = StatCardRow(
            self,
            stats=[
                ("Squad Size", "-"),
                ("Avg Age", "-"),
                ("Total Value", "-"),
                ("Avg Caps", "-"),
                ("Most Valuable", "-"),
            ],
        )
        self.cards.pack(fill="x", padx=8, pady=8)

        # Manager card
        self.manager_card = ManagerCard(self)
        self.manager_card.pack(fill="x", padx=8, pady=(0, 8))
        self.manager_card.pack_forget()  # Hidden until data is set

        # Position filter
        filter_frame = ctk.CTkFrame(self, fg_color="transparent")
        filter_frame.pack(fill="x", padx=8, pady=(0, 4))

        ctk.CTkLabel(filter_frame, text="Filter by Position:", font=ctk.CTkFont(size=14)).pack(side="left")
        self.pos_filter = ctk.CTkComboBox(
            filter_frame,
            values=["All", "GK", "DF", "MF", "FW"],
            command=self._on_filter_change,
            font=ctk.CTkFont(size=14),
            width=140,
        )
        self.pos_filter.set("All")
        self.pos_filter.pack(side="left", padx=8)

        # Composition summary
        self.composition_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=14),
            justify="left",
        )
        self.composition_label.pack(anchor="w", padx=12, pady=(0, 4))

        # Player table
        self.player_table = PlayerTable(self)
        self.player_table.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Status bar
        self.status_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=13))
        self.status_label.pack(anchor="w", padx=12, pady=(0, 8))

    def set_team(
        self,
        team: Team,
        assessment: Optional[ManagerAssessment] = None,
    ):
        self._team = team
        self._all_players = list(team.squad)
        self.team_label.configure(text=f"{team.name}  ({team.confederation})")
        self.pos_filter.set("All")

        summary = squad_summary(team)

        def fmt_val(v):
            if v >= 1_000_000_000:
                return f"€{v / 1_000_000_000:.2f}bn"
            elif v >= 1_000_000:
                return f"€{v / 1_000_000:.1f}m"
            elif v >= 1_000:
                return f"€{v / 1_000:.0f}k"
            return f"€{v:.0f}"

        self.cards.update_stats([
            ("Squad Size", summary["squad_size"]),
            ("Avg Age", summary["average_age"]),
            ("Total Value", fmt_val(summary["total_value"])),
            ("Avg Caps", summary["average_caps"]),
            ("Most Valuable", summary["most_valuable"]),
        ])

        pos = summary["position_breakdown"]
        age = summary["age_distribution"]
        comp_text = (
            f"Positions: GK {pos.get('GK', 0)} | DF {pos.get('DF', 0)} "
            f"| MF {pos.get('MF', 0)} | FW {pos.get('FW', 0)}     "
            f"Ages: U21 {age.get('U21', 0)} | 21-25 {age.get('21-25', 0)} "
            f"| 26-29 {age.get('26-29', 0)} | 30+ {age.get('30+', 0)}"
        )
        self.composition_label.configure(text=comp_text)

        # Manager card
        if assessment:
            self.manager_card.set_assessment(assessment)
            self.manager_card.pack(fill="x", padx=8, pady=(0, 8), after=self.cards)
        else:
            self.manager_card.pack_forget()

        self.player_table.set_players(self._all_players)
        self.status_label.configure(text=f"{len(self._all_players)} players")

    def _on_filter_change(self, value: str):
        if value == "All":
            filtered = self._all_players
        else:
            filtered = [p for p in self._all_players if p.position == value]
        self.player_table.set_players(filtered)
        self.status_label.configure(text=f"{len(filtered)} players shown")

    def _go_back(self):
        if self._on_back:
            self._on_back()

    def _export_csv(self):
        if not self._team:
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Export Squad",
            initialfile=f"{self._team.name.lower().replace(' ', '_')}_squad.csv",
        )
        if filepath:
            players = self.player_table.get_players()
            df = players_dataframe(players)
            export_to_csv(df, filepath)
            self.status_label.configure(text=f"Exported to {filepath}")
