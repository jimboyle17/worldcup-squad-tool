import tkinter as tk
from typing import Callable, List, Optional

import customtkinter as ctk
from tksheet import Sheet

from ...models.player import Player


class PlayerTable(ctk.CTkFrame):
    """Reusable sortable player table using tksheet."""

    COLUMNS = ["Name", "Pos", "Detail", "Age", "Club", "Market Value", "Caps", "Goals"]

    def __init__(self, master, on_sort: Optional[Callable] = None, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="transparent")
        self._players: List[Player] = []
        self._sort_col = None
        self._sort_asc = True

        self.sheet = Sheet(
            self,
            headers=self.COLUMNS,
            show_x_scrollbar=True,
            show_y_scrollbar=True,
            height=400,
            width=900,
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
        self.sheet.pack(fill="both", expand=True, padx=4, pady=4)

    def set_players(self, players: List[Player]):
        self._players = list(players)
        self._refresh_data()

    def _refresh_data(self):
        data = []
        for p in self._players:
            data.append([
                p.name,
                p.position,
                p.position_detail,
                p.age,
                p.club,
                p.market_value_display,
                p.appearances,
                p.goals,
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
            "Name": lambda p: p.name.lower(),
            "Pos": lambda p: p.position,
            "Detail": lambda p: p.position_detail,
            "Age": lambda p: p.age,
            "Club": lambda p: p.club.lower(),
            "Market Value": lambda p: p.market_value,
            "Caps": lambda p: p.appearances,
            "Goals": lambda p: p.goals,
        }

        key_fn = key_map.get(col_name)
        if key_fn:
            self._players.sort(key=key_fn, reverse=not self._sort_asc)
            self._refresh_data()

    def get_players(self) -> List[Player]:
        return self._players

    def filter_by_position(self, position: str):
        """Filter display to show only players of a given position. Empty string shows all."""
        # Re-read from original — caller should keep a full list.
        pass
