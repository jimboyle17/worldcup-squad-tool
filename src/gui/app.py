import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

import customtkinter as ctk

from ..analysis.manager_assessment import ManagerAssessment
from ..models.manager import Manager
from ..models.team import Team
from ..scraper.cache import ScraperCache
from ..scraper.transfermarkt import TransfermarktScraper
from ..scraper.wikipedia import WikipediaScraper
from .views.compare import CompareView
from .views.manager_view import ManagerView
from .views.squad_view import SquadView
from .views.teams_list import TeamsListView

logger = logging.getLogger(__name__)


class App(ctk.CTk):
    """Main application window."""

    def __init__(
        self,
        config: dict,
        teams_meta: List[dict],
        managers_meta: Optional[List[dict]] = None,
    ):
        super().__init__()
        self.title(config.get("app_name", "World Cup 2026 Squad Tool"))
        self.geometry(f"{config.get('window_width', 1280)}x{config.get('window_height', 800)}")
        self.minsize(960, 600)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._config = config
        self._teams_meta = teams_meta
        self._teams: List[Team] = []
        self._teams_dict: Dict[str, Team] = {}

        # Parse manager metadata
        self._managers_meta = managers_meta or []
        self._managers_dict: Dict[str, Manager] = {}
        for m in self._managers_meta:
            try:
                mgr = Manager.from_dict(m)
                self._managers_dict[mgr.team_name] = mgr
            except Exception as e:
                logger.error(f"Failed to parse manager data for {m.get('team_name', '?')}: {e}")

        # Assessments computed after teams are loaded
        self._assessments: Dict[str, ManagerAssessment] = {}

        # Initialize scraper
        base_dir = Path(__file__).resolve().parent.parent.parent
        db_path = str(base_dir / "data" / "cache.db")
        self._cache = ScraperCache(
            db_path,
            expiry_hours=config.get("cache_expiry_hours", 24),
        )
        self._scraper = TransfermarktScraper(
            self._cache,
            delay=config.get("scrape_delay_seconds", 2),
        )
        self._wiki_scraper = WikipediaScraper(
            self._cache,
            delay=1.0,
        )

        self._build_layout()
        self._show_teams_list()
        self._load_teams_data()

    def _build_layout(self):
        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        ctk.CTkLabel(
            self.sidebar,
            text="WC 2026",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(padx=16, pady=(20, 24))

        self.btn_teams = ctk.CTkButton(
            self.sidebar,
            text="Teams",
            font=ctk.CTkFont(size=15),
            height=38,
            command=self._show_teams_list,
        )
        self.btn_teams.pack(padx=16, pady=4, fill="x")

        self.btn_compare = ctk.CTkButton(
            self.sidebar,
            text="Compare",
            font=ctk.CTkFont(size=15),
            height=38,
            command=self._show_compare,
        )
        self.btn_compare.pack(padx=16, pady=4, fill="x")

        self.btn_managers = ctk.CTkButton(
            self.sidebar,
            text="Managers",
            font=ctk.CTkFont(size=15),
            height=38,
            command=self._show_managers,
        )
        self.btn_managers.pack(padx=16, pady=4, fill="x")

        # Separator
        ctk.CTkFrame(self.sidebar, height=1, fg_color="gray40").pack(
            fill="x", padx=16, pady=16
        )

        self.btn_refresh = ctk.CTkButton(
            self.sidebar,
            text="Refresh Data",
            font=ctk.CTkFont(size=14),
            height=36,
            fg_color=("gray70", "gray30"),
            command=self._refresh_all_data,
        )
        self.btn_refresh.pack(padx=16, pady=4, fill="x")

        # Appearance toggle
        self.appearance_menu = ctk.CTkComboBox(
            self.sidebar,
            values=["Dark", "Light", "System"],
            command=self._change_appearance,
            font=ctk.CTkFont(size=14),
            width=160,
        )
        self.appearance_menu.set("Dark")
        self.appearance_menu.pack(padx=16, pady=(16, 4))

        # Loading indicator
        self.loading_label = ctk.CTkLabel(
            self.sidebar,
            text="",
            font=ctk.CTkFont(size=13),
            text_color="gray50",
        )
        self.loading_label.pack(padx=16, pady=(8, 4), anchor="w")

        # Content area
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(side="right", fill="both", expand=True)

        # Create views
        self.teams_list_view = TeamsListView(
            self.content, on_team_select=self._on_team_selected
        )
        self.squad_view = SquadView(
            self.content, on_back=self._show_teams_list
        )
        self.compare_view = CompareView(self.content)
        self.manager_view = ManagerView(
            self.content, on_team_select=self._on_team_selected
        )

        self._current_view = None

    def _show_view(self, view):
        if self._current_view:
            self._current_view.pack_forget()
        view.pack(fill="both", expand=True)
        self._current_view = view

    def _show_teams_list(self):
        self._show_view(self.teams_list_view)
        if self._teams:
            self.teams_list_view.set_teams(self._teams, self._assessments)

    def _show_compare(self):
        self._show_view(self.compare_view)
        if self._teams:
            self.compare_view.set_teams(self._teams, self._assessments)

    def _show_managers(self):
        self._show_view(self.manager_view)
        if self._teams:
            self.manager_view.set_data(self._teams, self._assessments)

    def _on_team_selected(self, team: Team):
        assessment = self._assessments.get(team.name)
        self.squad_view.set_team(team, assessment)
        self._show_view(self.squad_view)

    def _change_appearance(self, value: str):
        ctk.set_appearance_mode(value.lower())

    def _load_teams_data(self):
        """Load team data in a background thread."""
        self.loading_label.configure(text="Loading teams...")
        self.btn_refresh.configure(state="disabled")

        def worker():
            teams = []
            total = len(self._teams_meta)
            for i, meta in enumerate(self._teams_meta):
                try:
                    team = self._scraper.scrape_team(meta)
                    teams.append(team)
                    self.after(0, lambda idx=i: self.loading_label.configure(
                        text=f"Squads... {idx + 1}/{total}"
                    ))
                except Exception as e:
                    logger.error(f"Failed to load {meta['name']}: {e}")
                    teams.append(Team(
                        name=meta["name"],
                        country_code=meta.get("country_code", ""),
                        confederation=meta.get("confederation", ""),
                        fifa_ranking=meta.get("fifa_ranking", 0),
                        transfermarkt_id=meta["transfermarkt_id"],
                    ))

            # Second pass: scrape caps/goals from Wikipedia
            for i, team in enumerate(teams):
                try:
                    self._wiki_scraper.update_team_players(team)
                    self.after(0, lambda idx=i: self.loading_label.configure(
                        text=f"Caps/goals... {idx + 1}/{total}"
                    ))
                except Exception as e:
                    logger.error(f"Failed to load Wikipedia data for {team.name}: {e}")

            # Third pass: attach managers to teams
            for team in teams:
                mgr = self._managers_dict.get(team.name)
                if mgr:
                    team.manager = mgr

            self.after(0, lambda: self._on_teams_loaded(teams))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _on_teams_loaded(self, teams: List[Team]):
        self._teams = sorted(teams, key=lambda t: t.fifa_ranking)
        self._teams_dict = {t.name: t for t in self._teams}

        # Compute manager assessments
        self._assessments = {}
        for team in self._teams:
            if team.manager:
                self._assessments[team.name] = ManagerAssessment(
                    manager=team.manager,
                    current_fifa_ranking=team.fifa_ranking,
                )

        self.loading_label.configure(text=f"{len(self._teams)} teams loaded")
        self.btn_refresh.configure(state="normal")

        if self._current_view == self.teams_list_view:
            self.teams_list_view.set_teams(self._teams, self._assessments)
        if self._current_view == self.compare_view:
            self.compare_view.set_teams(self._teams, self._assessments)
        if self._current_view == self.manager_view:
            self.manager_view.set_data(self._teams, self._assessments)

    def _refresh_all_data(self):
        """Force re-scrape of all team data."""
        self._cache.clear()
        self._load_teams_data()
