import json
import logging
import re
import time
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

from ..models.player import Player
from ..models.team import Team
from .cache import ScraperCache

logger = logging.getLogger(__name__)


class TransfermarktScraper:
    """Scrapes squad data from Transfermarkt."""

    BASE_URL = "https://www.transfermarkt.com"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(self, cache: ScraperCache, delay: float = 2.0):
        self.cache = cache
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._last_request_time = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request_time = time.time()

    def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch a page with rate limiting."""
        self._rate_limit()
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    @staticmethod
    def _parse_market_value(value_str: str) -> float:
        """Parse Transfermarkt market value string to float in EUR."""
        if not value_str or value_str == "-" or value_str == "N/A":
            return 0.0

        value_str = value_str.strip().replace("€", "").strip()

        multiplier = 1.0
        if "bn" in value_str.lower():
            multiplier = 1_000_000_000
            value_str = re.sub(r"[bB]n", "", value_str)
        elif "m" in value_str.lower():
            multiplier = 1_000_000
            value_str = value_str.lower().replace("m", "")
        elif "k" in value_str.lower() or "th." in value_str.lower():
            multiplier = 1_000
            value_str = value_str.lower().replace("k", "").replace("th.", "")

        try:
            return float(value_str.strip()) * multiplier
        except ValueError:
            return 0.0

    @staticmethod
    def _map_position(pos_str: str) -> tuple:
        """Map Transfermarkt position string to (position, position_detail)."""
        pos_str = pos_str.strip().lower() if pos_str else ""

        mapping = {
            "goalkeeper": ("GK", "GK"),
            "centre-back": ("DF", "CB"),
            "centre back": ("DF", "CB"),
            "left-back": ("DF", "LB"),
            "left back": ("DF", "LB"),
            "right-back": ("DF", "RB"),
            "right back": ("DF", "RB"),
            "defensive midfield": ("MF", "CDM"),
            "central midfield": ("MF", "CM"),
            "attacking midfield": ("MF", "CAM"),
            "left midfield": ("MF", "LM"),
            "right midfield": ("MF", "RM"),
            "left winger": ("FW", "LW"),
            "right winger": ("FW", "RW"),
            "centre-forward": ("FW", "CF"),
            "centre forward": ("FW", "CF"),
            "second striker": ("FW", "SS"),
        }

        for key, val in mapping.items():
            if key in pos_str:
                return val

        if "back" in pos_str or "defender" in pos_str:
            return ("DF", "DF")
        if "midfield" in pos_str:
            return ("MF", "MF")
        if "forward" in pos_str or "striker" in pos_str or "wing" in pos_str:
            return ("FW", "FW")

        return ("MF", "MF")  # Default fallback

    def scrape_squad(self, team_meta: dict, force_refresh: bool = False) -> List[Player]:
        """Scrape squad for a team. Returns list of Player objects."""
        tm_id = team_meta["transfermarkt_id"]
        tm_path = team_meta.get("transfermarkt_path", team_meta["name"].lower().replace(" ", "-"))
        cache_key = f"squad_{tm_id}"

        if not force_refresh:
            cached = self.cache.get_json(cache_key)
            if cached is not None:
                return [Player(**p) for p in cached]

        url = f"{self.BASE_URL}/{tm_path}/startseite/verein/{tm_id}/saison_id/2025"
        logger.info(f"Scraping squad from: {url}")

        html = self._fetch_page(url)
        if html is None:
            return []

        players = self._parse_squad_page(html, team_meta.get("name", ""))
        if players:
            self.cache.set_json(cache_key, [self._player_to_dict(p) for p in players])

        return players

    @staticmethod
    def _player_to_dict(player: Player) -> dict:
        return {
            "name": player.name,
            "position": player.position,
            "position_detail": player.position_detail,
            "age": player.age,
            "club": player.club,
            "nationality": player.nationality,
            "market_value": player.market_value,
            "appearances": player.appearances,
            "goals": player.goals,
            "assists": player.assists,
            "transfermarkt_id": player.transfermarkt_id,
        }

    def _parse_squad_page(self, html: str, team_name: str) -> List[Player]:
        """Parse the squad page HTML and extract player data."""
        soup = BeautifulSoup(html, "lxml")
        players = []

        # Find player rows in the squad table
        table = soup.find("table", class_="items")
        if not table:
            logger.warning(f"No squad table found for {team_name}")
            return []

        tbody = table.find("tbody")
        if not tbody:
            return []

        rows = tbody.find_all("tr", class_=["odd", "even"])
        for row in rows:
            player = self._parse_player_row(row, team_name)
            if player:
                players.append(player)

        return players

    def _parse_player_row(self, row, team_name: str) -> Optional[Player]:
        """Parse a single player row from the squad table.

        Expected td layout per row:
          [0] shirt number (zentriert)
          [1] player image/inline-table
          [2] nationality flag
          [3] player name (hauptlink)
          [4] position text
          [5] date of birth + age, e.g. "02/09/1992 (32)"  (zentriert)
          [6] nationality flag duplicate (zentriert)
          [7] market value (rechts hauptlink)
        """
        try:
            tds = row.find_all("td")
            if len(tds) < 7:
                return None

            # Player name — td with class hauptlink
            name_cell = row.find("td", class_="hauptlink")
            if not name_cell:
                return None
            name_link = name_cell.find("a")
            name = name_link.get_text(strip=True) if name_link else ""
            if not name:
                return None

            # Transfermarkt ID from link
            tm_id = ""
            if name_link and name_link.get("href"):
                href = name_link["href"]
                parts = href.strip("/").split("/")
                if parts:
                    tm_id = parts[-1]

            # Position — look for a td whose text matches a known position
            pos_str = ""
            for td in tds:
                text = td.get_text(strip=True)
                if text and any(kw in text.lower() for kw in [
                    "goalkeeper", "back", "midfield", "forward",
                    "winger", "striker",
                ]):
                    pos_str = text
                    break

            position, position_detail = self._map_position(pos_str)

            # Age — find zentriert td containing "(age)" pattern
            age = 0
            for td in tds:
                if "zentriert" not in (td.get("class") or []):
                    continue
                text = td.get_text(strip=True)
                # Format: "DD/MM/YYYY (age)" e.g. "02/09/1992 (32)"
                match = re.search(r"\((\d{1,2})\)", text)
                if match:
                    parsed = int(match.group(1))
                    if 14 <= parsed <= 50:
                        age = parsed
                        break

            # Market value — td with classes rechts + hauptlink
            value_cell = row.find("td", class_="rechts hauptlink")
            if not value_cell:
                value_cell = row.find("td", class_="rechts")
            value_str = value_cell.get_text(strip=True) if value_cell else "0"
            market_value = self._parse_market_value(value_str)

            return Player(
                name=name,
                position=position,
                position_detail=position_detail,
                age=age,
                club=team_name,  # National team context
                nationality=team_name,
                market_value=market_value,
                transfermarkt_id=tm_id,
            )
        except Exception as e:
            logger.warning(f"Failed to parse player row: {e}")
            return None

    def scrape_team(self, team_meta: dict, force_refresh: bool = False) -> Team:
        """Scrape and build a full Team object."""
        players = self.scrape_squad(team_meta, force_refresh)
        return Team(
            name=team_meta["name"],
            country_code=team_meta.get("country_code", ""),
            confederation=team_meta.get("confederation", ""),
            fifa_ranking=team_meta.get("fifa_ranking", 0),
            transfermarkt_id=team_meta["transfermarkt_id"],
            squad=players,
        )
