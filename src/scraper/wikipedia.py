import logging
import re
import time
import unicodedata
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from .cache import ScraperCache

logger = logging.getLogger(__name__)

# Maps team name -> Wikipedia article name
WIKI_TEAM_MAP = {
    "Argentina": "Argentina_national_football_team",
    "France": "France_national_football_team",
    "Spain": "Spain_national_football_team",
    "England": "England_national_football_team",
    "Brazil": "Brazil_national_football_team",
    "Portugal": "Portugal_national_football_team",
    "Netherlands": "Netherlands_national_football_team",
    "Belgium": "Belgium_national_football_team",
    "Italy": "Italy_national_football_team",
    "Germany": "Germany_national_football_team",
    "Uruguay": "Uruguay_national_football_team",
    "Colombia": "Colombia_national_football_team",
    "Croatia": "Croatia_national_football_team",
    "Japan": "Japan_national_football_team",
    "Morocco": "Morocco_national_football_team",
    "USA": "United_States_men%27s_national_soccer_team",
    "Mexico": "Mexico_national_football_team",
    "Senegal": "Senegal_national_football_team",
    "Ecuador": "Ecuador_national_football_team",
    "Austria": "Austria_national_football_team",
    "Switzerland": "Switzerland_national_football_team",
    "Denmark": "Denmark_national_football_team",
    "Australia": "Australia_men%27s_national_soccer_team",
    "Korea Republic": "South_Korea_national_football_team",
    "Serbia": "Serbia_national_football_team",
    "Turkey": "Turkey_national_football_team",
    "Ukraine": "Ukraine_national_football_team",
    "Wales": "Wales_national_football_team",
    "Hungary": "Hungary_national_football_team",
    "Iran": "Iran_national_football_team",
    "Cameroon": "Cameroon_national_football_team",
    "Canada": "Canada_men%27s_national_soccer_team",
    "Nigeria": "Nigeria_national_football_team",
    "Saudi Arabia": "Saudi_Arabia_national_football_team",
    "Paraguay": "Paraguay_national_football_team",
    "Scotland": "Scotland_national_football_team",
    "Egypt": "Egypt_national_football_team",
    "Chile": "Chile_national_football_team",
    "Venezuela": "Venezuela_national_football_team",
    "Bolivia": "Bolivia_national_football_team",
    "Qatar": "Qatar_national_football_team",
    "Ghana": "Ghana_national_football_team",
    "Ivory Coast": "Ivory_Coast_national_football_team",
    "Jamaica": "Jamaica_national_football_team",
    "Honduras": "Honduras_national_football_team",
    "Panama": "Panama_national_football_team",
    "New Zealand": "New_Zealand_men%27s_national_football_team",
    "Trinidad and Tobago": "Trinidad_and_Tobago_national_football_team",
}


def _normalize(name: str) -> str:
    """Normalize a player name for matching: lowercase, strip accents, remove punctuation."""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower().strip()
    name = re.sub(r"[^a-z\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def _name_match_score(name_a: str, name_b: str) -> int:
    """Score how well two player names match. 0 = no match, higher = better.

    3 = exact normalized match
    2 = multi-word match (first+last or all parts of shorter in longer)
    1 = single-word last-name-only match
    """
    na = _normalize(name_a)
    nb = _normalize(name_b)

    if na == nb:
        return 3

    parts_a = na.split()
    parts_b = nb.split()
    if not parts_a or not parts_b:
        return 0

    # If both have 2+ parts, require first AND last to match
    if len(parts_a) >= 2 and len(parts_b) >= 2:
        if parts_a[0] == parts_b[0] and parts_a[-1] == parts_b[-1]:
            return 3
        # Check if all parts of the shorter appear in the longer
        shorter, longer = (parts_a, parts_b) if len(parts_a) <= len(parts_b) else (parts_b, parts_a)
        if all(s in longer for s in shorter):
            return 2
        return 0

    # One name is a single word — allow last name match
    if parts_a[-1] == parts_b[-1]:
        return 1

    # Check if the single-word name appears in the multi-word name
    shorter, longer = (parts_a, parts_b) if len(parts_a) <= len(parts_b) else (parts_b, parts_a)
    if len(shorter) == 1 and shorter[0] in longer:
        return 1

    return 0


class WikipediaScraper:
    """Scrapes player caps and goals from Wikipedia national team pages."""

    BASE_URL = "https://en.wikipedia.org/wiki"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    def __init__(self, cache: ScraperCache, delay: float = 1.0):
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
        self._rate_limit()
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def scrape_caps_goals(self, team_name: str, force_refresh: bool = False) -> Dict[str, Tuple[int, int]]:
        """Scrape caps and goals for a team from Wikipedia.

        Returns dict mapping normalized player name -> (caps, goals).
        """
        cache_key = f"wiki_caps_{team_name.replace(' ', '_')}"

        if not force_refresh:
            cached = self.cache.get_json(cache_key)
            if cached is not None:
                return {k: tuple(v) for k, v in cached.items()}

        wiki_path = WIKI_TEAM_MAP.get(team_name)
        if not wiki_path:
            logger.warning(f"No Wikipedia mapping for {team_name}")
            return {}

        url = f"{self.BASE_URL}/{wiki_path}"
        logger.info(f"Scraping Wikipedia caps/goals: {url}")

        html = self._fetch_page(url)
        if html is None:
            return {}

        result = self._parse_squad_tables(html)
        if result:
            # Cache as JSON-serializable format
            self.cache.set_json(cache_key, {k: list(v) for k, v in result.items()})

        return result

    def _parse_squad_tables(self, html: str) -> Dict[str, Tuple[int, int]]:
        """Parse squad tables from Wikipedia page. Returns {player_name: (caps, goals)}."""
        soup = BeautifulSoup(html, "lxml")
        result = {}

        # Find all sortable tables that have Caps and Goals columns
        tables = soup.find_all("table", class_="sortable")

        for table in tables:
            header_row = table.find("tr")
            if not header_row:
                continue

            ths = header_row.find_all("th")
            headers = [th.get_text(strip=True).lower() for th in ths]

            # We need both Caps and Goals columns, and a Player column
            caps_idx = None
            goals_idx = None
            player_idx = None

            for i, h in enumerate(headers):
                if h == "caps":
                    caps_idx = i
                elif h == "goals":
                    goals_idx = i
                elif h in ("player", "name"):
                    player_idx = i

            if caps_idx is None or goals_idx is None or player_idx is None:
                continue

            # Also check this is a current squad table, not all-time records
            # All-time tables usually have "Career" or "Rank" columns
            if any(h in ("rank", "career", "ratio") for h in headers):
                continue

            rows = table.find_all("tr")[1:]
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) <= max(caps_idx, goals_idx, player_idx):
                    continue

                player_cell = cells[player_idx]
                # Get player name, prefer link text
                link = player_cell.find("a")
                name = link.get_text(strip=True) if link else player_cell.get_text(strip=True)
                if not name or len(name) < 2:
                    continue

                try:
                    caps_text = cells[caps_idx].get_text(strip=True)
                    caps = int(re.sub(r"[^\d]", "", caps_text)) if caps_text else 0
                except (ValueError, IndexError):
                    caps = 0

                try:
                    goals_text = cells[goals_idx].get_text(strip=True)
                    goals = int(re.sub(r"[^\d]", "", goals_text)) if goals_text else 0
                except (ValueError, IndexError):
                    goals = 0

                result[name] = (caps, goals)

        return result

    def update_team_players(self, team, force_refresh: bool = False):
        """Update a Team's players with caps and goals from Wikipedia."""
        caps_goals = self.scrape_caps_goals(team.name, force_refresh)
        if not caps_goals:
            logger.info(f"No Wikipedia data found for {team.name}")
            return

        wiki_names = list(caps_goals.keys())
        matched = 0

        for player in team.squad:
            best_match = None
            best_score = 0
            for wiki_name in wiki_names:
                score = _name_match_score(player.name, wiki_name)
                if score > best_score:
                    best_score = score
                    best_match = wiki_name

            if best_match and best_score >= 1:
                caps, goals = caps_goals[best_match]
                player.appearances = caps
                player.goals = goals
                matched += 1

        logger.info(f"{team.name}: matched {matched}/{len(team.squad)} players with Wikipedia data")
