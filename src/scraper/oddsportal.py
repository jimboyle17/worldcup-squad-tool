"""Scraper for OddsPortal match odds using Playwright (headless Chromium).

OddsPortal renders via JavaScript, so a real browser is required.
Odds are displayed in American money line format and converted to decimal.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .cache import ScraperCache

logger = logging.getLogger(__name__)


@dataclass
class MatchOdds:
    home_team: str
    away_team: str
    date: str
    competition: str
    home_odds: float
    draw_odds: float
    away_odds: float


# Competitions to scrape for WC 2026 power rankings
# URLs verified against live OddsPortal site structure
COMPETITIONS = {
    "nations_league": "/football/europe/uefa-nations-league/",
    "world_cup_2022": "/football/world/world-cup-2022/",
    "copa_america_2024": "/football/south-america/copa-america/",
    "afcon_2024": "/football/africa/africa-cup-of-nations/",
    "afcon_2023": "/football/africa/africa-cup-of-nations-2023/",
    "asian_cup": "/football/asia/asian-cup/",
}

# Normalize OddsPortal team names to canonical names
TEAM_NAME_MAP: Dict[str, str] = {
    "South Korea": "Korea Republic",
    "USA": "United States",
    "Ivory Coast": "Cote d'Ivoire",
    "Iran": "IR Iran",
    "Czech Republic": "Czechia",
    "Turkiye": "Turkey",
    "DR Congo": "Congo DR",
    "D.R. Congo": "Congo DR",
    "North Korea": "Korea DPR",
    "Cape Verde": "Cabo Verde",
    "China": "China PR",
}

BASE_URL = "https://www.oddsportal.com"


def _american_to_decimal(american: str) -> float:
    """Convert American money line odds string to decimal odds.

    +200 → 3.00  (win $200 on a $100 bet → total return $300)
    -150 → 1.667 (bet $150 to win $100 → total return $250)
    """
    try:
        val = int(american)
    except ValueError:
        return 0.0
    if val > 0:
        return round(1 + val / 100, 4)
    elif val < 0:
        return round(1 + 100 / abs(val), 4)
    return 0.0


class OddsPortalScraper:
    """Scrapes match odds from OddsPortal using Playwright."""

    def __init__(self, cache: ScraperCache, delay: float = 3.0):
        self.cache = cache
        self.delay = delay
        self._browser = None
        self._playwright = None

    def _ensure_browser(self):
        """Lazily initialize Playwright browser."""
        if self._browser is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=True)
            logger.info("Playwright browser launched")
        except Exception as e:
            logger.error(f"Failed to launch Playwright browser: {e}")
            raise

    def close(self):
        """Close browser and Playwright."""
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None

    def _normalize_team_name(self, name: str) -> str:
        """Normalize an OddsPortal team name to canonical form."""
        name = name.strip()
        return TEAM_NAME_MAP.get(name, name)

    def _accept_cookies(self, page):
        """Click the cookie consent button if present."""
        try:
            btn = page.query_selector('button:has-text("I Accept")')
            if btn:
                btn.click()
                time.sleep(2)
                page.wait_for_load_state("networkidle")
                logger.info("Accepted cookie consent")
        except Exception as e:
            logger.debug(f"Cookie consent handling: {e}")

    def scrape_competition(self, comp_key: str, force_refresh: bool = False) -> List[MatchOdds]:
        """Scrape all completed match results for a competition.

        Returns list of MatchOdds with decimal odds.
        """
        cache_key = f"odds_{comp_key}"

        if not force_refresh:
            cached = self.cache.get_json(cache_key)
            if cached is not None:
                return [MatchOdds(**m) for m in cached]

        if comp_key not in COMPETITIONS:
            logger.warning(f"Unknown competition key: {comp_key}")
            return []

        path = COMPETITIONS[comp_key]
        url = f"{BASE_URL}{path}results/"

        self._ensure_browser()
        matches = self._scrape_results_page(url, comp_key)

        if matches:
            self.cache.set_json(cache_key, [
                {
                    "home_team": m.home_team,
                    "away_team": m.away_team,
                    "date": m.date,
                    "competition": m.competition,
                    "home_odds": m.home_odds,
                    "draw_odds": m.draw_odds,
                    "away_odds": m.away_odds,
                }
                for m in matches
            ])

        return matches

    def _scrape_results_page(self, url: str, comp_key: str) -> List[MatchOdds]:
        """Scrape a results page and follow pagination."""
        matches: List[MatchOdds] = []

        try:
            page = self._browser.new_page()
            page.set_default_timeout(20000)
            logger.info(f"Navigating to {url}")
            page.goto(url, wait_until="networkidle")
            time.sleep(self.delay)

            # Accept cookie consent on first page
            self._accept_cookies(page)
            time.sleep(self.delay)

            # Parse matches from the current page
            page_matches = self._parse_results_page(page, comp_key)
            matches.extend(page_matches)

            # Handle pagination — try page 2-5
            page_num = 2
            max_pages = 5
            while page_num <= max_pages:
                next_url = f"{url}#/page/{page_num}/"
                try:
                    page.goto(next_url, wait_until="networkidle")
                    time.sleep(self.delay)
                    new_matches = self._parse_results_page(page, comp_key)
                    if not new_matches:
                        break
                    matches.extend(new_matches)
                    page_num += 1
                except Exception:
                    break

            page.close()
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")

        return matches

    def _parse_results_page(self, page, comp_key: str) -> List[MatchOdds]:
        """Parse match rows from a results page.

        OddsPortal uses a React/Tailwind UI where match rows are nested
        inside div containers. Each row has:
        - p.participant-name elements for home/away team names
        - American money line odds as text (e.g., +200, -150)
        - Date headers interspersed as "DD MMM YYYY" text

        We locate div.group.flex elements (match rows), walk up to a
        common ancestor, and iterate children to track dates.
        """
        matches: List[MatchOdds] = []

        try:
            # Find the first match row to locate the ancestor container
            first_row = page.query_selector("div.group.flex")
            if not first_row:
                logger.warning(f"No match rows found for {comp_key}")
                return []

            # The great-grandparent contains all date headers and match rows
            ancestor = first_row.evaluate_handle(
                "e => e.parentElement.parentElement.parentElement"
            ).as_element()
            if not ancestor:
                return []

            children = ancestor.query_selector_all(":scope > *")
            current_date = ""
            date_pattern = re.compile(r"(\d{2} \w{3} \d{4})")

            for child in children:
                text = child.inner_text()
                lines = text.strip().split("\n")
                first_line = lines[0].strip() if lines else ""

                # Check if this child contains a date header
                date_match = date_pattern.search(first_line)
                if date_match:
                    current_date = date_match.group(1)

                # Check if this child contains a match (has participant names)
                names = child.query_selector_all("p.participant-name")
                if len(names) < 2:
                    continue

                home_name = names[0].inner_text().strip()
                away_name = names[1].inner_text().strip()

                # Extract American odds from text using regex
                odds_matches = re.findall(r'[+-]\d+', text)
                if len(odds_matches) < 3:
                    continue

                home_dec = _american_to_decimal(odds_matches[0])
                draw_dec = _american_to_decimal(odds_matches[1])
                away_dec = _american_to_decimal(odds_matches[2])

                if home_dec <= 1.0 or draw_dec <= 1.0 or away_dec <= 1.0:
                    continue

                matches.append(MatchOdds(
                    home_team=self._normalize_team_name(home_name),
                    away_team=self._normalize_team_name(away_name),
                    date=current_date,
                    competition=comp_key,
                    home_odds=home_dec,
                    draw_odds=draw_dec,
                    away_odds=away_dec,
                ))

        except Exception as e:
            logger.error(f"Failed to parse results page for {comp_key}: {e}")

        logger.info(f"Parsed {len(matches)} matches from {comp_key}")
        return matches

    def scrape_all_competitions(self, force_refresh: bool = False) -> List[MatchOdds]:
        """Scrape all configured competitions. Returns combined match list."""
        all_matches: List[MatchOdds] = []
        for comp_key in COMPETITIONS:
            try:
                matches = self.scrape_competition(comp_key, force_refresh)
                all_matches.extend(matches)
                logger.info(f"Scraped {len(matches)} matches from {comp_key}")
            except Exception as e:
                logger.error(f"Failed to scrape {comp_key}: {e}")
        return all_matches
