"""
Example web scraping utilities for gathering match stats.

NOTE:
- This module is illustrative and may not work out-of-the-box for any
  particular website without adapting the URL and HTML selectors.
- Always respect the website's Terms of Service and robots.txt rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup

from footymind.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ScrapedMatch:
    """Container for a single scraped match row."""

    date: str
    season: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    home_xG: float | None = None
    away_xG: float | None = None
    home_possession: float | None = None
    away_possession: float | None = None
    home_shots: int | None = None
    away_shots: int | None = None
    home_shots_on_target: int | None = None
    away_shots_on_target: int | None = None
    home_corners: int | None = None
    away_corners: int | None = None
    home_yellow_cards: int | None = None
    away_yellow_cards: int | None = None
    home_red_cards: int | None = None
    away_red_cards: int | None = None


def scrape_matches_from_example_site(
    season: str,
    base_url: str = "https://example.com/premier-league",
) -> pd.DataFrame:
    """
    Example scraping function that demonstrates how you might pull match
    statistics from a website.

    This function will NOT work as-is on real sites. You must:
    - Update `base_url` to the real site's URL.
    - Inspect the HTML with your browser dev tools.
    - Adjust selectors (e.g., table IDs, class names).

    Parameters
    ----------
    season : str
        Season identifier (e.g., "2023-2024").
    base_url : str
        Base URL of the site to scrape.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per match and similar columns to the raw schema.
    """
    url = f"{base_url}/{season}/results"
    logger.info("Scraping matches for season %s from %s", season, url)

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Example: assume matches are in a <table> with <tr> rows
    table = soup.find("table")
    if table is None:
        logger.warning("No table found on the page. Check your selectors.")
        return pd.DataFrame()

    rows = table.find_all("tr")
    scraped_matches: List[ScrapedMatch] = []

    for row in rows[1:]:  # skip header row if present
        cells = row.find_all("td")
        if len(cells) < 5:
            # Not enough cells to parse a match row
            continue

        try:
            # The exact index of each cell depends on the site's HTML.
            # The below is purely illustrative; you must adjust it.
            date_str = cells[0].get_text(strip=True)
            home_team = cells[1].get_text(strip=True)
            score_text = cells[2].get_text(strip=True)  # e.g. "3–1"
            away_team = cells[3].get_text(strip=True)

            if "–" in score_text:
                home_goals_str, away_goals_str = score_text.split("–")
            elif "-" in score_text:
                home_goals_str, away_goals_str = score_text.split("-")
            else:
                # Unrecognized score format
                continue

            home_goals = int(home_goals_str)
            away_goals = int(away_goals_str)

            # Dummy placeholders for advanced stats (xG, possession, etc.)
            # In a real scraper, you'd extract these from specific columns.
            scraped_match = ScrapedMatch(
                date=date_str,
                season=season,
                home_team=home_team,
                away_team=away_team,
                home_goals=home_goals,
                away_goals=away_goals,
            )
            scraped_matches.append(scraped_match)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse row: %s", exc)
            continue

    if not scraped_matches:
        logger.warning("No matches were successfully scraped.")
        return pd.DataFrame()

    df = pd.DataFrame([m.__dict__ for m in scraped_matches])
    logger.info("Scraped %d matches for season %s.", len(df), season)
    return df
