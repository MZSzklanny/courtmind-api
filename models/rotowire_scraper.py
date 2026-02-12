# -*- coding: utf-8 -*-
"""
Rotowire NBA Lineup Scraper
===========================
Automatically scrapes starting lineups and injury info from Rotowire daily.
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json
from pathlib import Path
import re

ROTOWIRE_URL = "https://www.rotowire.com/basketball/nba-lineups.php"
CACHE_FILE = Path('C:/Users/user/CourtMind/data/rotowire_lineups.json')

# Ensure data directory exists
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

# Team name to abbreviation mapping
TEAM_ABBREV = {
    'hawks': 'ATL', 'celtics': 'BOS', 'nets': 'BKN', 'hornets': 'CHA',
    'bulls': 'CHI', 'cavaliers': 'CLE', 'mavericks': 'DAL', 'nuggets': 'DEN',
    'pistons': 'DET', 'warriors': 'GSW', 'rockets': 'HOU', 'pacers': 'IND',
    'clippers': 'LAC', 'lakers': 'LAL', 'grizzlies': 'MEM', 'heat': 'MIA',
    'bucks': 'MIL', 'timberwolves': 'MIN', 'pelicans': 'NOP', 'knicks': 'NYK',
    'thunder': 'OKC', 'magic': 'ORL', '76ers': 'PHI', 'suns': 'PHX',
    'trail blazers': 'POR', 'blazers': 'POR', 'kings': 'SAC', 'spurs': 'SAS',
    'raptors': 'TOR', 'jazz': 'UTA', 'wizards': 'WAS',
    # Also map abbreviations to themselves
    'atl': 'ATL', 'bos': 'BOS', 'bkn': 'BKN', 'cha': 'CHA', 'chi': 'CHI',
    'cle': 'CLE', 'dal': 'DAL', 'den': 'DEN', 'det': 'DET', 'gsw': 'GSW',
    'hou': 'HOU', 'ind': 'IND', 'lac': 'LAC', 'lal': 'LAL', 'mem': 'MEM',
    'mia': 'MIA', 'mil': 'MIL', 'min': 'MIN', 'nop': 'NOP', 'nyk': 'NYK',
    'okc': 'OKC', 'orl': 'ORL', 'phi': 'PHI', 'phx': 'PHX', 'por': 'POR',
    'sac': 'SAC', 'sas': 'SAS', 'tor': 'TOR', 'uta': 'UTA', 'was': 'WAS',
}

def get_team_abbrev(team_text):
    """Convert team name/abbreviation to standard abbreviation."""
    team_lower = team_text.lower().strip()
    # Check direct abbreviation first
    if team_lower in TEAM_ABBREV:
        return TEAM_ABBREV[team_lower]
    # Check if team name contains a known team
    for name, abbrev in TEAM_ABBREV.items():
        if name in team_lower:
            return abbrev
    return team_text.upper()[:3]


def scrape_rotowire_lineups():
    """Scrape today's lineups from Rotowire."""
    print(f"[SCRAPER] Fetching lineups from Rotowire...")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    try:
        response = requests.get(ROTOWIRE_URL, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"[SCRAPER] Error fetching Rotowire: {e}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all lineup cards
    lineup_cards = soup.find_all('div', class_='lineup__box')

    if not lineup_cards:
        # Try alternative selectors
        lineup_cards = soup.find_all('div', class_='lineup')

    print(f"[SCRAPER] Found {len(lineup_cards)} lineup cards")

    lineups = {}
    games = []

    for card in lineup_cards:
        try:
            # Get team abbreviations from the card
            team_elements = card.find_all('a', class_='lineup__abbr') or card.find_all('div', class_='lineup__abbr')

            if len(team_elements) < 2:
                # Try finding team names differently
                team_elements = card.find_all('span', class_='lineup__abbr')

            if len(team_elements) < 2:
                continue

            away_team = get_team_abbrev(team_elements[0].get_text(strip=True))
            home_team = get_team_abbrev(team_elements[1].get_text(strip=True))

            # Find lineup lists (away and home)
            lineup_lists = card.find_all('ul', class_='lineup__list')

            if len(lineup_lists) < 2:
                continue

            away_list = lineup_lists[0]
            home_list = lineup_lists[1]

            # Extract starters and OUT players from each lineup list
            away_starters = []
            home_starters = []
            away_out = []
            home_out = []

            def parse_lineup_list(lineup_list):
                """Parse a single team's lineup list for starters and Out players."""
                starters = []
                out_players = []

                players = lineup_list.find_all('li', class_='lineup__player')
                starter_count = 0

                for player_li in players:
                    player_link = player_li.find('a')
                    if not player_link:
                        continue

                    name = player_link.get_text(strip=True)
                    # Clean up name (remove position prefix like "PG ", "SG ", etc.)
                    name = re.sub(r'^[PGSFPC]{1,2}\s+', '', name)

                    classes = player_li.get('class', [])

                    # is-pct-play-0 = definitely Out
                    # OFS = out for season (also skip)
                    inj_span = player_li.find('span', class_='lineup__inj')
                    status = inj_span.get_text(strip=True) if inj_span else ''

                    if 'is-pct-play-0' in classes and status == 'Out':
                        out_players.append(name)
                    elif status not in ('OFS',) and starter_count < 5 and 'is-pct-play-0' not in classes:
                        # Count as starter if in first 5 non-OFS players
                        # Only add to starters if before MAY NOT PLAY section
                        title_before = lineup_list.find('li', class_='lineup__title')
                        if title_before:
                            # Get all items before the MAY NOT PLAY title
                            before_title = []
                            for item in lineup_list.find_all('li'):
                                if 'lineup__title' in item.get('class', []):
                                    break
                                before_title.append(item)
                            if player_li in before_title and 'lineup__player' in classes:
                                starters.append(name)
                                starter_count += 1
                        else:
                            if 'lineup__player' in classes and starter_count < 5:
                                starters.append(name)
                                starter_count += 1

                return starters, out_players

            away_starters, away_out = parse_lineup_list(away_list)
            home_starters, home_out = parse_lineup_list(home_list)

            # Store lineups
            if away_starters:
                lineups[away_team] = {
                    'opponent': home_team,
                    'home': False,
                    'starters': away_starters,
                    'out': away_out
                }

            if home_starters:
                lineups[home_team] = {
                    'opponent': away_team,
                    'home': True,
                    'starters': home_starters,
                    'out': home_out
                }

            games.append({
                'away': away_team,
                'home': home_team,
                'away_starters': away_starters,
                'home_starters': home_starters
            })

            print(f"[SCRAPER] {away_team} @ {home_team}: {len(away_starters)} vs {len(home_starters)} starters")

        except Exception as e:
            print(f"[SCRAPER] Error parsing lineup card: {e}")
            continue

    return {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'scraped_at': datetime.now().isoformat(),
        'lineups': lineups,
        'games': games,
        'game_count': len(games)
    }


def save_lineups(data):
    """Save scraped lineups to JSON cache."""
    if data:
        with open(CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[SCRAPER] Saved {len(data.get('lineups', {}))} team lineups to {CACHE_FILE}")
        return True
    return False


def load_cached_lineups():
    """Load lineups from cache file."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"[SCRAPER] Error loading cache: {e}")
    return None


def get_todays_lineups(force_refresh=False):
    """Get today's lineups, scraping if needed."""
    today = datetime.now().strftime('%Y-%m-%d')

    # Check cache first
    if not force_refresh:
        cached = load_cached_lineups()
        if cached and cached.get('date') == today:
            print(f"[SCRAPER] Using cached lineups from {cached.get('scraped_at')}")
            return cached.get('lineups', {})

    # Scrape fresh data
    data = scrape_rotowire_lineups()
    if data and data.get('lineups'):
        save_lineups(data)
        return data.get('lineups', {})

    # Fall back to cache if scraping fails
    cached = load_cached_lineups()
    if cached:
        print(f"[SCRAPER] Scraping failed, using stale cache from {cached.get('date')}")
        return cached.get('lineups', {})

    return {}


def get_team_starters(team_abbrev):
    """Get starters for a specific team."""
    lineups = get_todays_lineups()
    team_data = lineups.get(team_abbrev, {})
    return team_data.get('starters', [])


def get_all_todays_starters():
    """Get flat list of all players starting today."""
    lineups = get_todays_lineups()
    all_starters = []
    for team, data in lineups.items():
        all_starters.extend(data.get('starters', []))
    return all_starters


def get_out_players():
    """Get dict of all players listed as OUT today."""
    lineups = get_todays_lineups()
    out_players = {}
    for team, data in lineups.items():
        out_players[team] = data.get('out', [])
    return out_players


if __name__ == "__main__":
    print("=" * 60)
    print("Rotowire NBA Lineup Scraper")
    print("=" * 60)

    # Force refresh to test
    data = scrape_rotowire_lineups()

    if data:
        save_lineups(data)
        print(f"\nScraped {data['game_count']} games for {data['date']}")
        print("\nLineups:")
        for team, info in sorted(data['lineups'].items()):
            print(f"\n{team} vs {info['opponent']} ({'HOME' if info['home'] else 'AWAY'}):")
            for i, player in enumerate(info['starters'], 1):
                print(f"  {i}. {player}")
            if info.get('out'):
                print(f"  OUT: {', '.join(info['out'])}")
    else:
        print("Failed to scrape lineups")
