# -*- coding: utf-8 -*-
"""
NBA Official Lineups Fetcher
============================
Fetches starting lineups from NBA.com
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json
from pathlib import Path

CACHE_FILE = Path('C:/Users/user/CourtMind/nba_lineups_cache.json')
CACHE_DURATION_MINUTES = 30

# Team abbreviation mappings
TEAM_ABBREV = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS',
}

# Today's lineups - TBD until updated
# Rosters will be populated when official lineups are released
TODAYS_LINEUPS = {}


def get_todays_official_lineups():
    """Get today's official starting lineups"""
    return TODAYS_LINEUPS


def get_team_starters(team_abbrev):
    """Get starters for a specific team"""
    return TODAYS_LINEUPS.get(team_abbrev, {}).get('starters', [])


def get_all_todays_starters():
    """Get flat list of all players starting today"""
    all_starters = []
    for team, data in TODAYS_LINEUPS.items():
        all_starters.extend(data.get('starters', []))
    return all_starters


if __name__ == "__main__":
    print("Today's Official NBA Lineups:")
    print("=" * 50)
    for team, data in sorted(TODAYS_LINEUPS.items()):
        print(f"\n{team} vs {data['opponent']}:")
        for i, player in enumerate(data['starters'], 1):
            print(f"  {i}. {player}")
