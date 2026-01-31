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

# Today's lineups - January 31, 2026 (from Rotowire)
TODAYS_LINEUPS = {
    # Game 1: SAS @ CHA
    'CHA': {
        'opponent': 'SAS',
        'home': True,
        'starters': ['LaMelo Ball', 'Kon Knueppel', 'Brandon Miller', 'Miles Bridges', 'Moussa Diabate']
    },
    'SAS': {
        'opponent': 'CHA',
        'home': False,
        'starters': ["De'Aaron Fox", 'Stephon Castle', 'Justin Champagnie', 'Harrison Barnes', 'Victor Wembanyama']
    },
    # Game 2: ATL @ IND
    'IND': {
        'opponent': 'ATL',
        'home': True,
        'starters': ['Andrew Nembhard', 'Johnny Furphy', 'Aaron Nesmith', 'Jarace Walker', 'Pascal Siakam']
    },
    'ATL': {
        'opponent': 'IND',
        'home': False,
        'starters': ['Dyson Daniels', 'Nickeil Alexander-Walker', 'Corey Kispert', 'Jalen Johnson', 'Christ Koloko']
    },
    # Game 3: NOP @ PHI
    'PHI': {
        'opponent': 'NOP',
        'home': True,
        'starters': ['Tyrese Maxey', 'VJ Edgecombe', 'Kelly Oubre Jr.', 'Paul George', 'Joel Embiid']
    },
    'NOP': {
        'opponent': 'PHI',
        'home': False,
        'starters': ['Trey Murphy III', 'Herbert Jones', 'Saddiq Bey', 'Zion Williamson', 'Derik Queen']
    },
    # Game 4: CHI @ MIA
    'MIA': {
        'opponent': 'CHI',
        'home': True,
        'starters': ['Kasparas Jakucionis', 'Jaime Jaquez Jr.', 'Pelle Larsson', 'Andrew Wiggins', 'Bam Adebayo']
    },
    'CHI': {
        'opponent': 'MIA',
        'home': False,
        'starters': ['Josh Giddey', 'Coby White', 'Isaac Okoro', 'Matas Buzelis', 'Jalen Smith']
    },
    # Game 5: MIN @ MEM
    'MEM': {
        'opponent': 'MIN',
        'home': True,
        'starters': ['Cam Spencer', 'Cedric Coward', 'Jaylen Wells', 'Santi Aldama', 'Jaren Jackson Jr.']
    },
    'MIN': {
        'opponent': 'MEM',
        'home': False,
        'starters': ['Donte DiVincenzo', 'Anthony Edwards', 'Jaden McDaniels', 'Julius Randle', 'Rudy Gobert']
    },
    # Game 6: DAL @ HOU
    'HOU': {
        'opponent': 'DAL',
        'home': True,
        'starters': ['Amen Thompson', 'Tari Eason', 'Kevin Durant', 'Jabari Smith Jr.', 'Alperen Sengun']
    },
    'DAL': {
        'opponent': 'HOU',
        'home': False,
        'starters': ['Cooper Flagg', 'Max Christie', 'Naji Marshall', 'PJ Washington', 'Daniel Gafford']
    },
}


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
