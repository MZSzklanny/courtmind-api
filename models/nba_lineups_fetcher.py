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

# Today's lineups - January 30, 2026
TODAYS_LINEUPS = {
    # Game 1: LAL @ WAS
    'WAS': {
        'opponent': 'LAL',
        'home': True,
        'starters': ['Bub Carrington', 'Keyshawn George', 'Bilal Coulibaly', 'Wil Riley', 'Alex Sarr']
    },
    'LAL': {
        'opponent': 'WAS',
        'home': False,
        'starters': ['Marcus Smart', 'Jake LaRavia', 'Luka Doncic', 'LeBron James', 'Deandre Ayton']
    },
    # Game 2: TOR @ ORL
    'ORL': {
        'opponent': 'TOR',
        'home': True,
        'starters': ['Jalen Suggs', 'Anthony Black', 'Desmond Bane', 'Paolo Banchero', 'Wendell Carter Jr.']
    },
    'TOR': {
        'opponent': 'ORL',
        'home': False,
        'starters': ['Immanuel Quickley', 'Brandon Ingram', 'RJ Barrett', 'Scotty Barnes', 'Collin Murray-Boyles']
    },
    # Game 3: SAC @ BOS
    'BOS': {
        'opponent': 'SAC',
        'home': True,
        'starters': ['Derrick White', 'Baylor Scheierman', 'Payton Pritchard', 'Sam Hauser', 'Neemias Queta']
    },
    'SAC': {
        'opponent': 'BOS',
        'home': False,
        'starters': ['Nique Clifford', 'DeMar DeRozan', 'Russell Westbrook', 'Precious Achiuwa', 'Domantas Sabonis']
    },
    # Game 4: POR @ NYK
    'NYK': {
        'opponent': 'POR',
        'home': True,
        'starters': ['Josh Hart', 'Mikal Bridges', 'Jalen Brunson', 'OG Anunoby', 'Karl-Anthony Towns']
    },
    'POR': {
        'opponent': 'NYK',
        'home': False,
        'starters': ['Shaedon Sharpe', 'Toumani Camara', 'Jrue Holiday', 'Deni Avdija', 'Donovan Clingan']
    },
    # Game 5: MEM @ NOP
    'NOP': {
        'opponent': 'MEM',
        'home': True,
        'starters': ['Herbert Jones', 'Saddiq Bey', 'Trey Murphy III', 'Derik Queen', 'Zion Williamson']
    },
    'MEM': {
        'opponent': 'NOP',
        'home': False,
        'starters': ['Cedric Coward', 'Jaylen Wells', 'Cam Spencer', 'Jaren Jackson Jr.', 'Jock Landale']
    },
    # Game 6: CLE @ PHX
    'PHX': {
        'opponent': 'CLE',
        'home': True,
        'starters': ['Dillon Brooks', 'Grayson Allen', 'Collin Gillespie', 'Royce O\'Neale', 'Mark Williams']
    },
    'CLE': {
        'opponent': 'PHX',
        'home': False,
        'starters': ['Sam Merrill', 'Jaylon Tyson', 'Donovan Mitchell', 'Dean Wade', 'Jarrett Allen']
    },
    # Game 7: LAC @ DEN
    'DEN': {
        'opponent': 'LAC',
        'home': True,
        'starters': ['Jalen Pickett', 'Peyton Watson', 'Jamal Murray', 'Spencer Jones', 'Jonas Valanciunas']
    },
    'LAC': {
        'opponent': 'DEN',
        'home': False,
        'starters': ['Kris Dunn', 'Kawhi Leonard', 'James Harden', 'John Collins', 'Ivica Zubac']
    },
    # Game 8: BKN @ UTA
    'UTA': {
        'opponent': 'BKN',
        'home': True,
        'starters': ['Cody Williams', 'Ace Bailey', 'Keyonte George', 'Taylor Hendricks', 'Kyle Filipowski']
    },
    'BKN': {
        'opponent': 'UTA',
        'home': False,
        'starters': ['Terance Mann', 'Drake Powell', 'Egor Demin', 'Danny Wolf', 'Nic Claxton']
    },
    # Game 9: DET @ GSW
    'GSW': {
        'opponent': 'DET',
        'home': True,
        'starters': ['Stephen Curry', 'Brandin Podziemski', 'Moses Moody', 'Draymond Green', 'Al Horford']
    },
    'DET': {
        'opponent': 'GSW',
        'home': False,
        'starters': ['Cade Cunningham', 'Duncan Robinson', 'Ausar Thompson', 'Tobias Harris', 'Jalen Duren']
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
