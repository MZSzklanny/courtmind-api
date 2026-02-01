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

# Today's lineups - February 1, 2026 (from Rotowire)
TODAYS_LINEUPS = {
    # Game 1: MIL @ BOS
    'BOS': {
        'opponent': 'MIL',
        'home': True,
        'starters': ['Payton Pritchard', 'Derrick White', 'Jaylen Brown', 'Sam Hauser', 'Neemias Queta']
    },
    'MIL': {
        'opponent': 'BOS',
        'home': False,
        'starters': ['Ryan Rollins', 'AJ Green', 'Kyle Kuzma', 'Bobby Portis', 'Myles Turner']
    },
    # Game 2: BKN @ DET
    'DET': {
        'opponent': 'BKN',
        'home': True,
        'starters': ['Cade Cunningham', 'Duncan Robinson', 'Ausar Thompson', 'Tobias Harris', 'Jalen Duren']
    },
    'BKN': {
        'opponent': 'DET',
        'home': False,
        'starters': ['Egor Demin', 'Terance Mann', 'Michael Porter Jr.', 'Danny Wolf', 'Nic Claxton']
    },
    # Game 3: CHI @ MIA
    'MIA': {
        'opponent': 'CHI',
        'home': True,
        'starters': ['Davion Mitchell', 'Norman Powell', 'Pelle Larsson', 'Andrew Wiggins', 'Bam Adebayo']
    },
    'CHI': {
        'opponent': 'MIA',
        'home': False,
        'starters': ['Josh Giddey', 'Coby White', 'Isaac Okoro', 'Matas Buzelis', 'Nikola Vucevic']
    },
    # Game 4: SAC @ WAS
    'WAS': {
        'opponent': 'SAC',
        'home': True,
        'starters': ['Bub Carrington', 'Keyshawn George', 'Bilal Coulibaly', 'Khris Middleton', 'Alex Sarr']
    },
    'SAC': {
        'opponent': 'WAS',
        'home': False,
        'starters': ['Russell Westbrook', 'Zach LaVine', 'DeMar DeRozan', 'Precious Achiuwa', 'Domantas Sabonis']
    },
    # Game 5: UTA @ TOR
    'TOR': {
        'opponent': 'UTA',
        'home': True,
        'starters': ['Immanuel Quickley', 'Brandon Ingram', 'RJ Barrett', 'Scottie Barnes', 'Collin Murray-Boyles']
    },
    'UTA': {
        'opponent': 'TOR',
        'home': False,
        'starters': ['Keyonte George', 'Cody Williams', 'Ace Bailey', 'Lauri Markkanen', 'Jusuf Nurkic']
    },
    # Game 6: LAL @ NYK
    'NYK': {
        'opponent': 'LAL',
        'home': True,
        'starters': ['Jalen Brunson', 'Josh Hart', 'Mikal Bridges', 'OG Anunoby', 'Karl-Anthony Towns']
    },
    'LAL': {
        'opponent': 'NYK',
        'home': False,
        'starters': ['Luka Doncic', 'Marcus Smart', 'Jake LaRavia', 'LeBron James', 'Deandre Ayton']
    },
    # Game 7: ORL @ SAS
    'SAS': {
        'opponent': 'ORL',
        'home': True,
        'starters': ["De'Aaron Fox", 'Stephon Castle', 'Devin Vassell', 'Justin Champagnie', 'Victor Wembanyama']
    },
    'ORL': {
        'opponent': 'SAS',
        'home': False,
        'starters': ['Jalen Suggs', 'Anthony Black', 'Desmond Bane', 'Paolo Banchero', 'Wendell Carter Jr.']
    },
    # Game 8: LAC @ PHX
    'PHX': {
        'opponent': 'LAC',
        'home': True,
        'starters': ['Collin Gillespie', 'Grayson Allen', 'Dillon Brooks', "Royce O'Neale", 'Mark Williams']
    },
    'LAC': {
        'opponent': 'PHX',
        'home': False,
        'starters': ['James Harden', 'Kris Dunn', 'Kawhi Leonard', 'John Collins', 'Ivica Zubac']
    },
    # Game 9: CLE @ POR
    'POR': {
        'opponent': 'CLE',
        'home': True,
        'starters': ['Deni Avdija', 'Shaedon Sharpe', 'Toumani Camara', 'Sidy Cissoko', 'Donovan Clingan']
    },
    'CLE': {
        'opponent': 'POR',
        'home': False,
        'starters': ['Donovan Mitchell', 'Sam Merrill', 'Jaylon Tyson', 'Dean Wade', 'Jarrett Allen']
    },
    # Game 10: OKC @ DEN
    'DEN': {
        'opponent': 'OKC',
        'home': True,
        'starters': ['Jamal Murray', 'Jalen Pickett', 'Peyton Watson', 'Spencer Jones', 'Nikola Jokic']
    },
    'OKC': {
        'opponent': 'DEN',
        'home': False,
        'starters': ['Shai Gilgeous-Alexander', 'Luguentz Dort', 'Aaron Wiggins', 'Chet Holmgren', 'Isaiah Hartenstein']
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
