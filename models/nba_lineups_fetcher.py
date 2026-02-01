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

# Today's lineups - February 1, 2026 (from Rotowire) - UPDATED with injury changes
TODAYS_LINEUPS = {
    # Game 1: MIL @ BOS (3:30 PM ET) - Both Confirmed
    'BOS': {
        'opponent': 'MIL',
        'home': True,
        'starters': ['Payton Pritchard', 'Derrick White', 'Jaylen Brown', 'Sam Hauser', 'Neemias Queta']
        # OUT: Jayson Tatum
    },
    'MIL': {
        'opponent': 'BOS',
        'home': False,
        'starters': ['Ryan Rollins', 'AJ Green', 'Kyle Kuzma', 'Bobby Portis', 'Myles Turner']
        # OUT: Giannis Antetokounmpo, Gary Harris, Kevin Porter, Taurean Prince
    },
    # Game 2: BKN @ DET (6:00 PM ET) - Both Confirmed
    'DET': {
        'opponent': 'BKN',
        'home': True,
        'starters': ['Cade Cunningham', 'Duncan Robinson', 'Ausar Thompson', 'Tobias Harris', 'Jalen Duren']
        # OUT: Caris LeVert
    },
    'BKN': {
        'opponent': 'DET',
        'home': False,
        'starters': ['Egor Demin', 'Nolan Traore', 'Terance Mann', 'Danny Wolf', 'Nic Claxton']
        # OUT: Noah Clowney, Haywood Highsmith, Michael Porter Jr., Ziaire Williams
    },
    # Game 3: CHI @ MIA (6:00 PM ET) - Both Confirmed
    'MIA': {
        'opponent': 'CHI',
        'home': True,
        'starters': ['Davion Mitchell', 'Pelle Larsson', 'Sandro Fontecchio', 'Andrew Wiggins', 'Bam Adebayo']
        # OUT: Tyler Herro, Nikola Jovic, Norman Powell, Terry Rozier
    },
    'CHI': {
        'opponent': 'MIA',
        'home': False,
        'starters': ['Coby White', 'Ayo Dosunmu', 'Isaac Okoro', 'Matas Buzelis', 'Nikola Vucevic']
        # OUT: Zach Collins, Josh Giddey, Kevin Huerter, Tre Jones, Jalen Smith
    },
    # Game 4: SAC @ WAS (6:00 PM ET) - Both Confirmed
    'WAS': {
        'opponent': 'SAC',
        'home': True,
        'starters': ['Bub Carrington', 'Keyshawn George', 'Bilal Coulibaly', 'Khris Middleton', 'Marvin Bagley III']
        # OUT: Tristan Johnson, Alex Sarr, Tristan Vukcevic, Trae Young
    },
    'SAC': {
        'opponent': 'WAS',
        'home': False,
        'starters': ['Nick Clifford', 'Zach LaVine', 'DeMar DeRozan', 'Precious Achiuwa', 'Mason Raynaud']
        # OUT: De'Andre Hunter, Keegan Murray, Domantas Sabonis, Russell Westbrook
    },
    # Game 5: UTA @ TOR (6:00 PM ET) - Both Confirmed
    'TOR': {
        'opponent': 'UTA',
        'home': True,
        'starters': ['Immanuel Quickley', 'Brandon Ingram', 'RJ Barrett', 'Scottie Barnes', 'Collin Murray-Boyles']
        # OUT: Jakob Poeltl
    },
    'UTA': {
        'opponent': 'TOR',
        'home': False,
        'starters': ['Isaiah Collier', 'Cody Williams', 'Ace Bailey', 'Lauri Markkanen', 'Jusuf Nurkic']
        # OUT: Keyonte George, E. Harkless, Georges Niang
    },
    # Game 6: LAL @ NYK (7:00 PM ET) - Expected
    'NYK': {
        'opponent': 'LAL',
        'home': True,
        'starters': ['Jalen Brunson', 'Josh Hart', 'Mikal Bridges', 'OG Anunoby', 'Karl-Anthony Towns']
        # OUT: Miles McBride
    },
    'LAL': {
        'opponent': 'NYK',
        'home': False,
        'starters': ['Luka Doncic', 'Marcus Smart', 'Jake LaRavia', 'LeBron James', 'Deandre Ayton']
        # Questionable: Austin Reaves | OUT: Bronny James, A. Thierry
    },
    # Game 7: LAC @ PHX (8:00 PM ET) - Expected
    'PHX': {
        'opponent': 'LAC',
        'home': True,
        'starters': ['Collin Gillespie', 'Grayson Allen', 'Dillon Brooks', "Royce O'Neale", 'Mark Williams']
        # OUT: Devin Booker, Josh Green
    },
    'LAC': {
        'opponent': 'PHX',
        'home': False,
        'starters': ['Kris Dunn', 'Jordan Miller', 'Kawhi Leonard', 'John Collins', 'Ivica Zubac']
        # OUT: James Harden, Derrick Jones Jr., Chris Paul, Terance Washington
    },
    # Game 8: CLE @ POR (9:00 PM ET) - Expected
    'POR': {
        'opponent': 'CLE',
        'home': True,
        'starters': ['Deni Avdija', 'Shaedon Sharpe', 'Toumani Camara', 'Sidy Cissoko', 'Donovan Clingan']
        # Questionable: Deni Avdija, Blake Wesley, Robert Williams III
        # OUT: Scoot Henderson, Jrue Holiday, Vit Krejci, Keegan Murray, Matisse Thybulle
    },
    'CLE': {
        'opponent': 'POR',
        'home': False,
        'starters': ['Donovan Mitchell', 'Sam Merrill', 'Jaylon Tyson', 'Dean Wade', 'Jarrett Allen']
        # Questionable: Craig Porter Jr. | OUT: Keon Ellis, Darius Garland, Evan Mobley, Dennis Schroder, Max Strus
    },
    # Game 9: ORL @ SAS (9:00 PM ET) - Expected
    'SAS': {
        'opponent': 'ORL',
        'home': True,
        'starters': ["De'Aaron Fox", 'Stephon Castle', 'Devin Vassell', 'Justin Champagnie', 'Victor Wembanyama']
        # Questionable: Stephon Castle, Victor Wembanyama
    },
    'ORL': {
        'opponent': 'SAS',
        'home': False,
        'starters': ['Jalen Suggs', 'Anthony Black', 'Desmond Bane', 'Paolo Banchero', 'Wendell Carter Jr.']
    },
    # Game 10: OKC @ DEN (9:30 PM ET) - Expected
    'DEN': {
        'opponent': 'OKC',
        'home': True,
        'starters': ['Jamal Murray', 'Jalen Pickett', 'Peyton Watson', 'Spencer Jones', 'Nikola Jokic']
        # Probable: Jamal Murray, Nikola Jokic
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
