# -*- coding: utf-8 -*-
"""
CourtMind Lineup Fetcher
========================
Fetches today's starting lineups from NBA.com and other sources.
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import json
import os

CACHE_FILE = 'C:/Users/user/CourtMind/lineups_cache.json'

# Default minutes by position (when actual data unavailable)
DEFAULT_MINUTES = {
    'starter': 32,
    'sixth_man': 24,
    'rotation': 18,
    'bench': 10
}

# Team rosters with positions and typical roles
# This gets updated by the lineup scraper
TEAM_ROSTERS = {}

# Full team names to abbreviations
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
    # Handle variations
    'Los Angeles Clippers': 'LAC', 'L.A. Clippers': 'LAC', 'L.A. Lakers': 'LAL'
}

ABBREV_TO_FULL = {v: k for k, v in TEAM_ABBREV.items() if 'L.A.' not in k and 'Los Angeles Clippers' not in k}


def fetch_nba_lineups():
    """
    Fetch today's lineups from NBA.com API.
    Returns dict of games with starters for each team.
    """
    # NBA.com stats API for today's games
    today = datetime.now().strftime('%Y-%m-%d')

    url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Referer': 'https://www.nba.com/'
    }

    games = []

    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()

            # Find today's games
            for date_group in data.get('leagueSchedule', {}).get('gameDates', []):
                game_date = date_group.get('gameDate', '')[:10]
                if game_date == today:
                    for game in date_group.get('games', []):
                        home = game.get('homeTeam', {})
                        away = game.get('awayTeam', {})

                        games.append({
                            'game_id': game.get('gameId'),
                            'game_time': game.get('gameDateTimeUTC'),
                            'home_team': home.get('teamTricode', ''),
                            'away_team': away.get('teamTricode', ''),
                            'home_name': home.get('teamName', ''),
                            'away_name': away.get('teamName', ''),
                        })
                    break
    except Exception as e:
        print(f"Error fetching NBA schedule: {e}")

    return games


def fetch_rotowire_lineups():
    """
    Scrape projected lineups from RotoWire (backup source).
    """
    url = "https://www.rotowire.com/basketball/nba-lineups.php"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    lineups = {}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find lineup boxes
            lineup_boxes = soup.find_all('div', class_='lineup__box')

            for box in lineup_boxes:
                # Get team name
                team_elem = box.find('div', class_='lineup__team')
                if not team_elem:
                    continue

                team_link = team_elem.find('a')
                if team_link:
                    team_name = team_link.get_text(strip=True)
                    team_abbrev = TEAM_ABBREV.get(team_name, team_name[:3].upper())
                else:
                    continue

                # Get players
                players = []
                player_elems = box.find_all('li', class_='lineup__player')

                for i, player_elem in enumerate(player_elems[:5]):  # First 5 are starters
                    player_link = player_elem.find('a')
                    if player_link:
                        player_name = player_link.get_text(strip=True)
                        pos_elem = player_elem.find('div', class_='lineup__pos')
                        pos = pos_elem.get_text(strip=True) if pos_elem else ['PG', 'SG', 'SF', 'PF', 'C'][i]

                        players.append({
                            'name': player_name,
                            'position': pos,
                            'role': 'starter',
                            'projected_minutes': DEFAULT_MINUTES['starter']
                        })

                if players:
                    lineups[team_abbrev] = {
                        'starters': players,
                        'bench': [],
                        'source': 'rotowire'
                    }

    except Exception as e:
        print(f"Error fetching RotoWire lineups: {e}")

    return lineups


def get_team_roster_from_data(df, team):
    """
    Build team roster from our game data.
    Returns list of players with their avg minutes.
    """
    team_games = df[df['team'] == team].copy()
    if len(team_games) == 0:
        return []

    # Get recent games (last 10)
    recent_game_ids = team_games['game_id'].unique()[-10:]
    recent = team_games[team_games['game_id'].isin(recent_game_ids)]

    # Aggregate player stats
    player_stats = recent.groupby('player').agg({
        'minutes': 'mean',
        'pts': 'mean',
        'game_id': 'nunique'
    }).reset_index()

    player_stats.columns = ['name', 'avg_minutes', 'avg_pts', 'games']
    player_stats = player_stats.sort_values('avg_minutes', ascending=False)

    roster = []
    for i, row in player_stats.iterrows():
        if row['games'] >= 3:  # Played at least 3 of last 10
            role = 'starter' if row['avg_minutes'] >= 25 else 'rotation' if row['avg_minutes'] >= 15 else 'bench'
            roster.append({
                'name': row['name'],
                'avg_minutes': round(row['avg_minutes'], 1),
                'avg_pts': round(row['avg_pts'], 1),
                'games': row['games'],
                'role': role
            })

    return roster[:15]  # Top 15 by minutes


def get_todays_lineups(df=None):
    """
    Get today's games with projected lineups.
    Combines NBA.com schedule with RotoWire lineups.
    """
    # Check cache first (valid for 1 hour)
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache = json.load(f)
                cache_time = datetime.fromisoformat(cache.get('timestamp', '2000-01-01'))
                if datetime.now() - cache_time < timedelta(hours=1):
                    return cache.get('data', {})
        except:
            pass

    # Fetch fresh data
    games = fetch_nba_lineups()
    lineups = fetch_rotowire_lineups()

    result = {
        'games': [],
        'updated': datetime.now().isoformat()
    }

    for game in games:
        home = game['home_team']
        away = game['away_team']

        game_data = {
            'game_id': game['game_id'],
            'game_time': game['game_time'],
            'home_team': home,
            'away_team': away,
            'home_lineup': lineups.get(home, {'starters': [], 'bench': []}),
            'away_lineup': lineups.get(away, {'starters': [], 'bench': []})
        }

        # If we have our data, enhance with our stats
        if df is not None:
            game_data['home_roster'] = get_team_roster_from_data(df, home)
            game_data['away_roster'] = get_team_roster_from_data(df, away)

        result['games'].append(game_data)

    # Cache results
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump({'timestamp': datetime.now().isoformat(), 'data': result}, f)
    except:
        pass

    return result


def calculate_team_projection(roster, predictor, opponent, minutes_overrides=None):
    """
    Calculate team total projection based on lineup and minutes.

    Args:
        roster: List of player dicts with 'name' and 'minutes'
        predictor: PlayerPredictor instance
        opponent: Opponent team abbreviation
        minutes_overrides: Dict of player_name -> custom minutes

    Returns:
        Dict with team projection details
    """
    minutes_overrides = minutes_overrides or {}

    total_pts = 0
    total_reb = 0
    total_ast = 0
    player_projections = []

    for player in roster:
        name = player['name']
        base_minutes = player.get('avg_minutes', player.get('projected_minutes', 25))
        minutes = minutes_overrides.get(name, base_minutes)

        if minutes <= 0:
            continue

        # Get player prediction
        pred = predictor.predict(name, opponent)

        if pred:
            # Scale by minutes ratio (assuming prediction is for ~32 min)
            scale = minutes / 32.0

            pts = pred['pts'] * scale
            reb = pred['reb'] * scale
            ast = pred['ast'] * scale

            total_pts += pts
            total_reb += reb
            total_ast += ast

            player_projections.append({
                'name': name,
                'minutes': minutes,
                'pts': round(pts, 1),
                'reb': round(reb, 1),
                'ast': round(ast, 1),
                'base_projection': pred['pts'],
                'confidence': pred['confidence']
            })

    return {
        'total_pts': round(total_pts, 1),
        'total_reb': round(total_reb, 1),
        'total_ast': round(total_ast, 1),
        'players': player_projections
    }


if __name__ == "__main__":
    print("Testing lineup fetcher...")

    games = fetch_nba_lineups()
    print(f"\nFound {len(games)} games today:")
    for g in games:
        print(f"  {g['away_team']} @ {g['home_team']}")

    lineups = fetch_rotowire_lineups()
    print(f"\nGot lineups for {len(lineups)} teams")
    for team, data in list(lineups.items())[:3]:
        print(f"\n{team} starters:")
        for p in data['starters']:
            print(f"  {p['position']}: {p['name']}")
