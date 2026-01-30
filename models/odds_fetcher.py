# -*- coding: utf-8 -*-
"""
CourtMind Odds Fetcher
======================
Fetch real-time odds from DraftKings and FanDuel via The Odds API
"""

import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
import os

# Get base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Cache file for odds (avoid hitting API too often)
CACHE_FILE = BASE_DIR / 'odds_cache.json'
CACHE_DURATION = timedelta(hours=1)

# The Odds API
ODDS_API_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

# You can get a free API key at https://the-odds-api.com/
# Free tier: 500 requests/month
API_KEY = os.environ.get('ODDS_API_KEY', '')


def get_api_key():
    """Get API key from environment or file."""
    global API_KEY
    if API_KEY:
        return API_KEY

    # Try to read from file
    key_file = BASE_DIR / 'odds_api_key.txt'
    if key_file.exists():
        API_KEY = key_file.read_text().strip()
        return API_KEY

    return None


def load_cache():
    """Load cached odds data."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                cache = json.load(f)
                cache_time = datetime.fromisoformat(cache.get('timestamp', '2000-01-01'))
                if datetime.now() - cache_time < CACHE_DURATION:
                    return cache.get('data', {})
        except:
            pass
    return None


def save_cache(data):
    """Save odds data to cache."""
    cache = {
        'timestamp': datetime.now().isoformat(),
        'data': data
    }
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)


def fetch_game_odds():
    """
    Fetch NBA game odds (spreads, totals, moneylines) from DK and FD.

    Returns:
        dict: Game odds keyed by matchup
    """
    # Check cache first
    cached = load_cache()
    if cached and 'games' in cached:
        return cached

    api_key = get_api_key()
    if not api_key:
        return {'games': [], 'error': 'No API key. Get one free at the-odds-api.com'}

    try:
        # Fetch odds
        url = f"{ODDS_API_URL}/sports/{SPORT}/odds"
        params = {
            'apiKey': api_key,
            'regions': 'us',
            'markets': 'spreads,totals,h2h',
            'bookmakers': 'draftkings,fanduel',
            'oddsFormat': 'american'
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 401:
            return {'games': [], 'error': 'Invalid API key'}
        elif response.status_code == 429:
            return {'games': [], 'error': 'API rate limit reached'}
        elif response.status_code != 200:
            return {'games': [], 'error': f'API error: {response.status_code}'}

        games_data = response.json()

        # Parse into cleaner format
        games = []
        for game in games_data:
            game_info = {
                'id': game['id'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'commence_time': game['commence_time'],
                'bookmakers': {}
            }

            for bookmaker in game.get('bookmakers', []):
                book_key = bookmaker['key']
                book_data = {'title': bookmaker['title']}

                for market in bookmaker.get('markets', []):
                    market_key = market['key']
                    outcomes = {o['name']: o for o in market['outcomes']}
                    book_data[market_key] = outcomes

                game_info['bookmakers'][book_key] = book_data

            games.append(game_info)

        result = {'games': games, 'updated': datetime.now().isoformat()}
        save_cache(result)
        return result

    except requests.Timeout:
        return {'games': [], 'error': 'API timeout'}
    except Exception as e:
        return {'games': [], 'error': str(e)}


def fetch_player_props():
    """
    Fetch NBA player props from DK and FD.

    Returns:
        dict: Player prop odds organized by player
    """
    api_key = get_api_key()
    if not api_key:
        return {'props': {}, 'error': 'No API key'}

    # Check cache
    props_cache = BASE_DIR / 'props_cache.json'
    if props_cache.exists():
        try:
            with open(props_cache, 'r') as f:
                cache = json.load(f)
                cache_time = datetime.fromisoformat(cache.get('timestamp', '2000-01-01'))
                if datetime.now() - cache_time < CACHE_DURATION:
                    return cache.get('data', {})
        except:
            pass

    try:
        # First get today's events
        events_url = f"{ODDS_API_URL}/sports/{SPORT}/events"
        params = {'apiKey': api_key}

        response = requests.get(events_url, params=params, timeout=10)
        if response.status_code != 200:
            return {'props': {}, 'error': f'API error: {response.status_code}'}

        events = response.json()

        all_props = {}
        markets = [
            'player_points', 'player_rebounds', 'player_assists',
            'player_threes', 'player_points_rebounds_assists',
            'player_steals', 'player_blocks'
        ]

        for event in events[:8]:  # Limit to avoid rate limits
            event_id = event['id']
            home_team = event.get('home_team', '')
            away_team = event.get('away_team', '')

            # Fetch props for this event
            props_url = f"{ODDS_API_URL}/sports/{SPORT}/events/{event_id}/odds"
            props_params = {
                'apiKey': api_key,
                'regions': 'us',
                'markets': ','.join(markets),
                'bookmakers': 'draftkings,fanduel',
                'oddsFormat': 'american'
            }

            try:
                props_response = requests.get(props_url, params=props_params, timeout=15)
                if props_response.status_code == 200:
                    props_data = props_response.json()

                    # Parse bookmakers
                    for bookmaker in props_data.get('bookmakers', []):
                        book_key = bookmaker['key']

                        for market in bookmaker.get('markets', []):
                            market_key = market['key']

                            for outcome in market.get('outcomes', []):
                                player_name = outcome.get('description', '')
                                if not player_name:
                                    continue

                                if player_name not in all_props:
                                    all_props[player_name] = {
                                        'player': player_name,
                                        'game': f"{away_team} @ {home_team}",
                                        'props': {}
                                    }

                                prop_type = market_key.replace('player_', '')
                                if prop_type not in all_props[player_name]['props']:
                                    all_props[player_name]['props'][prop_type] = {}

                                # Over or Under
                                side = outcome.get('name', 'Over')
                                line = outcome.get('point', 0)
                                odds = outcome.get('price', -110)

                                book_short = 'dk' if book_key == 'draftkings' else 'fd'

                                if book_short not in all_props[player_name]['props'][prop_type]:
                                    all_props[player_name]['props'][prop_type][book_short] = {}

                                all_props[player_name]['props'][prop_type][book_short][side.lower()] = {
                                    'line': line,
                                    'odds': odds
                                }

            except Exception as e:
                print(f"Error fetching props for event {event_id}: {e}")
                continue

        result = {'props': all_props, 'updated': datetime.now().isoformat()}

        # Cache results
        with open(props_cache, 'w') as f:
            json.dump({'timestamp': datetime.now().isoformat(), 'data': result}, f)

        return result

    except Exception as e:
        return {'props': {}, 'error': str(e)}


def get_player_prop_line(player_name, prop_type='points'):
    """
    Get a specific player's prop line.

    Args:
        player_name: Player name
        prop_type: 'points', 'rebounds', 'assists', 'threes', 'points_rebounds_assists'

    Returns:
        dict with dk and fd lines
    """
    props_data = fetch_player_props()

    if 'error' in props_data:
        return None

    # Try exact match first, then fuzzy
    props = props_data.get('props', {})

    # Exact match
    if player_name in props:
        player_props = props[player_name].get('props', {})
        if prop_type in player_props:
            return player_props[prop_type]

    # Fuzzy match (last name)
    last_name = player_name.split()[-1].lower()
    for name, data in props.items():
        if last_name in name.lower():
            player_props = data.get('props', {})
            if prop_type in player_props:
                return player_props[prop_type]

    return None


def get_all_player_props():
    """Get all available player props formatted for display."""
    props_data = fetch_player_props()

    if 'error' in props_data and props_data.get('error'):
        return {'error': props_data['error'], 'players': []}

    players = []
    for player_name, data in props_data.get('props', {}).items():
        player_info = {
            'name': player_name,
            'game': data.get('game', ''),
            'lines': {}
        }

        for prop_type, books in data.get('props', {}).items():
            player_info['lines'][prop_type] = {
                'dk': books.get('dk', {}).get('over', {}).get('line', '-'),
                'fd': books.get('fd', {}).get('over', {}).get('line', '-'),
                'dk_odds': books.get('dk', {}).get('over', {}).get('odds', -110),
                'fd_odds': books.get('fd', {}).get('over', {}).get('odds', -110),
            }

        if player_info['lines']:
            players.append(player_info)

    return {'players': players, 'updated': props_data.get('updated', '')}


def get_game_line(home_team, away_team, odds_data):
    """
    Extract odds for a specific game.

    Args:
        home_team: Home team name (e.g., "Oklahoma City Thunder")
        away_team: Away team name
        odds_data: Data from fetch_game_odds()

    Returns:
        dict with DK and FD odds
    """
    if not odds_data or 'games' not in odds_data:
        return None

    for game in odds_data['games']:
        if (home_team in game['home_team'] or game['home_team'] in home_team) and \
           (away_team in game['away_team'] or game['away_team'] in away_team):

            result = {
                'home': game['home_team'],
                'away': game['away_team'],
                'time': game['commence_time'],
                'dk': None,
                'fd': None
            }

            for book_key in ['draftkings', 'fanduel']:
                if book_key in game['bookmakers']:
                    book = game['bookmakers'][book_key]
                    book_short = 'dk' if book_key == 'draftkings' else 'fd'

                    result[book_short] = {
                        'spread': None,
                        'total': None,
                        'moneyline': None
                    }

                    # Spreads
                    if 'spreads' in book:
                        home_spread = book['spreads'].get(game['home_team'], {})
                        result[book_short]['spread'] = {
                            'line': home_spread.get('point', 0),
                            'odds': home_spread.get('price', -110)
                        }

                    # Totals
                    if 'totals' in book:
                        over = book['totals'].get('Over', {})
                        result[book_short]['total'] = {
                            'line': over.get('point', 0),
                            'over_odds': over.get('price', -110),
                            'under_odds': book['totals'].get('Under', {}).get('price', -110)
                        }

                    # Moneyline
                    if 'h2h' in book:
                        result[book_short]['moneyline'] = {
                            'home': book['h2h'].get(game['home_team'], {}).get('price', 0),
                            'away': book['h2h'].get(game['away_team'], {}).get('price', 0)
                        }

            return result

    return None


# Team name mappings (API uses full names)
TEAM_FULL_NAMES = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'Los Angeles Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
}


def get_todays_odds():
    """
    Get all of today's NBA game odds formatted for display.

    Returns:
        list of dicts with game odds
    """
    odds_data = fetch_game_odds()

    if 'error' in odds_data:
        return {'error': odds_data['error'], 'games': []}

    games = []
    for game in odds_data.get('games', []):
        # Parse time
        try:
            game_time = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00'))
        except:
            game_time = None

        game_info = {
            'home': game['home_team'],
            'away': game['away_team'],
            'time': game_time.strftime('%I:%M %p') if game_time else 'TBD',
            'dk': {'spread': '-', 'total': '-', 'ml_home': '-', 'ml_away': '-'},
            'fd': {'spread': '-', 'total': '-', 'ml_home': '-', 'ml_away': '-'}
        }

        for book_key, book_short in [('draftkings', 'dk'), ('fanduel', 'fd')]:
            if book_key in game.get('bookmakers', {}):
                book = game['bookmakers'][book_key]

                # Spread
                if 'spreads' in book:
                    home_spread = book['spreads'].get(game['home_team'], {})
                    line = home_spread.get('point', 0)
                    game_info[book_short]['spread'] = f"{line:+.1f}" if line else '-'

                # Total
                if 'totals' in book:
                    over = book['totals'].get('Over', {})
                    total = over.get('point', 0)
                    game_info[book_short]['total'] = f"{total:.1f}" if total else '-'

                # Moneyline
                if 'h2h' in book:
                    ml_home = book['h2h'].get(game['home_team'], {}).get('price', 0)
                    ml_away = book['h2h'].get(game['away_team'], {}).get('price', 0)
                    game_info[book_short]['ml_home'] = f"{ml_home:+d}" if ml_home else '-'
                    game_info[book_short]['ml_away'] = f"{ml_away:+d}" if ml_away else '-'

        games.append(game_info)

    return {'games': games, 'updated': odds_data.get('updated', '')}


if __name__ == "__main__":
    print("Testing Odds Fetcher...")

    # Check for API key
    api_key = get_api_key()
    if api_key:
        print(f"API Key found: {api_key[:8]}...")
        odds = get_todays_odds()

        if 'error' in odds:
            print(f"Error: {odds['error']}")
        else:
            print(f"\nFound {len(odds['games'])} games:")
            for game in odds['games']:
                print(f"\n{game['away']} @ {game['home']} ({game['time']})")
                print(f"  DK: Spread {game['dk']['spread']}, O/U {game['dk']['total']}")
                print(f"  FD: Spread {game['fd']['spread']}, O/U {game['fd']['total']}")
    else:
        print("No API key found.")
        print("Get a free key at: https://the-odds-api.com/")
        print("Then save it to: C:/Users/user/CourtMind/odds_api_key.txt")
