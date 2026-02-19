# -*- coding: utf-8 -*-
"""
CourtMind Stats Fetcher
=======================
Fetches real NBA box scores and player stats for grading predictions.
"""

import pandas as pd
from datetime import datetime, timedelta
from nba_api.stats.endpoints import scoreboardv2, boxscoretraditionalv3
from nba_api.stats.static import teams
import time


# Team abbreviation mapping
NBA_TEAMS = {team['abbreviation']: team['id'] for team in teams.get_teams()}
TEAM_NAMES = {team['full_name']: team['abbreviation'] for team in teams.get_teams()}


def get_games_for_date(game_date):
    """
    Get all NBA games for a specific date.

    Args:
        game_date: str in format 'YYYY-MM-DD' or datetime object

    Returns:
        List of game dictionaries with game_id, home_team, away_team, scores
    """
    if isinstance(game_date, str):
        game_date = datetime.strptime(game_date, '%Y-%m-%d')

    date_str = game_date.strftime('%m/%d/%Y')

    try:
        scoreboard = scoreboardv2.ScoreboardV2(game_date=date_str)
        games_df = scoreboard.get_data_frames()[0]  # GameHeader
        line_score = scoreboard.get_data_frames()[1]  # LineScore

        games = []
        for _, game in games_df.iterrows():
            game_id = game['GAME_ID']

            # Get scores from line_score
            game_lines = line_score[line_score['GAME_ID'] == game_id]

            home_row = game_lines[game_lines['TEAM_ID'] == game['HOME_TEAM_ID']]
            away_row = game_lines[game_lines['TEAM_ID'] == game['VISITOR_TEAM_ID']]

            games.append({
                'game_id': game_id,
                'home_team': home_row['TEAM_ABBREVIATION'].values[0] if len(home_row) > 0 else None,
                'away_team': away_row['TEAM_ABBREVIATION'].values[0] if len(away_row) > 0 else None,
                'home_score': int(home_row['PTS'].values[0]) if len(home_row) > 0 and pd.notna(home_row['PTS'].values[0]) else None,
                'away_score': int(away_row['PTS'].values[0]) if len(away_row) > 0 and pd.notna(away_row['PTS'].values[0]) else None,
                'status': game['GAME_STATUS_TEXT']
            })

        return games

    except Exception as e:
        print(f"Error fetching games for {date_str}: {e}")
        return []


def get_player_stats_for_game(game_id):
    """
    Get all player stats for a specific game using V3 API.

    Args:
        game_id: NBA game ID string

    Returns:
        DataFrame with player stats
    """
    try:
        time.sleep(0.6)  # Rate limiting
        boxscore = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
        data = boxscore.get_dict()

        # V3 returns nested structure
        players_data = []
        if 'boxScoreTraditional' in data:
            # Get home and away players
            for team_key in ['homeTeam', 'awayTeam']:
                team_data = data['boxScoreTraditional'].get(team_key, {})
                team_abbrev = team_data.get('teamTricode', '')

                for player in team_data.get('players', []):
                    stats = player.get('statistics', {})
                    # Build full name from firstName + familyName
                    first_name = player.get('firstName', '')
                    last_name = player.get('familyName', '')
                    full_name = f"{first_name} {last_name}".strip()

                    players_data.append({
                        'player': full_name,
                        'team': team_abbrev,
                        'minutes': stats.get('minutes', '0:00'),
                        'points': stats.get('points', 0),
                        'rebounds': stats.get('reboundsTotal', 0),
                        'assists': stats.get('assists', 0),
                        'threes': stats.get('threePointersMade', 0),
                        'steals': stats.get('steals', 0),
                        'blocks': stats.get('blocks', 0),
                        'turnovers': stats.get('turnovers', 0),
                    })

        stats = pd.DataFrame(players_data)

        # Convert numeric columns
        numeric_cols = ['points', 'rebounds', 'assists', 'threes', 'steals', 'blocks', 'turnovers']
        for col in numeric_cols:
            if col in stats.columns:
                stats[col] = pd.to_numeric(stats[col], errors='coerce').fillna(0).astype(int)

        return stats

    except Exception as e:
        print(f"Error fetching box score for game {game_id}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def get_all_player_stats_for_date(game_date):
    """
    Get all player stats for all games on a specific date.

    Args:
        game_date: str in format 'YYYY-MM-DD' or datetime object

    Returns:
        DataFrame with all player stats for the date
    """
    games = get_games_for_date(game_date)

    if not games:
        print(f"No games found for {game_date}")
        return pd.DataFrame(), {}

    all_stats = []
    game_results = {}

    for game in games:
        if game['status'] != 'Final':
            print(f"Skipping {game['away_team']} @ {game['home_team']} - not final ({game['status']})")
            continue

        print(f"Fetching stats for {game['away_team']} @ {game['home_team']}...")

        # Store game result
        matchup = f"{game['away_team']} @ {game['home_team']}"
        game_results[matchup] = {
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'home_score': game['home_score'],
            'away_score': game['away_score'],
            'total': (game['home_score'] or 0) + (game['away_score'] or 0),
            'margin': (game['home_score'] or 0) - (game['away_score'] or 0)  # Positive = home won
        }

        stats = get_player_stats_for_game(game['game_id'])
        if not stats.empty:
            stats['game_date'] = game_date if isinstance(game_date, str) else game_date.strftime('%Y-%m-%d')
            stats['matchup'] = matchup
            all_stats.append(stats)

    if all_stats:
        return pd.concat(all_stats, ignore_index=True), game_results

    return pd.DataFrame(), {}


def fetch_stats_for_grading(game_date):
    """
    Fetch all stats needed for grading predictions.

    Args:
        game_date: str in format 'YYYY-MM-DD'

    Returns:
        dict with 'player_stats' DataFrame and 'game_results' dict
    """
    print(f"\n=== Fetching NBA Stats for {game_date} ===\n")

    player_stats, game_results = get_all_player_stats_for_date(game_date)

    return {
        'player_stats': player_stats,
        'game_results': game_results,
        'date': game_date
    }


if __name__ == "__main__":
    # Test with yesterday's date
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"Testing stats fetch for {yesterday}")

    result = fetch_stats_for_grading(yesterday)

    if not result['player_stats'].empty:
        print(f"\nFetched {len(result['player_stats'])} player stat lines")
        print(f"Games: {len(result['game_results'])}")
        print("\nSample player stats:")
        print(result['player_stats'].head(10))
        print("\nGame results:")
        for matchup, scores in result['game_results'].items():
            print(f"  {matchup}: {scores['away_score']}-{scores['home_score']}")
    else:
        print("No stats found")
