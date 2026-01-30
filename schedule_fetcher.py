"""
CourtMind Schedule Fetcher
==========================
Fetches tonight's NBA games and identifies B2B situations.
"""

import pandas as pd
from datetime import datetime, timedelta
from nba_api.stats.endpoints import scoreboardv2, leaguegamefinder
from nba_api.stats.static import teams
import time


def get_all_teams():
    """Get dictionary mapping team abbreviation to team info."""
    nba_teams = teams.get_teams()
    return {team['abbreviation']: team for team in nba_teams}


def get_todays_games(date=None):
    """
    Fetch today's NBA games.

    Args:
        date: datetime object or None for today

    Returns:
        List of dicts with game info:
        - game_id, home_team, away_team, game_time, home_team_id, away_team_id
    """
    if date is None:
        date = datetime.now()

    date_str = date.strftime('%Y-%m-%d')

    try:
        time.sleep(0.6)  # Rate limiting
        scoreboard = scoreboardv2.ScoreboardV2(game_date=date_str)
        games_df = scoreboard.get_data_frames()[0]  # GameHeader

        if games_df.empty:
            print(f"[CourtMind] No games found for {date_str}")
            return []

        games = []
        for _, game in games_df.iterrows():
            game_info = {
                'game_id': game['GAME_ID'],
                'game_date': date_str,
                'game_time': game.get('GAME_STATUS_TEXT', 'TBD'),
                'home_team_id': game['HOME_TEAM_ID'],
                'away_team_id': game['VISITOR_TEAM_ID'],
                'home_team': get_team_abbrev(game['HOME_TEAM_ID']),
                'away_team': get_team_abbrev(game['VISITOR_TEAM_ID']),
            }
            games.append(game_info)

        print(f"[CourtMind] Found {len(games)} games for {date_str}")
        return games

    except Exception as e:
        print(f"[CourtMind] Error fetching games: {e}")
        return []


def get_team_abbrev(team_id):
    """Convert team ID to abbreviation."""
    nba_teams = teams.get_teams()
    for team in nba_teams:
        if team['id'] == team_id:
            return team['abbreviation']
    return 'UNK'


def get_team_recent_games(team_abbrev, days=3):
    """
    Get team's games in the last N days.

    Used to identify back-to-back situations.
    """
    try:
        nba_teams = teams.get_teams()
        team_info = next((t for t in nba_teams if t['abbreviation'] == team_abbrev), None)

        if not team_info:
            return []

        time.sleep(0.6)
        finder = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=team_info['id'],
            season_nullable='2025-26',
            season_type_nullable='Regular Season'
        )
        games_df = finder.get_data_frames()[0]

        if games_df.empty:
            return []

        # Parse dates and filter to recent
        games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
        cutoff = datetime.now() - timedelta(days=days)
        recent = games_df[games_df['GAME_DATE'] >= cutoff]

        return recent.to_dict('records')

    except Exception as e:
        print(f"[CourtMind] Error fetching recent games for {team_abbrev}: {e}")
        return []


def identify_b2b_teams(games):
    """
    Identify teams playing on a back-to-back tonight.

    Args:
        games: List of game dicts from get_todays_games()

    Returns:
        Set of team abbreviations on B2B
    """
    b2b_teams = set()
    yesterday = datetime.now() - timedelta(days=1)

    # Get all teams playing today
    teams_today = set()
    for game in games:
        teams_today.add(game['home_team'])
        teams_today.add(game['away_team'])

    # Check each team for yesterday game
    for team in teams_today:
        recent = get_team_recent_games(team, days=2)
        for game in recent:
            game_date = game.get('GAME_DATE')
            if isinstance(game_date, str):
                game_date = pd.to_datetime(game_date)
            if game_date and game_date.date() == yesterday.date():
                b2b_teams.add(team)
                break

    if b2b_teams:
        print(f"[CourtMind] B2B teams tonight: {', '.join(b2b_teams)}")

    return b2b_teams


def get_players_in_games(games):
    """
    Get list of all players expected to play in tonight's games.

    Returns list of dicts with player info.
    """
    from CourtMind.config import DATA_PATHS

    try:
        # Load from our combined data
        df = pd.read_parquet(DATA_PATHS['combined_data'])

        # Get unique players by team
        players = []
        teams_playing = set()
        for game in games:
            teams_playing.add(game['home_team'])
            teams_playing.add(game['away_team'])

        # Get recent players from each team
        for team in teams_playing:
            team_df = df[df['team'] == team]
            if team_df.empty:
                continue

            # Get players with games in last 30 days
            recent = team_df.sort_values('game_date', ascending=False)
            recent_players = recent['player'].unique()[:15]  # Top 15 by recency

            for player in recent_players:
                players.append({
                    'player': player,
                    'team': team,
                })

        return players

    except Exception as e:
        print(f"[CourtMind] Error getting players: {e}")
        return []


if __name__ == "__main__":
    # Test the fetcher
    print("Testing schedule fetcher...")
    games = get_todays_games()
    print(f"\nGames today: {len(games)}")
    for g in games:
        print(f"  {g['away_team']} @ {g['home_team']} - {g['game_time']}")

    if games:
        b2b = identify_b2b_teams(games)
        print(f"\nB2B teams: {b2b}")
