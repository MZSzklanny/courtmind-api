# -*- coding: utf-8 -*-
"""
Update NBA_Game_PRODUCTION.parquet with home/away data for recent games
"""

import pandas as pd
import requests
import time
from datetime import datetime
import sys
sys.stdout.reconfigure(encoding='utf-8')

# NBA API headers
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true'
}

def get_game_boxscore(game_id):
    """Fetch boxscore to get home/away teams."""
    url = f"https://stats.nba.com/stats/boxscoresummaryv2?GameID={game_id}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            data = response.json()

            # GameSummary contains home/away team info
            game_summary = data['resultSets'][0]
            headers_list = game_summary['headers']
            rows = game_summary['rowSet']

            if rows:
                row = rows[0]
                home_idx = headers_list.index('HOME_TEAM_ID')
                away_idx = headers_list.index('VISITOR_TEAM_ID')

                return {
                    'home_team_id': row[home_idx],
                    'away_team_id': row[away_idx]
                }
    except Exception as e:
        print(f"  Error fetching {game_id}: {e}")

    return None


# Team ID to abbreviation mapping
TEAM_ID_MAP = {
    1610612737: 'ATL', 1610612738: 'BOS', 1610612751: 'BKN', 1610612766: 'CHA',
    1610612741: 'CHI', 1610612739: 'CLE', 1610612742: 'DAL', 1610612743: 'DEN',
    1610612765: 'DET', 1610612744: 'GSW', 1610612745: 'HOU', 1610612754: 'IND',
    1610612746: 'LAC', 1610612747: 'LAL', 1610612763: 'MEM', 1610612748: 'MIA',
    1610612749: 'MIL', 1610612750: 'MIN', 1610612740: 'NOP', 1610612752: 'NYK',
    1610612760: 'OKC', 1610612753: 'ORL', 1610612755: 'PHI', 1610612756: 'PHX',
    1610612757: 'POR', 1610612758: 'SAC', 1610612759: 'SAS', 1610612761: 'TOR',
    1610612762: 'UTA', 1610612764: 'WAS'
}


def main():
    print("=" * 60)
    print("UPDATING GAME DATA WITH HOME/AWAY INFO")
    print("=" * 60)

    # Load data
    prod_df = pd.read_parquet('C:/Users/user/NBA_PRODUCTION.parquet')
    game_df = pd.read_parquet('C:/Users/user/NBA_Game_PRODUCTION.parquet')

    prod_df['game_date'] = pd.to_datetime(prod_df['game_date'])
    game_df['game_date'] = pd.to_datetime(game_df['game_date'])

    missing_start = game_df['game_date'].max()
    print(f"Current game data ends: {missing_start}")
    print(f"Production data ends: {prod_df['game_date'].max()}")

    # Get missing games
    missing_games = prod_df[prod_df['game_date'] > missing_start].copy()
    unique_game_ids = missing_games['game_id'].unique()
    print(f"\nGames to update: {len(unique_game_ids)}")

    # Fetch home/away for each game
    game_info = {}

    for i, game_id in enumerate(unique_game_ids):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(unique_game_ids)}] games processed...")

        info = get_game_boxscore(game_id)
        if info:
            home_team = TEAM_ID_MAP.get(info['home_team_id'], 'UNK')
            away_team = TEAM_ID_MAP.get(info['away_team_id'], 'UNK')
            game_info[game_id] = {'home_team': home_team, 'away_team': away_team}

        time.sleep(0.4)  # Rate limiting

    print(f"\nSuccessfully fetched: {len(game_info)} games")

    # Create new game records from production data
    new_records = []

    for _, row in missing_games.iterrows():
        game_id = row['game_id']
        if game_id in game_info:
            info = game_info[game_id]
            is_home = row['team'] == info['home_team']

            new_records.append({
                'player': row['player'],
                'game_id': game_id,
                'game_date': row['game_date'],
                'team': row['team'],
                'season': row['season'],
                'win_loss': '',  # Would need additional fetch
                'pts': row['pts'],
                'trb': row['trb'],
                'ast': row['ast'],
                'stl': row['stl'],
                'blk': row['blk'],
                'tov': row['tov'],
                'pf': row['pf'],
                'fgm': row['fgm'],
                'fga': row['fga'],
                'minutes': row['minutes'],
                'home_team': info['home_team'],
                'away_team': info['away_team'],
                'fg_pct': row['fgm'] / row['fga'] if row['fga'] > 0 else 0,
                'pts_per_min': row['pts'] / row['minutes'] if row['minutes'] > 0 else 0,
                'is_home': is_home,
                'win': None
            })

    if new_records:
        new_df = pd.DataFrame(new_records)
        updated_game_df = pd.concat([game_df, new_df], ignore_index=True)
        updated_game_df.to_parquet('C:/Users/user/NBA_Game_PRODUCTION.parquet', index=False)
        print(f"\nAdded {len(new_records)} records to game data")
        print(f"New total: {len(updated_game_df)} records")
        print(f"Date range: {updated_game_df['game_date'].min()} to {updated_game_df['game_date'].max()}")
    else:
        print("\nNo new records to add")

    print("\n" + "=" * 60)
    print("UPDATE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
