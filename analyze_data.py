# -*- coding: utf-8 -*-
"""
CourtMind Data Analysis - Identify Top 50 Players
"""
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load production data
df = pd.read_parquet('C:/Users/user/NBA_PRODUCTION.parquet')
print('=== NBA_PRODUCTION.parquet ===')
print(f'Records: {len(df):,}')
print(f'Columns: {list(df.columns)}')
print(f'Date range: {df["game_date"].min()} to {df["game_date"].max()}')
print(f'Unique players: {df["player"].nunique()}')
print()

# Top 50 players by PPG (min 20 games)
top_players = df.groupby('player').agg({
    'pts': ['sum', 'mean', 'count'],
    'trb': 'mean',
    'ast': 'mean',
    'stl': 'mean',
    'blk': 'mean',
    'fg3m': 'mean',
    'minutes': 'mean'
}).round(1)
top_players.columns = ['total_pts', 'ppg', 'games', 'rpg', 'apg', 'spg', 'bpg', '3pm', 'mpg']
top_players = top_players[top_players['games'] >= 20].sort_values('ppg', ascending=False).head(50)
print('=== TOP 50 PLAYERS (by PPG, min 20 games) ===')
for i, (player, row) in enumerate(top_players.iterrows(), 1):
    print(f'{i:2}. {player[:25]:25} | {row["ppg"]:5.1f} PPG | {row["rpg"]:4.1f} RPG | {row["apg"]:4.1f} APG | {row["games"]:.0f} GP')

# Save top 50 list
top_50_list = top_players.index.tolist()
print(f'\n=== TOP 50 PLAYER LIST ===')
print(top_50_list)

# Check for team info
print(f'\n=== TEAMS ===')
print(df['team'].unique())

# Check Quarter data for more features
try:
    qdf = pd.read_parquet('C:/Users/user/NBA_Quarter_ALL_Combined.parquet')
    print(f'\n=== NBA_Quarter_ALL_Combined.parquet ===')
    print(f'Records: {len(qdf):,}')
    print(f'Columns: {list(qdf.columns)}')
except Exception as e:
    print(f'Quarter data error: {e}')

# Check for game-level data
try:
    gdf = pd.read_parquet('C:/Users/user/NBA_Game_PRODUCTION.parquet')
    print(f'\n=== NBA_Game_PRODUCTION.parquet ===')
    print(f'Records: {len(gdf):,}')
    print(f'Columns: {list(gdf.columns)}')
except Exception as e:
    print(f'Game data error: {e}')
