# -*- coding: utf-8 -*-
"""
CourtMind Daily Predictions Logger
==================================
Automatically logs daily predictions before games start.
Run via scheduled task at 6:30 PM ET.
"""

import sys
import os
from datetime import datetime

# Add paths
sys.path.insert(0, 'C:/Users/user')
sys.path.insert(0, 'C:/Users/user/CourtMind')

import pandas as pd
from models.bet_tracker import log_daily_picks, generate_daily_predictions, grade_daily_picks
from models.odds_fetcher import fetch_game_odds, get_api_key
from models.nba_lineups_fetcher import get_todays_official_lineups

LOG_FILE = 'C:/Users/user/CourtMind/daily_log.txt'


def log_message(msg):
    """Log to file and print."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(full_msg + '\n')


def main():
    log_message("=" * 50)
    log_message("Starting daily predictions log...")

    try:
        # Load data
        log_message("Loading data...")
        df = pd.read_parquet('C:/Users/user/NBA_PRODUCTION.parquet')
        log_message(f"Loaded {len(df)} records, data through {df['game_date'].max()}")

        # Get odds
        log_message("Fetching odds...")
        if not get_api_key():
            log_message("ERROR: No odds API key found!")
            return

        odds_data = fetch_game_odds()
        games = odds_data.get('games', [])
        log_message(f"Found {len(games)} games today")

        if len(games) == 0:
            log_message("No games today, nothing to log.")
            return

        # Get lineups
        log_message("Getting lineups...")
        try:
            lineups = get_todays_official_lineups()
            log_message(f"Got lineups for {len(lineups)} teams")
        except:
            lineups = {}
            log_message("WARNING: Could not get lineups, props may be limited")

        # Generate predictions
        log_message("Generating predictions with 3%+ edge...")
        game_preds, prop_preds = generate_daily_predictions(df, odds_data, lineups)

        log_message(f"Generated {len(game_preds)} game predictions")
        log_message(f"Generated {len(prop_preds)} prop predictions")

        # Log game picks
        for gp in game_preds:
            picks = []
            if gp.get('spread_pick'):
                picks.append(f"SPREAD: {gp['spread_pick']} ({gp['spread_edge']}% edge)")
            if gp.get('ou_pick'):
                picks.append(f"O/U: {gp['ou_pick']} {gp['dk_total']} ({gp['ou_edge']}% edge)")
            if gp.get('ml_pick'):
                picks.append(f"ML: {gp['ml_pick']} ({gp['ml_edge']}% edge)")
            if picks:
                log_message(f"  {gp['away_team']} @ {gp['home_team']}: {', '.join(picks)}")

        # Log prop picks
        for pp in prop_preds[:10]:  # Show first 10
            log_message(f"  PROP: {pp['player']} {pp['pick']} {pp['dk_line'] or pp['fd_line']} {pp['stat']} ({pp['edge']}% edge)")

        # Save to tracking
        log_message("Saving to tracking...")
        result = log_daily_picks(game_preds, prop_preds)
        log_message(f"Logged {result['games']} game picks and {result['props']} prop picks")

        # Also grade yesterday's picks if not done
        log_message("Checking yesterday's grades...")
        grade_result = grade_daily_picks(df)
        if 'error' not in grade_result:
            log_message(f"Graded {grade_result.get('graded_games', 0)} games, {grade_result.get('graded_props', 0)} props")
        else:
            log_message(f"Grade result: {grade_result}")

        log_message("Daily predictions log complete!")

    except Exception as e:
        log_message(f"ERROR: {str(e)}")
        import traceback
        log_message(traceback.format_exc())


if __name__ == "__main__":
    main()
