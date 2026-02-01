# -*- coding: utf-8 -*-
"""
CourtMind Predictions Export
============================
Exports daily predictions to Excel for tracking and model tuning.
Run after 5pm to log predictions, run next morning to grade results.
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import requests

BASE_DIR = Path(__file__).resolve().parent
TRACKING_FILE = BASE_DIR / 'predictions_tracking.xlsx'

API_BASE = "https://courtmind-api.onrender.com"


def get_todays_predictions():
    """Fetch today's predictions from API"""
    # Get top picks
    picks_resp = requests.get(f"{API_BASE}/api/top-picks")
    picks_data = picks_resp.json()

    # Get games
    games_resp = requests.get(f"{API_BASE}/api/games")
    games_data = games_resp.json()

    return picks_data, games_data


def log_predictions_to_excel():
    """Log today's predictions to Excel tracking file"""
    today = datetime.now().strftime('%Y-%m-%d')

    picks_data, games_data = get_todays_predictions()

    rows = []

    # Player props
    for pick in picks_data.get('player_props', picks_data.get('picks', [])):
        if pick.get('type') != 'player_prop':
            continue
        rows.append({
            'date': today,
            'type': 'PLAYER_PROP',
            'matchup': pick.get('team', ''),
            'player': pick['player'],
            'stat': pick['stat'],
            'line': pick['line'],
            'direction': pick['direction'],
            'projection': pick['projection'],
            'edge': pick['edge'],
            'confidence': pick['confidence'],
            'book': pick.get('book', 'DK'),
            'actual': None,
            'hit': None,
            'logged_at': datetime.now().isoformat()
        })

    # Game spreads/totals
    for pick in picks_data.get('game_props', []):
        rows.append({
            'date': today,
            'type': pick.get('stat', 'SPREAD'),
            'matchup': pick['player'],
            'player': pick.get('team', ''),
            'stat': pick['stat'],
            'line': pick['line'],
            'direction': pick['direction'],
            'projection': pick['projection'],
            'edge': pick['edge'],
            'confidence': pick['confidence'],
            'book': pick.get('book', 'DK'),
            'actual': None,
            'hit': None,
            'logged_at': datetime.now().isoformat()
        })

    # Game predictions (spread/total for all games)
    for game in games_data.get('games', []):
        pred = game.get('prediction', {})
        odds = game.get('odds', {})
        matchup = f"{game['away_team']} @ {game['home_team']}"

        # Spread pick
        if odds.get('dk_spread') and pred.get('spread'):
            model_spread = pred['spread']
            dk_spread = odds['dk_spread']
            spread_diff = abs(model_spread - dk_spread)

            rows.append({
                'date': today,
                'type': 'GAME_SPREAD',
                'matchup': matchup,
                'player': game['home_team'],
                'stat': 'SPREAD',
                'line': dk_spread,
                'direction': f"{game['home_team']} {dk_spread:+.1f}",
                'projection': model_spread,
                'edge': round(spread_diff, 1),
                'confidence': round(pred.get('home_win_prob', 50)),
                'book': 'DK',
                'actual': None,
                'hit': None,
                'logged_at': datetime.now().isoformat()
            })

        # Total pick
        if odds.get('dk_total') and pred.get('total'):
            model_total = pred['total']
            dk_total = odds['dk_total']
            total_diff = model_total - dk_total
            direction = 'OVER' if total_diff > 0 else 'UNDER'

            rows.append({
                'date': today,
                'type': 'GAME_TOTAL',
                'matchup': matchup,
                'player': '',
                'stat': 'TOTAL',
                'line': dk_total,
                'direction': direction,
                'projection': round(model_total, 1),
                'edge': round(abs(total_diff / dk_total * 100), 1),
                'confidence': 65,
                'book': 'DK',
                'actual': None,
                'hit': None,
                'logged_at': datetime.now().isoformat()
            })

    new_df = pd.DataFrame(rows)

    # Load existing or create new
    if TRACKING_FILE.exists():
        try:
            existing_df = pd.read_excel(TRACKING_FILE)
            if 'date' in existing_df.columns and len(existing_df) > 0:
                # Remove today's entries if re-running
                existing_df = existing_df[existing_df['date'] != today]
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
        except:
            combined_df = new_df
    else:
        combined_df = new_df

    # Save
    combined_df.to_excel(TRACKING_FILE, index=False)
    print(f"[{today}] Logged {len(rows)} predictions to {TRACKING_FILE}")
    return len(rows)


def grade_predictions(date=None):
    """
    Grade predictions for a specific date by fetching actual results.
    Run this the morning after games.
    """
    if date is None:
        date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    if not TRACKING_FILE.exists():
        print("No tracking file found!")
        return

    df = pd.read_excel(TRACKING_FILE)

    # Get yesterday's rows that need grading
    mask = (df['date'] == date) & (df['hit'].isna())
    to_grade = df[mask]

    if len(to_grade) == 0:
        print(f"No predictions to grade for {date}")
        return

    print(f"Found {len(to_grade)} predictions to grade for {date}")
    print("Please manually enter actual results in the Excel file.")
    print(f"File: {TRACKING_FILE}")

    # TODO: Auto-fetch results from basketball-reference or other source
    # For now, manual entry is required

    return len(to_grade)


def get_tracking_summary():
    """Get summary stats from tracking file"""
    if not TRACKING_FILE.exists():
        return {"error": "No tracking file"}

    df = pd.read_excel(TRACKING_FILE)
    graded = df[df['hit'].notna()]

    summary = {
        'total_predictions': len(df),
        'graded': len(graded),
        'pending': len(df) - len(graded),
        'hits': int(graded['hit'].sum()) if len(graded) > 0 else 0,
        'misses': len(graded) - int(graded['hit'].sum()) if len(graded) > 0 else 0,
        'hit_rate': round(graded['hit'].mean() * 100, 1) if len(graded) > 0 else 0,
        'by_type': {}
    }

    for ptype in df['type'].unique():
        type_graded = graded[graded['type'] == ptype]
        if len(type_graded) > 0:
            summary['by_type'][ptype] = {
                'total': len(df[df['type'] == ptype]),
                'graded': len(type_graded),
                'hits': int(type_graded['hit'].sum()),
                'hit_rate': round(type_graded['hit'].mean() * 100, 1)
            }

    return summary


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'grade':
        date = sys.argv[2] if len(sys.argv) > 2 else None
        grade_predictions(date)
    elif len(sys.argv) > 1 and sys.argv[1] == 'summary':
        print(get_tracking_summary())
    else:
        log_predictions_to_excel()
