# -*- coding: utf-8 -*-
"""
CourtMind Auto Grader
=====================
Automatically grades predictions using real NBA stats.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from .stats_fetcher import fetch_stats_for_grading

BASE_DIR = Path(__file__).resolve().parent.parent
PREDICTIONS_FILE = BASE_DIR / 'predictions_log.json'


def load_predictions():
    """Load predictions from file."""
    if PREDICTIONS_FILE.exists():
        with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_predictions(predictions):
    """Save predictions to file."""
    with open(PREDICTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, default=str)


def normalize_player_name(name):
    """Normalize player name for matching."""
    if not name:
        return ''
    # Remove special characters and extra spaces
    name = name.strip().lower()
    # Handle common variations
    name = name.replace("'", "'").replace("'", "'")
    return name


def find_player_stats(player_name, stats_df):
    """Find a player's stats in the dataframe, handling name variations."""
    if stats_df.empty or not player_name:
        return None

    player_lower = normalize_player_name(player_name)
    player_parts = player_lower.split()

    if len(player_parts) < 2:
        return None

    first_name = player_parts[0]
    last_name = player_parts[-1]

    # Try exact match first
    for _, row in stats_df.iterrows():
        row_name = normalize_player_name(row['player'])
        if row_name == player_lower:
            # Check if player actually played (has minutes)
            minutes = row.get('minutes', '0:00')
            if minutes == '0:00' or minutes == '' or minutes is None:
                return None  # DNP - Did Not Play
            return row

    # Try matching first name + last name (handles middle names, suffixes)
    for _, row in stats_df.iterrows():
        row_name = normalize_player_name(row['player'])
        row_parts = row_name.split()

        if len(row_parts) < 2:
            continue

        row_first = row_parts[0]
        row_last = row_parts[-1]

        # Must match FULL first name and last name
        # This prevents "Miles Bridges" matching "Mikal Bridges"
        if first_name == row_first and last_name == row_last:
            # Check if player actually played
            minutes = row.get('minutes', '0:00')
            if minutes == '0:00' or minutes == '' or minutes is None:
                return None  # DNP
            return row

    # No match found - player likely didn't play or wasn't in the game
    return None


def grade_player_prop(pred, stats_df):
    """
    Grade a player prop prediction.

    Returns:
        tuple: (actual_value, hit_bool) or (None, None) if can't grade
    """
    player = pred.get('player')
    stat = pred.get('stat', '').upper()
    line = pred.get('line')
    direction = pred.get('direction', '').upper()

    if not player or not stat or line is None:
        return None, None

    player_stats = find_player_stats(player, stats_df)
    if player_stats is None:
        return None, None

    # Map stat types to dataframe columns
    stat_map = {
        'POINTS': 'points',
        'PTS': 'points',
        'REBOUNDS': 'rebounds',
        'REB': 'rebounds',
        'ASSISTS': 'assists',
        'AST': 'assists',
        '3PM': 'threes',
        'THREES': 'threes',
        '3PT': 'threes',
        'STEALS': 'steals',
        'STL': 'steals',
        'BLOCKS': 'blocks',
        'BLK': 'blocks',
    }

    col = stat_map.get(stat)
    if not col or col not in player_stats.index:
        return None, None

    actual = float(player_stats[col])

    # Determine hit/miss
    if 'OVER' in direction:
        hit = actual > line
    elif 'UNDER' in direction:
        hit = actual < line
    else:
        hit = None

    return actual, hit


def grade_game_spread(pred, game_results):
    """
    Grade a game spread prediction.

    Returns:
        tuple: (actual_margin, hit_bool) or (None, None) if can't grade
    """
    matchup = pred.get('matchup', '')
    direction = pred.get('direction', '')
    line = pred.get('line')

    if not matchup or not direction:
        return None, None

    # Find the game
    game = None
    for m, result in game_results.items():
        if matchup in m or m in matchup:
            game = result
            break
        # Try matching team abbreviations
        for team in [result['home_team'], result['away_team']]:
            if team in matchup:
                game = result
                break

    if not game:
        return None, None

    margin = game['margin']  # Positive = home won

    # Parse the spread direction
    # Format could be "BOS -12.5" or "BOS" with line separate
    if '-' in direction:
        # Favorite (negative spread)
        parts = direction.split('-')
        team = parts[0].strip()
        spread = -float(parts[1]) if len(parts) > 1 else -line
    elif '+' in direction:
        # Underdog (positive spread)
        parts = direction.split('+')
        team = parts[0].strip()
        spread = float(parts[1]) if len(parts) > 1 else line
    else:
        team = direction.strip()
        spread = -line if line else 0

    # Determine if team covered
    if team == game['home_team']:
        actual_margin = margin
    elif team == game['away_team']:
        actual_margin = -margin
    else:
        # Try to find team in matchup
        actual_margin = margin  # Default to home perspective

    # Hit if actual + spread > 0
    hit = (actual_margin + spread) > 0

    return actual_margin, hit


def grade_game_total(pred, game_results):
    """
    Grade a game total (O/U) prediction.

    Returns:
        tuple: (actual_total, hit_bool) or (None, None) if can't grade
    """
    matchup = pred.get('matchup', '')
    direction = pred.get('direction', '').upper()
    line = pred.get('line')

    if not matchup or not direction or line is None:
        return None, None

    # Find the game
    game = None
    for m, result in game_results.items():
        if matchup in m or m in matchup:
            game = result
            break
        # Try matching team abbreviations
        for team in [result['home_team'], result['away_team']]:
            if team in matchup:
                game = result
                break

    if not game:
        return None, None

    actual_total = game['total']

    if 'OVER' in direction:
        hit = actual_total > line
    elif 'UNDER' in direction:
        hit = actual_total < line
    else:
        hit = None

    return actual_total, hit


def grade_predictions_for_date(game_date):
    """
    Grade all predictions for a specific date using real stats.

    Args:
        game_date: str in format 'YYYY-MM-DD'

    Returns:
        dict with grading results
    """
    print(f"\n=== Grading Predictions for {game_date} ===\n")

    # Fetch real stats
    stats_data = fetch_stats_for_grading(game_date)
    player_stats = stats_data['player_stats']
    game_results = stats_data['game_results']

    if player_stats.empty:
        return {'error': f'No stats found for {game_date}', 'graded': 0}

    # Load predictions
    predictions = load_predictions()

    graded = 0
    updated = 0

    for pred in predictions:
        if pred.get('game_date') != game_date:
            continue

        pred_type = pred.get('type', 'PLAYER_PROP')
        actual = None
        hit = None

        # Grade based on type
        if pred_type in ['PLAYER_PROP', 'TOP_PICK']:
            actual, hit = grade_player_prop(pred, player_stats)
        elif pred_type == 'GAME_SPREAD':
            actual, hit = grade_game_spread(pred, game_results)
        elif pred_type == 'GAME_TOTAL':
            actual, hit = grade_game_total(pred, game_results)

        if actual is not None:
            old_result = pred.get('result')
            old_hit = pred.get('hit')

            pred['result'] = round(actual, 1) if isinstance(actual, float) else actual
            pred['hit'] = bool(hit) if hit is not None else None

            graded += 1
            if old_result != pred['result'] or old_hit != pred['hit']:
                updated += 1
                status = "HIT" if hit else "MISS" if hit is not None else "N/A"
                player = pred.get('player', pred.get('matchup', 'Unknown'))
                stat = pred.get('stat', pred_type)
                line = pred.get('line', '')
                direction = pred.get('direction', '')
                print(f"  {player} {stat} {direction} {line}: actual={actual} -> {status}")

    # Save updated predictions
    save_predictions(predictions)

    print(f"\nGraded {graded} predictions ({updated} updated)")

    return {
        'date': game_date,
        'graded': graded,
        'updated': updated,
        'total_predictions': len([p for p in predictions if p.get('game_date') == game_date])
    }


def grade_all_ungraded():
    """Grade all ungraded predictions."""
    predictions = load_predictions()

    # Find unique dates with ungraded predictions
    ungraded_dates = set()
    for pred in predictions:
        if pred.get('result') is None and pred.get('game_date'):
            ungraded_dates.add(pred['game_date'])

    if not ungraded_dates:
        print("No ungraded predictions found")
        return []

    results = []
    for date in sorted(ungraded_dates):
        # Only grade past dates
        if date >= datetime.now().strftime('%Y-%m-%d'):
            print(f"Skipping {date} - games may not be complete")
            continue

        result = grade_predictions_for_date(date)
        results.append(result)

    return results


def regrade_date(game_date):
    """
    Re-grade all predictions for a date (even if already graded).

    Args:
        game_date: str in format 'YYYY-MM-DD'
    """
    predictions = load_predictions()

    # Clear existing grades for this date
    for pred in predictions:
        if pred.get('game_date') == game_date:
            pred['result'] = None
            pred['hit'] = None

    save_predictions(predictions)

    # Re-grade
    return grade_predictions_for_date(game_date)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        date = sys.argv[1]
        print(f"Grading predictions for {date}")
        result = regrade_date(date)
    else:
        # Grade yesterday by default
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"Grading predictions for {yesterday}")
        result = grade_predictions_for_date(yesterday)

    print(f"\nResult: {result}")
