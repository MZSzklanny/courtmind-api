# -*- coding: utf-8 -*-
"""
CourtMind Bet Tracker
=====================
Tracks predictions vs actual results for performance analysis.
Includes: Game spreads, O/U, ML, and player props.
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Get the base directory (CourtMind folder)
BASE_DIR = Path(__file__).resolve().parent.parent

PREDICTIONS_FILE = BASE_DIR / 'predictions_log.json'
RESULTS_FILE = BASE_DIR / 'results_log.json'
DAILY_TRACKING_FILE = BASE_DIR / 'daily_tracking.json'

# Minimum edge to log a prediction (3%)
MIN_EDGE_THRESHOLD = 3.0


def load_predictions():
    """Load all historical predictions."""
    if PREDICTIONS_FILE.exists():
        with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_predictions(predictions):
    """Save predictions to file."""
    with open(PREDICTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, default=str)


def log_daily_predictions(picks):
    """
    Log today's predictions.

    Args:
        picks: List of dicts with keys:
            - player, team, opponent, stat, line, direction, projection, edge, confidence, game_date
    """
    predictions = load_predictions()
    today = datetime.now().strftime('%Y-%m-%d')

    # Get all dates from the incoming picks
    import_dates = set()
    for pick in picks:
        if pick.get('game_date'):
            import_dates.add(pick['game_date'])
        else:
            import_dates.add(today)

    # Remove any existing predictions for dates being imported (in case of re-run)
    predictions = [p for p in predictions if p.get('game_date') not in import_dates]

    # Add new predictions (filter out low lines < 4.5)
    for pick in picks:
        # Filter out low-quality props (lines under 4.5)
        line = pick.get('line', 0)
        if line > 0 and line < 4.5:
            continue  # Skip this pick

        # Preserve existing game_date if provided (for historical imports)
        if not pick.get('game_date'):
            pick['game_date'] = today
        # Preserve existing logged_at if provided
        if not pick.get('logged_at'):
            pick['logged_at'] = datetime.now().isoformat()
        # Preserve existing result/hit if provided (for graded historical data)
        if 'result' not in pick:
            pick['result'] = None
        if 'hit' not in pick:
            pick['hit'] = None
        predictions.append(pick)

    save_predictions(predictions)
    return len(picks)


def check_results(df, game_date=None):
    """
    Check actual results against predictions.

    Args:
        df: DataFrame with actual game stats
        game_date: Date to check (defaults to yesterday)
    """
    if game_date is None:
        game_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    predictions = load_predictions()
    updated = 0

    for pred in predictions:
        if pred.get('game_date') != game_date:
            continue
        if pred.get('result') is not None:
            continue  # Already checked

        player = pred['player']
        stat = pred['stat'].lower()

        # Map stat names
        stat_col = {
            'pts': 'pts',
            'points': 'pts',
            'reb': 'trb',
            'rebounds': 'trb',
            'ast': 'ast',
            'assists': 'ast',
            '3pm': 'fg3m',
            'threes': 'fg3m'
        }.get(stat, stat)

        # Find player's actual stats for that date
        player_games = df[(df['player'] == player) &
                          (df['game_date'].astype(str).str[:10] == game_date)]

        if len(player_games) == 0:
            continue

        # Sum stats across quarters
        actual = player_games[stat_col].sum()
        pred['result'] = round(actual, 1)

        # Determine hit/miss
        line = pred['line']
        direction = pred['direction'].upper()

        if direction == 'OVER':
            pred['hit'] = actual > line
        else:
            pred['hit'] = actual < line

        updated += 1

    save_predictions(predictions)
    return updated


def get_tracking_stats():
    """
    Calculate tracking statistics.

    Returns:
        dict with overall stats and recent picks
    """
    predictions = load_predictions()

    # Filter to those with results
    with_results = [p for p in predictions if p.get('result') is not None]

    if not with_results:
        return {
            'total_picks': len(predictions),
            'graded_picks': 0,
            'hits': 0,
            'misses': 0,
            'hit_rate': 0,
            'pending': len(predictions),
            'recent': predictions[-20:] if predictions else [],
            'by_confidence': {},
            'by_stat': {},
            'streak': 0
        }

    hits = sum(1 for p in with_results if p.get('hit'))
    misses = len(with_results) - hits
    hit_rate = (hits / len(with_results)) * 100 if with_results else 0

    # By confidence tier
    by_confidence = {}
    for tier in ['80+', '70-79', '60-69', '<60']:
        tier_picks = []
        for p in with_results:
            conf = p.get('confidence', 50)
            if tier == '80+' and conf >= 80:
                tier_picks.append(p)
            elif tier == '70-79' and 70 <= conf < 80:
                tier_picks.append(p)
            elif tier == '60-69' and 60 <= conf < 70:
                tier_picks.append(p)
            elif tier == '<60' and conf < 60:
                tier_picks.append(p)

        if tier_picks:
            tier_hits = sum(1 for p in tier_picks if p.get('hit'))
            by_confidence[tier] = {
                'picks': len(tier_picks),
                'hits': tier_hits,
                'rate': round((tier_hits / len(tier_picks)) * 100, 1)
            }

    # By stat type
    by_stat = {}
    for stat in ['PTS', 'REB', 'AST', '3PM']:
        stat_picks = [p for p in with_results if p.get('stat', '').upper() == stat]
        if stat_picks:
            stat_hits = sum(1 for p in stat_picks if p.get('hit'))
            by_stat[stat] = {
                'picks': len(stat_picks),
                'hits': stat_hits,
                'rate': round((stat_hits / len(stat_picks)) * 100, 1)
            }

    # Current streak
    streak = 0
    for p in reversed(with_results):
        if p.get('hit'):
            streak += 1
        else:
            break

    # Losing streak if no wins
    if streak == 0:
        for p in reversed(with_results):
            if not p.get('hit'):
                streak -= 1
            else:
                break

    return {
        'total_picks': len(predictions),
        'graded_picks': len(with_results),
        'hits': hits,
        'misses': misses,
        'hit_rate': round(hit_rate, 1),
        'pending': len(predictions) - len(with_results),
        'recent': list(reversed(with_results[-20:])),  # Most recent first
        'by_confidence': by_confidence,
        'by_stat': by_stat,
        'streak': streak
    }


def get_todays_predictions():
    """Get predictions logged for today."""
    predictions = load_predictions()
    today = datetime.now().strftime('%Y-%m-%d')
    return [p for p in predictions if p.get('game_date') == today]


def clear_old_predictions(days=30):
    """Remove predictions older than N days."""
    predictions = load_predictions()
    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    predictions = [p for p in predictions if p.get('game_date', '2000-01-01') >= cutoff]
    save_predictions(predictions)


# =============================================================================
# DAILY TRACKING SYSTEM (Games + Props)
# =============================================================================

def load_daily_tracking():
    """Load all daily tracking records."""
    if DAILY_TRACKING_FILE.exists():
        with open(DAILY_TRACKING_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_daily_tracking(records):
    """Save daily tracking records."""
    with open(DAILY_TRACKING_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, default=str)


def log_daily_picks(game_preds, prop_preds, date=None):
    """
    Log daily game and prop predictions.

    Args:
        game_preds: List of game predictions (spread, O/U, ML)
        prop_preds: List of player prop predictions
        date: Date string (defaults to today)
    """
    records = load_daily_tracking()
    date = date or datetime.now().strftime('%Y-%m-%d')

    # Remove existing record for this date
    records = [r for r in records if r.get('date') != date]

    # Create new record
    record = {
        'date': date,
        'logged_at': datetime.now().isoformat(),
        'game_predictions': game_preds,
        'prop_predictions': prop_preds,
        'graded': False
    }

    records.append(record)
    save_daily_tracking(records)

    return {
        'games': len(game_preds),
        'props': len(prop_preds)
    }


def generate_daily_predictions(df, odds_data, lineups, min_edge=MIN_EDGE_THRESHOLD):
    """
    Generate daily predictions with edge filtering.

    Returns game predictions and prop predictions that meet edge threshold.
    """
    from models.ensemble_model import GamePredictor
    from models.predictor import PlayerPredictor
    from models.odds_fetcher import get_player_prop_line, TEAM_FULL_NAMES

    TEAM_ABBREV = {v: k for k, v in TEAM_FULL_NAMES.items()}

    game_predictor = GamePredictor()
    player_predictor = PlayerPredictor(df)

    game_preds = []
    prop_preds = []

    for game in odds_data.get('games', []):
        home_full = game['home_team']
        away_full = game['away_team']
        home = TEAM_ABBREV.get(home_full, home_full[:3].upper())
        away = TEAM_ABBREV.get(away_full, away_full[:3].upper())

        # Get our prediction
        result = game_predictor.predict_game(df, home, away)
        if not result:
            continue

        # Get DK odds
        dk = game.get('bookmakers', {}).get('draftkings', {})
        dk_spread = dk.get('spreads', {}).get(home_full, {}).get('point')
        dk_total = dk.get('totals', {}).get('Over', {}).get('point')
        dk_ml_home = dk.get('h2h', {}).get(home_full, {}).get('odds')

        our_spread = result['spread']
        our_total = result['predicted_total']
        our_home_prob = result['home_win_prob']

        game_pred = {
            'home_team': home,
            'away_team': away,
            'our_spread': round(our_spread, 1),
            'dk_spread': dk_spread,
            'our_total': round(our_total, 1),
            'dk_total': dk_total,
            'our_home_prob': round(our_home_prob, 1),
            'home_score': None,
            'away_score': None,
            'spread_pick': None,
            'spread_edge': None,
            'spread_hit': None,
            'ou_pick': None,
            'ou_edge': None,
            'ou_hit': None,
            'ml_pick': None,
            'ml_edge': None,
            'ml_hit': None
        }

        # Calculate spread edge
        if dk_spread is not None:
            spread_diff = abs(our_spread - dk_spread)
            spread_edge = (spread_diff / abs(dk_spread)) * 100 if dk_spread != 0 else spread_diff * 10
            if spread_edge >= min_edge:
                game_pred['spread_pick'] = home if our_spread < dk_spread else away
                game_pred['spread_edge'] = round(spread_edge, 1)

        # Calculate O/U edge
        if dk_total is not None:
            total_diff = abs(our_total - dk_total)
            ou_edge = (total_diff / dk_total) * 100
            if ou_edge >= min_edge:
                game_pred['ou_pick'] = 'OVER' if our_total > dk_total else 'UNDER'
                game_pred['ou_edge'] = round(ou_edge, 1)

        # Calculate ML edge (using implied probability)
        if dk_ml_home is not None:
            if dk_ml_home > 0:
                dk_implied = 100 / (dk_ml_home + 100) * 100
            else:
                dk_implied = abs(dk_ml_home) / (abs(dk_ml_home) + 100) * 100
            ml_edge = abs(our_home_prob - dk_implied)
            if ml_edge >= min_edge:
                game_pred['ml_pick'] = home if our_home_prob > dk_implied else away
                game_pred['ml_edge'] = round(ml_edge, 1)

        # Only include if at least one pick has edge
        if game_pred['spread_pick'] or game_pred['ou_pick'] or game_pred['ml_pick']:
            game_preds.append(game_pred)

        # Get player props for starters
        for team, opp in [(home, away), (away, home)]:
            starters = lineups.get(team, {}).get('starters', [])
            for player in starters[:5]:
                try:
                    pred = player_predictor.predict(player, opp)
                    if not pred:
                        continue

                    pts_lines = get_player_prop_line(player, 'points')
                    if not pts_lines:
                        continue

                    dk_line = pts_lines.get('dk', {}).get('over', {}).get('line')
                    fd_line = pts_lines.get('fd', {}).get('over', {}).get('line')

                    best_line = None
                    if dk_line and fd_line:
                        best_line = min(dk_line, fd_line)
                    elif dk_line:
                        best_line = dk_line
                    elif fd_line:
                        best_line = fd_line

                    if best_line and best_line > 0:
                        edge = ((pred['pts'] - best_line) / best_line) * 100
                        if abs(edge) >= min_edge:
                            prop_preds.append({
                                'player': player,
                                'team': team,
                                'opponent': opp,
                                'stat': 'PTS',
                                'projection': round(pred['pts'], 1),
                                'dk_line': dk_line,
                                'fd_line': fd_line,
                                'pick': 'OVER' if edge > 0 else 'UNDER',
                                'edge': round(abs(edge), 1),
                                'confidence': pred['confidence'],
                                'result': None,
                                'hit': None
                            })
                except:
                    continue

    return game_preds, prop_preds


def grade_daily_picks(df, date=None):
    """
    Grade predictions for a specific date using actual results.

    Args:
        df: DataFrame with game results (needs home/away scores)
        date: Date to grade (defaults to yesterday)
    """
    records = load_daily_tracking()
    date = date or (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    # Find the record for this date
    record = next((r for r in records if r.get('date') == date), None)
    if not record:
        return {'error': f'No predictions found for {date}'}

    if record.get('graded'):
        return {'message': f'Already graded for {date}'}

    graded_games = 0
    graded_props = 0

    # Grade game predictions
    for game in record.get('game_predictions', []):
        home = game['home_team']
        away = game['away_team']

        # Get actual scores from data
        home_games = df[(df['team'] == home) & (df['game_date'].astype(str).str[:10] == date)]
        away_games = df[(df['team'] == away) & (df['game_date'].astype(str).str[:10] == date)]

        if len(home_games) == 0 or len(away_games) == 0:
            continue

        home_score = home_games['pts'].sum()
        away_score = away_games['pts'].sum()
        actual_spread = away_score - home_score  # Positive = home won by that much
        actual_total = home_score + away_score

        game['home_score'] = home_score
        game['away_score'] = away_score

        # Grade spread
        if game.get('spread_pick'):
            dk_spread = game['dk_spread']
            if game['spread_pick'] == home:
                # We picked home to cover (beat spread)
                game['spread_hit'] = (home_score - away_score) > -dk_spread
            else:
                # We picked away to cover
                game['spread_hit'] = (away_score - home_score) > dk_spread
            graded_games += 1

        # Grade O/U
        if game.get('ou_pick'):
            if game['ou_pick'] == 'OVER':
                game['ou_hit'] = actual_total > game['dk_total']
            else:
                game['ou_hit'] = actual_total < game['dk_total']
            graded_games += 1

        # Grade ML
        if game.get('ml_pick'):
            winner = home if home_score > away_score else away
            game['ml_hit'] = game['ml_pick'] == winner
            graded_games += 1

    # Grade prop predictions
    for prop in record.get('prop_predictions', []):
        player = prop['player']
        stat = prop['stat'].lower()

        stat_col = {'pts': 'pts', 'reb': 'trb', 'ast': 'ast'}.get(stat, stat)

        player_games = df[(df['player'] == player) &
                          (df['game_date'].astype(str).str[:10] == date)]

        if len(player_games) == 0:
            continue

        actual = player_games[stat_col].sum()
        prop['result'] = round(actual, 1)

        line = prop['dk_line'] or prop['fd_line']
        if prop['pick'] == 'OVER':
            prop['hit'] = actual > line
        else:
            prop['hit'] = actual < line

        graded_props += 1

    record['graded'] = True
    save_daily_tracking(records)

    return {
        'date': date,
        'graded_games': graded_games,
        'graded_props': graded_props
    }


def get_daily_tracking_stats():
    """
    Calculate comprehensive tracking statistics.
    """
    records = load_daily_tracking()

    # Initialize stats
    stats = {
        'total_days': len(records),
        'spread': {'picks': 0, 'hits': 0, 'rate': 0, 'pending': 0},
        'ou': {'picks': 0, 'hits': 0, 'rate': 0, 'pending': 0},
        'ml': {'picks': 0, 'hits': 0, 'rate': 0, 'pending': 0},
        'props': {'picks': 0, 'hits': 0, 'rate': 0, 'pending': 0},
        'by_date': [],
        'recent_picks': [],
        'pending_picks': []
    }

    for record in records:
        date = record['date']
        day_stats = {'date': date, 'spread': 0, 'ou': 0, 'ml': 0, 'props': 0}

        # Game predictions
        for game in record.get('game_predictions', []):
            # Count pending picks (not yet graded)
            if game.get('spread_pick') and game.get('spread_hit') is None:
                stats['spread']['pending'] += 1
                stats['pending_picks'].append({
                    'date': date,
                    'type': 'SPREAD',
                    'pick': f"{game['spread_pick']} {game['dk_spread']:+.1f}" if game.get('dk_spread') else game['spread_pick'],
                    'matchup': f"{game['away_team']} @ {game['home_team']}",
                    'edge': game['spread_edge']
                })
            if game.get('ou_pick') and game.get('ou_hit') is None:
                stats['ou']['pending'] += 1
                stats['pending_picks'].append({
                    'date': date,
                    'type': 'O/U',
                    'pick': f"{game['ou_pick']} {game['dk_total']}",
                    'matchup': f"{game['away_team']} @ {game['home_team']}",
                    'edge': game['ou_edge']
                })
            if game.get('ml_pick') and game.get('ml_hit') is None:
                stats['ml']['pending'] += 1
                stats['pending_picks'].append({
                    'date': date,
                    'type': 'ML',
                    'pick': game['ml_pick'],
                    'matchup': f"{game['away_team']} @ {game['home_team']}",
                    'edge': game['ml_edge']
                })

            # Count graded picks
            if game.get('spread_hit') is not None:
                stats['spread']['picks'] += 1
                if game['spread_hit']:
                    stats['spread']['hits'] += 1
                    day_stats['spread'] += 1
                stats['recent_picks'].append({
                    'date': date,
                    'type': 'SPREAD',
                    'pick': f"{game['spread_pick']} {game['dk_spread']:+.1f}",
                    'result': f"{game['away_team']} {game.get('away_score', '?')} @ {game['home_team']} {game.get('home_score', '?')}",
                    'hit': game['spread_hit'],
                    'edge': game['spread_edge']
                })

            if game.get('ou_hit') is not None:
                stats['ou']['picks'] += 1
                if game['ou_hit']:
                    stats['ou']['hits'] += 1
                    day_stats['ou'] += 1
                stats['recent_picks'].append({
                    'date': date,
                    'type': 'O/U',
                    'pick': f"{game['ou_pick']} {game['dk_total']}",
                    'result': f"Total: {(game.get('home_score', 0) or 0) + (game.get('away_score', 0) or 0)}",
                    'hit': game['ou_hit'],
                    'edge': game['ou_edge']
                })

            if game.get('ml_hit') is not None:
                stats['ml']['picks'] += 1
                if game['ml_hit']:
                    stats['ml']['hits'] += 1
                    day_stats['ml'] += 1
                stats['recent_picks'].append({
                    'date': date,
                    'type': 'ML',
                    'pick': game['ml_pick'],
                    'result': f"{game['away_team']} {game.get('away_score', '?')} @ {game['home_team']} {game.get('home_score', '?')}",
                    'hit': game['ml_hit'],
                    'edge': game['ml_edge']
                })

        # Prop predictions
        for prop in record.get('prop_predictions', []):
            if prop.get('hit') is not None:
                stats['props']['picks'] += 1
                if prop['hit']:
                    stats['props']['hits'] += 1
                    day_stats['props'] += 1
                stats['recent_picks'].append({
                    'date': date,
                    'type': 'PROP',
                    'pick': f"{prop['player']} {prop['pick']} {prop['dk_line'] or prop['fd_line']} {prop['stat']}",
                    'result': f"Actual: {prop.get('result', '?')}",
                    'hit': prop['hit'],
                    'edge': prop['edge']
                })

        stats['by_date'].append(day_stats)

    # Calculate rates
    for key in ['spread', 'ou', 'ml', 'props']:
        if stats[key]['picks'] > 0:
            stats[key]['rate'] = round((stats[key]['hits'] / stats[key]['picks']) * 100, 1)

    # Overall
    total_picks = stats['spread']['picks'] + stats['ou']['picks'] + stats['ml']['picks'] + stats['props']['picks']
    total_hits = stats['spread']['hits'] + stats['ou']['hits'] + stats['ml']['hits'] + stats['props']['hits']
    stats['overall'] = {
        'picks': total_picks,
        'hits': total_hits,
        'rate': round((total_hits / total_picks) * 100, 1) if total_picks > 0 else 0
    }

    # Sort recent picks by date (most recent first)
    stats['recent_picks'] = sorted(stats['recent_picks'], key=lambda x: x['date'], reverse=True)[:50]

    return stats


if __name__ == "__main__":
    # Test
    stats = get_tracking_stats()
    print(f"Total picks: {stats['total_picks']}")
    print(f"Graded: {stats['graded_picks']}")
    print(f"Hit rate: {stats['hit_rate']}%")
    print(f"Streak: {stats['streak']}")

    print("\n--- Daily Tracking Stats ---")
    daily_stats = get_daily_tracking_stats()
    print(f"Spread: {daily_stats['spread']['hits']}/{daily_stats['spread']['picks']} ({daily_stats['spread']['rate']}%)")
    print(f"O/U: {daily_stats['ou']['hits']}/{daily_stats['ou']['picks']} ({daily_stats['ou']['rate']}%)")
    print(f"ML: {daily_stats['ml']['hits']}/{daily_stats['ml']['picks']} ({daily_stats['ml']['rate']}%)")
    print(f"Props: {daily_stats['props']['hits']}/{daily_stats['props']['picks']} ({daily_stats['props']['rate']}%)")
