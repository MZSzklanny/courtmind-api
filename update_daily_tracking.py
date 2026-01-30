# -*- coding: utf-8 -*-
"""Update daily tracking with corrected predictions"""

import sys
sys.path.insert(0, 'C:/Users/user/CourtMind')

import pandas as pd
from datetime import datetime
from models.predictor import PlayerPredictor
from models.ensemble_model import GamePredictor
from models.nba_lineups_fetcher import TODAYS_LINEUPS
from models.odds_fetcher import get_todays_odds, get_all_player_props
from models.bet_tracker import log_daily_picks

df = pd.read_parquet('C:/Users/user/NBA_PRODUCTION.parquet')
predictor = PlayerPredictor(df)
game_predictor = GamePredictor()

# Get market data
odds = get_todays_odds()
props = get_all_player_props()

# Build lookups
abbrev_map = {
    'Washington Wizards': 'WAS', 'Los Angeles Lakers': 'LAL', 'Boston Celtics': 'BOS',
    'Sacramento Kings': 'SAC', 'New York Knicks': 'NYK', 'Portland Trail Blazers': 'POR',
    'Orlando Magic': 'ORL', 'Toronto Raptors': 'TOR', 'New Orleans Pelicans': 'NOP',
    'Memphis Grizzlies': 'MEM', 'Phoenix Suns': 'PHX', 'Cleveland Cavaliers': 'CLE',
    'Denver Nuggets': 'DEN', 'Los Angeles Clippers': 'LAC', 'Utah Jazz': 'UTA',
    'Brooklyn Nets': 'BKN', 'Golden State Warriors': 'GSW', 'Detroit Pistons': 'DET'
}

market_spreads = {}
market_totals = {}
for game in odds.get('games', []):
    home = game['home']
    away = game['away']
    dk_spread = game['dk'].get('spread', '-')
    dk_total = game['dk'].get('total', '-')
    home_abbrev = abbrev_map.get(home, home[:3].upper())
    away_abbrev = abbrev_map.get(away, away[:3].upper())

    if dk_spread != '-':
        try:
            market_spreads[(home_abbrev, away_abbrev)] = float(dk_spread.replace('+', ''))
        except:
            pass
    if dk_total != '-':
        try:
            market_totals[(home_abbrev, away_abbrev)] = float(dk_total)
        except:
            pass

props_lookup = {}
for p in props.get('players', []):
    lines = p.get('lines', {}).get('points', {})
    dk_line = lines.get('dk')
    fd_line = lines.get('fd')
    if dk_line and dk_line != '-':
        props_lookup[p['name']] = float(dk_line)
    elif fd_line and fd_line != '-':
        props_lookup[p['name']] = float(fd_line)

# Build CORRECTED game predictions
game_preds = []
for team, data in TODAYS_LINEUPS.items():
    if data['home']:
        home = team
        away = data['opponent']
        result = game_predictor.predict_game(df, home, away)
        if result:
            our_spread = result['spread']
            our_total = result['predicted_total']

            market_spread = market_spreads.get((home, away))
            market_total = market_totals.get((home, away))

            game_pred = {
                'home_team': home,
                'away_team': away,
                'our_spread': round(our_spread, 1),
                'dk_spread': market_spread,
                'our_total': round(our_total, 1),
                'dk_total': market_total,
                'our_home_prob': round(result['home_win_prob'], 1),
                'our_home_score': round(result['predicted_home_score'], 1),
                'our_away_score': round(result['predicted_away_score'], 1),
                'spread_pick': None,
                'spread_edge': None,
                'ou_pick': None,
                'ou_edge': None,
                'home_score': None,
                'away_score': None,
                'spread_hit': None,
                'ou_hit': None,
            }

            # CORRECTED spread edge
            if market_spread is not None:
                if our_spread >= 0:
                    our_fav = home
                    our_margin = our_spread
                else:
                    our_fav = away
                    our_margin = abs(our_spread)

                if market_spread > 0:
                    market_fav = away
                    market_margin = market_spread
                elif market_spread < 0:
                    market_fav = home
                    market_margin = abs(market_spread)
                else:
                    market_fav = None
                    market_margin = 0

                if our_fav == market_fav and market_fav is not None:
                    edge = our_margin - market_margin
                    if abs(edge) >= 2:
                        game_pred['spread_pick'] = our_fav
                        game_pred['spread_edge'] = round(abs(edge), 1)
                elif market_fav is not None:
                    edge = our_margin + market_margin
                    game_pred['spread_pick'] = our_fav
                    game_pred['spread_edge'] = round(edge, 1)

            # O/U edge
            if market_total is not None:
                total_diff = our_total - market_total
                if abs(total_diff) >= 3:
                    game_pred['ou_pick'] = 'OVER' if total_diff > 0 else 'UNDER'
                    game_pred['ou_edge'] = round(abs(total_diff), 1)

            game_preds.append(game_pred)

# Build prop predictions (5%+ edge)
prop_preds = []
for team, data in TODAYS_LINEUPS.items():
    opponent = data['opponent']
    is_home = data['home']

    for player in data['starters']:
        try:
            pred = predictor.predict(player, opponent, is_home=is_home)
            if pred:
                line = props_lookup.get(player)
                if line and line > 0:
                    edge_pct = ((pred['pts'] - line) / line) * 100
                    if abs(edge_pct) >= 5:
                        direction = 'OVER' if edge_pct > 0 else 'UNDER'
                        prop_preds.append({
                            'player': player,
                            'team': team,
                            'opponent': opponent,
                            'stat': 'PTS',
                            'projection': round(pred['pts'], 1),
                            'dk_line': line,
                            'fd_line': line,
                            'pick': direction,
                            'edge': round(abs(edge_pct), 1),
                            'confidence': pred['confidence'],
                            'result': None,
                            'hit': None
                        })
        except:
            pass

# Log to daily tracking
result = log_daily_picks(game_preds, prop_preds)
print(f"Logged {result['games']} game predictions and {result['props']} prop predictions")
print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")

# Show summary
print("\nSPREAD PICKS LOGGED:")
for g in game_preds:
    if g['spread_pick']:
        print(f"  {g['away_team']}@{g['home_team']}: {g['spread_pick']} ({g['spread_edge']:.1f} pt edge)")

print("\nO/U PICKS LOGGED:")
for g in game_preds:
    if g['ou_pick']:
        print(f"  {g['away_team']}@{g['home_team']}: {g['ou_pick']} {g['dk_total']} ({g['ou_edge']:.1f} pts off)")

print("\nTOP PROP PICKS:")
prop_preds_sorted = sorted(prop_preds, key=lambda x: x['edge'], reverse=True)
for p in prop_preds_sorted[:10]:
    print(f"  {p['player']}: {p['pick']} {p['dk_line']} ({p['projection']:.1f} proj, {p['edge']:.1f}% edge)")
