"""
CourtMind Prediction Engine
===========================
Generates player predictions for tonight's games using the SPRS neural model.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats as scipy_stats

from CourtMind.config import DATA_PATHS, CONTENT_SETTINGS


def load_model_and_data():
    """
    Load the LSTM model, scalers, and all prediction data.

    Returns:
        dict with model, scalers, df, player_stats, player_vs_team, team_matchups
    """
    try:
        import torch
        import joblib
    except ImportError:
        print("[CourtMind] Warning: torch/joblib not available")
        return None

    data = {}

    # Load main dataset
    if os.path.exists(DATA_PATHS['combined_data']):
        data['df'] = pd.read_parquet(DATA_PATHS['combined_data'])
    else:
        print(f"[CourtMind] Warning: Combined data not found")
        return None

    # Load model
    try:
        from sdis_neural_models import PlayerPerformanceModel
        data['scalers'] = joblib.load(DATA_PATHS['scalers'])
        target_cols = data['scalers'].get('target_cols', ['total_pts', 'total_fga', 'ts_pct', 'tov_rate', 'game_score', 'spm'])

        model = PlayerPerformanceModel(
            seq_input_size=len(data['scalers']['seq_features']),
            static_input_size=3,
            hidden_size=128,
            num_targets=len(target_cols)
        )
        model.load_state_dict(torch.load(DATA_PATHS['lstm_model'], weights_only=True))
        model.eval()
        data['model'] = model
        data['target_cols'] = target_cols
    except Exception as e:
        print(f"[CourtMind] Warning: Could not load model: {e}")
        data['model'] = None

    # Load player historical stats
    if os.path.exists(DATA_PATHS['player_stats']):
        with open(DATA_PATHS['player_stats'], 'rb') as f:
            data['player_stats'] = pickle.load(f)
    else:
        data['player_stats'] = {}

    # Load player vs team matchup stats
    if os.path.exists(DATA_PATHS['player_vs_team']):
        with open(DATA_PATHS['player_vs_team'], 'rb') as f:
            data['player_vs_team'] = pickle.load(f)
    else:
        data['player_vs_team'] = pd.DataFrame()

    # Load team matchups
    if os.path.exists(DATA_PATHS['team_matchups']):
        with open(DATA_PATHS['team_matchups'], 'r') as f:
            data['team_matchups'] = json.load(f)
    else:
        data['team_matchups'] = {}

    return data


def prepare_player_game_history(df, player, min_games=5):
    """
    Prepare player's game history for prediction.

    Returns DataFrame aggregated to game level with all required features.
    """
    player_df = df[df['player'] == player].copy()

    if len(player_df) == 0:
        return pd.DataFrame()

    # Aggregate quarters to game level
    game_agg = player_df.groupby(['game_id', 'game_date', 'team', 'opponent']).agg({
        'pts': 'sum',
        'fgm': 'sum',
        'fga': 'sum',
        'trb': 'sum',
        'ast': 'sum',
        'stl': 'sum',
        'blk': 'sum',
        'tov': 'sum',
        'pf': 'sum',
        'minutes': 'sum',
        'is_b2b': 'first',
    }).reset_index()

    game_agg = game_agg.sort_values('game_date')

    # Rename to expected column names
    game_agg = game_agg.rename(columns={
        'pts': 'total_pts',
        'fgm': 'total_fgm',
        'fga': 'total_fga',
        'trb': 'total_trb',
        'ast': 'total_ast',
        'stl': 'total_stl',
        'blk': 'total_blk',
        'tov': 'total_tov',
        'pf': 'total_pf',
        'minutes': 'total_minutes',
    })

    # Calculate derived stats
    game_agg['ts_pct'] = np.where(
        game_agg['total_fga'] > 0,
        game_agg['total_pts'] / (2 * game_agg['total_fga']),
        0.5
    )

    game_agg['tov_rate'] = np.where(
        (game_agg['total_fga'] + game_agg['total_tov']) > 0,
        game_agg['total_tov'] / (game_agg['total_fga'] + game_agg['total_tov']),
        0.1
    )

    # Game score (simplified)
    game_agg['game_score'] = (
        game_agg['total_pts'] +
        0.4 * game_agg['total_fgm'] -
        0.7 * game_agg['total_fga'] +
        0.7 * game_agg['total_trb'] +
        0.7 * game_agg['total_ast'] +
        game_agg['total_stl'] +
        0.7 * game_agg['total_blk'] -
        0.4 * game_agg['total_pf'] -
        game_agg['total_tov']
    )

    # SPM placeholder
    game_agg['spm'] = 0.0

    # Rolling averages
    for col in ['total_pts', 'total_fga', 'total_trb', 'total_ast']:
        game_agg[f'{col}_rolling_5'] = game_agg[col].rolling(5, min_periods=1).mean()

    # Days rest
    game_agg['game_date'] = pd.to_datetime(game_agg['game_date'])
    game_agg['prev_game_date'] = game_agg['game_date'].shift(1)
    game_agg['days_rest'] = (game_agg['game_date'] - game_agg['prev_game_date']).dt.days.fillna(3)

    if len(game_agg) < min_games:
        return pd.DataFrame()

    return game_agg


def predict_player_next_game(model, scalers, player_history, target_cols):
    """Generate prediction for a player's next game using LSTM model."""
    import torch

    if len(player_history) < 5:
        return None

    seq_scaler = scalers['seq_scaler']
    target_scaler = scalers['target_scaler']
    seq_features = scalers['seq_features']

    # Get last 10 games (or pad if less)
    recent = player_history.tail(10).copy()

    # Ensure all required sequence features exist
    for feat in seq_features:
        if feat not in recent.columns:
            recent[feat] = 0.0

    sequence = recent[seq_features].values.astype(np.float32)

    if len(sequence) < 10:
        padding = np.zeros((10 - len(sequence), len(seq_features)))
        sequence = np.vstack([padding, sequence])

    # Scale sequence
    sequence_scaled = seq_scaler.transform(sequence)

    # Prepare tensors
    seq_tensor = torch.tensor(sequence_scaled, dtype=torch.float32).unsqueeze(0)
    static_tensor = torch.tensor([0, 2, 27], dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        predictions = model(seq_tensor, static_tensor)

    # Denormalize
    pred_array = np.array([[predictions[name].numpy()[0, 0] for name in target_cols]])
    pred_denorm = target_scaler.inverse_transform(pred_array)[0]

    return {name: pred_denorm[i] for i, name in enumerate(target_cols)}


def get_player_vs_team_boost(player_vs_team_df, player, opponent):
    """Get player's historical performance boost against opponent."""
    if player_vs_team_df is None or len(player_vs_team_df) == 0:
        return None

    if isinstance(player_vs_team_df, dict):
        key = f"{player}_vs_{opponent}"
        return player_vs_team_df.get(key)

    try:
        match = player_vs_team_df[
            (player_vs_team_df['player'] == player) &
            (player_vs_team_df['opponent'] == opponent)
        ]
        if len(match) > 0:
            row = match.iloc[0]
            return {
                'pts_diff_pct': row.get('pts_diff_pct', 0),
                'games': row.get('games', 0),
            }
    except:
        pass

    return None


def calculate_90th_percentile_odds(proj_pts, pctile_90, hist_std, hot_hand_mult=1.0):
    """
    Calculate probability of reaching 90th percentile.

    Returns:
        combined_odds: Weighted average of statistical and MC methods
        ci_low, ci_high: Confidence interval
    """
    if hist_std <= 0 or proj_pts <= 0:
        return 10.0, 5.0, 15.0

    # Statistical method
    z_score = (pctile_90 - proj_pts) / hist_std
    stat_odds = (1 - scipy_stats.norm.cdf(z_score)) * 100

    # Hot hand adjustment
    hot_hand_odds_mult = 1.0
    if hot_hand_mult > 1.05:
        hot_hand_odds_mult = 1.0 + (hot_hand_mult - 1.0) * 2
    elif hot_hand_mult < 0.95:
        hot_hand_odds_mult = 0.7

    stat_odds_adjusted = min(stat_odds * hot_hand_odds_mult, 50)

    # Monte Carlo
    np.random.seed(42)
    mc_samples = np.random.normal(proj_pts, hist_std, 200)
    mc_odds = (mc_samples >= pctile_90).sum() / 200 * 100
    mc_odds_adjusted = min(mc_odds * hot_hand_odds_mult, 50)

    # Combined
    combined_odds = 0.6 * stat_odds_adjusted + 0.4 * mc_odds_adjusted
    combined_odds = max(2, min(combined_odds, 45))

    ci_low = max(0, combined_odds - 5)
    ci_high = min(50, combined_odds + 5)

    return combined_odds, ci_low, ci_high


def generate_player_prediction(player, team, opponent, data, is_b2b=False):
    """
    Generate full prediction for a single player.

    Returns:
        dict with all prediction fields or None if insufficient data
    """
    df = data['df']
    model = data.get('model')
    scalers = data.get('scalers')
    target_cols = data.get('target_cols', [])
    player_vs_team = data.get('player_vs_team')

    # Get player history
    player_history = prepare_player_game_history(df, player)
    if len(player_history) < CONTENT_SETTINGS['min_games_for_player']:
        return None

    recent_5 = player_history.tail(5)
    recent_3 = player_history.tail(3)

    # Base projection
    pts_col = 'total_pts'
    if model and scalers:
        pred = predict_player_next_game(model, scalers, player_history, target_cols)
        if pred:
            proj_pts = pred.get('total_pts', recent_5[pts_col].mean())
        else:
            proj_pts = recent_5[pts_col].mean()
    else:
        proj_pts = recent_5[pts_col].mean()

    # Historical stats
    floor_pts = player_history[pts_col].quantile(0.10)
    ceiling_pts = player_history[pts_col].quantile(0.90)
    max_pts = player_history[pts_col].max()
    season_avg = player_history[pts_col].mean()
    hist_std = player_history[pts_col].std()

    # Hot hand
    recent_3_avg = recent_3[pts_col].mean() if len(recent_3) >= 3 else season_avg
    hot_hand_mult = 1.0
    if season_avg > 0:
        hot_hand_ratio = recent_3_avg / season_avg
        if hot_hand_ratio > 1.05:
            hot_hand_mult = min(1.15, 0.85 + (hot_hand_ratio * 0.15))
        elif hot_hand_ratio < 0.95:
            hot_hand_mult = max(0.92, hot_hand_ratio * 0.97)

    # Matchup boost
    matchup_adj = 1.0
    matchup_info = ""
    vs_team = get_player_vs_team_boost(player_vs_team, player, opponent)
    if vs_team and vs_team.get('games', 0) >= 3:
        boost_pct = vs_team.get('pts_diff_pct', 0)
        matchup_adj = 1.0 + (boost_pct / 100) * 0.7
        matchup_adj = max(0.85, min(1.20, matchup_adj))
        if boost_pct > 5:
            matchup_info = f"+{boost_pct:.0f}%"
        elif boost_pct < -5:
            matchup_info = f"{boost_pct:.0f}%"

    # B2B adjustment
    b2b_adj = 0.95 if is_b2b else 1.0

    # Combined adjustment
    combined_adj = hot_hand_mult * matchup_adj * b2b_adj

    # Final projections
    final_proj_pts = proj_pts * combined_adj
    final_90th = max(ceiling_pts * (0.7 + 0.3 * combined_adj), final_proj_pts * 1.15)
    final_floor = floor_pts * (0.5 + 0.5 * combined_adj)
    final_max = max(max_pts, final_90th)

    # 90th percentile odds
    odds, odds_low, odds_high = calculate_90th_percentile_odds(
        final_proj_pts, final_90th, hist_std, hot_hand_mult
    )

    # Calculate SGDR (fatigue indicator)
    q4_games = df[(df['player'] == player) & (df['qtr'] == 'Q4')]
    q1_games = df[(df['player'] == player) & (df['qtr'] == 'Q1')]
    if len(q4_games) > 5 and len(q1_games) > 5:
        q4_fg_pct = q4_games['fgm'].sum() / q4_games['fga'].sum() * 100 if q4_games['fga'].sum() > 0 else 0
        q1_fg_pct = q1_games['fgm'].sum() / q1_games['fga'].sum() * 100 if q1_games['fga'].sum() > 0 else 0
        q4_drop = q1_fg_pct - q4_fg_pct
        sgdr = min(100, max(0, 50 + q4_drop * 5))  # Scale to 0-100
    else:
        sgdr = 50  # Neutral

    # Recent stats
    avg_reb = recent_5['total_trb'].mean()
    avg_ast = recent_5['total_ast'].mean()
    avg_mins = recent_5['total_minutes'].mean()

    return {
        'player': player,
        'team': team,
        'opponent': opponent,
        'proj_pts': round(final_proj_pts, 1),
        'floor_pts': round(final_floor, 1),
        'pctile_90': round(final_90th, 1),
        'pctile_90_odds': round(odds, 0),
        'pctile_90_odds_ci': f"{odds_low:.0f}-{odds_high:.0f}%",
        'max_pts': round(final_max, 1),
        'proj_reb': round(avg_reb * combined_adj, 1),
        'proj_ast': round(avg_ast * combined_adj, 1),
        'avg_mins': round(avg_mins, 1),
        'sgdr': round(sgdr, 0),
        'hot_hand_mult': hot_hand_mult,
        'matchup_adj': matchup_adj,
        'matchup_info': matchup_info,
        'is_b2b': is_b2b,
        'is_hot': hot_hand_mult > 1.05,
        'is_cold': hot_hand_mult < 0.95,
        'games_played': len(player_history),
    }


def generate_all_predictions(games, b2b_teams=None):
    """
    Generate predictions for all players in tonight's games.

    Args:
        games: List of game dicts from schedule_fetcher
        b2b_teams: Set of team abbreviations on back-to-back

    Returns:
        DataFrame with all predictions
    """
    if b2b_teams is None:
        b2b_teams = set()

    print("[CourtMind] Loading model and data...")
    data = load_model_and_data()
    if data is None:
        print("[CourtMind] Error: Could not load data")
        return pd.DataFrame()

    predictions = []
    df = data['df']

    # Get all teams playing tonight
    teams_playing = set()
    team_opponents = {}
    for game in games:
        teams_playing.add(game['home_team'])
        teams_playing.add(game['away_team'])
        team_opponents[game['home_team']] = game['away_team']
        team_opponents[game['away_team']] = game['home_team']

    print(f"[CourtMind] Generating predictions for {len(teams_playing)} teams...")

    # Get players from each team
    for team in teams_playing:
        team_df = df[df['team'] == team]
        if team_df.empty:
            continue

        # Get recent players (last 30 days)
        recent_games = team_df.sort_values('game_date', ascending=False)
        recent_players = recent_games['player'].unique()[:12]  # Top 12 rotation

        opponent = team_opponents.get(team, '')
        is_b2b = team in b2b_teams

        for player in recent_players:
            pred = generate_player_prediction(player, team, opponent, data, is_b2b)
            if pred:
                predictions.append(pred)

    print(f"[CourtMind] Generated {len(predictions)} player predictions")
    return pd.DataFrame(predictions)


def select_top_storylines(predictions_df, n_posts=3):
    """
    Select the top storylines for today's content.

    Returns:
        List of dicts, each representing a post storyline
    """
    if predictions_df.empty:
        return []

    storylines = []

    # 1. Best ceiling game (highest 90th odds with good matchup)
    ceiling_df = predictions_df[predictions_df['pctile_90_odds'] >= 15].copy()
    if not ceiling_df.empty:
        ceiling_df['ceiling_score'] = (
            ceiling_df['pctile_90_odds'] * 0.5 +
            ceiling_df['pctile_90'] * 0.3 +
            (ceiling_df['matchup_adj'] - 1) * 100 * 0.2
        )
        best_ceiling = ceiling_df.nlargest(1, 'ceiling_score').iloc[0]
        storylines.append({
            'type': 'ceiling_watch',
            'player': best_ceiling['player'],
            'team': best_ceiling['team'],
            'opponent': best_ceiling['opponent'],
            'data': best_ceiling.to_dict(),
        })

    # 2. Top 5 picks tonight (for summary graphic)
    top_5 = predictions_df.nlargest(5, 'proj_pts')
    if len(top_5) >= 3:
        storylines.append({
            'type': 'top_picks',
            'players': top_5.to_dict('records'),
        })

    # 3. Fatigue alerts (high SGDR on B2B)
    fatigue_df = predictions_df[
        (predictions_df['is_b2b']) &
        (predictions_df['sgdr'] >= CONTENT_SETTINGS['sgdr_alert_threshold'])
    ]
    if not fatigue_df.empty:
        fatigue_players = fatigue_df.nlargest(3, 'sgdr')
        storylines.append({
            'type': 'fatigue_alert',
            'players': fatigue_players.to_dict('records'),
        })

    # 4. Hot streak player
    hot_df = predictions_df[predictions_df['is_hot']]
    if not hot_df.empty:
        hottest = hot_df.nlargest(1, 'hot_hand_mult').iloc[0]
        if hottest['hot_hand_mult'] > 1.08:  # At least 8% hot
            storylines.append({
                'type': 'hot_streak',
                'player': hottest['player'],
                'team': hottest['team'],
                'opponent': hottest['opponent'],
                'data': hottest.to_dict(),
            })

    return storylines[:n_posts]


if __name__ == "__main__":
    # Test the prediction engine
    print("Testing prediction engine...")

    from CourtMind.schedule_fetcher import get_todays_games, identify_b2b_teams

    games = get_todays_games()
    if games:
        b2b = identify_b2b_teams(games)
        preds = generate_all_predictions(games, b2b)
        print(f"\nGenerated {len(preds)} predictions")
        if not preds.empty:
            print("\nTop 5 projected scorers:")
            print(preds.nlargest(5, 'proj_pts')[['player', 'team', 'opponent', 'proj_pts', 'pctile_90', 'pctile_90_odds']])

            storylines = select_top_storylines(preds)
            print(f"\n{len(storylines)} storylines selected")
            for s in storylines:
                print(f"  - {s['type']}")
    else:
        print("No games today")
