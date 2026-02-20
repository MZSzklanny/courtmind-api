# -*- coding: utf-8 -*-
"""
CourtMind Ensemble Model
========================
Multi-model architecture combining LSTM, XGBoost, and Ridge regression.

LSTM: Best for capturing trends and streaks over time
XGBoost: Best for contextual features (matchups, rest, fatigue)
Ridge: Baseline sanity check and regularized predictions
"""

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from models.feature_engineering import (
    engineer_player_features,
    prepare_training_data,
    calculate_opponent_defensive_rating,
    calculate_player_fatigue_index
)


# =============================================================================
# LSTM MODEL (only defined if PyTorch is available)
# =============================================================================

if TORCH_AVAILABLE:
    class LSTMPredictor(nn.Module):
        """LSTM model for time-series player performance prediction."""

        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            super(LSTMPredictor, self).__init__()

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )

            self.attention = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
                nn.Softmax(dim=1)
            )

            self.fc = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            # x shape: (batch, seq_len, features)
            lstm_out, _ = self.lstm(x)

            # Attention weights
            attn_weights = self.attention(lstm_out)
            context = torch.sum(attn_weights * lstm_out, dim=1)

            # Final prediction
            out = self.fc(context)
            return out.squeeze()


class LSTMWrapper:
    """Wrapper for training and using LSTM model."""

    def __init__(self, seq_length=10, hidden_size=64):
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_cols = None

    def prepare_sequences(self, df, player, target='pts'):
        """Prepare sequential data for a player."""
        player_games = df[df['player'] == player].copy()

        # Aggregate to game level
        game_agg = player_games.groupby(['game_id', 'game_date']).agg({
            'pts': 'sum',
            'trb': 'sum',
            'ast': 'sum',
            'stl': 'sum',
            'blk': 'sum',
            'tov': 'sum',
            'fgm': 'sum',
            'fga': 'sum',
            'minutes': 'sum',
        }).reset_index().sort_values('game_date')

        if len(game_agg) < self.seq_length + 1:
            return None, None

        # Calculate derived features
        game_agg['fg_pct'] = game_agg['fgm'] / game_agg['fga'].replace(0, 1)
        game_agg['pts_per_min'] = game_agg['pts'] / game_agg['minutes'].replace(0, 1)

        # Rolling features
        for stat in ['pts', 'trb', 'ast']:
            game_agg[f'{stat}_rolling_3'] = game_agg[stat].rolling(3, min_periods=1).mean()
            game_agg[f'{stat}_rolling_5'] = game_agg[stat].rolling(5, min_periods=1).mean()

        feature_cols = ['pts', 'trb', 'ast', 'fgm', 'fga', 'minutes', 'fg_pct',
                        'pts_rolling_3', 'pts_rolling_5', 'trb_rolling_3', 'ast_rolling_3']

        self.feature_cols = feature_cols

        # Create sequences
        sequences = []
        targets = []

        for i in range(self.seq_length, len(game_agg)):
            seq = game_agg.iloc[i-self.seq_length:i][feature_cols].values
            target_val = game_agg.iloc[i][target]
            sequences.append(seq)
            targets.append(target_val)

        return np.array(sequences), np.array(targets)

    def train(self, df, players, epochs=50, lr=0.001):
        """Train LSTM on multiple players."""
        if not TORCH_AVAILABLE:
            print("PyTorch not available - skipping LSTM training")
            return

        all_sequences = []
        all_targets = []

        print(f"Preparing sequences for {len(players)} players...")
        for player in players:
            seqs, tgts = self.prepare_sequences(df, player)
            if seqs is not None and len(seqs) > 0:
                all_sequences.append(seqs)
                all_targets.append(tgts)

        if not all_sequences:
            print("No valid sequences found")
            return

        X = np.vstack(all_sequences)
        y = np.concatenate(all_targets)

        print(f"Training on {len(X)} sequences")

        # Scale
        n_samples, seq_len, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        X_scaled = self.scaler_X.fit_transform(X_flat).reshape(n_samples, seq_len, n_features)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).squeeze()

        # Convert to tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

        # Create model
        self.model = LSTMPredictor(input_size=n_features, hidden_size=self.hidden_size)

        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

        self.model.eval()

    def predict(self, df, player):
        """Predict next game for a player."""
        if self.model is None or not TORCH_AVAILABLE:
            return None

        seqs, _ = self.prepare_sequences(df, player)
        if seqs is None or len(seqs) == 0:
            return None

        # Use most recent sequence
        seq = seqs[-1:]
        seq_flat = seq.reshape(-1, seq.shape[-1])
        seq_scaled = self.scaler_X.transform(seq_flat).reshape(1, self.seq_length, -1)

        seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32)

        with torch.no_grad():
            pred_scaled = self.model(seq_tensor).item()

        pred = self.scaler_y.inverse_transform([[pred_scaled]])[0, 0]
        return pred


# =============================================================================
# XGBOOST MODEL
# =============================================================================

class XGBoostPredictor:
    """XGBoost model for contextual feature-based prediction."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None

    def train(self, X, y):
        """Train XGBoost on prepared features."""
        if not XGB_AVAILABLE:
            print("XGBoost not available - skipping training")
            return

        self.feature_cols = list(X.columns)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train XGBoost
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        )

        self.model.fit(X_scaled, y)

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
        print(f"XGBoost CV MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

    def predict(self, features_dict):
        """Predict from feature dictionary."""
        if self.model is None or not XGB_AVAILABLE:
            return None

        # Convert dict to array in correct order
        X = np.array([[features_dict.get(col, 0) for col in self.feature_cols]])
        X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled)[0]

    def get_feature_importance(self):
        """Get feature importance ranking."""
        if self.model is None:
            return {}

        importance = dict(zip(self.feature_cols, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


# =============================================================================
# RIDGE/LASSO MODEL
# =============================================================================

class RidgePredictor:
    """Ridge regression baseline model."""

    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.feature_cols = None

    def train(self, X, y):
        """Train Ridge regression."""
        self.feature_cols = list(X.columns)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
        print(f"Ridge CV MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

    def predict(self, features_dict):
        """Predict from feature dictionary."""
        if self.feature_cols is None:
            return None

        X = np.array([[features_dict.get(col, 0) for col in self.feature_cols]])
        X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled)[0]


# =============================================================================
# ENSEMBLE MODEL
# =============================================================================

class CourtMindEnsemble:
    """
    Ensemble predictor combining LSTM, XGBoost, and Ridge.

    Weights are adjusted based on backtest performance.
    """

    def __init__(self, weights=None):
        self.lstm = LSTMWrapper() if TORCH_AVAILABLE else None
        self.xgb = XGBoostPredictor() if XGB_AVAILABLE else None
        self.ridge = RidgePredictor()

        # Default weights (can be tuned)
        self.weights = weights or {
            'lstm': 0.35,
            'xgb': 0.45,
            'ridge': 0.20
        }

        self.is_trained = False
        self.training_stats = {}

    def train(self, df, top_players, target='pts'):
        """Train all models in the ensemble."""
        print("=" * 60)
        print("TRAINING COURTMIND ENSEMBLE")
        print("=" * 60)

        # Prepare training data for XGBoost and Ridge
        print("\nPreparing training data...")
        X, y = prepare_training_data(df, target_stat=target)
        print(f"Training samples: {len(X)}")

        # Train XGBoost
        if self.xgb:
            print("\n[1/3] Training XGBoost...")
            self.xgb.train(X, y)

        # Train Ridge
        print("\n[2/3] Training Ridge...")
        self.ridge.train(X, y)

        # Train LSTM
        if self.lstm:
            print("\n[3/3] Training LSTM...")
            self.lstm.train(df, top_players, epochs=30)

        self.is_trained = True
        print("\n" + "=" * 60)
        print("ENSEMBLE TRAINING COMPLETE")
        print("=" * 60)

    def predict(self, df, player, opponent, game_date, is_home=True):
        """
        Generate ensemble prediction for a player.

        Returns:
            dict with predictions from each model and ensemble
        """
        if not self.is_trained:
            print("Warning: Model not trained")
            return None

        # Get features
        features = engineer_player_features(df, player, opponent, game_date, is_home)
        if features is None:
            return None

        predictions = {}
        valid_weights = {}

        # LSTM prediction
        if self.lstm:
            lstm_pred = self.lstm.predict(df, player)
            if lstm_pred is not None:
                predictions['lstm'] = lstm_pred
                valid_weights['lstm'] = self.weights['lstm']

        # XGBoost prediction
        if self.xgb:
            xgb_pred = self.xgb.predict(features)
            if xgb_pred is not None:
                predictions['xgb'] = xgb_pred
                valid_weights['xgb'] = self.weights['xgb']

        # Ridge prediction
        ridge_pred = self.ridge.predict(features)
        if ridge_pred is not None:
            predictions['ridge'] = ridge_pred
            valid_weights['ridge'] = self.weights['ridge']

        if not predictions:
            return None

        # Normalize weights
        total_weight = sum(valid_weights.values())
        normalized_weights = {k: v/total_weight for k, v in valid_weights.items()}

        # Ensemble prediction
        ensemble_pred = sum(predictions[k] * normalized_weights[k] for k in predictions)

        return {
            'player': player,
            'opponent': opponent,
            'predictions': predictions,
            'weights_used': normalized_weights,
            'ensemble': ensemble_pred,
            'features': features
        }

    def predict_all_stats(self, df, player, opponent, game_date, is_home=True):
        """
        Predict all stats (pts, trb, ast, etc.) for a player.
        Note: This uses pts model for now - full implementation would train separate models.
        """
        base_pred = self.predict(df, player, opponent, game_date, is_home)
        if base_pred is None:
            return None

        features = base_pred['features']

        # Use feature averages for other stats
        result = {
            'player': player,
            'opponent': opponent,
            'is_home': is_home,
            'pts_proj': base_pred['ensemble'],
            'pts_models': base_pred['predictions'],
            'trb_proj': features.get('trb_avg_5', 5.0),
            'ast_proj': features.get('ast_avg_5', 3.0),
            'fatigue_index': features.get('fatigue_index', 50),
            'opp_def_rating': features.get('opp_def_rating', 1.0),
            'vs_opp_factor': features.get('vs_opp_factor', 1.0),
            'ceiling_90': features.get('ceiling_90', base_pred['ensemble'] * 1.3),
            'floor_10': features.get('floor_10', base_pred['ensemble'] * 0.6),
            'confidence': self._calculate_confidence(base_pred)
        }

        return result

    def _calculate_confidence(self, pred):
        """Calculate prediction confidence based on model agreement."""
        if len(pred['predictions']) < 2:
            return 0.5

        values = list(pred['predictions'].values())
        mean_val = np.mean(values)
        std_val = np.std(values)

        # Coefficient of variation (lower = more confident)
        cv = std_val / mean_val if mean_val > 0 else 1.0

        # Convert to confidence score (0-100)
        confidence = max(0, min(100, (1 - cv) * 100))

        return confidence

    def save(self, path):
        """Save ensemble model."""
        with open(path, 'wb') as f:
            pickle.dump({
                'xgb': self.xgb,
                'ridge': self.ridge,
                'lstm_scaler_X': self.lstm.scaler_X if self.lstm else None,
                'lstm_scaler_y': self.lstm.scaler_y if self.lstm else None,
                'weights': self.weights,
                'is_trained': self.is_trained
            }, f)

        # Save LSTM model separately if exists
        if self.lstm and self.lstm.model:
            torch.save(self.lstm.model.state_dict(), path.replace('.pkl', '_lstm.pth'))

        print(f"Ensemble saved to {path}")

    def load(self, path):
        """Load ensemble model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.xgb = data['xgb']
        self.ridge = data['ridge']
        self.weights = data['weights']
        self.is_trained = data['is_trained']

        # Load LSTM
        lstm_path = path.replace('.pkl', '_lstm.pth')
        if self.lstm and os.path.exists(lstm_path):
            self.lstm.scaler_X = data['lstm_scaler_X']
            self.lstm.scaler_y = data['lstm_scaler_y']
            # Note: Would need to reconstruct model architecture

        print(f"Ensemble loaded from {path}")


# =============================================================================
# GAME PREDICTION (W/L, O/U)
# =============================================================================

class GamePredictor:
    """Predict game outcomes (win/loss, over/under, spread) using team standings."""

    def __init__(self):
        self.win_model = None
        self.total_model = None
        self.scaler = StandardScaler()
        self.team_records = {}

    def calculate_team_records(self, df):
        """Calculate team records from game data."""
        try:
            gdf = pd.read_parquet('C:/Users/user/NBA_Game_PRODUCTION.parquet')
            gdf['game_date'] = pd.to_datetime(gdf['game_date'])
            current = gdf[gdf['game_date'] >= '2025-10-01'].copy()

            valid_teams = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
                           'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
                           'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']

            for team in valid_teams:
                team_games = current[current['team'] == team].drop_duplicates('game_id')
                wins = len(team_games[team_games['win_loss'] == 'W'])
                losses = len(team_games[team_games['win_loss'] == 'L'])
                total = wins + losses
                if total > 0:
                    self.team_records[team] = {
                        'wins': wins,
                        'losses': losses,
                        'win_pct': wins / total,
                        'games': total
                    }
        except Exception as e:
            print(f"Error calculating records: {e}")

    def prepare_team_features(self, df, team, opponent, n_games=10):
        """Prepare team-level features for game prediction."""
        team_games = df[df['team'] == team].copy()
        opp_games = df[df['team'] == opponent].copy()

        if len(team_games) == 0 or len(opp_games) == 0:
            return None

        # Team stats (last n games) - use actual game scores, not summed player pts
        # (player pts are quarter-level and incomplete)
        team_recent = team_games.groupby('game_id').agg({
            'pts': 'sum',
            'trb': 'sum',
            'ast': 'sum',
            'fgm': 'sum',
            'fga': 'sum',
            'tov': 'sum',
            'home_team': 'first',
            'away_team': 'first',
            'home_score': 'first',
            'away_score': 'first'
        }).tail(n_games)

        opp_recent = opp_games.groupby('game_id').agg({
            'pts': 'sum',
            'trb': 'sum',
            'ast': 'sum',
            'fgm': 'sum',
            'fga': 'sum',
            'tov': 'sum',
            'home_team': 'first',
            'away_team': 'first',
            'home_score': 'first',
            'away_score': 'first'
        }).tail(n_games)

        # Calculate actual PPG using home_score/away_score columns
        def get_actual_ppg(games_df, team_abbr):
            scores = []
            for _, row in games_df.iterrows():
                if row['home_team'] == team_abbr:
                    scores.append(row['home_score'])
                else:
                    scores.append(row['away_score'])
            return np.mean(scores) if scores else 110.0  # Default to league avg

        team_ppg = get_actual_ppg(team_recent, team)
        opp_ppg = get_actual_ppg(opp_recent, opponent)

        features = {
            'team_ppg': team_ppg,
            'team_rpg': team_recent['trb'].mean(),
            'team_apg': team_recent['ast'].mean(),
            'team_fg_pct': team_recent['fgm'].sum() / team_recent['fga'].sum() if team_recent['fga'].sum() > 0 else 0.45,
            'team_tov': team_recent['tov'].mean(),
            'opp_ppg': opp_ppg,
            'opp_rpg': opp_recent['trb'].mean(),
            'opp_fg_pct': opp_recent['fgm'].sum() / opp_recent['fga'].sum() if opp_recent['fga'].sum() > 0 else 0.45,
            'opp_tov': opp_recent['tov'].mean(),
            'opp_def_rating': calculate_opponent_defensive_rating(df, opponent),
        }

        # Differentials
        features['ppg_diff'] = features['team_ppg'] - features['opp_ppg']
        features['fg_pct_diff'] = features['team_fg_pct'] - features['opp_fg_pct']

        return features

    def predict_game(self, df, home_team, away_team):
        """
        Predict game outcome using team stats AND standings.

        Returns dict with:
        - home_win_prob: Probability of home team winning
        - predicted_home_score: Projected home team score
        - predicted_away_score: Projected away team score
        - predicted_total: Projected total points
        - spread: Projected point differential
        """
        # Calculate records if not done
        if not self.team_records:
            self.calculate_team_records(df)

        home_features = self.prepare_team_features(df, home_team, away_team)
        away_features = self.prepare_team_features(df, away_team, home_team)

        if home_features is None or away_features is None:
            return None

        # Get team records
        home_record = self.team_records.get(home_team, {'win_pct': 0.5})
        away_record = self.team_records.get(away_team, {'win_pct': 0.5})

        # Base prediction from PPG and defense
        home_avg_ppg = home_features['team_ppg']
        away_avg_ppg = away_features['team_ppg']

        # Adjust for opponent defense
        # Home team scores against away team's defense (home_features has away team's def)
        # Away team scores against home team's defense (away_features has home team's def)
        home_adj = home_avg_ppg * home_features['opp_def_rating']
        away_adj = away_avg_ppg * away_features['opp_def_rating']

        # Home court advantage (~3 points)
        home_adj += 3.0

        # Win probability combining multiple factors:
        # 1. PPG-based spread (20% weight) - useful but noisy
        spread_from_ppg = home_adj - away_adj
        prob_from_spread = 1 / (1 + np.exp(-spread_from_ppg / 6))

        # 2. Win percentage differential (60% weight) - best predictor
        # OKC (0.833) vs LAL (0.667) = 0.167 diff -> OKC should be ~65% favorite
        win_pct_diff = home_record['win_pct'] - away_record['win_pct']
        prob_from_record = 0.5 + (win_pct_diff * 1.0)  # Full scaling
        prob_from_record = max(0.15, min(0.85, prob_from_record))

        # 3. Home court bonus (20% weight) - about 55% home win rate historically
        home_court_prob = 0.55

        # Combine - records matter most
        home_win_prob = (
            0.20 * prob_from_spread +
            0.60 * prob_from_record +
            0.20 * home_court_prob
        )

        # Adjust spread based on win probability
        # Convert probability back to spread equivalent
        if home_win_prob > 0.5:
            implied_spread = -np.log((1 - home_win_prob) / home_win_prob) * 4
        else:
            implied_spread = np.log(home_win_prob / (1 - home_win_prob)) * 4

        # Blend PPG spread with implied spread
        final_spread = 0.6 * spread_from_ppg + 0.4 * implied_spread

        # Adjust scores to be consistent with final spread
        # Keep total the same, shift based on spread difference
        total = home_adj + away_adj
        spread_adjustment = (final_spread - spread_from_ppg) / 2
        final_home_score = home_adj + spread_adjustment
        final_away_score = away_adj - spread_adjustment

        # Derive final win probability from spread (ensures consistency)
        # Positive spread = home favored, negative = away favored
        # ~2.5 pts = 10% swing in win probability
        final_home_win_prob = 1 / (1 + np.exp(-final_spread / 5))

        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_win_prob': round(final_home_win_prob * 100, 1),
            'away_win_prob': round((1 - final_home_win_prob) * 100, 1),
            'predicted_home_score': round(final_home_score, 1),
            'predicted_away_score': round(final_away_score, 1),
            'predicted_total': round(total, 1),
            'spread': round(final_spread, 1),
            'home_record': f"{home_record.get('wins', 0)}-{home_record.get('losses', 0)}",
            'away_record': f"{away_record.get('wins', 0)}-{away_record.get('losses', 0)}",
        }


if __name__ == "__main__":
    print("Testing CourtMind Ensemble...")

    # Load data
    df = pd.read_parquet('C:/Users/user/NBA_PRODUCTION.parquet')

    # Top 50 players
    top_50 = [
        'Shai Gilgeous-Alexander', 'Luka Dončić', 'Giannis Antetokounmpo',
        'Anthony Edwards', 'Nikola Jokić', 'Tyrese Maxey', 'Jalen Brunson',
        'Kevin Durant', 'Cade Cunningham', 'Jayson Tatum'
    ]

    # Train ensemble
    ensemble = CourtMindEnsemble()
    ensemble.train(df, top_50)

    # Test prediction
    print("\nTesting prediction for SGA...")
    pred = ensemble.predict_all_stats(df, 'Shai Gilgeous-Alexander', 'LAL', '2026-01-28')
    if pred:
        print(f"  PTS Projection: {pred['pts_proj']:.1f}")
        print(f"  Ceiling (90th): {pred['ceiling_90']:.1f}")
        print(f"  Floor (10th): {pred['floor_10']:.1f}")
        print(f"  Confidence: {pred['confidence']:.1f}%")

    # Test game prediction
    print("\nTesting game prediction OKC vs LAL...")
    game_pred = GamePredictor()
    game = game_pred.predict_game(df, 'OKC', 'LAL')
    if game:
        print(f"  OKC Win Prob: {game['home_win_prob']:.1f}%")
        print(f"  Predicted: OKC {game['predicted_home_score']:.1f} - LAL {game['predicted_away_score']:.1f}")
        print(f"  Total: {game['predicted_total']:.1f}")
        print(f"  Spread: OKC {game['spread']:+.1f}")
