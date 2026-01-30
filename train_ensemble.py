# -*- coding: utf-8 -*-
"""
CourtMind Ensemble Training Script
==================================
Train the multi-model ensemble on production data.
"""

import sys
import os
sys.path.insert(0, 'C:/Users/user')
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from datetime import datetime

# Top 50 players to train on
TOP_50_PLAYERS = [
    'Shai Gilgeous-Alexander', 'Luka Dončić', 'Giannis Antetokounmpo',
    'Anthony Edwards', 'Nikola Jokić', 'Tyrese Maxey', 'Jalen Brunson',
    'Kevin Durant', 'Cade Cunningham', 'Jayson Tatum', 'Devin Booker',
    'Joel Embiid', 'Stephen Curry', 'Donovan Mitchell', 'Paolo Banchero',
    'Kawhi Leonard', 'Jaylen Brown', 'Kyrie Irving', 'LeBron James',
    'Damian Lillard', 'Victor Wembanyama', 'Jamal Murray', 'Trae Young',
    'James Harden', 'Tyler Herro', 'Zion Williamson', 'Ja Morant',
    'Zach LaVine', 'Franz Wagner', 'Anthony Davis', 'Lauri Markkanen',
    'Karl-Anthony Towns', 'LaMelo Ball', 'Austin Reaves', 'Jalen Green',
    'Norman Powell', 'Trey Murphy III', 'Brandon Ingram', 'DeMar DeRozan',
    'Brandon Miller', 'Jalen Johnson', 'Pascal Siakam', 'Jaren Jackson Jr.',
    "De'Aaron Fox", 'Deni Avdija', 'RJ Barrett', 'CJ McCollum',
    'Coby White', 'Darius Garland', 'Jalen Williams'
]

def main():
    print("=" * 70)
    print("COURTMIND ENSEMBLE TRAINING")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print()

    # Load production data
    print("[1/4] Loading production data...")
    df = pd.read_parquet('C:/Users/user/NBA_PRODUCTION.parquet')
    print(f"  Records: {len(df):,}")
    print(f"  Players: {df['player'].nunique()}")
    print(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")

    # Filter to top 50 players for training
    print(f"\n[2/4] Filtering to top 50 players...")
    df_top50 = df[df['player'].isin(TOP_50_PLAYERS)]
    print(f"  Records after filter: {len(df_top50):,}")

    # Import ensemble
    print("\n[3/4] Initializing ensemble models...")
    from CourtMind.models.ensemble_model import CourtMindEnsemble, GamePredictor

    # Create and train ensemble
    ensemble = CourtMindEnsemble(weights={
        'lstm': 0.35,
        'xgb': 0.45,
        'ridge': 0.20
    })

    # Train on full dataset (using all players for broader patterns)
    print("\n[4/4] Training ensemble...")
    ensemble.train(df, TOP_50_PLAYERS)

    # Save models
    model_path = 'C:/Users/user/CourtMind/models/ensemble_model.pkl'
    ensemble.save(model_path)

    # Test predictions on sample players
    print("\n" + "=" * 70)
    print("VALIDATION - Sample Predictions")
    print("=" * 70)

    test_matchups = [
        ('Shai Gilgeous-Alexander', 'LAL'),
        ('Tyrese Maxey', 'BOS'),
        ('LeBron James', 'PHI'),
        ('Giannis Antetokounmpo', 'MIA'),
        ('Stephen Curry', 'DAL'),
    ]

    for player, opponent in test_matchups:
        pred = ensemble.predict_all_stats(df, player, opponent, '2026-01-28')
        if pred:
            print(f"\n{player} vs {opponent}:")
            print(f"  PTS: {pred['pts_proj']:.1f} (Floor: {pred['floor_10']:.1f}, Ceiling: {pred['ceiling_90']:.1f})")
            print(f"  TRB: {pred['trb_proj']:.1f}, AST: {pred['ast_proj']:.1f}")
            print(f"  Confidence: {pred['confidence']:.0f}%")
            print(f"  Model predictions: {', '.join(f'{k}: {v:.1f}' for k,v in pred['pts_models'].items())}")

    # Test game predictions
    print("\n" + "=" * 70)
    print("GAME PREDICTIONS")
    print("=" * 70)

    game_pred = GamePredictor()
    games = [
        ('OKC', 'LAL'),
        ('BOS', 'PHI'),
        ('MIL', 'MIA'),
        ('DAL', 'DEN'),
    ]

    for home, away in games:
        result = game_pred.predict_game(df, home, away)
        if result:
            print(f"\n{away} @ {home}:")
            print(f"  {home} Win Prob: {result['home_win_prob']:.0f}%")
            print(f"  Projected: {home} {result['predicted_home_score']:.0f} - {away} {result['predicted_away_score']:.0f}")
            print(f"  Total: {result['predicted_total']:.0f} | Spread: {home} {result['spread']:+.1f}")

    print("\n" + "=" * 70)
    print(f"Training complete: {datetime.now()}")
    print(f"Model saved to: {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
