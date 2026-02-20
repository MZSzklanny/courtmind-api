# -*- coding: utf-8 -*-
"""
CourtMind Feature Engineering Pipeline
======================================
Travel distance, fatigue metrics, opponent adjustments, and contextual features.
"""

import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime, timedelta

# Team arena locations (lat, lon)
TEAM_LOCATIONS = {
    'ATL': (33.7573, -84.3963),
    'BOS': (42.3662, -71.0621),
    'BKN': (40.6826, -73.9754),
    'CHA': (35.2251, -80.8392),
    'CHI': (41.8807, -87.6742),
    'CLE': (41.4965, -81.6882),
    'DAL': (32.7905, -96.8103),
    'DEN': (39.7487, -105.0077),
    'DET': (42.3410, -83.0550),
    'GSW': (37.7680, -122.3879),
    'HOU': (29.7508, -95.3621),
    'IND': (39.7640, -86.1555),
    'LAC': (34.0430, -118.2673),
    'LAL': (34.0430, -118.2673),
    'MEM': (35.1382, -90.0505),
    'MIA': (25.7814, -80.1870),
    'MIL': (43.0451, -87.9172),
    'MIN': (44.9795, -93.2760),
    'NOP': (29.9490, -90.0821),
    'NYK': (40.7505, -73.9934),
    'OKC': (35.4634, -97.5151),
    'ORL': (28.5392, -81.3839),
    'PHI': (39.9012, -75.1720),
    'PHX': (33.4457, -112.0712),
    'POR': (45.5316, -122.6668),
    'SAC': (38.5802, -121.4997),
    'SAS': (29.4270, -98.4375),
    'TOR': (43.6435, -79.3791),
    'UTA': (40.7683, -111.9011),
    'WAS': (38.8981, -77.0209),
}


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in miles."""
    R = 3959  # Earth's radius in miles

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c


def calculate_travel_distance(prev_team_loc, curr_team_loc, is_home):
    """
    Calculate travel distance for a game.

    If home game: travel from prev location to home
    If away game: travel from prev location to away venue
    """
    if prev_team_loc is None:
        return 0

    prev_coords = TEAM_LOCATIONS.get(prev_team_loc)
    curr_coords = TEAM_LOCATIONS.get(curr_team_loc)

    if prev_coords is None or curr_coords is None:
        return 0

    return haversine_distance(prev_coords[0], prev_coords[1],
                              curr_coords[0], curr_coords[1])


def calculate_team_pace(df, team, n_games=10):
    """Calculate team's recent pace (possessions per game estimate)."""
    team_games = df[df['team'] == team].copy()
    if len(team_games) == 0:
        return 100.0  # League average

    # Aggregate to game level
    game_stats = team_games.groupby('game_id').agg({
        'fga': 'sum',
        'tov': 'sum',
        'trb': 'sum',  # Using for offensive rebounds estimate
    }).tail(n_games)

    if len(game_stats) == 0:
        return 100.0

    # Pace = FGA + TOV (simplified)
    pace = (game_stats['fga'] + game_stats['tov']).mean()
    return pace


def calculate_opponent_defensive_rating(df, opponent, n_games=10):
    """
    Calculate opponent's defensive rating.
    Lower = better defense.
    Returns factor relative to league average (1.0 = average).
    """
    # Get games where opponent was playing
    opp_games = df[df['team'] == opponent].copy()
    if len(opp_games) == 0:
        return 1.0  # League average

    # Get unique game IDs
    game_ids = opp_games['game_id'].unique()[-n_games:]

    # Get opponents' scoring in those games
    all_games = df[df['game_id'].isin(game_ids)]
    opp_allowed = all_games[all_games['team'] != opponent]

    if len(opp_allowed) == 0:
        return 1.0

    # Points allowed per game
    pts_allowed = opp_allowed.groupby('game_id')['pts'].sum().mean()

    # League average is ~115 PPG
    league_avg = 115.0

    return pts_allowed / league_avg


def calculate_player_fatigue_index(df, player, current_date):
    """
    Calculate player fatigue index based on:
    - Recent minutes load
    - Games in last 7 days
    - Back-to-back situations
    - Travel distance

    Returns value 0-100 (higher = more fatigued)
    """
    player_games = df[df['player'] == player].copy()
    if len(player_games) == 0:
        return 50.0

    player_games['game_date'] = pd.to_datetime(player_games['game_date'])
    current_date = pd.to_datetime(current_date)

    # Games in last 7 days
    last_7_days = player_games[player_games['game_date'] >= current_date - timedelta(days=7)]
    games_in_week = len(last_7_days['game_id'].unique())

    # Games in last 3 days
    last_3_days = player_games[player_games['game_date'] >= current_date - timedelta(days=3)]
    games_in_3days = len(last_3_days['game_id'].unique())

    # Average minutes in last 5 games
    recent_games = player_games.drop_duplicates('game_id').tail(5)
    if 'minutes' in recent_games.columns:
        recent_mins = recent_games['minutes'].mean()
    else:
        recent_mins = 32.0  # Default

    # Fatigue calculation
    fatigue = 0

    # Games load (max 30 points)
    fatigue += min(30, games_in_week * 6)

    # B2B detection (max 20 points)
    fatigue += min(20, games_in_3days * 10)

    # Minutes load (max 25 points)
    if recent_mins > 36:
        fatigue += 25
    elif recent_mins > 34:
        fatigue += 20
    elif recent_mins > 32:
        fatigue += 10

    # Age factor would go here if we had age data
    # Injury history factor would go here

    return min(100, max(0, fatigue))


def calculate_home_away_split(df, player, is_home):
    """Calculate player's home vs away performance split."""
    player_games = df[df['player'] == player].copy()

    if 'is_home' not in player_games.columns:
        return 1.0  # No adjustment

    home_games = player_games[player_games['is_home'] == True]
    away_games = player_games[player_games['is_home'] == False]

    if len(home_games) < 3 or len(away_games) < 3:
        return 1.0

    home_ppg = home_games.groupby('game_id')['pts'].sum().mean()
    away_ppg = away_games.groupby('game_id')['pts'].sum().mean()

    if is_home:
        return home_ppg / ((home_ppg + away_ppg) / 2)
    else:
        return away_ppg / ((home_ppg + away_ppg) / 2)


def calculate_rest_days(df, player, current_date):
    """Calculate days since last game."""
    player_games = df[df['player'] == player].copy()
    if len(player_games) == 0:
        return 3  # Default

    player_games['game_date'] = pd.to_datetime(player_games['game_date'])
    current_date = pd.to_datetime(current_date)

    last_game = player_games['game_date'].max()
    rest_days = (current_date - last_game).days

    return max(0, min(rest_days, 14))  # Cap at 14


def calculate_player_vs_opponent(df, player, opponent, stat='pts'):
    """Calculate player's historical performance vs specific opponent."""
    # This requires opponent data in the dataframe
    if 'opponent' not in df.columns:
        return {'avg': None, 'games': 0, 'factor': 1.0}

    player_vs_opp = df[(df['player'] == player) & (df['opponent'] == opponent)]

    if len(player_vs_opp) < 2:
        return {'avg': None, 'games': 0, 'factor': 1.0}

    # Aggregate to game level
    vs_opp_games = player_vs_opp.groupby('game_id')[stat].sum()
    all_games = df[df['player'] == player].groupby('game_id')[stat].sum()

    vs_opp_avg = vs_opp_games.mean()
    career_avg = all_games.mean()

    factor = vs_opp_avg / career_avg if career_avg > 0 else 1.0

    return {
        'avg': vs_opp_avg,
        'games': len(vs_opp_games),
        'factor': factor
    }


def calculate_spm(df, player, n_games=20, quarter_df=None):
    """
    Calculate Statistical Plus-Minus (SPM) based on Q4 vs Q1 performance.

    Can use either:
    - df with 'qtr' column (quarter data)
    - separate quarter_df if main df doesn't have quarters

    Returns:
        spm_score: Higher = more fatigue resistant (good)
        q4_degradation: Percentage drop from Q1 to Q4
    """
    # Determine which dataframe has quarter data
    qtr_data = None
    if 'qtr' in df.columns:
        qtr_data = df
    elif quarter_df is not None and 'qtr' in quarter_df.columns:
        qtr_data = quarter_df

    if qtr_data is None:
        return {'spm': 0, 'q4_degradation': 0}

    player_games = qtr_data[qtr_data['player'] == player].copy()
    if len(player_games) < 20:
        return {'spm': 0, 'q4_degradation': 0}

    # Get recent games
    recent_game_ids = player_games['game_id'].unique()[-n_games:]
    recent = player_games[player_games['game_id'].isin(recent_game_ids)]

    q1_data = recent[recent['qtr'] == 'Q1']
    q4_data = recent[recent['qtr'] == 'Q4']

    if len(q1_data) < 5 or len(q4_data) < 5:
        return {'spm': 0, 'q4_degradation': 0}

    # Points per minute in each quarter
    q1_ppm = q1_data['pts'].sum() / q1_data['minutes'].sum() if q1_data['minutes'].sum() > 0 else 0
    q4_ppm = q4_data['pts'].sum() / q4_data['minutes'].sum() if q4_data['minutes'].sum() > 0 else 0

    # FG% in each quarter
    q1_fg = q1_data['fgm'].sum() / q1_data['fga'].sum() if q1_data['fga'].sum() > 0 else 0.45
    q4_fg = q4_data['fgm'].sum() / q4_data['fga'].sum() if q4_data['fga'].sum() > 0 else 0.45

    # Degradation (negative = worse in Q4)
    ppm_degradation = (q4_ppm - q1_ppm) / q1_ppm * 100 if q1_ppm > 0 else 0
    fg_degradation = (q4_fg - q1_fg) / q1_fg * 100 if q1_fg > 0 else 0

    # SPM score: positive = fatigue resistant
    spm = (ppm_degradation + fg_degradation) / 2

    return {
        'spm': spm,
        'q4_degradation': -ppm_degradation,  # Positive = worse in Q4
        'q1_fg_pct': q1_fg,
        'q4_fg_pct': q4_fg
    }


# Global cache for supplemental data (loaded once)
_QUARTER_DATA_CACHE = None
_GAME_DATA_CACHE = None

def get_quarter_data():
    """Load quarter data (cached) - for SPM calculations."""
    global _QUARTER_DATA_CACHE
    if _QUARTER_DATA_CACHE is None:
        try:
            _QUARTER_DATA_CACHE = pd.read_parquet('C:/Users/user/NBA_Quarter_ALL_Combined.parquet')
        except:
            _QUARTER_DATA_CACHE = pd.DataFrame()
    return _QUARTER_DATA_CACHE


def get_game_data():
    """Load game-level data with home/away info (cached) - for travel calculations."""
    global _GAME_DATA_CACHE
    if _GAME_DATA_CACHE is None:
        try:
            _GAME_DATA_CACHE = pd.read_parquet('C:/Users/user/NBA_Game_PRODUCTION.parquet')
        except:
            _GAME_DATA_CACHE = pd.DataFrame()
    return _GAME_DATA_CACHE


def calculate_schedule_density(df, player, game_date, team):
    """
    Calculate schedule density and travel burden.

    Uses game-level data with home/away info for travel calculations.

    Returns:
        games_in_week: Games played in last 7 days
        games_in_3days: Games in last 3 days (B2B indicator)
        travel_miles_week: Total travel distance in last 7 days
        is_b2b: True if back-to-back
    """
    player_games = df[df['player'] == player].copy()
    if len(player_games) == 0:
        return {'games_in_week': 0, 'games_in_3days': 0, 'travel_miles_week': 0, 'is_b2b': False}

    player_games['game_date'] = pd.to_datetime(player_games['game_date'])
    game_date = pd.to_datetime(game_date)

    # Games in last 7 days
    week_ago = game_date - timedelta(days=7)
    last_week = player_games[player_games['game_date'] >= week_ago]
    games_in_week = len(last_week['game_id'].unique())

    # Games in last 3 days
    three_days_ago = game_date - timedelta(days=3)
    last_3days = player_games[player_games['game_date'] >= three_days_ago]
    games_in_3days = len(last_3days['game_id'].unique())

    # Back-to-back check
    yesterday = game_date - timedelta(days=1)
    is_b2b = len(player_games[player_games['game_date'] == yesterday]) > 0

    # Calculate travel distance using game data with home/away info
    travel_miles_week = 0
    game_df = get_game_data()

    if len(game_df) > 0:
        player_game_data = game_df[game_df['player'] == player].copy()
        player_game_data['game_date'] = pd.to_datetime(player_game_data['game_date'])

        recent_games = player_game_data[player_game_data['game_date'] >= week_ago]
        recent_games = recent_games.drop_duplicates('game_id').sort_values('game_date')

        if len(recent_games) > 1 and 'home_team' in recent_games.columns:
            prev_location = None
            for _, game in recent_games.iterrows():
                # Determine game location (where they played)
                if game.get('is_home', False):
                    curr_location = game.get('home_team', team)
                else:
                    # Away game - they traveled to the home team's city
                    curr_location = game.get('home_team', team)

                if prev_location and curr_location and prev_location != curr_location:
                    if prev_location in TEAM_LOCATIONS and curr_location in TEAM_LOCATIONS:
                        travel_miles_week += haversine_distance(
                            TEAM_LOCATIONS[prev_location][0], TEAM_LOCATIONS[prev_location][1],
                            TEAM_LOCATIONS[curr_location][0], TEAM_LOCATIONS[curr_location][1]
                        )
                prev_location = curr_location

    return {
        'games_in_week': games_in_week,
        'games_in_3days': games_in_3days,
        'travel_miles_week': travel_miles_week,
        'is_b2b': is_b2b
    }


def engineer_player_features(df, player, opponent, game_date, is_home=True):
    """
    Create full feature set for a player prediction.

    NOW INCLUDES:
    - Travel distance
    - SPM (Statistical Plus-Minus)
    - Schedule density
    - Opponent pace

    Returns dict with all engineered features.
    """
    features = {}

    # Basic recent performance (last 5, 10 games)
    player_games = df[df['player'] == player].copy()
    if len(player_games) == 0:
        return None

    team = player_games['team'].mode().iloc[0] if len(player_games) > 0 else None

    # Aggregate to game level first
    agg_dict = {
        'pts': 'sum',
        'trb': 'sum',
        'ast': 'sum',
        'stl': 'sum',
        'blk': 'sum',
        'tov': 'sum',
        'fgm': 'sum',
        'fga': 'sum',
        'minutes': 'sum',
    }
    if 'fg3m' in player_games.columns:
        agg_dict['fg3m'] = 'sum'

    game_agg = player_games.groupby(['game_id', 'game_date']).agg(agg_dict).reset_index().sort_values('game_date')

    if len(game_agg) < 5:
        return None

    recent_5 = game_agg.tail(5)
    recent_10 = game_agg.tail(10)

    # Rolling stats
    features['pts_avg_5'] = recent_5['pts'].mean()
    features['pts_avg_10'] = recent_10['pts'].mean()
    features['pts_std_10'] = recent_10['pts'].std()
    features['trb_avg_5'] = recent_5['trb'].mean()
    features['ast_avg_5'] = recent_5['ast'].mean()
    features['minutes_avg_5'] = recent_5['minutes'].mean()

    # Trend features
    if len(recent_10) >= 5:
        first_half = recent_10.head(5)['pts'].mean()
        second_half = recent_10.tail(5)['pts'].mean()
        features['pts_trend'] = (second_half - first_half) / first_half if first_half > 0 else 0
    else:
        features['pts_trend'] = 0

    # Efficiency
    total_fga = recent_5['fga'].sum()
    total_pts = recent_5['pts'].sum()
    features['ts_pct_5'] = total_pts / (2 * total_fga) if total_fga > 0 else 0.5

    # =====================================================
    # SCHEDULE & TRAVEL FEATURES (NEW)
    # =====================================================
    schedule = calculate_schedule_density(df, player, game_date, team)
    features['games_in_week'] = schedule['games_in_week']
    features['games_in_3days'] = schedule['games_in_3days']
    features['travel_miles_week'] = schedule['travel_miles_week']
    features['is_b2b'] = 1 if schedule['is_b2b'] else 0

    # Travel fatigue factor (0-1, higher = more fatigued)
    # 2000+ miles in a week = significant fatigue
    features['travel_fatigue'] = min(1.0, schedule['travel_miles_week'] / 3000)

    # =====================================================
    # SPM FEATURES (NEW) - Uses quarter data if available
    # =====================================================
    quarter_df = get_quarter_data()
    spm_data = calculate_spm(df, player, quarter_df=quarter_df)
    features['spm'] = spm_data['spm']
    features['q4_degradation'] = spm_data['q4_degradation']

    # Context features
    features['rest_days'] = calculate_rest_days(df, player, game_date)
    features['fatigue_index'] = calculate_player_fatigue_index(df, player, game_date)
    features['home_away_factor'] = calculate_home_away_split(df, player, is_home)
    features['is_home'] = 1 if is_home else 0

    # =====================================================
    # OPPONENT FEATURES
    # =====================================================
    features['opp_def_rating'] = calculate_opponent_defensive_rating(df, opponent)
    features['opp_pace'] = calculate_team_pace(df, opponent)

    # Player vs opponent history
    vs_opp = calculate_player_vs_opponent(df, player, opponent)
    features['vs_opp_factor'] = vs_opp['factor']
    features['vs_opp_games'] = vs_opp['games']

    # Season averages (for baseline)
    features['season_ppg'] = game_agg['pts'].mean()
    features['season_games'] = len(game_agg)

    # Ceiling/floor from historical
    features['ceiling_90'] = game_agg['pts'].quantile(0.90)
    features['floor_10'] = game_agg['pts'].quantile(0.10)

    return features


def prepare_training_data(df, target_stat='pts'):
    """
    Prepare training data for models.

    NOW INCLUDES:
    - Travel distance features
    - Schedule density
    - Back-to-back detection

    Creates feature matrix X and target vector y.
    Each row is a game with features calculated from prior games.
    """
    all_features = []
    all_targets = []

    players = df['player'].unique()
    total_players = len(players)

    for idx, player in enumerate(players):
        if idx % 100 == 0:
            print(f"  Processing player {idx}/{total_players}...")

        player_games = df[df['player'] == player].copy()
        team = player_games['team'].mode().iloc[0] if len(player_games) > 0 else None

        # Aggregate to game level
        game_agg = player_games.groupby(['game_id', 'game_date', 'team']).agg({
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

        if len(game_agg) < 10:
            continue

        # For each game (starting from game 10), create features from prior games
        for i in range(10, len(game_agg)):
            current_game = game_agg.iloc[i]
            prior_games = game_agg.iloc[:i]

            # Features from prior 5 and 10 games
            recent_5 = prior_games.tail(5)
            recent_10 = prior_games.tail(10)

            features = {
                'pts_avg_5': recent_5['pts'].mean(),
                'pts_avg_10': recent_10['pts'].mean(),
                'pts_std_10': recent_10['pts'].std(),
                'trb_avg_5': recent_5['trb'].mean(),
                'ast_avg_5': recent_5['ast'].mean(),
                'minutes_avg_5': recent_5['minutes'].mean(),
            }

            # Trend
            first_half = recent_10.head(5)['pts'].mean()
            second_half = recent_10.tail(5)['pts'].mean()
            features['pts_trend'] = (second_half - first_half) / first_half if first_half > 0 else 0

            # Efficiency
            total_fga = recent_5['fga'].sum()
            total_pts = recent_5['pts'].sum()
            features['ts_pct_5'] = total_pts / (2 * total_fga) if total_fga > 0 else 0.5

            # Days rest
            current_date = pd.to_datetime(current_game['game_date'])
            prev_date = pd.to_datetime(prior_games.iloc[-1]['game_date'])
            features['rest_days'] = (current_date - prev_date).days

            # =====================================================
            # SCHEDULE DENSITY FEATURES (NEW)
            # =====================================================
            # Games in last 7 days
            week_ago = current_date - timedelta(days=7)
            week_games = prior_games[pd.to_datetime(prior_games['game_date']) >= week_ago]
            features['games_in_week'] = len(week_games)

            # Games in last 3 days (B2B indicator)
            three_days_ago = current_date - timedelta(days=3)
            recent_3d = prior_games[pd.to_datetime(prior_games['game_date']) >= three_days_ago]
            features['games_in_3days'] = len(recent_3d)

            # Back-to-back detection
            features['is_b2b'] = 1 if features['rest_days'] <= 1 else 0

            # =====================================================
            # TRAVEL DISTANCE (NEW)
            # =====================================================
            travel_miles = 0
            if len(week_games) > 1:
                prev_loc = None
                for _, g in week_games.iterrows():
                    curr_loc = g['team']
                    if prev_loc and curr_loc and prev_loc != curr_loc:
                        if prev_loc in TEAM_LOCATIONS and curr_loc in TEAM_LOCATIONS:
                            travel_miles += haversine_distance(
                                TEAM_LOCATIONS[prev_loc][0], TEAM_LOCATIONS[prev_loc][1],
                                TEAM_LOCATIONS[curr_loc][0], TEAM_LOCATIONS[curr_loc][1]
                            )
                    prev_loc = curr_loc

            features['travel_miles_week'] = travel_miles
            features['travel_fatigue'] = min(1.0, travel_miles / 3000)

            all_features.append(features)
            all_targets.append(current_game[target_stat])

    X = pd.DataFrame(all_features)
    y = np.array(all_targets)

    return X, y


if __name__ == "__main__":
    # Test feature engineering
    print("Testing feature engineering...")

    df = pd.read_parquet('C:/Users/user/NBA_PRODUCTION.parquet')
    print(f"Loaded {len(df):,} records")

    # Test on a top player
    features = engineer_player_features(df, 'Shai Gilgeous-Alexander', 'LAL', '2026-01-28')
    if features:
        print("\nFeatures for SGA vs LAL:")
        for k, v in features.items():
            print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    # Test training data prep
    print("\nPreparing training data (sample)...")
    sample_df = df[df['player'].isin(['Shai Gilgeous-Alexander', 'Tyrese Maxey', 'LeBron James'])]
    X, y = prepare_training_data(sample_df)
    print(f"Training samples: {len(X)}")
    print(f"Features: {list(X.columns)}")
