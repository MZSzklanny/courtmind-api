# -*- coding: utf-8 -*-
"""
CourtMind Predictor v2.0
========================
Advanced player projections with sophisticated defensive modeling:
- Position-specific defense
- Player vs team matchup history
- Pace-adjusted defensive ratings
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class PlayerPredictor:
    """
    Player stat predictor with advanced defensive modeling.
    """

    def __init__(self, df, game_df=None):
        self.df = df
        self.game_df = game_df or self._load_game_df()
        self._cache = {}
        self._def_cache = {}
        self._position_cache = {}

        # Pre-calculate league averages for pace adjustment
        self._calc_league_averages()

    def _load_game_df(self):
        try:
            return pd.read_parquet('C:/Users/user/NBA_Game_PRODUCTION.parquet')
        except:
            return pd.DataFrame()

    def _calc_league_averages(self):
        """Calculate league-wide averages for normalization."""
        # Use recent season data
        recent = self.df[self.df['game_date'] >= '2025-10-01']
        if len(recent) == 0:
            recent = self.df

        game_totals = recent.groupby('game_id').agg({
            'pts': 'sum',
            'fga': 'sum',
            'tov': 'sum',
            'trb': 'sum',
            'ast': 'sum'
        })

        # Average per game (both teams combined)
        self.league_avg_pts = game_totals['pts'].mean() / 2  # Per team
        self.league_avg_pace = (game_totals['fga'].mean() + game_totals['tov'].mean()) / 2

    def get_player_position(self, player):
        """
        Infer player position from their stat profile.
        Returns: 'guard', 'wing', or 'big'
        """
        if player in self._position_cache:
            return self._position_cache[player]

        player_games = self.df[self.df['player'] == player]
        if len(player_games) == 0:
            return 'wing'  # Default

        # Aggregate stats
        avg_stats = player_games.groupby('player').agg({
            'ast': 'mean',
            'trb': 'mean',
            'fg3m': 'mean',
            'blk': 'mean',
            'minutes': 'mean'
        }).iloc[0]

        # Normalize per 36 minutes
        if avg_stats['minutes'] > 0:
            per36_ast = avg_stats['ast'] * 36 / avg_stats['minutes']
            per36_reb = avg_stats['trb'] * 36 / avg_stats['minutes']
            per36_blk = avg_stats['blk'] * 36 / avg_stats['minutes']
            per36_3pm = avg_stats['fg3m'] * 36 / avg_stats['minutes']
        else:
            per36_ast, per36_reb, per36_blk, per36_3pm = 3, 5, 0.5, 1

        # Classification logic
        if per36_ast >= 5 and per36_reb < 6:
            position = 'guard'
        elif per36_reb >= 7 or per36_blk >= 1.2:
            position = 'big'
        else:
            position = 'wing'

        self._position_cache[player] = position
        return position

    def get_position_defense(self, opponent, n_games=10):
        """
        Calculate position-specific defensive ratings.
        Returns dict with defense vs guards, wings, bigs.
        """
        cache_key = f"{opponent}_pos_def"
        if cache_key in self._def_cache:
            return self._def_cache[cache_key]

        # Get opponent's recent games
        opp_games = self.df[self.df['team'] == opponent]
        if len(opp_games) == 0:
            return {'guard': 1.0, 'wing': 1.0, 'big': 1.0}

        game_ids = opp_games['game_id'].unique()[-n_games:]

        # Get all players who played against this opponent
        games_vs_opp = self.df[
            (self.df['game_id'].isin(game_ids)) &
            (self.df['team'] != opponent)
        ]

        if len(games_vs_opp) == 0:
            return {'guard': 1.0, 'wing': 1.0, 'big': 1.0}

        # Classify each player and sum their points
        position_pts = {'guard': [], 'wing': [], 'big': []}

        for player in games_vs_opp['player'].unique():
            pos = self.get_player_position(player)
            player_pts = games_vs_opp[games_vs_opp['player'] == player]['pts'].sum()
            player_games = len(games_vs_opp[games_vs_opp['player'] == player]['game_id'].unique())
            if player_games > 0:
                position_pts[pos].append(player_pts / player_games)

        # Calculate averages and compare to league average
        result = {}
        league_pos_avg = {'guard': 14, 'wing': 12, 'big': 13}  # Rough league averages

        for pos in ['guard', 'wing', 'big']:
            if position_pts[pos]:
                allowed = np.mean(position_pts[pos])
                # Rating > 1 means bad defense (allows more than average)
                result[pos] = allowed / league_pos_avg[pos]
            else:
                result[pos] = 1.0

        self._def_cache[cache_key] = result
        return result

    def get_matchup_history(self, player, opponent, n_games=10):
        """
        Get player's historical performance vs specific opponent.
        Returns multiplier based on over/under performance vs this team.
        """
        cache_key = f"{player}_vs_{opponent}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        player_games = self.df[self.df['player'] == player].copy()
        if len(player_games) == 0:
            return {'multiplier': 1.0, 'games': 0, 'avg_vs': 0, 'season_avg': 0}

        # Get player's season average
        season_avg = player_games.groupby('game_id')['pts'].sum().mean()

        # Find games vs this opponent
        opp_game_ids = self.df[self.df['team'] == opponent]['game_id'].unique()
        vs_opp = player_games[player_games['game_id'].isin(opp_game_ids)]

        if len(vs_opp) == 0:
            return {'multiplier': 1.0, 'games': 0, 'avg_vs': season_avg, 'season_avg': season_avg}

        # Calculate average vs this opponent
        avg_vs_opp = vs_opp.groupby('game_id')['pts'].sum().mean()
        games_vs = len(vs_opp['game_id'].unique())

        # Calculate multiplier (capped for stability)
        if season_avg > 0:
            raw_mult = avg_vs_opp / season_avg
            # Weight by sample size - more games = trust history more
            weight = min(games_vs / 5, 1.0)  # Full weight at 5+ games
            multiplier = 1.0 + (raw_mult - 1.0) * weight * 0.5  # Dampen effect
            multiplier = max(0.85, min(1.15, multiplier))
        else:
            multiplier = 1.0

        result = {
            'multiplier': multiplier,
            'games': games_vs,
            'avg_vs': round(avg_vs_opp, 1),
            'season_avg': round(season_avg, 1)
        }

        self._cache[cache_key] = result
        return result

    def get_pace_adjusted_defense(self, opponent, n_games=10):
        """
        Calculate pace-adjusted defensive rating (points per 100 possessions).
        More accurate than raw points allowed.
        """
        cache_key = f"{opponent}_pace_def"
        if cache_key in self._def_cache:
            return self._def_cache[cache_key]

        opp_games = self.df[self.df['team'] == opponent]
        if len(opp_games) == 0:
            return {'def_rtg': 110, 'pace': 100, 'raw_allowed': 115, 'pace_factor': 1.0}

        game_ids = opp_games['game_id'].unique()[-n_games:]

        # Calculate possessions and points allowed per game
        game_stats = []
        for gid in game_ids:
            game_data = self.df[self.df['game_id'] == gid]
            opp_data = game_data[game_data['team'] == opponent]
            vs_data = game_data[game_data['team'] != opponent]

            if len(opp_data) == 0 or len(vs_data) == 0:
                continue

            # Estimate possessions (FGA + 0.44*FTA + TOV - ORB)
            # Simplified: FGA + TOV
            opp_poss = opp_data['fga'].sum() + opp_data['tov'].sum()
            vs_poss = vs_data['fga'].sum() + vs_data['tov'].sum()
            avg_poss = (opp_poss + vs_poss) / 2

            pts_allowed = vs_data['pts'].sum()

            if avg_poss > 0:
                def_rtg = (pts_allowed / avg_poss) * 100
                game_stats.append({
                    'def_rtg': def_rtg,
                    'pace': avg_poss,
                    'pts_allowed': pts_allowed
                })

        if not game_stats:
            return {'def_rtg': 110, 'pace': 100, 'raw_allowed': 115, 'pace_factor': 1.0}

        avg_def_rtg = np.mean([g['def_rtg'] for g in game_stats])
        avg_pace = np.mean([g['pace'] for g in game_stats])
        avg_pts = np.mean([g['pts_allowed'] for g in game_stats])

        # League average defensive rating is ~110
        # pace_factor adjusts for team speed
        pace_factor = avg_pace / self.league_avg_pace if self.league_avg_pace > 0 else 1.0

        result = {
            'def_rtg': round(avg_def_rtg, 1),
            'pace': round(avg_pace, 1),
            'raw_allowed': round(avg_pts, 1),
            'pace_factor': round(pace_factor, 2)
        }

        self._def_cache[cache_key] = result
        return result

    def get_player_stats(self, player):
        """Get comprehensive player stats."""
        cache_key = f"{player}_stats"
        if cache_key in self._cache:
            return self._cache[cache_key]

        player_games = self.df[self.df['player'] == player].copy()
        if len(player_games) == 0:
            return None

        # Aggregate to game level
        games = player_games.groupby(['game_id', 'game_date', 'team']).agg({
            'pts': 'sum',
            'trb': 'sum',
            'ast': 'sum',
            'stl': 'sum',
            'blk': 'sum',
            'tov': 'sum',
            'fg3m': 'sum',
            'fgm': 'sum',
            'fga': 'sum',
            'minutes': 'sum',
        }).reset_index().sort_values('game_date')

        if len(games) < 5:
            return None

        stats = {
            'player': player,
            'team': games.iloc[-1]['team'],
            'games': len(games),
            'position': self.get_player_position(player),
            # Season averages
            'season_ppg': games['pts'].mean(),
            'season_rpg': games['trb'].mean(),
            'season_apg': games['ast'].mean(),
            'season_3pm': games['fg3m'].mean(),
            # Recent form
            'last_5_ppg': games.tail(5)['pts'].mean(),
            'last_10_ppg': games.tail(10)['pts'].mean(),
            'last_5_rpg': games.tail(5)['trb'].mean(),
            'last_5_apg': games.tail(5)['ast'].mean(),
            # Variability
            'pts_std': games['pts'].std(),
            'minutes_std': games['minutes'].std(),
            'ceiling': games['pts'].quantile(0.90),
            'floor': games['pts'].quantile(0.10),
            # Efficiency
            'fg_pct': games['fgm'].sum() / games['fga'].sum() if games['fga'].sum() > 0 else 0.45,
            'minutes_avg': games['minutes'].mean(),
        }

        # Recent trend
        if len(games) >= 10:
            first_half = games.tail(10).head(5)['pts'].mean()
            second_half = games.tail(5)['pts'].mean()
            stats['trend'] = (second_half - first_half) / first_half if first_half > 0 else 0
        else:
            stats['trend'] = 0

        self._cache[cache_key] = stats
        return stats

    def get_schedule_context(self, player, game_date):
        """Get schedule and travel context."""
        game_date = pd.to_datetime(game_date)

        if len(self.game_df) > 0:
            player_games = self.game_df[self.game_df['player'] == player].copy()
            player_games['game_date'] = pd.to_datetime(player_games['game_date'])
        else:
            player_games = self.df[self.df['player'] == player].copy()
            player_games['game_date'] = pd.to_datetime(player_games['game_date'])

        if len(player_games) == 0:
            return {'rest_days': 2, 'games_in_week': 3, 'is_b2b': False}

        week_ago = game_date - timedelta(days=7)
        recent = player_games[player_games['game_date'] >= week_ago]
        games_in_week = len(recent['game_id'].unique())

        last_game_date = player_games['game_date'].max()
        rest_days = (game_date - last_game_date).days
        is_b2b = rest_days <= 1

        return {
            'rest_days': max(0, rest_days),
            'games_in_week': games_in_week,
            'is_b2b': is_b2b,
        }

    def load_lineups(self):
        """Load today's lineups from rotowire."""
        import json
        from pathlib import Path
        # Use paths relative to this file to work both locally and on production
        base_dir = Path(__file__).resolve().parent.parent
        lineups_path = base_dir / 'data' / 'rotowire_lineups.json'
        try:
            with open(lineups_path, 'r') as f:
                data = json.load(f)
                return data.get('lineups', {})
        except:
            return {}

    def get_injury_boost(self, player, team):
        """
        Calculate usage boost when key teammates are injured.

        Logic:
        - 15+ PPG players trigger boost
        - 55% of missing production redistributed (moderate)
        - Weighted by PPG of remaining starters

        Returns multiplier (e.g., 1.15 = 15% boost)
        """
        lineups = self.load_lineups()

        if team not in lineups:
            return 1.0

        team_data = lineups[team]
        out_players = team_data.get('out', [])
        starters = team_data.get('starters', [])

        if not out_players:
            return 1.0

        # Calculate total missing production (15+ PPG only)
        total_missing_ppg = 0
        injured_detail = []

        for out_player in out_players:
            # Try to get their stats
            stats = self.get_player_stats(out_player)
            if stats and stats['season_ppg'] >= 15:
                total_missing_ppg += stats['season_ppg']
                injured_detail.append({
                    'player': out_player,
                    'ppg': stats['season_ppg']
                })

        if total_missing_ppg == 0:
            return 1.0

        # 55% redistribution (moderate setting)
        redistributed_pts = total_missing_ppg * 0.55

        # Get PPG for all remaining starters
        starter_ppgs = {}
        total_weight = 0

        for starter in starters:
            stats = self.get_player_stats(starter)
            if stats:
                ppg = stats['season_ppg']
                starter_ppgs[starter] = ppg
                # Use squared weights (non-linear) so high-PPG players get more
                total_weight += ppg ** 2

        if total_weight == 0 or player not in starter_ppgs:
            return 1.0

        # Calculate this player's weighted share (higher PPG = bigger % boost)
        player_ppg = starter_ppgs[player]
        player_weight = player_ppg ** 2
        player_share = player_weight / total_weight
        player_boost_pts = redistributed_pts * player_share

        # Convert to multiplier
        if player_ppg > 0:
            multiplier = 1.0 + (player_boost_pts / player_ppg)
        else:
            multiplier = 1.0

        # Cap at reasonable range (max 30% boost)
        multiplier = min(1.30, multiplier)

        return multiplier

    def predict(self, player, opponent, game_date=None, is_home=True):
        """
        Generate prediction with advanced defensive modeling.

        New features:
        - Position-specific defense adjustment
        - Player vs team matchup history
        - Pace-adjusted defensive rating
        """
        game_date = game_date or datetime.now().strftime('%Y-%m-%d')

        stats = self.get_player_stats(player)
        if stats is None:
            return None

        # Get all defensive factors
        pos_def = self.get_position_defense(opponent)
        matchup = self.get_matchup_history(player, opponent)
        pace_def = self.get_pace_adjusted_defense(opponent)
        ctx = self.get_schedule_context(player, game_date)

        # Get injury boost (usage increase when teammates are out)
        injury_boost = self.get_injury_boost(player, stats['team'])

        # === POINTS PROJECTION ===
        season_avg = stats['season_ppg']
        recent_avg = stats['last_5_ppg']

        # Baseline with regression to mean
        baseline = (0.55 * season_avg) + (0.35 * recent_avg) + (0.10 * stats['last_10_ppg'])

        # Trend adjustment
        trend_adj = 1.0 + (stats['trend'] * 0.15)
        trend_adj = max(0.92, min(1.08, trend_adj))

        # === SOPHISTICATED DEFENSE ADJUSTMENT ===

        # 1. Position-specific defense (weight: 40%)
        player_pos = stats['position']
        pos_def_rating = pos_def.get(player_pos, 1.0)
        pos_adj = 0.85 + (pos_def_rating * 0.15)  # Scale to ~0.85-1.15 range
        pos_adj = max(0.88, min(1.12, pos_adj))

        # 2. Pace-adjusted defense rating (weight: 35%)
        # League avg def rtg ~110, range ~100-120
        def_rtg = pace_def['def_rtg']
        pace_def_adj = def_rtg / 110.0
        pace_def_adj = max(0.90, min(1.10, pace_def_adj))

        # 3. Matchup history (weight: 25%)
        matchup_adj = matchup['multiplier']

        # Combined defense adjustment (weighted average)
        def_adj = (pos_adj * 0.40) + (pace_def_adj * 0.35) + (matchup_adj * 0.25)
        def_adj = max(0.85, min(1.15, def_adj))

        # Pace factor - faster teams = more possessions = more stats
        pace_factor = pace_def['pace_factor']
        pace_boost = 1.0 + (pace_factor - 1.0) * 0.3  # Dampen pace effect
        pace_boost = max(0.95, min(1.08, pace_boost))

        # Rest/fatigue adjustment
        if ctx['is_b2b']:
            rest_adj = 0.94
        elif ctx['rest_days'] == 0:
            rest_adj = 0.90
        elif ctx['rest_days'] >= 4:
            rest_adj = 1.03
        elif ctx['rest_days'] >= 2:
            rest_adj = 1.01
        else:
            rest_adj = 1.0

        if ctx['games_in_week'] >= 4:
            rest_adj *= 0.96

        # Home/away
        home_adj = 1.02 if is_home else 0.98

        # Final projection (with injury boost)
        pts_proj = baseline * trend_adj * def_adj * pace_boost * rest_adj * home_adj * injury_boost

        # === OTHER STATS ===
        reb_base = stats['season_rpg'] * 0.7 + stats['last_5_rpg'] * 0.3
        ast_base = stats['season_apg'] * 0.7 + stats['last_5_apg'] * 0.3

        # Apply injury boost to other stats (slightly dampened)
        injury_boost_other = 1.0 + ((injury_boost - 1.0) * 0.6)  # 60% of pts boost

        reb_proj = reb_base * pace_boost * injury_boost_other
        ast_proj = ast_base * pace_boost * injury_boost_other
        three_proj = stats['season_3pm'] * def_adj * injury_boost_other

        # === IMPROVED CONFIDENCE ===
        # Factor in: consistency, sample size, minutes stability, matchup history
        consistency = 1 - (stats['pts_std'] / stats['season_ppg']) if stats['season_ppg'] > 0 else 0.5
        sample_score = min(stats['games'] / 40, 1.0)  # Max confidence at 40+ games
        minutes_stability = 1 - (stats['minutes_std'] / stats['minutes_avg']) if stats['minutes_avg'] > 0 else 0.5
        matchup_confidence = min(matchup['games'] / 3, 1.0) * 0.5 + 0.5  # Boost if matchup history exists

        # Weighted confidence
        confidence = (
            consistency * 0.40 +
            sample_score * 0.25 +
            minutes_stability * 0.20 +
            matchup_confidence * 0.15
        ) * 100
        confidence = max(45, min(92, confidence))

        # Calculate effective defensive rank from our new metrics
        eff_def_rank = int(15 + (def_adj - 1.0) * 50)  # Center at 15, adjust by def_adj
        eff_def_rank = max(1, min(30, eff_def_rank))

        return {
            'player': player,
            'team': stats['team'],
            'opponent': opponent,
            'position': player_pos,
            'pts': round(pts_proj, 1),
            'reb': round(reb_proj, 1),
            'ast': round(ast_proj, 1),
            '3pm': round(three_proj, 1),
            'ceiling': round(stats['ceiling'] * def_adj, 1),
            'floor': round(stats['floor'] * def_adj, 1),
            'confidence': round(confidence),
            # Context
            'season_avg': round(season_avg, 1),
            'last_5': round(recent_avg, 1),
            'opp_def': round(def_adj, 2),
            'opp_def_rank': eff_def_rank,
            'opp_pts_allowed': round(pace_def['raw_allowed'], 1),
            'is_b2b': ctx['is_b2b'],
            'rest_days': ctx['rest_days'],
            'trend': 'hot' if stats['trend'] > 0.05 else 'cold' if stats['trend'] < -0.05 else 'neutral',
            'def_adj': round(def_adj, 3),
            # New detailed breakdown
            'pos_def_adj': round(pos_adj, 3),
            'pace_def_adj': round(pace_def_adj, 3),
            'matchup_adj': round(matchup_adj, 3),
            'matchup_games': matchup['games'],
            'matchup_avg': matchup['avg_vs'],
            'pace_factor': pace_def['pace_factor'],
            'injury_boost': round(injury_boost, 3),
        }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    df = pd.read_parquet('C:/Users/user/NBA_PRODUCTION.parquet')
    predictor = PlayerPredictor(df)

    print("=" * 70)
    print("COURTMIND v2.0 - ADVANCED DEFENSIVE MODELING")
    print("=" * 70)

    # Test with today's matchups
    test_cases = [
        ('Shai Gilgeous-Alexander', 'MIN'),
        ('Anthony Edwards', 'OKC'),
        ('Joel Embiid', 'SAC'),
        ('Tyrese Maxey', 'SAC'),
        ('Cade Cunningham', 'PHX'),
    ]

    for player, opp in test_cases:
        pred = predictor.predict(player, opp)
        if pred:
            print(f"\n{player} vs {opp}")
            print(f"  Position: {pred['position'].upper()}")
            print(f"  Projection: {pred['pts']} pts (Season: {pred['season_avg']}, L5: {pred['last_5']})")
            print(f"  Range: {pred['floor']}-{pred['ceiling']}")
            print(f"  Confidence: {pred['confidence']}%")
            print(f"  --- Defense Breakdown ---")
            print(f"  Position Defense: {pred['pos_def_adj']:.3f}")
            print(f"  Pace-Adj Defense: {pred['pace_def_adj']:.3f}")
            print(f"  Matchup History:  {pred['matchup_adj']:.3f} ({pred['matchup_games']} games, avg {pred['matchup_avg']})")
            print(f"  Combined Def Adj: {pred['def_adj']:.3f}")
