"""
CourtMind Daily Simulations
===========================
Run 2000 Monte Carlo simulations per player for today's games.
Identify standout ceiling plays and over/under opportunities.
"""

import sys
import os

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams
import time

from CourtMind.config import DATA_PATHS, OUTPUT_DIR, TEAM_COLORS
from CourtMind.infographic_generator import (
    create_player_prediction_card,
    create_top_picks_graphic
)
from CourtMind.tweet_formatter import (
    format_player_prediction_tweet,
    format_top_picks_tweet,
    generate_insight_text
)

N_SIMULATIONS = 2000


def get_team_abbrev(team_id):
    """Convert team ID to abbreviation."""
    nba_teams = teams.get_teams()
    for team in nba_teams:
        if team['id'] == team_id:
            return team['abbreviation']
    return 'UNK'


def get_todays_games():
    """Fetch today's games with team info."""
    date_str = datetime.now().strftime('%Y-%m-%d')
    print(f"[CourtMind] Fetching games for {date_str}...")

    time.sleep(0.6)
    scoreboard = scoreboardv2.ScoreboardV2(game_date=date_str)
    games_df = scoreboard.get_data_frames()[0]

    games = []
    seen = set()
    for _, game in games_df.iterrows():
        game_id = game['GAME_ID']
        if game_id in seen:
            continue
        seen.add(game_id)

        home = get_team_abbrev(game['HOME_TEAM_ID'])
        away = get_team_abbrev(game['VISITOR_TEAM_ID'])

        games.append({
            'game_id': game_id,
            'home_team': home,
            'away_team': away,
            'game_time': game.get('GAME_STATUS_TEXT', 'TBD'),
        })

    print(f"[CourtMind] Found {len(games)} games")
    return games


def load_player_data():
    """Load player historical data."""
    print("[CourtMind] Loading player data...")
    df = pd.read_parquet(DATA_PATHS['combined_data'])
    return df


def get_player_stats(df, player, min_games=10):
    """Get player's scoring statistics for simulations."""
    player_df = df[df['player'] == player].copy()
    if len(player_df) == 0:
        return None

    # Aggregate to game level
    game_stats = player_df.groupby(['game_id', 'game_date']).agg({
        'pts': 'sum',
        'fga': 'sum',
        'fgm': 'sum',
        'trb': 'sum',
        'ast': 'sum',
        'minutes': 'sum',
    }).reset_index()

    if len(game_stats) < min_games:
        return None

    pts = game_stats['pts']
    return {
        'player': player,
        'games': len(game_stats),
        'pts_mean': pts.mean(),
        'pts_std': pts.std(),
        'pts_min': pts.min(),
        'pts_max': pts.max(),
        'pts_10': pts.quantile(0.10),
        'pts_25': pts.quantile(0.25),
        'pts_50': pts.quantile(0.50),
        'pts_75': pts.quantile(0.75),
        'pts_90': pts.quantile(0.90),
        'recent_5_avg': game_stats.tail(5)['pts'].mean(),
        'total_game_avg': (pts + game_stats['trb'] + game_stats['ast']).mean(),
    }


def run_monte_carlo_player(stats, n_sims=N_SIMULATIONS):
    """
    Run Monte Carlo simulation for a player's points.

    Returns distribution statistics.
    """
    if stats is None:
        return None

    mean = stats['pts_mean']
    std = stats['pts_std']

    # Adjust for recent form (hot hand)
    recent_ratio = stats['recent_5_avg'] / mean if mean > 0 else 1.0
    hot_adjustment = 1.0
    if recent_ratio > 1.05:
        hot_adjustment = min(1.15, recent_ratio * 0.5 + 0.5)
    elif recent_ratio < 0.95:
        hot_adjustment = max(0.90, recent_ratio * 0.5 + 0.5)

    adjusted_mean = mean * hot_adjustment

    # Run simulation
    np.random.seed(int(hash(stats['player']) % 2**31))
    samples = np.random.normal(adjusted_mean, std, n_sims)
    samples = np.maximum(samples, 0)  # Can't score negative

    # Calculate percentiles
    pctile_90 = np.percentile(samples, 90)
    odds_over_90th = (samples >= stats['pts_90']).mean() * 100

    return {
        'player': stats['player'],
        'sim_mean': samples.mean(),
        'sim_std': samples.std(),
        'sim_90th': pctile_90,
        'odds_over_hist_90th': odds_over_90th,
        'hist_90th': stats['pts_90'],
        'hot_adjustment': hot_adjustment,
        'is_hot': hot_adjustment > 1.05,
        'floor': np.percentile(samples, 10),
        'ceiling': np.percentile(samples, 95),
        'samples': samples,
    }


def run_monte_carlo_game_total(team1_players, team2_players, n_sims=N_SIMULATIONS):
    """
    Run Monte Carlo simulation for game total points.

    Returns over/under analysis.
    """
    np.random.seed(42)

    # Simulate each player and sum
    team1_totals = np.zeros(n_sims)
    team2_totals = np.zeros(n_sims)

    for p in team1_players:
        if p['sim_result'] is not None:
            team1_totals += p['sim_result']['samples']

    for p in team2_players:
        if p['sim_result'] is not None:
            team2_totals += p['sim_result']['samples']

    game_totals = team1_totals + team2_totals

    return {
        'mean_total': game_totals.mean(),
        'std_total': game_totals.std(),
        'p10': np.percentile(game_totals, 10),
        'p25': np.percentile(game_totals, 25),
        'p50': np.percentile(game_totals, 50),
        'p75': np.percentile(game_totals, 75),
        'p90': np.percentile(game_totals, 90),
    }


def run_daily_simulations():
    """Main function to run all simulations for today."""
    print("=" * 60)
    print("CourtMind Daily Monte Carlo Simulations")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Simulations per player: {N_SIMULATIONS}")
    print("=" * 60)

    # Get games
    games = get_todays_games()
    if not games:
        print("No games today!")
        return None

    for g in games:
        print(f"  {g['away_team']} @ {g['home_team']} - {g['game_time']}")

    # Load data
    df = load_player_data()

    # Get all teams playing
    teams_playing = set()
    team_opponents = {}
    for g in games:
        teams_playing.add(g['home_team'])
        teams_playing.add(g['away_team'])
        team_opponents[g['home_team']] = g['away_team']
        team_opponents[g['away_team']] = g['home_team']

    # Run simulations for all players
    print(f"\n[CourtMind] Running {N_SIMULATIONS} simulations per player...")
    all_results = []

    for team in teams_playing:
        team_df = df[df['team'] == team]
        if team_df.empty:
            continue

        # Get rotation players (recent games)
        recent = team_df.sort_values('game_date', ascending=False)
        players = recent['player'].unique()[:10]  # Top 10 rotation

        for player in players:
            stats = get_player_stats(df, player)
            if stats:
                sim_result = run_monte_carlo_player(stats)
                if sim_result:
                    opponent = team_opponents.get(team, '')
                    all_results.append({
                        'player': player,
                        'team': team,
                        'opponent': opponent,
                        'games': stats['games'],
                        'proj_pts': round(sim_result['sim_mean'], 1),
                        'floor_pts': round(sim_result['floor'], 1),
                        'pctile_90': round(sim_result['sim_90th'], 1),
                        'pctile_90_odds': round(sim_result['odds_over_hist_90th'], 0),
                        'max_pts': round(sim_result['ceiling'], 1),
                        'is_hot': sim_result['is_hot'],
                        'hot_mult': sim_result['hot_adjustment'],
                        'sim_result': sim_result,
                    })

    results_df = pd.DataFrame(all_results)
    print(f"[CourtMind] Simulated {len(results_df)} players")

    # Find standout ceiling plays
    print("\n" + "=" * 60)
    print("STANDOUT CEILING PLAYS (High 90th pctile odds)")
    print("=" * 60)

    standouts = results_df[results_df['pctile_90_odds'] >= 15].nlargest(5, 'pctile_90_odds')
    for _, p in standouts.iterrows():
        hot_str = " [HOT]" if p['is_hot'] else ""
        print(f"  {p['player']} ({p['team']} vs {p['opponent']}){hot_str}")
        print(f"    Proj: {p['proj_pts']} | 90th: {p['pctile_90']} | Odds: {p['pctile_90_odds']:.0f}%")

    # Find over/under opportunities
    print("\n" + "=" * 60)
    print("GAME TOTAL ANALYSIS (Monte Carlo)")
    print("=" * 60)

    game_totals = []
    for game in games:
        home = game['home_team']
        away = game['away_team']

        home_players = [r for _, r in results_df[results_df['team'] == home].iterrows()]
        away_players = [r for _, r in results_df[results_df['team'] == away].iterrows()]

        if home_players and away_players:
            total_sim = run_monte_carlo_game_total(
                [{'sim_result': p.get('sim_result')} for p in home_players],
                [{'sim_result': p.get('sim_result')} for p in away_players]
            )
            game_totals.append({
                'game': f"{away} @ {home}",
                'home': home,
                'away': away,
                **total_sim
            })
            print(f"\n  {away} @ {home}")
            print(f"    Projected Total: {total_sim['mean_total']:.1f} (std: {total_sim['std_total']:.1f})")
            print(f"    Range: {total_sim['p10']:.0f} - {total_sim['p90']:.0f}")

    return {
        'players': results_df,
        'standouts': standouts,
        'game_totals': game_totals,
        'games': games,
    }


def generate_infographics(results):
    """Generate infographics from simulation results."""
    if results is None:
        return []

    today = datetime.now().strftime('%Y-%m-%d')
    output_dir = os.path.join(OUTPUT_DIR, today)
    os.makedirs(output_dir, exist_ok=True)

    posts = []
    standouts = results['standouts']

    # 1. Top ceiling player card
    if len(standouts) > 0:
        top = standouts.iloc[0]

        insight = generate_insight_text(
            player=top['player'],
            matchup_adj=1.0 + (0.05 if top['is_hot'] else 0),
            sgdr=50,
            is_b2b=False,
            is_hot=top['is_hot'],
            opponent=top['opponent']
        )

        path = create_player_prediction_card(
            player_name=top['player'],
            team=top['team'],
            opponent=top['opponent'],
            proj_pts=top['proj_pts'],
            pctile_90=top['pctile_90'],
            pctile_90_odds=top['pctile_90_odds'],
            floor_pts=top['floor_pts'],
            max_pts=top['max_pts'],
            sgdr=50,
            matchup_adj=1.0,
            is_hot=top['is_hot'],
            is_b2b=False,
            insight_text=insight,
            game_time="Tonight",
            output_path=os.path.join(output_dir, f"ceiling_{top['player'].replace(' ', '_')}.png")
        )

        tweet = format_player_prediction_tweet(
            player=top['player'],
            team=top['team'],
            opponent=top['opponent'],
            proj_pts=top['proj_pts'],
            pctile_90=top['pctile_90'],
            pctile_90_odds=top['pctile_90_odds'],
            is_hot=top['is_hot'],
            insight=insight
        )

        posts.append({
            'type': 'ceiling_player',
            'image_path': path,
            'tweet_text': tweet,
            'player': top['player'],
        })
        print(f"\n[CourtMind] Created player card: {path}")

    # 2. Top 5 ceiling picks
    top5 = results['players'].nlargest(5, 'pctile_90_odds')
    if len(top5) >= 3:
        path = create_top_picks_graphic(
            players=top5.to_dict('records'),
            title="TOP CEILING GAMES TONIGHT",
            output_path=os.path.join(output_dir, "top_5_ceiling.png")
        )

        tweet = format_top_picks_tweet(top5.to_dict('records'))

        posts.append({
            'type': 'top_picks',
            'image_path': path,
            'tweet_text': tweet,
        })
        print(f"[CourtMind] Created top picks: {path}")

    return posts


if __name__ == "__main__":
    results = run_daily_simulations()

    if results:
        print("\n" + "=" * 60)
        print("GENERATING INFOGRAPHICS")
        print("=" * 60)
        posts = generate_infographics(results)

        print("\n" + "=" * 60)
        print(f"COMPLETE - {len(posts)} posts ready")
        print("=" * 60)

        for p in posts:
            print(f"\n--- {p['type'].upper()} ---")
            print(f"Image: {p['image_path']}")
            print(f"Tweet:\n{p['tweet_text']}")
