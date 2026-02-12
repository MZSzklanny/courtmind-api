# -*- coding: utf-8 -*-
"""
CourtMind AI - FastAPI Backend
==============================
REST API for the Next.js frontend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Get the directory where this script is located
BASE_DIR = Path(__file__).resolve().parent

# Add paths for imports
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR.parent))  # For local dev

from models.predictor import PlayerPredictor
from models.odds_fetcher import (
    fetch_game_odds, get_player_prop_line, get_api_key, TEAM_FULL_NAMES
)
from models.nba_lineups_fetcher import (
    TODAYS_LINEUPS, get_todays_official_lineups, refresh_lineups_from_rotowire
)
from models.rotowire_scraper import scrape_rotowire_lineups, save_lineups
from models.bet_tracker import (
    get_tracking_stats, check_results, get_todays_predictions,
    log_daily_predictions, get_daily_tracking_stats, log_daily_picks,
    generate_daily_predictions, grade_daily_picks
)
from models.auto_grader import grade_predictions_for_date, regrade_date, grade_all_ungraded
from models.model_feedback import analyze_performance, calculate_calibration_adjustments, save_feedback

app = FastAPI(
    title="CourtMind AI API",
    description="Neural Network Powered NBA Analytics",
    version="2.4.2"
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://courtmind.bet",
        "https://www.courtmind.bet"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


STARTUP_CACHE_REGEN = True  # Set to False to disable auto-regeneration


# Load data once at startup
DATA_FILE = BASE_DIR / 'data' / 'NBA_PRODUCTION.parquet'
if not DATA_FILE.exists():
    # Fallback for local development
    DATA_FILE = Path('C:/Users/user/NBA_PRODUCTION.parquet')
df = pd.read_parquet(DATA_FILE)

# CRITICAL: Add fg3m column if missing (required for predictor)
if 'fg3m' not in df.columns:
    print("[STARTUP] WARNING: fg3m column missing - adding placeholder")
    df['fg3m'] = 0

predictor = PlayerPredictor(df)

TEAMS = {
    'ATL': 'Hawks', 'BOS': 'Celtics', 'BKN': 'Nets', 'CHA': 'Hornets',
    'CHI': 'Bulls', 'CLE': 'Cavaliers', 'DAL': 'Mavericks', 'DEN': 'Nuggets',
    'DET': 'Pistons', 'GSW': 'Warriors', 'HOU': 'Rockets', 'IND': 'Pacers',
    'LAC': 'Clippers', 'LAL': 'Lakers', 'MEM': 'Grizzlies', 'MIA': 'Heat',
    'MIL': 'Bucks', 'MIN': 'Timberwolves', 'NOP': 'Pelicans', 'NYK': 'Knicks',
    'OKC': 'Thunder', 'ORL': 'Magic', 'PHI': '76ers', 'PHX': 'Suns',
    'POR': 'Trail Blazers', 'SAC': 'Kings', 'SAS': 'Spurs', 'TOR': 'Raptors',
    'UTA': 'Jazz', 'WAS': 'Wizards'
}

TEAM_ABBREV = {v: k for k, v in TEAM_FULL_NAMES.items()}


# =============================================================================
# RESPONSE MODELS
# =============================================================================
class PlayerPrediction(BaseModel):
    player: str
    team: str
    pts: float
    reb: float
    ast: float
    threes: float
    floor: float
    ceiling: float
    season_avg: float
    last_5: float
    opp_def: float
    opp_def_rank: int
    rest_days: int
    is_b2b: bool
    trend: str
    confidence: int


class PropLine(BaseModel):
    dk_line: Optional[float]
    dk_odds: Optional[int]
    fd_line: Optional[float]
    fd_odds: Optional[int]


class GamePrediction(BaseModel):
    home_team: str
    away_team: str
    home_score: float
    away_score: float
    home_win_prob: float
    away_win_prob: float
    spread: float
    total: float
    dk_spread: Optional[str]
    dk_total: Optional[str]
    top_plays: List[Dict[str, Any]]


# =============================================================================
# ENDPOINTS
# =============================================================================
@app.get("/")
def root():
    return {
        "name": "CourtMind AI API",
        "version": "2.4.2",
        "status": "online",
        "data_through": df['game_date'].max().strftime('%Y-%m-%d'),
        "total_records": len(df)
    }


@app.get("/api/teams")
def get_teams():
    """Get all NBA teams"""
    return {"teams": TEAMS}


@app.get("/api/debug/odds")
def debug_odds():
    """Debug endpoint for odds API"""
    api_key = get_api_key()
    odds_data = fetch_game_odds()
    return {
        "has_api_key": bool(api_key),
        "api_key_preview": api_key[:8] + "..." if api_key and len(api_key) > 8 else "none",
        "games_count": len(odds_data.get('games', [])),
        "error": odds_data.get('error'),
        "sample_game": odds_data.get('games', [{}])[0] if odds_data.get('games') else None
    }


@app.get("/api/teams/playing")
def get_teams_playing_today():
    """Get teams playing today"""
    # Use lineups as primary source (TODAYS_LINEUPS imported at top)
    if TODAYS_LINEUPS:
        return {"teams": list(TODAYS_LINEUPS.keys())}
    # Fallback to odds API
    try:
        odds_data = fetch_game_odds()
        teams_playing = set()
        for game in odds_data.get('games', []):
            home = TEAM_ABBREV.get(game['home_team'], game['home_team'][:3].upper())
            away = TEAM_ABBREV.get(game['away_team'], game['away_team'][:3].upper())
            teams_playing.add(home)
            teams_playing.add(away)
        return {"teams": list(teams_playing)}
    except:
        return {"teams": []}


@app.get("/api/players/{team}")
def get_players_by_team(team: str):
    """Get players for a specific team"""
    current_season = df[df['game_date'] >= '2025-10-01']
    team_df = current_season[current_season['team'] == team.upper()]

    if team_df.empty:
        raise HTTPException(status_code=404, detail=f"No players found for team {team}")

    # Get players sorted by minutes
    player_minutes = team_df.groupby('player')['minutes'].mean().sort_values(ascending=False)
    players = player_minutes.index.tolist()

    # Remove bottom 4 by minutes
    if len(players) > 4:
        players = players[:-4]

    return {"team": team.upper(), "players": players}


@app.get("/api/lineups")
def get_todays_lineups():
    """Get today's official lineups"""
    try:
        from models.nba_lineups_fetcher import get_todays_official_lineups
        lineups = get_todays_official_lineups()
        return {"lineups": lineups}
    except Exception as e:
        return {"lineups": {}, "error": str(e)}


@app.get("/api/predict/{player}")
def predict_player(player: str, opponent: str = "LAL"):
    """Get prediction for a player vs opponent"""
    pred = predictor.predict(player, opponent.upper())

    if not pred:
        raise HTTPException(status_code=404, detail=f"No prediction available for {player}")

    return {
        "player": pred['player'],
        "team": pred['team'],
        "opponent": opponent.upper(),
        "stats": {
            "points": {"projection": pred['pts'], "floor": pred['floor'], "ceiling": pred['ceiling']},
            "rebounds": {"projection": pred['reb']},
            "assists": {"projection": pred['ast']},
            "threes": {"projection": pred['3pm']}
        },
        "context": {
            "season_avg": pred['season_avg'],
            "last_5": pred['last_5'],
            "opp_defense": pred['opp_def'],
            "opp_def_rank": pred.get('opp_def_rank', 15),
            "rest_days": pred['rest_days'],
            "is_b2b": pred['is_b2b']
        },
        "trend": pred['trend'],
        "confidence": pred['confidence']
    }


@app.get("/api/props/{player}")
def get_player_props(player: str):
    """Get betting lines for a player"""
    pts_lines = get_player_prop_line(player, 'points')
    reb_lines = get_player_prop_line(player, 'rebounds')
    ast_lines = get_player_prop_line(player, 'assists')

    def extract_line(lines, book):
        if lines and book in lines:
            return {
                "line": lines[book].get('over', {}).get('line'),
                "odds": lines[book].get('over', {}).get('odds', -110)
            }
        return {"line": None, "odds": None}

    return {
        "player": player,
        "points": {
            "dk": extract_line(pts_lines, 'dk'),
            "fd": extract_line(pts_lines, 'fd')
        },
        "rebounds": {
            "dk": extract_line(reb_lines, 'dk'),
            "fd": extract_line(reb_lines, 'fd')
        },
        "assists": {
            "dk": extract_line(ast_lines, 'dk'),
            "fd": extract_line(ast_lines, 'fd')
        }
    }


@app.get("/api/games")
def get_todays_games():
    """Get cached games data (fast read from cache)."""
    cached = load_games_cache()
    if cached:
        return cached
    return {"games": [], "count": 0, "message": "No games cached. Run POST /api/games/generate first."}


# ============================================================================
# LINEUP ENDPOINTS
# ============================================================================

@app.get("/api/lineups")
def get_lineups():
    """Get today's starting lineups."""
    try:
        lineups = get_todays_official_lineups()
        return {
            "lineups": lineups,
            "count": len(lineups),
            "date": datetime.now().strftime('%Y-%m-%d')
        }
    except Exception as e:
        return {"lineups": {}, "count": 0, "error": str(e)}


@app.post("/api/lineups/refresh")
def refresh_lineups():
    """Force refresh lineups from Rotowire."""
    try:
        print("[API] Refreshing lineups from Rotowire...")
        data = scrape_rotowire_lineups()
        if data and data.get('lineups'):
            save_lineups(data)
            return {
                "success": True,
                "message": f"Refreshed {len(data['lineups'])} team lineups",
                "lineups": data['lineups'],
                "count": len(data['lineups']),
                "scraped_at": data.get('scraped_at')
            }
        return {"success": False, "message": "No lineups found on Rotowire"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/props/refresh")
def refresh_props():
    """Force refresh player props from Odds API (clears cache)."""
    from pathlib import Path
    props_cache = Path('C:/Users/user/CourtMind/models/props_cache.json')

    # Clear cache
    if props_cache.exists():
        props_cache.unlink()
        print("[API] Cleared props cache")

    # Fetch fresh props
    from models.odds_fetcher import fetch_player_props
    props_data = fetch_player_props()

    if 'error' in props_data and props_data.get('error'):
        return {"success": False, "error": props_data['error']}

    player_count = len(props_data.get('props', {}))
    return {
        "success": True,
        "message": f"Refreshed props for {player_count} players",
        "player_count": player_count,
        "updated": props_data.get('updated')
    }


@app.post("/api/games/generate")
def generate_todays_games():
    """Generate and cache today's games with predictions and odds. This is slow - run daily."""
    from models.ensemble_model import GamePredictor

    game_predictor = GamePredictor()
    odds_data = fetch_game_odds() if get_api_key() else {'games': []}
    games_list = odds_data.get('games', [])

    # Get lineups
    try:
        from models.nba_lineups_fetcher import get_todays_official_lineups
        official_lineups = get_todays_official_lineups()
    except:
        official_lineups = {}

    results = []

    # Build games from lineups if odds API returns empty
    if not games_list and TODAYS_LINEUPS:
        for team, data in TODAYS_LINEUPS.items():
            if data.get('home'):
                home_team = team
                away_team = data['opponent']
                games_list.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'from_lineups': True
                })

    for game in games_list:
        if game.get('from_lineups'):
            home_team = game['home_team']
            away_team = game['away_team']
            abbrev_to_full = {v: k for k, v in TEAM_ABBREV.items()}
            home_full = abbrev_to_full.get(home_team, home_team)
            away_full = abbrev_to_full.get(away_team, away_team)
        else:
            home_full = game['home_team']
            away_full = game['away_team']
            home_team = TEAM_ABBREV.get(home_full, home_full[:3].upper())
            away_team = TEAM_ABBREV.get(away_full, away_full[:3].upper())

        result = game_predictor.predict_game(df, home_team, away_team)

        # Find top plays for this game (points, rebounds, assists)
        top_plays = []
        game_players = []
        if home_team in official_lineups:
            game_players.extend([(p, home_team, away_team) for p in official_lineups[home_team].get('starters', [])])
        if away_team in official_lineups:
            game_players.extend([(p, away_team, home_team) for p in official_lineups[away_team].get('starters', [])])

        MIN_EDGE_GAME = 10  # 10% minimum edge for game top plays (allows more picks)

        for player_name, team, opp in game_players:
            try:
                pred = predictor.predict(player_name, opp)
                if not pred:
                    continue

                # Check points, rebounds, assists
                prop_types = [
                    ('points', 'pts', 'POINTS'),
                    ('rebounds', 'reb', 'REBOUNDS'),
                    ('assists', 'ast', 'ASSISTS')
                ]

                for prop_key, stat_key, stat_display in prop_types:
                    prop_line = get_player_prop_line(player_name, prop_key)
                    if not prop_line:
                        continue

                    dk_data = prop_line.get('dk', {}).get('over', {})
                    fd_data = prop_line.get('fd', {}).get('over', {})

                    dk_line = dk_data.get('line', 0)
                    fd_line = fd_data.get('line', 0)
                    dk_odds = dk_data.get('odds', -110)
                    fd_odds = fd_data.get('odds', -110)

                    # Use best available line (lower for OVER)
                    if dk_line > 0 and fd_line > 0:
                        best_line = min(dk_line, fd_line)
                        best_book = 'DK' if dk_line <= fd_line else 'FD'
                    elif dk_line > 0:
                        best_line = dk_line
                        best_book = 'DK'
                    elif fd_line > 0:
                        best_line = fd_line
                        best_book = 'FD'
                    else:
                        continue

                    projection = pred[stat_key]
                    edge = ((projection - best_line) / best_line) * 100

                    # Filter: 15%+ edge, <80% edge (bad data filter)
                    if abs(edge) >= MIN_EDGE_GAME and abs(edge) < 80:
                        score = abs(edge) * (pred['confidence'] / 100)

                        top_plays.append({
                            'player': player_name,
                            'team': team,
                            'stat': stat_display,
                            'projection': round(projection, 1),
                            'line': best_line,
                            'dk_line': dk_line if dk_line > 0 else None,
                            'fd_line': fd_line if fd_line > 0 else None,
                            'dk_odds': dk_odds if dk_line > 0 else None,
                            'fd_odds': fd_odds if fd_line > 0 else None,
                            'book': best_book,
                            'edge': round(edge, 1),
                            'direction': 'OVER' if edge > 0 else 'UNDER',
                            'confidence': pred['confidence'],
                            'score': round(score, 2)
                        })
            except Exception as e:
                continue

        # Sort by score and take top 3
        top_plays = sorted(top_plays, key=lambda x: x['score'], reverse=True)[:3]

        dk = game.get('bookmakers', {}).get('draftkings', {})
        dk_spread = dk.get('spreads', {}).get(home_full, {}).get('point', None)
        dk_total = dk.get('totals', {}).get('Over', {}).get('point', None)

        game_data = {
            "home_team": home_team,
            "away_team": away_team,
            "home_full": home_full,
            "away_full": away_full,
            "prediction": None,
            "odds": {
                "dk_spread": dk_spread,
                "dk_total": dk_total
            },
            "top_plays": top_plays
        }

        if result:
            game_data["prediction"] = {
                "home_score": round(result['predicted_home_score'], 1),
                "away_score": round(result['predicted_away_score'], 1),
                "home_win_prob": round(result['home_win_prob'], 1),
                "away_win_prob": round(result['away_win_prob'], 1),
                "spread": round(result['spread'], 1),
                "total": round(result['predicted_total'], 1)
            }

        results.append(game_data)

    response = {
        "games": results,
        "count": len(results),
        "generated_at": datetime.now().isoformat()
    }

    # Save to cache
    save_games_cache(response)

    return response


@app.get("/api/tracking")
def get_tracking():
    """Get bet tracking statistics"""
    stats = get_tracking_stats()
    return {
        "total_picks": stats['total_picks'],
        "graded_picks": stats['graded_picks'],
        "hits": stats['hits'],
        "misses": stats['misses'],
        "hit_rate": stats['hit_rate'],
        "pending": stats['pending'],
        "streak": stats['streak'],
        "by_confidence": stats['by_confidence'],
        "by_stat": stats['by_stat'],
        "recent": stats['recent'][:20]
    }


@app.get("/api/tracking/today")
def get_todays_picks():
    """Get today's logged predictions"""
    picks = get_todays_predictions()
    return {"picks": picks, "count": len(picks)}


@app.post("/api/tracking/log")
def log_picks(picks: List[Dict[str, Any]]):
    """Log predictions for tracking"""
    count = log_daily_predictions(picks)
    return {"logged": count, "message": f"Logged {count} picks"}


@app.post("/api/tracking/update")
def update_results():
    """Update results for graded picks"""
    updated = check_results(df)
    return {"updated": updated, "message": f"Updated {updated} picks"}


@app.get("/api/props/scan")
def scan_props(stat: str = "points", min_edge: float = 3.0):
    """Scan for value props across all players"""
    try:
        from models.nba_lineups_fetcher import get_all_todays_starters
        players = get_all_todays_starters()
    except:
        players = []

    stat_map = {
        'points': ('pts', 'points'),
        'rebounds': ('reb', 'rebounds'),
        'assists': ('ast', 'assists'),
        'threes': ('3pm', 'threes')
    }

    if stat not in stat_map:
        raise HTTPException(status_code=400, detail=f"Invalid stat: {stat}")

    stat_key, prop_key = stat_map[stat]
    props = []

    for player in players[:40]:
        try:
            pred = predictor.predict(player, 'LAL')
            if not pred:
                continue

            proj = pred[stat_key]
            lines = get_player_prop_line(player, prop_key)

            if lines:
                dk_line = lines.get('dk', {}).get('over', {}).get('line', 0)
                fd_line = lines.get('fd', {}).get('over', {}).get('line', 0)
            else:
                continue

            best_line = min(dk_line, fd_line) if dk_line > 0 and fd_line > 0 else max(dk_line, fd_line)
            if best_line <= 0:
                continue

            edge = ((proj - best_line) / best_line) * 100

            if abs(edge) >= min_edge:
                props.append({
                    'player': player,
                    'team': pred['team'],
                    'stat': stat,
                    'projection': proj,
                    'dk_line': dk_line if dk_line > 0 else None,
                    'fd_line': fd_line if fd_line > 0 else None,
                    'edge': round(edge, 1),
                    'direction': 'OVER' if edge > 0 else 'UNDER',
                    'confidence': pred['confidence'],
                    'trend': pred['trend']
                })
        except:
            continue

    props = sorted(props, key=lambda x: abs(x['edge']), reverse=True)
    return {"props": props, "count": len(props), "stat": stat, "min_edge": min_edge}


@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": len(df) > 0,
        "odds_api": get_api_key() is not None,
        "df_shape": list(df.shape),
        "has_fg3m": 'fg3m' in df.columns
    }


# Cache files
TOP_PICKS_CACHE = BASE_DIR / 'top_picks_cache.json'
GAMES_CACHE = BASE_DIR / 'games_cache.json'


def load_top_picks_cache():
    """Load cached top picks."""
    if TOP_PICKS_CACHE.exists():
        try:
            with open(TOP_PICKS_CACHE, 'r') as f:
                return json.load(f)
        except:
            pass
    return None


def save_top_picks_cache(data):
    """Save top picks to cache."""
    with open(TOP_PICKS_CACHE, 'w') as f:
        json.dump(data, f)


def load_games_cache():
    """Load cached games data."""
    if GAMES_CACHE.exists():
        try:
            with open(GAMES_CACHE, 'r') as f:
                return json.load(f)
        except:
            pass
    return None


def save_games_cache(data):
    """Save games data to cache."""
    with open(GAMES_CACHE, 'w') as f:
        json.dump(data, f)


@app.get("/api/top-picks")
def get_top_picks(limit: int = 10):
    """
    Get cached top picks of the day (fast read from cache).
    Use POST /api/top-picks/generate to refresh the cache.
    """
    cached = load_top_picks_cache()
    if cached:
        # Return cached picks, limited to requested count
        picks = cached.get('picks', [])[:limit]
        return {
            "picks": picks,
            "count": len(picks),
            "total_evaluated": cached.get('total_evaluated', 0),
            "date": cached.get('date', ''),
            "generated_at": cached.get('generated_at', ''),
            "criteria": "Score = |Edge%| × (Confidence/100)"
        }
    return {
        "picks": [],
        "count": 0,
        "total_evaluated": 0,
        "date": datetime.now().strftime('%Y-%m-%d'),
        "message": "No picks cached. Run POST /api/top-picks/generate first."
    }


@app.post("/api/top-picks/generate")
def generate_top_picks(limit: int = 10):
    """
    Generate and cache top picks of the day.
    This is slow (~30-60s) - run once daily, not on every request.

    Applies calibration adjustments based on model feedback:
    - Confidence multiplier: 0.85 (model is slightly overconfident)
    - Minimum edge thresholds by type
    - Stat-specific score multipliers
    - Filters out 10-20% edge player props (historically underperform)
    """
    all_picks = []

    # ==========================================================================
    # CALIBRATION ADJUSTMENTS (based on model_feedback analysis)
    # ==========================================================================
    CALIBRATION = {
        'confidence_multiplier': 0.85,  # Model is slightly overconfident
        'min_edge': {
            'player_prop': 20,  # 20%+ edge required (data: <20% loses money)
            'spread': 20,       # 20%+ edge required
            'total': 20         # 20%+ edge required
        },
        'stat_multipliers': {
            'POINTS': 0.89,     # Points underperform (42.9%)
            'REBOUNDS': 1.25,   # Rebounds outperform (60%)
            'ASSISTS': 1.0,     # Neutral (limited data)
            'SPREAD': 1.11,     # Spreads outperform (53.3%)
            'TOTAL': 0.83       # Totals underperform (40%)
        },
        'avoid_edge_range': None  # No longer needed - 20% min handles this
    }

    # Get lineups for player props
    try:
        from models.nba_lineups_fetcher import get_all_todays_starters, get_todays_official_lineups
        players = get_all_todays_starters()
        lineups = get_todays_official_lineups()
    except:
        players = []
        lineups = {}

    # Get odds data for game lines
    odds_data = fetch_game_odds() if get_api_key() else {'games': []}

    # ==========================================================================
    # PLAYER PROPS
    # ==========================================================================
    stat_types = [
        ('points', 'pts'),
        ('rebounds', 'reb'),
        ('assists', 'ast')
    ]

    for player in players[:50]:  # Limit to avoid timeout
        try:
            # Find opponent
            opponent = 'AVG'
            for team, data in lineups.items():
                if player in data.get('starters', []):
                    opponent = data.get('opponent', 'AVG')
                    break

            pred = predictor.predict(player, opponent)
            if not pred:
                continue

            for prop_type, stat_key in stat_types:
                proj = pred[stat_key]

                # Filter out low projections - these props likely don't exist on books
                if proj <= 5:
                    continue

                lines = get_player_prop_line(player, prop_type)

                if not lines:
                    continue

                dk_line = lines.get('dk', {}).get('over', {}).get('line', 0)
                fd_line = lines.get('fd', {}).get('over', {}).get('line', 0)
                dk_odds = lines.get('dk', {}).get('over', {}).get('odds', -110)
                fd_odds = lines.get('fd', {}).get('over', {}).get('odds', -110)

                # Use best available line
                best_line = None
                best_book = None
                if dk_line > 0 and fd_line > 0:
                    # For OVER, prefer lower line; for UNDER, prefer higher line
                    best_line = min(dk_line, fd_line)
                    best_book = 'DK' if dk_line <= fd_line else 'FD'
                elif dk_line > 0:
                    best_line = dk_line
                    best_book = 'DK'
                elif fd_line > 0:
                    best_line = fd_line
                    best_book = 'FD'

                if not best_line or best_line <= 0:
                    continue

                edge = ((proj - best_line) / best_line) * 100
                raw_confidence = pred['confidence']

                # Apply confidence calibration (model is overconfident)
                confidence = round(raw_confidence * CALIBRATION['confidence_multiplier'])

                # Filter out suspicious data:
                # - Edge over 80% is likely bad data (line doesn't exist or wrong player match)
                # - Projection way below line for UNDER (e.g., proj 1.0 vs line 6.5) is suspicious
                if abs(edge) > 80:
                    continue  # Skip - likely bad data

                # Apply minimum edge threshold for player props
                min_edge = CALIBRATION['min_edge']['player_prop']
                if abs(edge) < min_edge:
                    continue

                # Filter out low edge range if configured
                if CALIBRATION['avoid_edge_range']:
                    avoid_low, avoid_high = CALIBRATION['avoid_edge_range']
                    if avoid_low <= abs(edge) < avoid_high:
                        continue  # Skip this edge range

                # Apply stat-specific multiplier
                stat_upper = prop_type.upper()
                stat_mult = CALIBRATION['stat_multipliers'].get(stat_upper, 1.0)

                # Score = edge * confidence weight * stat multiplier
                score = abs(edge) * (confidence / 100) * stat_mult

                # Only include picks that pass all filters
                if abs(edge) >= min_edge:
                    all_picks.append({
                        'type': 'player_prop',
                        'player': player,
                        'team': pred['team'],
                        'stat': prop_type.upper(),
                        'projection': round(proj, 1),
                        'line': best_line,
                        'book': best_book,
                        'dk_line': dk_line if dk_line > 0 else None,
                        'fd_line': fd_line if fd_line > 0 else None,
                        'edge': round(edge, 1),
                        'direction': 'OVER' if edge > 0 else 'UNDER',
                        'confidence': confidence,
                        'score': round(score, 2)
                    })
        except Exception as e:
            continue

    # ==========================================================================
    # GAME LINES (Spread & Total)
    # ==========================================================================
    from models.ensemble_model import GamePredictor
    game_predictor = GamePredictor()

    for game in odds_data.get('games', []):
        try:
            home_full = game['home_team']
            away_full = game['away_team']
            home_team = TEAM_ABBREV.get(home_full, home_full[:3].upper())
            away_team = TEAM_ABBREV.get(away_full, away_full[:3].upper())

            result = game_predictor.predict_game(df, home_team, away_team)
            if not result:
                continue

            dk = game.get('bookmakers', {}).get('draftkings', {})

            # SPREAD
            dk_spread = dk.get('spreads', {}).get(home_full, {}).get('point')
            if dk_spread is not None:
                model_spread = result['spread']  # Positive = home wins by X
                # dk_spread is from home perspective: negative = home favored
                # model_spread: positive = home wins by X
                # Edge in points: how much better is model vs line
                # If model=-5 (home wins by 5) and dk=-3 (home -3), edge = 2 pts on home
                spread_edge_pts = abs(model_spread) - abs(dk_spread)

                # Convert to percentage (2 pts edge on a 5pt spread = 40% edge)
                # But cap at reasonable values
                edge = (abs(spread_edge_pts) / max(abs(dk_spread), 3)) * 100
                edge = min(edge, 50)  # Cap at 50%

                # Apply minimum edge threshold for spreads
                min_edge_spread = CALIBRATION['min_edge']['spread']
                if edge >= min_edge_spread and abs(spread_edge_pts) >= 1.5:
                    # Determine direction based on model vs line
                    if model_spread < dk_spread:
                        # Model has home winning by more (or losing by less)
                        pick_team = home_team
                        pick_spread = dk_spread
                    else:
                        # Model has away covering
                        pick_team = away_team
                        pick_spread = -dk_spread  # Flip for away

                    raw_conf = round(result['home_win_prob'] if pick_team == home_team else result['away_win_prob'])
                    # Apply confidence calibration
                    conf = round(raw_conf * CALIBRATION['confidence_multiplier'])

                    # Apply spread multiplier (spreads outperform)
                    spread_mult = CALIBRATION['stat_multipliers'].get('SPREAD', 1.0)
                    score = edge * (conf / 100) * spread_mult

                    all_picks.append({
                        'type': 'spread',
                        'player': f"{away_team} @ {home_team}",
                        'team': pick_team,
                        'stat': 'SPREAD',
                        'projection': round(model_spread, 1),
                        'line': dk_spread,
                        'book': 'DK',
                        'dk_line': dk_spread,
                        'fd_line': None,
                        'edge': round(edge, 1),
                        'direction': f"{pick_team} {'+' if pick_spread > 0 else ''}{pick_spread}",
                        'confidence': conf,
                        'score': round(score, 2)
                    })

            # TOTAL
            dk_total = dk.get('totals', {}).get('Over', {}).get('point')
            if dk_total is not None:
                model_total = result['predicted_total']
                edge = ((model_total - dk_total) / dk_total) * 100

                # Apply minimum edge threshold for totals (they underperform)
                min_edge_total = CALIBRATION['min_edge']['total']
                if abs(edge) >= min_edge_total:
                    # Apply confidence calibration
                    raw_conf = 65  # Base confidence for totals
                    conf = round(raw_conf * CALIBRATION['confidence_multiplier'])

                    # Apply total multiplier (totals underperform significantly)
                    total_mult = CALIBRATION['stat_multipliers'].get('TOTAL', 1.0)
                    score = abs(edge) * (conf / 100) * total_mult

                    all_picks.append({
                        'type': 'total',
                        'player': f"{away_team} @ {home_team}",
                        'team': f"{away_team}/{home_team}",
                        'stat': 'TOTAL',
                        'projection': round(model_total, 1),
                        'line': dk_total,
                        'book': 'DK',
                        'dk_line': dk_total,
                        'fd_line': None,
                        'edge': round(abs(edge), 1),
                        'direction': 'OVER' if edge > 0 else 'UNDER',
                        'confidence': conf,
                        'score': round(score, 2)
                    })
        except:
            continue

    # Separate player props from game props
    player_props = [p for p in all_picks if p['type'] == 'player_prop']
    game_props = [p for p in all_picks if p['type'] in ('spread', 'total')]

    # Sort each by score and take top N
    player_props = sorted(player_props, key=lambda x: x['score'], reverse=True)[:10]
    game_props = sorted(game_props, key=lambda x: x['score'], reverse=True)[:5]

    # Combine for the response
    top_picks = player_props + game_props

    result = {
        "picks": top_picks,
        "player_props": player_props,
        "game_props": game_props,
        "count": len(top_picks),
        "player_prop_count": len(player_props),
        "game_prop_count": len(game_props),
        "total_evaluated": len(all_picks),
        "date": datetime.now().strftime('%Y-%m-%d'),
        "generated_at": datetime.now().isoformat(),
        "criteria": "Score = |Edge%| × (Calibrated Confidence) × Stat Multiplier",
        "calibration_applied": {
            "confidence_multiplier": CALIBRATION['confidence_multiplier'],
            "min_edge_all_types": "20%+",
            "note": "Only recommending 20%+ edge picks (data shows <20% edge loses money)"
        }
    }

    # Save to cache
    save_top_picks_cache(result)

    return result


# =============================================================================
# DAILY TRACKING ENDPOINTS
# =============================================================================

@app.get("/api/daily-tracking")
def get_daily_tracking():
    """Get comprehensive daily tracking statistics from predictions_log"""
    try:
        from models.bet_tracker import load_predictions
        tracking = get_tracking_stats()
        predictions = load_predictions()

        # Build stats in the format frontend expects
        stats = {
            'total_days': len(set(p.get('game_date', '') for p in predictions)),
            'spread': {'picks': 0, 'hits': 0, 'rate': 0, 'pending': 0},
            'ou': {'picks': 0, 'hits': 0, 'rate': 0, 'pending': 0},
            'ml': {'picks': 0, 'hits': 0, 'rate': 0, 'pending': 0},
            'props': {'picks': 0, 'hits': 0, 'rate': 0, 'pending': 0},
            'by_date': [],
            'recent_picks': [],
            'pending_picks': [],
            'overall': {
                'picks': tracking.get('graded_picks', 0),
                'hits': tracking.get('hits', 0),
                'rate': tracking.get('hit_rate', 0)
            }
        }

        for pred in predictions:
            ptype = pred.get('type', 'PLAYER_PROP')
            hit = pred.get('hit')
            result = pred.get('result')

            # Map type to category
            if ptype == 'GAME_SPREAD':
                cat = 'spread'
            elif ptype == 'GAME_TOTAL':
                cat = 'ou'
            elif ptype == 'GAME_ML':
                cat = 'ml'
            else:
                cat = 'props'

            # Format pick string based on type
            if ptype == 'GAME_SPREAD':
                # Direction already has team + spread (e.g., "MIA -5.5")
                pick_str = f"{pred.get('matchup', '')} | {pred.get('direction', '')}"
            elif ptype == 'GAME_TOTAL':
                # Show matchup + over/under + line
                pick_str = f"{pred.get('matchup', '')} | {pred.get('direction', '')} {pred.get('line', '')}"
            else:
                # Player props: player + direction + line + stat
                pick_str = f"{pred.get('player', '')} {pred.get('direction', '')} {pred.get('line', '')} {pred.get('stat', '')}"

            if hit is not None:
                stats[cat]['picks'] += 1
                if hit:
                    stats[cat]['hits'] += 1
                stats['recent_picks'].append({
                    'date': pred.get('game_date'),
                    'type': ptype,
                    'pick': pick_str,
                    'result': f"Actual: {result}",
                    'hit': hit,
                    'edge': pred.get('edge', 0)
                })
            else:
                stats[cat]['pending'] += 1
                stats['pending_picks'].append({
                    'date': pred.get('game_date'),
                    'type': ptype,
                    'pick': pick_str,
                    'edge': pred.get('edge', 0)
                })

        # Calculate rates
        for cat in ['spread', 'ou', 'ml', 'props']:
            if stats[cat]['picks'] > 0:
                stats[cat]['rate'] = round((stats[cat]['hits'] / stats[cat]['picks']) * 100, 1)

        # Sort recent picks by date
        stats['recent_picks'] = sorted(stats['recent_picks'], key=lambda x: x.get('date', ''), reverse=True)[:50]

        # Add model feedback metrics
        try:
            feedback = analyze_performance()
            calibration = calculate_calibration_adjustments()

            # Convert numpy types for JSON
            def convert_numpy(obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(i) for i in obj]
                return obj

            stats['model_feedback'] = {
                'by_confidence': convert_numpy(feedback.get('by_confidence', {})),
                'by_stat': convert_numpy(feedback.get('by_stat', {})),
                'by_edge': convert_numpy(feedback.get('by_edge', {})),
                'recommendations': feedback.get('recommendations', []),
                'calibration': convert_numpy(calibration)
            }
        except:
            stats['model_feedback'] = None

        return stats

    except Exception as e:
        return {
            'total_days': 0,
            'spread': {'picks': 0, 'hits': 0, 'rate': 0, 'pending': 0},
            'ou': {'picks': 0, 'hits': 0, 'rate': 0, 'pending': 0},
            'ml': {'picks': 0, 'hits': 0, 'rate': 0, 'pending': 0},
            'props': {'picks': 0, 'hits': 0, 'rate': 0, 'pending': 0},
            'by_date': [],
            'recent_picks': [],
            'pending_picks': [],
            'overall': {'picks': 0, 'hits': 0, 'rate': 0},
            'error': str(e)
        }


@app.post("/api/daily-tracking/log")
def log_todays_predictions():
    """Auto-generate and log today's predictions based on edge threshold"""
    try:
        # Get today's odds
        odds_data = fetch_game_odds() if get_api_key() else {'games': []}

        # Get lineups
        try:
            from models.nba_lineups_fetcher import get_todays_official_lineups
            lineups = get_todays_official_lineups()
        except:
            lineups = {}

        # Generate predictions
        game_preds, prop_preds = generate_daily_predictions(df, odds_data, lineups)

        # Log them
        result = log_daily_picks(game_preds, prop_preds)

        return {
            "status": "success",
            "logged": result,
            "message": f"Logged {result['games']} game predictions and {result['props']} prop predictions"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/daily-tracking/grade")
def grade_yesterdays_predictions(date: str = None):
    """Grade predictions for a specific date (defaults to yesterday)"""
    result = grade_daily_picks(df, date)
    return result


@app.post("/api/tracking/auto-grade")
def auto_grade_predictions(date: str = None, regrade: bool = False):
    """
    Auto-grade predictions using real NBA stats from NBA API.

    Args:
        date: Date to grade (YYYY-MM-DD format, defaults to yesterday)
        regrade: If True, re-grade even if already graded
    """
    try:
        from datetime import datetime, timedelta
        if date is None:
            date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        if regrade:
            result = regrade_date(date)
        else:
            result = grade_predictions_for_date(date)

        return {"status": "success", **result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/tracking/auto-grade-all")
def auto_grade_all():
    """Auto-grade all ungraded predictions using real NBA stats."""
    try:
        results = grade_all_ungraded()
        return {
            "status": "success",
            "dates_graded": len(results),
            "results": results
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/model/feedback")
def get_model_feedback():
    """
    Get model performance analysis and calibration recommendations.

    Returns performance by confidence, stat type, edge, and suggested adjustments.
    """
    try:
        analysis = analyze_performance()
        adjustments = calculate_calibration_adjustments()

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj

        return {
            "status": "success",
            "analysis": convert_numpy(analysis),
            "calibration": convert_numpy(adjustments)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/model/save-feedback")
def save_model_feedback():
    """Save current model feedback analysis to file."""
    try:
        feedback = save_feedback()
        return {"status": "success", "message": "Feedback saved", "generated_at": feedback['generated_at']}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# =============================================================================
# STARTUP EVENT - Regenerate cache on server start (only if needed)
# =============================================================================
@app.on_event("startup")
async def startup_regenerate_cache():
    """
    Regenerate games and picks cache on server startup.
    1. First refresh lineups from Rotowire
    2. Only regenerates if cache is empty/stale to avoid unnecessary API calls.
    """
    import asyncio

    async def regen():
        await asyncio.sleep(3)  # Wait for server to be ready
        try:
            today = datetime.now().strftime('%Y-%m-%d')

            # STEP 1: Refresh lineups from Rotowire
            print("[STARTUP] Refreshing lineups from Rotowire...")
            try:
                lineups = refresh_lineups_from_rotowire()
                if lineups:
                    print(f"[STARTUP] Loaded {len(lineups)} team lineups from Rotowire")
                else:
                    print("[STARTUP] Using cached/fallback lineups")
            except Exception as e:
                print(f"[STARTUP] Lineup refresh failed: {e}, using fallback")

            # STEP 2: Check if games cache exists and is from today
            games_cache = load_games_cache()

            if games_cache and games_cache.get('date') == today:
                print(f"[STARTUP] Games cache exists for {today}, skipping regeneration")
            else:
                print("[STARTUP] No valid games cache, regenerating...")
                generate_games()
                print("[STARTUP] Games cache regenerated")

            # Check picks cache
            picks_cache = load_top_picks_cache()
            if picks_cache and picks_cache.get('date') == today:
                print(f"[STARTUP] Picks cache exists for {today}, skipping regeneration")
            else:
                print("[STARTUP] No valid picks cache, regenerating...")
                generate_top_picks(limit=10)
                print("[STARTUP] Picks cache regenerated")

        except Exception as e:
            print(f"[STARTUP] Cache check/regen error: {e}")

    asyncio.create_task(regen())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
