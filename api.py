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
from models.bet_tracker import (
    get_tracking_stats, check_results, get_todays_predictions,
    log_daily_predictions, get_daily_tracking_stats, log_daily_picks,
    generate_daily_predictions, grade_daily_picks
)

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

# Load data once at startup
DATA_FILE = BASE_DIR / 'data' / 'NBA_PRODUCTION.parquet'
if not DATA_FILE.exists():
    # Fallback for local development
    DATA_FILE = Path('C:/Users/user/NBA_PRODUCTION.parquet')
df = pd.read_parquet(DATA_FILE)
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


@app.get("/api/teams/playing")
def get_teams_playing_today():
    """Get teams playing today"""
    from models.nba_lineups_fetcher import TODAYS_LINEUPS
    try:
        # Use lineups as primary source
        if TODAYS_LINEUPS:
            return {"teams": list(TODAYS_LINEUPS.keys())}
        # Fallback to odds API
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
    """Get today's games with predictions and odds"""
    from models.ensemble_model import GamePredictor
    from models.nba_lineups_fetcher import TODAYS_LINEUPS

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
        # Create game entries from lineups (home teams only to avoid duplicates)
        for team, data in TODAYS_LINEUPS.items():
            if data.get('home'):
                home_team = team
                away_team = data['opponent']
                games_list.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'from_lineups': True  # Flag to indicate source
                })

    for game in games_list:
        # Handle both Odds API format (full names) and lineups format (abbreviations)
        if game.get('from_lineups'):
            home_team = game['home_team']
            away_team = game['away_team']
            # Reverse lookup for full names
            abbrev_to_full = {v: k for k, v in TEAM_ABBREV.items()}
            home_full = abbrev_to_full.get(home_team, home_team)
            away_full = abbrev_to_full.get(away_team, away_team)
        else:
            home_full = game['home_team']
            away_full = game['away_team']
            home_team = TEAM_ABBREV.get(home_full, home_full[:3].upper())
            away_team = TEAM_ABBREV.get(away_full, away_full[:3].upper())

        # Get prediction
        result = game_predictor.predict_game(df, home_team, away_team)

        # Find top plays
        top_plays = []
        game_players = []
        if home_team in official_lineups:
            game_players.extend([(p, home_team, away_team) for p in official_lineups[home_team].get('starters', [])])
        if away_team in official_lineups:
            game_players.extend([(p, away_team, home_team) for p in official_lineups[away_team].get('starters', [])])

        for player_name, team, opp in game_players:
            try:
                pred = predictor.predict(player_name, opp)
                if not pred:
                    continue
                pts_line = get_player_prop_line(player_name, 'points')
                if pts_line:
                    dk_line = pts_line.get('dk', {}).get('over', {}).get('line', 0)
                    fd_line = pts_line.get('fd', {}).get('over', {}).get('line', 0)
                    best_line = min(dk_line, fd_line) if dk_line > 0 and fd_line > 0 else max(dk_line, fd_line)
                    if best_line > 0:
                        edge = ((pred['pts'] - best_line) / best_line) * 100
                        score = abs(edge) * (pred['confidence'] / 100)
                        if abs(edge) >= 3:
                            top_plays.append({
                                'player': player_name,
                                'team': team,
                                'stat': 'PTS',
                                'projection': pred['pts'],
                                'dk_line': dk_line if dk_line > 0 else None,
                                'fd_line': fd_line if fd_line > 0 else None,
                                'edge': round(edge, 1),
                                'direction': 'OVER' if edge > 0 else 'UNDER',
                                'confidence': pred['confidence'],
                                'score': score
                            })
            except:
                continue

        top_plays = sorted(top_plays, key=lambda x: x['score'], reverse=True)[:3]

        # Get odds
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

    return {"games": results, "count": len(results)}


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
        "odds_api": get_api_key() is not None
    }


# =============================================================================
# DAILY TRACKING ENDPOINTS
# =============================================================================

@app.get("/api/daily-tracking")
def get_daily_tracking():
    """Get comprehensive daily tracking statistics"""
    stats = get_daily_tracking_stats()
    return stats


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
