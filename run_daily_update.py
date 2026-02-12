# -*- coding: utf-8 -*-
"""
CourtMind Daily Update
======================
Complete workflow for daily predictions with Odds API integration

Run this once per day (morning or evening) to:
1. Refresh player props from Odds API
2. Refresh game odds from Odds API
3. Generate top picks with DK/FD lines
4. Save to cache for production API
"""

import sys
import requests
import time
from datetime import datetime

# API endpoint (use production or local)
API_BASE = "https://courtmind-api.onrender.com"
# API_BASE = "http://localhost:8000"  # Uncomment for local testing

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}")

def call_endpoint(method, path, description):
    """Call API endpoint and return success status."""
    log(f"{description}...")
    url = f"{API_BASE}{path}"

    try:
        if method == "POST":
            response = requests.post(url, timeout=120)
        else:
            response = requests.get(url, timeout=30)

        if response.status_code == 200:
            data = response.json()
            log(f"SUCCESS: {description}")
            return data
        else:
            log(f"FAILED: {description} ({response.status_code})")
            return None
    except requests.Timeout:
        log(f"TIMEOUT: {description}")
        return None
    except Exception as e:
        log(f"ERROR: {description} - {e}")
        return None

def main():
    log("=" * 70)
    log("COURTMIND DAILY UPDATE - STARTING")
    log("=" * 70)

    # Step 1: Health check
    health = call_endpoint("GET", "/api/health", "Health check")
    if not health:
        log("ERROR: API is not responding. Aborting.")
        return

    log(f"  Data loaded: {health.get('data_loaded')}")
    log(f"  Odds API available: {health.get('odds_api')}")

    # Step 2: Refresh lineups from Rotowire
    log("\n" + "=" * 70)
    log("STEP 1: Refresh Lineups")
    log("=" * 70)
    lineups = call_endpoint("POST", "/api/lineups/refresh", "Fetching lineups from Rotowire")
    if lineups and lineups.get('success'):
        log(f"  Lineups for {lineups.get('count', 0)} teams")

    time.sleep(2)

    # Step 3: Refresh player props from Odds API
    log("\n" + "=" * 70)
    log("STEP 2: Refresh Player Props from Odds API")
    log("=" * 70)
    props = call_endpoint("POST", "/api/props/refresh", "Fetching player props from Odds API")
    if props and props.get('success'):
        log(f"  Props cached for {props.get('player_count', 0)} players")
        log(f"  From {props.get('game_count', 0)} games")

    time.sleep(2)

    # Step 4: Generate games data with odds
    log("\n" + "=" * 70)
    log("STEP 3: Generate Games Data")
    log("=" * 70)
    games = call_endpoint("POST", "/api/games/generate", "Generating game predictions with odds")
    if games:
        log(f"  Generated predictions for {games.get('count', 0)} games")

    time.sleep(3)

    # Step 5: Generate TOP PICKS (calibrated, filtered)
    log("\n" + "=" * 70)
    log("STEP 4: Generate TOP 10 PICKS")
    log("=" * 70)
    log("  Using model calibration:")
    log("    - 20%+ edge requirement")
    log("    - Confidence multiplier: 0.85")
    log("    - Stat-specific scoring")
    log("    - Bad data filtered (>80% edge)")

    top_picks = call_endpoint("POST", "/api/top-picks/generate?limit=10",
                               "Generating top 10 picks")

    if top_picks:
        picks = top_picks.get('picks', [])
        player_props = [p for p in picks if p.get('type') == 'player_prop']
        game_props = [p for p in picks if p.get('type') in ['spread', 'total']]

        log(f"  Total picks generated: {len(picks)}")
        log(f"    Player props: {len(player_props)}")
        log(f"    Game props: {len(game_props)}")

        # Show top 5 picks
        if picks:
            log("\n  TOP 5 PICKS:")
            log("  " + "-" * 66)
            for i, pick in enumerate(picks[:5], 1):
                ptype = pick.get('type', '').upper()
                if ptype == 'PLAYER_PROP':
                    player = pick.get('player', '')
                    stat = pick.get('stat', '')
                    direction = pick.get('direction', '')
                    line = pick.get('line', 0)
                    proj = pick.get('projection', 0)
                    edge = pick.get('edge', 0)
                    book = pick.get('book', '')
                    score = pick.get('score', 0)

                    log(f"  {i}. {player} {stat} {direction} {line} ({book})")
                    log(f"     Proj: {proj} | Edge: {edge:.1f}% | Score: {score:.2f}")
                else:
                    matchup = pick.get('player', '')
                    stat = pick.get('stat', '')
                    direction = pick.get('direction', '')
                    line = pick.get('line', 0)
                    edge = pick.get('edge', 0)
                    book = pick.get('book', '')
                    score = pick.get('score', 0)

                    log(f"  {i}. {matchup} - {stat} {direction}")
                    log(f"     Line: {line} | Edge: {edge:.1f}% | Score: {score:.2f}")

    # Step 5: Log picks to tracking
    log("\n" + "=" * 70)
    log("STEP 5: Log Picks to Tracking")
    log("=" * 70)

    if top_picks and top_picks.get('picks'):
        from models.bet_tracker import log_daily_predictions
        logged_count = log_daily_predictions(top_picks['picks'])
        log(f"  Logged {logged_count} picks to tracking")

    time.sleep(2)

    # Step 6: Summary
    log("\n" + "=" * 70)
    log("DAILY UPDATE COMPLETE")
    log("=" * 70)
    log(f"API Base: {API_BASE}")
    log(f"Timestamp: {datetime.now().isoformat()}")
    log("\nData is cached and ready for frontend consumption.")
    log("Endpoints updated:")
    log("  - GET /api/games")
    log("  - GET /api/top-picks")
    log("  - GET /api/lineups")
    log("  - GET /api/props/{player}")
    log("=" * 70)

if __name__ == "__main__":
    main()
