# -*- coding: utf-8 -*-
"""
CourtMind Afternoon Lineup Update - 4PM Daily
==============================================
Fetches latest starting lineups from NBA.com before games start.
Run via Task Scheduler at 4:00 PM daily.
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time

COURTMIND_DIR = r"C:\Users\user\CourtMind"
LINEUPS_FILE = os.path.join(COURTMIND_DIR, "todays_lineups.json")
LOG_FILE = os.path.join(COURTMIND_DIR, "lineup_update_log.txt")

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def fetch_nba_schedule():
    """Get today's games from NBA.com"""
    today = datetime.now().strftime('%Y-%m-%d')
    url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
    headers = {'User-Agent': 'Mozilla/5.0'}

    games = []
    try:
        response = requests.get(url, headers=headers, timeout=15)
        data = response.json()

        for date_group in data.get('leagueSchedule', {}).get('gameDates', []):
            game_date = date_group.get('gameDate', '')[:10]
            if game_date == today:
                for game in date_group.get('games', []):
                    home = game.get('homeTeam', {})
                    away = game.get('awayTeam', {})
                    games.append({
                        'game_id': game.get('gameId'),
                        'home_team': home.get('teamTricode', ''),
                        'away_team': away.get('teamTricode', ''),
                        'game_time': game.get('gameDateTimeUTC'),
                    })
                break
    except Exception as e:
        log(f"Error fetching NBA schedule: {e}")

    return games


def fetch_rotowire_lineups():
    """Scrape latest lineups from RotoWire"""
    url = "https://www.rotowire.com/basketball/nba-lineups.php"
    headers = {'User-Agent': 'Mozilla/5.0'}

    lineups = {}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Team abbreviation mapping
        team_map = {
            'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
            'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
            'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
            'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
            'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
            'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
            'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
            'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
            'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
            'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS',
        }

        lineup_boxes = soup.find_all('div', class_='lineup__box')

        for box in lineup_boxes:
            team_elem = box.find('div', class_='lineup__team')
            if not team_elem:
                continue

            team_link = team_elem.find('a')
            if not team_link:
                continue

            team_name = team_link.get_text(strip=True)
            team_abbrev = team_map.get(team_name)
            if not team_abbrev:
                continue

            starters = []
            bench = []

            player_elems = box.find_all('li', class_='lineup__player')
            positions = ['PG', 'SG', 'SF', 'PF', 'C']

            for i, player_elem in enumerate(player_elems):
                player_link = player_elem.find('a')
                if not player_link:
                    continue

                player_name = player_link.get_text(strip=True)

                # Check for injury indicator
                injury_elem = player_elem.find('span', class_='lineup__inj')
                injury_status = injury_elem.get_text(strip=True) if injury_elem else None

                player_data = {
                    'name': player_name,
                    'position': positions[i] if i < 5 else 'BENCH',
                    'injury_status': injury_status
                }

                if i < 5:
                    starters.append(player_data)
                else:
                    bench.append(player_data)

            lineups[team_abbrev] = {
                'starters': starters,
                'bench': bench[:5],  # Top 5 bench
                'source': 'rotowire',
                'updated': datetime.now().isoformat()
            }

            log(f"  {team_abbrev}: {len(starters)} starters, {len(bench)} bench")

    except Exception as e:
        log(f"Error fetching RotoWire: {e}")

    return lineups


def fetch_fantasylabs_lineups():
    """Backup: Try FantasyLabs for lineups"""
    # This is a backup source if RotoWire fails
    url = "https://www.fantasylabs.com/nba/lineups/"
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        # Parse as needed...
        return {}
    except:
        return {}


def main():
    log("=" * 60)
    log("CourtMind Afternoon Lineup Update (4 PM)")
    log("=" * 60)

    # Get today's games
    games = fetch_nba_schedule()
    log(f"Found {len(games)} games today")

    if not games:
        log("No games today - skipping lineup fetch")
        return

    for g in games:
        log(f"  {g['away_team']} @ {g['home_team']}")

    # Fetch lineups
    log("\nFetching lineups from RotoWire...")
    lineups = fetch_rotowire_lineups()

    log(f"\nGot lineups for {len(lineups)} teams")

    # Save to file
    result = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'updated': datetime.now().isoformat(),
        'games': games,
        'lineups': lineups
    }

    with open(LINEUPS_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    log(f"\nSaved lineups to {LINEUPS_FILE}")
    log("Lineup update complete!")


if __name__ == "__main__":
    main()
