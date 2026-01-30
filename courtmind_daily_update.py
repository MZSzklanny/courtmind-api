# -*- coding: utf-8 -*-
"""
CourtMind Daily Update - 6AM Automation
========================================
1. Pull latest NBA game data
2. Scrape injury report from ESPN
3. Update INJURED_PLAYERS in app.py
4. Update production parquet files

Run via Task Scheduler at 6:00 AM daily.
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import os
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import time

# Paths
COURTMIND_DIR = r"C:\Users\user\CourtMind"
APP_FILE = os.path.join(COURTMIND_DIR, "app.py")
LOG_FILE = os.path.join(COURTMIND_DIR, "daily_update_log.txt")
DATA_DIR = r"C:\Users\user"

# Our tracked players (must match ALL_TOP_PLAYERS in app.py)
TRACKED_PLAYERS = [
    'Shai Gilgeous-Alexander', 'Luka Dončić', 'Giannis Antetokounmpo',
    'Anthony Edwards', 'Nikola Jokić', 'Jalen Brunson', 'Kevin Durant',
    'Cade Cunningham', 'Jayson Tatum', 'Devin Booker', 'Stephen Curry',
    'Donovan Mitchell', 'LeBron James', 'Trae Young', 'James Harden',
    'Anthony Davis', 'Karl-Anthony Towns', 'Damian Lillard', 'Jaylen Brown',
    'Jamal Murray', 'Darius Garland', "De'Aaron Fox", 'Alperen Sengun',
    'Lauri Markkanen', 'Jalen Williams', 'Tyler Herro', 'Desmond Bane',
    'Zach LaVine', 'Mikal Bridges', 'LaMelo Ball', 'Tyrese Haliburton',
    'Dejounte Murray', 'Evan Mobley', 'Jaren Jackson Jr.', 'Coby White',
    # Additional stars to track
    'Joel Embiid', 'Paul George', 'Kawhi Leonard', 'Zion Williamson',
    'Chet Holmgren', 'Ja Morant', 'Paolo Banchero', 'Franz Wagner',
    'Jimmy Butler', 'Kyrie Irving', 'Khris Middleton'
]

def log(msg):
    """Log message to console and file."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def scrape_espn_injuries():
    """Scrape injury data from ESPN."""
    log("Fetching ESPN injury report...")

    url = "https://www.espn.com/nba/injuries"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        injured_players = set()

        # Find all player entries
        # ESPN structure: tables with player names and injury status
        tables = soup.find_all('table')

        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    # First cell usually has player name
                    player_cell = cells[0]
                    player_link = player_cell.find('a')
                    if player_link:
                        player_name = player_link.get_text(strip=True)
                    else:
                        player_name = player_cell.get_text(strip=True)

                    # Check status (usually "Out", "Day-To-Day", etc.)
                    status_text = ' '.join(c.get_text(strip=True) for c in cells[1:])

                    # Only track if player is OUT or has significant injury
                    if any(kw in status_text.lower() for kw in ['out', 'season', 'acl', 'achilles', 'surgery']):
                        # Check if this is one of our tracked players
                        for tracked in TRACKED_PLAYERS:
                            # Fuzzy match (last name)
                            if tracked.split()[-1].lower() in player_name.lower():
                                injured_players.add(tracked)
                                log(f"  Found injured: {tracked} ({status_text[:50]}...)")
                                break

        return injured_players

    except Exception as e:
        log(f"Error scraping ESPN: {e}")
        return set()

def scrape_bbref_injuries():
    """Scrape injury data from Basketball Reference."""
    log("Fetching Basketball Reference injury report...")

    url = "https://www.basketball-reference.com/friv/injuries.fcgi"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        injured_players = set()

        # BBRef has a table with injuries
        table = soup.find('table', {'id': 'injuries'})
        if table:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 3:
                    player_cell = cells[0]
                    player_link = player_cell.find('a')
                    if player_link:
                        player_name = player_link.get_text(strip=True)

                        # Get injury description
                        desc = cells[2].get_text(strip=True) if len(cells) > 2 else ''

                        # Check if significant injury
                        if any(kw in desc.lower() for kw in ['out', 'season', 'acl', 'achilles', 'surgery', 'week']):
                            for tracked in TRACKED_PLAYERS:
                                if tracked.split()[-1].lower() in player_name.lower():
                                    injured_players.add(tracked)
                                    log(f"  Found injured: {tracked}")
                                    break

        return injured_players

    except Exception as e:
        log(f"Error scraping BBRef: {e}")
        return set()

def update_app_injuries(injured_players):
    """Update INJURED_PLAYERS in app.py."""
    log(f"Updating app.py with {len(injured_players)} injured players...")

    with open(APP_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # Build new INJURED_PLAYERS block
    today = datetime.now().strftime('%Y-%m-%d')
    new_block = f"# Injured players (auto-updated {today})\n"
    new_block += "# Source: ESPN/BBRef injury reports\n"
    new_block += "INJURED_PLAYERS = {\n"
    for player in sorted(injured_players):
        new_block += f"    '{player}',\n"
    new_block += "}"

    # Replace old block
    pattern = r'# Injured players \([^)]+\)\n# [^\n]+\nINJURED_PLAYERS = \{[^}]+\}'

    if re.search(pattern, content, flags=re.DOTALL):
        new_content = re.sub(pattern, new_block, content, flags=re.DOTALL)
    else:
        # Fallback pattern
        pattern2 = r'INJURED_PLAYERS = \{[^}]+\}'
        new_content = re.sub(pattern2, new_block, content, flags=re.DOTALL)

    with open(APP_FILE, 'w', encoding='utf-8') as f:
        f.write(new_content)

    log(f"Updated INJURED_PLAYERS with: {', '.join(sorted(injured_players))}")

def update_game_data():
    """Run the NBA daily data update."""
    log("Updating game data...")

    daily_update_script = os.path.join(DATA_DIR, "nba_daily_update.py")

    if os.path.exists(daily_update_script):
        try:
            # Run the daily update
            import subprocess
            result = subprocess.run(
                [sys.executable, daily_update_script],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            if result.returncode == 0:
                log("Game data update completed successfully")
            else:
                log(f"Game data update failed: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            log("Game data update timed out after 10 minutes")
        except Exception as e:
            log(f"Error running game data update: {e}")
    else:
        log(f"Daily update script not found: {daily_update_script}")

def update_production_parquet():
    """Convert updated Excel to production parquet."""
    log("Updating production parquet files...")

    try:
        # Quarter data
        quarter_xlsx = os.path.join(DATA_DIR, "NBA_Quarter_ALL_Combined.xlsx")
        quarter_parquet = os.path.join(DATA_DIR, "NBA_PRODUCTION.parquet")

        if os.path.exists(quarter_xlsx):
            df = pd.read_excel(quarter_xlsx)
            df.to_parquet(quarter_parquet, index=False)
            log(f"Updated {quarter_parquet} ({len(df)} rows)")

        # Game data
        game_xlsx = os.path.join(DATA_DIR, "NBA_Game_ALL_Combined.xlsx")
        game_parquet = os.path.join(DATA_DIR, "NBA_Game_PRODUCTION.parquet")

        if os.path.exists(game_xlsx):
            df = pd.read_excel(game_xlsx)
            df.to_parquet(game_parquet, index=False)
            log(f"Updated {game_parquet} ({len(df)} rows)")

    except Exception as e:
        log(f"Error updating parquet files: {e}")

def main():
    log("=" * 60)
    log("CourtMind Daily Update Starting")
    log("=" * 60)

    start_time = datetime.now()

    # 1. Update game data
    update_game_data()

    # 2. Scrape injuries from multiple sources
    espn_injured = scrape_espn_injuries()
    time.sleep(2)  # Be nice to servers
    bbref_injured = scrape_bbref_injuries()

    # Combine injury lists
    all_injured = espn_injured | bbref_injured

    # 3. Update app.py with injuries
    if all_injured:
        update_app_injuries(all_injured)
    else:
        log("No injuries found - keeping existing list")

    # 4. Update parquet files
    update_production_parquet()

    elapsed = (datetime.now() - start_time).total_seconds()
    log(f"Daily update completed in {elapsed:.1f} seconds")
    log("=" * 60)

if __name__ == "__main__":
    main()
