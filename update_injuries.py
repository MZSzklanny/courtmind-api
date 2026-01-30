# -*- coding: utf-8 -*-
"""
CourtMind Daily Injury Update
=============================
Run this script each morning to check and update injured players.

Usage: python update_injuries.py
"""

import webbrowser
import re
from datetime import datetime

APP_FILE = 'C:/Users/user/CourtMind/app.py'
INJURY_URLS = [
    'https://www.espn.com/nba/injuries',
    'https://www.basketball-reference.com/friv/injuries.fcgi'
]

def get_current_injured():
    """Read current injured players from app.py"""
    with open(APP_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find INJURED_PLAYERS set
    match = re.search(r'INJURED_PLAYERS = \{([^}]+)\}', content, re.DOTALL)
    if match:
        players_text = match.group(1)
        # Extract player names
        players = re.findall(r"'([^']+)'", players_text)
        return players
    return []

def update_injured_players(players):
    """Update INJURED_PLAYERS in app.py"""
    with open(APP_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # Build new INJURED_PLAYERS block
    today = datetime.now().strftime('%Y-%m-%d')
    new_block = f"# Injured players (updated {today} from ESPN/BBRef)\n"
    new_block += "# Check daily: https://www.espn.com/nba/injuries\n"
    new_block += "INJURED_PLAYERS = {\n"
    for player in sorted(players):
        new_block += f"    '{player}',\n"
    new_block += "}"

    # Replace old block
    pattern = r'# Injured players \(updated.*?\nINJURED_PLAYERS = \{[^}]+\}'
    new_content = re.sub(pattern, new_block, content, flags=re.DOTALL)

    with open(APP_FILE, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"Updated {len(players)} injured players in app.py")

def main():
    print("=" * 50)
    print("CourtMind Daily Injury Update")
    print("=" * 50)
    print()

    # Show current injured players
    current = get_current_injured()
    print(f"Current injured players ({len(current)}):")
    for p in sorted(current):
        print(f"  - {p}")
    print()

    # Open injury pages
    print("Opening injury report pages...")
    for url in INJURY_URLS:
        webbrowser.open(url)

    print()
    print("Review the injury pages, then update the list below.")
    print("Commands:")
    print("  + PlayerName  = Add player to injured list")
    print("  - PlayerName  = Remove player from injured list")
    print("  list          = Show current list")
    print("  save          = Save changes and exit")
    print("  quit          = Exit without saving")
    print()

    injured = set(current)

    while True:
        cmd = input("> ").strip()

        if cmd.lower() == 'quit':
            print("Exited without saving.")
            break
        elif cmd.lower() == 'save':
            update_injured_players(list(injured))
            print("Saved! Restart CourtMind to see changes.")
            break
        elif cmd.lower() == 'list':
            print(f"\nCurrent injured ({len(injured)}):")
            for p in sorted(injured):
                print(f"  - {p}")
            print()
        elif cmd.startswith('+'):
            player = cmd[1:].strip()
            if player:
                injured.add(player)
                print(f"Added: {player}")
        elif cmd.startswith('-'):
            player = cmd[1:].strip()
            if player in injured:
                injured.remove(player)
                print(f"Removed: {player}")
            else:
                print(f"Not found: {player}")
        else:
            print("Unknown command. Use +Name, -Name, list, save, or quit")

if __name__ == "__main__":
    main()
