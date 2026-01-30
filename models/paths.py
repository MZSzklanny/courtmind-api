# -*- coding: utf-8 -*-
"""
Centralized path configuration for CourtMind
"""
from pathlib import Path
import os

# Base directory is the CourtMind folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directory
DATA_DIR = BASE_DIR / 'data'

# Cache files
CACHE_DIR = BASE_DIR

# Data files - check both locations (deployed vs local dev)
def get_data_file(filename):
    """Get path to data file, checking multiple locations."""
    # First check data/ subdirectory (deployed)
    deployed_path = DATA_DIR / filename
    if deployed_path.exists():
        return deployed_path

    # Then check C:/Users/user (local dev on Windows)
    local_path = Path(f'C:/Users/user/{filename}')
    if local_path.exists():
        return local_path

    # Default to deployed path
    return deployed_path

# Common file paths
NBA_PRODUCTION_PARQUET = get_data_file('NBA_PRODUCTION.parquet')
NBA_GAME_PRODUCTION_PARQUET = get_data_file('NBA_Game_PRODUCTION.parquet')

# Cache and config files
ODDS_CACHE_FILE = CACHE_DIR / 'odds_cache.json'
PROPS_CACHE_FILE = CACHE_DIR / 'props_cache.json'
LINEUPS_CACHE_FILE = CACHE_DIR / 'lineups_cache.json'
NBA_LINEUPS_CACHE_FILE = CACHE_DIR / 'nba_lineups_cache.json'

# API key file
ODDS_API_KEY_FILE = CACHE_DIR / 'odds_api_key.txt'

# Check environment variable for API key (for cloud deployment)
def get_odds_api_key():
    """Get odds API key from env var or file."""
    # First check environment variable
    key = os.environ.get('ODDS_API_KEY')
    if key:
        return key

    # Then check file
    if ODDS_API_KEY_FILE.exists():
        return ODDS_API_KEY_FILE.read_text().strip()

    return None
