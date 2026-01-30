"""
CourtMind Configuration
=======================
Team colors, paths, typography, and settings for Twitter content generation.
"""

import os

# =============================================================================
# PATHS
# =============================================================================
DATA_DIR = r"C:\Users\user"
COURTMIND_DIR = r"C:\Users\user\CourtMind"
OUTPUT_DIR = os.path.join(COURTMIND_DIR, "daily_posts")

DATA_PATHS = {
    'combined_data': os.path.join(DATA_DIR, "NBA_Quarter_ALL_Combined.parquet"),
    'training_data': os.path.join(DATA_DIR, "NBA_Training_Full_5Seasons.parquet"),
    'lstm_model': os.path.join(DATA_DIR, "player_model_lstm.keras"),
    'scalers': os.path.join(DATA_DIR, "player_model_scalers.pkl"),
    'player_stats': os.path.join(DATA_DIR, "player_historical_stats.pkl"),
    'player_vs_team': os.path.join(DATA_DIR, "player_vs_team_stats.pkl"),
    'team_matchups': os.path.join(DATA_DIR, "team_matchup_stats.json"),
    'injuries': os.path.join(DATA_DIR, "NBA_Injuries_Combined.xlsx"),
}

# =============================================================================
# TWITTER SETTINGS
# =============================================================================
TWITTER_HANDLE = "@CourtMindAI"
TWITTER_LANDSCAPE = (1200, 675)  # 16:9 for optimal feed display
TWITTER_SQUARE = (1080, 1080)    # 1:1 for maximum engagement

# =============================================================================
# NEON CYBERPUNK STYLE - COLORS
# =============================================================================
COLORS = {
    'background': '#0a0a12',       # Deep dark blue-black
    'card_bg': '#12121f',          # Dark card background
    'text_primary': '#ffffff',     # White text
    'text_secondary': '#7a8599',   # Muted blue-gray text
    'accent_positive': '#00ff88',  # Neon green
    'accent_negative': '#ff3366',  # Neon pink/red
    'accent_neutral': '#ffaa00',   # Neon orange
    'accent_blue': '#00d4ff',      # Neon cyan
    'border': '#1e2a3a',           # Dark border
    'gold': '#00ff88',             # Neon green for highlights (was gold)
    'neon_green': '#00ff88',       # Bright neon green
    'neon_blue': '#00d4ff',        # Bright neon cyan
    'neon_purple': '#bf00ff',      # Neon purple
    'glow': '#00ff8844',           # Green glow (with alpha)
}

# Rating colors for SGDR, matchup ratings (Neon style)
RATING_COLORS = {
    'excellent': '#00ff88',   # Neon green - low fatigue, great matchup
    'good': '#00d4ff',        # Neon cyan
    'average': '#ffaa00',     # Neon orange
    'concerning': '#ff3366',  # Neon pink - high fatigue
    'critical': '#ff0044',    # Bright red
}

# =============================================================================
# NBA TEAM COLORS (All 30 Teams)
# =============================================================================
TEAM_COLORS = {
    # Atlantic Division
    'BOS': {'primary': '#007A33', 'secondary': '#BA9653', 'name': 'Celtics'},
    'BKN': {'primary': '#000000', 'secondary': '#FFFFFF', 'name': 'Nets'},
    'NYK': {'primary': '#006BB6', 'secondary': '#F58426', 'name': 'Knicks'},
    'PHI': {'primary': '#006BB6', 'secondary': '#ED174C', 'name': 'Sixers'},
    'TOR': {'primary': '#CE1141', 'secondary': '#000000', 'name': 'Raptors'},

    # Central Division
    'CHI': {'primary': '#CE1141', 'secondary': '#000000', 'name': 'Bulls'},
    'CLE': {'primary': '#860038', 'secondary': '#FDBB30', 'name': 'Cavaliers'},
    'DET': {'primary': '#C8102E', 'secondary': '#1D42BA', 'name': 'Pistons'},
    'IND': {'primary': '#002D62', 'secondary': '#FDBB30', 'name': 'Pacers'},
    'MIL': {'primary': '#00471B', 'secondary': '#EEE1C6', 'name': 'Bucks'},

    # Southeast Division
    'ATL': {'primary': '#E03A3E', 'secondary': '#C1D32F', 'name': 'Hawks'},
    'CHA': {'primary': '#1D1160', 'secondary': '#00788C', 'name': 'Hornets'},
    'MIA': {'primary': '#98002E', 'secondary': '#F9A01B', 'name': 'Heat'},
    'ORL': {'primary': '#0077C0', 'secondary': '#C4CED4', 'name': 'Magic'},
    'WAS': {'primary': '#002B5C', 'secondary': '#E31837', 'name': 'Wizards'},

    # Northwest Division
    'DEN': {'primary': '#0E2240', 'secondary': '#FEC524', 'name': 'Nuggets'},
    'MIN': {'primary': '#0C2340', 'secondary': '#236192', 'name': 'Timberwolves'},
    'OKC': {'primary': '#007AC1', 'secondary': '#EF3B24', 'name': 'Thunder'},
    'POR': {'primary': '#E03A3E', 'secondary': '#000000', 'name': 'Trail Blazers'},
    'UTA': {'primary': '#002B5C', 'secondary': '#00471B', 'name': 'Jazz'},

    # Pacific Division
    'GSW': {'primary': '#1D428A', 'secondary': '#FFC72C', 'name': 'Warriors'},
    'LAC': {'primary': '#C8102E', 'secondary': '#1D428A', 'name': 'Clippers'},
    'LAL': {'primary': '#552583', 'secondary': '#FDB927', 'name': 'Lakers'},
    'PHX': {'primary': '#1D1160', 'secondary': '#E56020', 'name': 'Suns'},
    'SAC': {'primary': '#5A2D81', 'secondary': '#63727A', 'name': 'Kings'},

    # Southwest Division
    'DAL': {'primary': '#00538C', 'secondary': '#002B5E', 'name': 'Mavericks'},
    'HOU': {'primary': '#CE1141', 'secondary': '#000000', 'name': 'Rockets'},
    'MEM': {'primary': '#5D76A9', 'secondary': '#12173F', 'name': 'Grizzlies'},
    'NOP': {'primary': '#0C2340', 'secondary': '#C8102E', 'name': 'Pelicans'},
    'SAS': {'primary': '#C4CED4', 'secondary': '#000000', 'name': 'Spurs'},
}

# Team abbreviation aliases (for API compatibility)
TEAM_ALIASES = {
    'PHX': 'PHO',  # Some APIs use PHO for Phoenix
    'BKN': 'BRK',  # Some APIs use BRK for Brooklyn
    'CHA': 'CHO',  # Some APIs use CHO for Charlotte
}

# =============================================================================
# HASHTAGS
# =============================================================================
HASHTAGS = {
    'always': ['#NBA', '#NBATwitter'],
    'fantasy': ['#FantasyBasketball', '#DFS', '#PrizePicks'],
    'betting': ['#NBABets', '#PropBets', '#GamblingTwitter'],
}

# Team-specific hashtags
TEAM_HASHTAGS = {
    'ATL': ['#TrueToAtlanta', '#Hawks'],
    'BOS': ['#BleedGreen', '#Celtics'],
    'BKN': ['#NetsWorld', '#Nets'],
    'CHA': ['#AllFly', '#Hornets'],
    'CHI': ['#BullsNation', '#Bulls'],
    'CLE': ['#LetEmKnow', '#Cavs'],
    'DAL': ['#MFFL', '#Mavs'],
    'DEN': ['#MileHighBasketball', '#Nuggets'],
    'DET': ['#Pistons'],
    'GSW': ['#DubNation', '#Warriors'],
    'HOU': ['#Rockets'],
    'IND': ['#Pacers'],
    'LAC': ['#ClipperNation', '#Clippers'],
    'LAL': ['#LakeShow', '#Lakers'],
    'MEM': ['#GrindCity', '#Grizzlies'],
    'MIA': ['#HeatCulture', '#Heat'],
    'MIL': ['#FearTheDeer', '#Bucks'],
    'MIN': ['#Timberwolves', '#Wolves'],
    'NOP': ['#WontBowDown', '#Pelicans'],
    'NYK': ['#NewYorkForever', '#Knicks'],
    'OKC': ['#ThunderUp', '#Thunder'],
    'ORL': ['#MagicTogether', '#Magic'],
    'PHI': ['#HereTheyCome', '#Sixers'],
    'PHX': ['#ValleyProud', '#Suns'],
    'POR': ['#RipCity', '#Blazers'],
    'SAC': ['#LightTheBeam', '#Kings'],
    'SAS': ['#PorVida', '#Spurs'],
    'TOR': ['#WeTheNorth', '#Raptors'],
    'UTA': ['#TakeNote', '#Jazz'],
    'WAS': ['#DCAboveAll', '#Wizards'],
}

# =============================================================================
# CONTENT SETTINGS
# =============================================================================
CONTENT_SETTINGS = {
    'morning_posts': 2,           # Number of morning posts
    'evening_posts': 1,           # Number of evening posts
    'min_games_for_player': 5,    # Minimum games for predictions
    'top_picks_count': 5,         # Players in "Top 5" graphic
    'sgdr_alert_threshold': 65,   # SGDR above this triggers fatigue alert
}

# =============================================================================
# TYPOGRAPHY (matplotlib font settings)
# =============================================================================
FONTS = {
    'title': {'family': 'sans-serif', 'weight': 'bold', 'size': 32},
    'subtitle': {'family': 'sans-serif', 'weight': 'normal', 'size': 18},
    'stat_value': {'family': 'sans-serif', 'weight': 'bold', 'size': 48},
    'stat_label': {'family': 'sans-serif', 'weight': 'normal', 'size': 14},
    'body': {'family': 'sans-serif', 'size': 16},
    'small': {'family': 'sans-serif', 'size': 12},
    'watermark': {'family': 'sans-serif', 'weight': 'bold', 'size': 14},
}

# =============================================================================
# CURRENT SEASON
# =============================================================================
CURRENT_SEASON = "2025-26"
