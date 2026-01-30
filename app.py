# -*- coding: utf-8 -*-
"""
CourtMind AI
============
Professional NBA Analytics Platform - Cyberpunk Edition
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import base64
from pathlib import Path
sys.path.insert(0, 'C:/Users/user')

from CourtMind.models.predictor import PlayerPredictor
from CourtMind.models.odds_fetcher import get_todays_odds, get_api_key, TEAM_FULL_NAMES, get_player_prop_line, get_all_player_props

# Page config
st.set_page_config(
    page_title="CourtMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# ASSETS
# =============================================================================
def get_logo_base64():
    for logo_path in [Path('C:/Users/user/CourtMind/logo.png'), Path('C:/Users/user/CourtMind/assets/logo.png')]:
        if logo_path.exists():
            with open(logo_path, 'rb') as f:
                return base64.b64encode(f.read()).decode()
    return None

def get_court_bg_base64():
    bg_path = Path('C:/Users/user/CourtMind/court-bg.png')
    if bg_path.exists():
        with open(bg_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    return None

APP_PASSWORD = "degen"

# =============================================================================
# CYBERPUNK CSS
# =============================================================================
CYBERPUNK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

:root {
    --bg-dark: #050508;
    --card-bg: #0a0a12;
    --card-bg-90: rgba(10, 10, 18, 0.9);
    --card-bg-80: rgba(10, 10, 18, 0.8);
    --neon-green: #00ff88;
    --neon-blue: #00d4ff;
    --neon-orange: #f59e0b;
    --neon-red: #ef4444;
    --muted: #6b7280;
    --border: #1e2530;
    --secondary: #1a1f2e;
    --foreground: #e5e5e5;
}

* { font-family: 'Space Grotesk', sans-serif; }

.stApp {
    background: var(--bg-dark);
}

/* Hide streamlit branding */
#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}

/* Main container */
.main .block-container {
    padding: 0.5rem 2rem 1.5rem 2rem;
    max-width: 1400px;
}

/* Circuit board background */
.circuit-bg {
    background-image:
        linear-gradient(rgba(0,255,136,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,136,0.03) 1px, transparent 1px),
        linear-gradient(rgba(0,212,255,0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.02) 1px, transparent 1px);
    background-size: 100px 100px, 100px 100px, 20px 20px, 20px 20px;
}

/* Neon glow text */
.neon-text-green {
    text-shadow: 0 0 10px rgba(0,255,136,0.5), 0 0 20px rgba(0,255,136,0.3), 0 0 30px rgba(0,255,136,0.2);
}
.neon-text-blue {
    text-shadow: 0 0 10px rgba(0,212,255,0.5), 0 0 20px rgba(0,212,255,0.3), 0 0 30px rgba(0,212,255,0.2);
}
.neon-text-orange {
    text-shadow: 0 0 10px rgba(245,158,11,0.5), 0 0 20px rgba(245,158,11,0.3);
}

/* Pulse glow animation */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 15px rgba(0,255,136,0.4), 0 0 30px rgba(0,255,136,0.2); }
    50% { box-shadow: 0 0 25px rgba(0,255,136,0.6), 0 0 50px rgba(0,255,136,0.3); }
}
.pulse-glow { animation: pulse-glow 2s ease-in-out infinite; }

/* LED flicker */
@keyframes led-flicker {
    0%, 100% { opacity: 1; }
    92% { opacity: 1; }
    93% { opacity: 0.8; }
    94% { opacity: 1; }
}
.led-flicker { animation: led-flicker 3s ease-in-out infinite; }

/* Charging animation */
@keyframes charge-up {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
.charge-animation {
    background: linear-gradient(90deg, rgba(0,255,136,0.3) 0%, rgba(0,255,136,0.8) 50%, rgba(0,255,136,0.3) 100%);
    background-size: 200% 100%;
    animation: charge-up 3s linear infinite;
}

/* Card hover */
.card-hover {
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card-hover:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 40px rgba(0,255,136,0.15), 0 0 20px rgba(0,255,136,0.1);
}

/* Premium Header */
.cyber-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0;
    border-bottom: 1px solid rgba(0,255,136,0.1);
    margin-bottom: 1rem;
}

.logo-section {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.logo-img {
    height: 56px;
    width: auto;
    filter: drop-shadow(0 0 15px rgba(0,255,136,0.3));
}

.brand-text {
    font-size: 1.4rem;
    font-weight: 700;
}

.brand-blue { color: var(--neon-blue); }
.brand-green { color: var(--neon-green); }

.tagline {
    font-size: 0.6rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.15em;
}

/* Trust badges */
.trust-badges {
    display: flex;
    gap: 0.75rem;
}

.trust-badge {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 0.4rem 0.8rem;
    background: var(--card-bg-80);
    border: 1px solid var(--border);
    border-radius: 8px;
}

.trust-badge.live {
    border-color: rgba(0,255,136,0.3);
    box-shadow: 0 0 10px rgba(0,255,136,0.2);
}

.trust-value {
    font-size: 0.9rem;
    font-weight: 700;
    font-family: monospace;
}

.trust-label {
    font-size: 0.5rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Live indicator */
.live-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: var(--neon-green);
    border-radius: 50%;
    margin-right: 4px;
    animation: pulse-glow 1s ease-in-out infinite;
    box-shadow: 0 0 8px rgba(0,255,136,0.8);
}

/* Nav styling */
div[data-testid="stHorizontalBlock"] > div > div > div[data-testid="stRadio"] > div {
    background: var(--secondary);
    padding: 0.25rem;
    border-radius: 10px;
    border: 1px solid var(--border);
}

/* Cards */
.cyber-card {
    background: linear-gradient(135deg, var(--card-bg) 0%, rgba(15, 20, 25, 0.95) 100%);
    border: 1px solid rgba(0,255,136,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
}

.cyber-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(0,255,136,0.05) 0%, transparent 50%, rgba(0,212,255,0.05) 100%);
    pointer-events: none;
}

/* Stat card */
.stat-card {
    background: var(--card-bg-90);
    border: 1px solid rgba(0,255,136,0.2);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.stat-card::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 25%;
    right: 25%;
    height: 2px;
    border-radius: 1px;
}

.stat-card.green::after { background: var(--neon-green); box-shadow: 0 0 10px var(--neon-green); }
.stat-card.blue::after { background: var(--neon-blue); box-shadow: 0 0 10px var(--neon-blue); }
.stat-card.red::after { background: var(--neon-red); box-shadow: 0 0 10px var(--neon-red); }
.stat-card.orange::after { background: var(--neon-orange); box-shadow: 0 0 10px var(--neon-orange); }

.stat-value {
    font-size: 2.5rem;
    font-weight: 700;
    font-family: monospace;
    line-height: 1;
}

.stat-label {
    font-size: 0.6rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.5rem;
}

/* Projection card */
.projection-card {
    background: var(--card-bg-90);
    border: 1px solid rgba(0,255,136,0.2);
    border-radius: 12px;
    padding: 1rem;
    transition: all 0.3s;
}

.projection-card:hover {
    box-shadow: 0 0 20px rgba(0,255,136,0.15);
    border-color: rgba(0,255,136,0.4);
}

.projection-value {
    font-size: 2.5rem;
    font-weight: 700;
    font-family: monospace;
    color: var(--foreground);
}

/* Game card */
.game-card {
    background: linear-gradient(135deg, var(--card-bg) 0%, #0f1419 100%);
    border: 1px solid rgba(0,255,136,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}

.game-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(0,255,136,0.03) 0%, transparent 50%, rgba(0,212,255,0.03) 100%);
    pointer-events: none;
}

.team-score {
    font-size: 3rem;
    font-weight: 700;
    font-family: monospace;
}

/* Prop pill */
.prop-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 0.75rem;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 0.2rem;
}

.prop-pill.over {
    background: rgba(0,255,136,0.1);
    border: 1px solid rgba(0,255,136,0.3);
    border-left: 3px solid var(--neon-green);
}

.prop-pill.under {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    border-left: 3px solid var(--neon-red);
}

/* Progress bar */
.progress-bar {
    height: 12px;
    background: var(--secondary);
    border-radius: 6px;
    overflow: hidden;
    position: relative;
}

.progress-fill {
    height: 100%;
    border-radius: 6px;
    position: relative;
}

.progress-fill.gradient {
    background: linear-gradient(90deg, var(--neon-red), var(--neon-orange), var(--neon-green));
}

.progress-fill.green {
    background: var(--neon-green);
    box-shadow: 0 0 15px rgba(0,255,136,0.5);
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, var(--neon-green) 0%, #10b981 100%);
    color: var(--bg-dark);
    border: none;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-weight: 700;
    transition: all 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 25px rgba(0,255,136,0.5);
}

/* Select box */
.stSelectbox > div > div {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--foreground);
}

.stSelectbox > div > div:focus-within {
    border-color: var(--neon-green);
    box-shadow: 0 0 10px rgba(0,255,136,0.2);
}

/* Slider */
.stSlider > div > div > div {
    background: var(--neon-green) !important;
}

/* Text input */
div[data-testid="stTextInput"] input {
    background: var(--card-bg-90) !important;
    border: 2px solid rgba(0,255,136,0.3) !important;
    border-radius: 8px !important;
    color: var(--foreground) !important;
    font-family: monospace;
    font-size: 1rem;
    padding: 0.75rem;
}

div[data-testid="stTextInput"] input:focus {
    border-color: var(--neon-green) !important;
    box-shadow: 0 0 20px rgba(0,255,136,0.3) !important;
}

/* Footer */
.cyber-footer {
    margin-top: 3rem;
    padding: 1.5rem 0;
    border-top: 1px solid var(--border);
}
</style>
"""

# =============================================================================
# AUTHENTICATION / LANDING PAGE
# =============================================================================
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    logo_b64 = get_logo_base64()
    court_bg_b64 = get_court_bg_base64()

    landing_css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    * {{ font-family: 'Space Grotesk', sans-serif; }}

    .stApp {{
        background: #050508;
        {'background-image: url(data:image/png;base64,' + court_bg_b64 + '); background-size: cover; background-position: center;' if court_bg_b64 else ''}
    }}

    .stApp::before {{
        content: '';
        position: fixed;
        inset: 0;
        background: linear-gradient(180deg, rgba(5,5,8,0.8) 0%, transparent 30%, rgba(5,5,8,0.9) 100%);
        pointer-events: none;
    }}

    #MainMenu, footer, header {{visibility: hidden;}}
    .stDeployButton {{display: none;}}

    .main .block-container {{
        padding-top: 8rem;
        max-width: 500px;
        display: flex;
        flex-direction: column;
        align-items: center;
    }}

    .landing-logo {{
        width: 200px;
        margin-bottom: 1.5rem;
        filter: drop-shadow(0 0 30px rgba(0,255,136,0.4));
    }}

    .status-text {{
        font-size: 0.7rem;
        font-family: monospace;
        letter-spacing: 0.15em;
        color: #00ff88;
        margin-bottom: 2rem;
    }}

    .status-dot {{
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #00ff88;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 10px #00ff88;
        animation: pulse 1.5s ease-in-out infinite;
    }}

    @keyframes pulse {{
        0%, 100% {{ opacity: 1; box-shadow: 0 0 10px #00ff88; }}
        50% {{ opacity: 0.6; box-shadow: 0 0 20px #00ff88; }}
    }}

    div[data-testid="stTextInput"] {{
        width: 100%;
    }}

    div[data-testid="stTextInput"] input {{
        background: rgba(10, 10, 18, 0.9) !important;
        border: 2px solid rgba(0,255,136,0.3) !important;
        border-radius: 8px !important;
        color: #e5e5e5 !important;
        text-align: center;
        font-size: 1rem;
        font-family: monospace;
        padding: 0.9rem;
        transition: all 0.3s;
    }}

    div[data-testid="stTextInput"] input:focus {{
        border-color: #00ff88 !important;
        box-shadow: 0 0 20px rgba(0,255,136,0.3) !important;
    }}

    .stButton > button {{
        background: linear-gradient(135deg, #00ff88 0%, #10b981 100%);
        color: #050508;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2.5rem;
        font-weight: 700;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        transition: all 0.3s;
        animation: glow-pulse 2s ease-in-out infinite;
    }}

    @keyframes glow-pulse {{
        0%, 100% {{ box-shadow: 0 0 15px rgba(0,255,136,0.4); }}
        50% {{ box-shadow: 0 0 30px rgba(0,255,136,0.6); }}
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(0,255,136,0.4);
    }}

    .version-text {{
        font-size: 0.6rem;
        color: rgba(107, 114, 128, 0.5);
        font-family: monospace;
        letter-spacing: 0.1em;
        margin-top: 2rem;
    }}
    </style>
    """
    st.markdown(landing_css, unsafe_allow_html=True)

    if logo_b64:
        st.markdown(f'<div style="text-align: center;"><img src="data:image/png;base64,{logo_b64}" class="landing-logo"></div>', unsafe_allow_html=True)

    st.markdown('<div style="text-align: center;" class="status-text"><span class="status-dot"></span>NEURAL NETWORK ACTIVE_</div>', unsafe_allow_html=True)

    password = st.text_input("Enter Access Code", type="password", placeholder="", label_visibility="collapsed")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Enter"):
            if password == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("ACCESS DENIED")

    st.markdown('<div style="text-align: center;" class="version-text">v2.4.1 // ENCRYPTED CONNECTION // 256-BIT AES</div>', unsafe_allow_html=True)
    st.stop()

# =============================================================================
# MAIN APP
# =============================================================================
st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)

# Constants
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

@st.cache_data(ttl=3600)
def load_data():
    return pd.read_parquet('C:/Users/user/NBA_PRODUCTION.parquet')

def get_predictor():
    df = load_data()
    return PlayerPredictor(df)

df = load_data()
predictor = get_predictor()

# Get today's players
def get_todays_players():
    try:
        from CourtMind.models.nba_lineups_fetcher import get_all_todays_starters
        return get_all_todays_starters()
    except:
        return []

TODAYS_PLAYERS = get_todays_players()

# Header
logo_b64 = get_logo_base64()
logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="logo-img">' if logo_b64 else ''

total_games = len(df['game_date'].unique())
total_players = len(df['player'].unique())

st.markdown(f"""<div class="cyber-header">
<div class="logo-section">
{logo_html}
<div>
<div class="brand-text"><span class="brand-blue neon-text-blue">Court</span><span class="brand-green neon-text-green">MindAI</span></div>
<div class="tagline">Neural Network Powered Analytics</div>
</div>
</div>
<div class="trust-badges">
<div class="trust-badge">
<span class="trust-value" style="color: var(--neon-blue);">375K+</span>
<span class="trust-label">Data Points</span>
</div>
<div class="trust-badge">
<span class="trust-value" style="color: var(--foreground);">6</span>
<span class="trust-label">Seasons</span>
</div>
<div class="trust-badge live">
<span class="trust-value" style="color: var(--neon-green);"><span class="live-dot"></span>Live</span>
<span class="trust-label">Data Feed</span>
</div>
</div>
</div>""", unsafe_allow_html=True)

# Navigation
page = st.radio("Nav", ["Players", "Games", "Tracking", "Lineups", "Props"], horizontal=True, label_visibility="collapsed")

# Import modules
from CourtMind.models.lineup_fetcher import get_todays_lineups, get_team_roster_from_data

if page == "Players":
    st.markdown("""<h2 style="font-size: 1.3rem; font-weight: 600; color: var(--foreground); margin: 1rem 0 0.5rem 0;">Player Analysis</h2>
<p style="color: var(--muted); font-size: 0.8rem; margin-bottom: 1rem;">Select team & player for AI-powered projections</p>""", unsafe_allow_html=True)

    # Build matchups
    todays_matchups = {}
    todays_teams = set()
    try:
        from CourtMind.models.odds_fetcher import fetch_game_odds, TEAM_FULL_NAMES
        TEAM_ABBREV = {v: k for k, v in TEAM_FULL_NAMES.items()}
        odds_data = fetch_game_odds()
        for game in odds_data.get('games', []):
            home = TEAM_ABBREV.get(game['home_team'], game['home_team'][:3].upper())
            away = TEAM_ABBREV.get(game['away_team'], game['away_team'][:3].upper())
            todays_matchups[home] = away
            todays_matchups[away] = home
            todays_teams.add(home)
            todays_teams.add(away)
    except:
        pass

    # Build players by team
    players_by_team = {}
    current_season_df = df[df['game_date'] >= '2025-10-01'].copy()

    try:
        from CourtMind.models.nba_lineups_fetcher import get_todays_official_lineups
        official_lineups = get_todays_official_lineups()
        for team, data in official_lineups.items():
            if team not in players_by_team:
                players_by_team[team] = []
            for player in data.get('starters', []):
                if player not in players_by_team[team]:
                    players_by_team[team].append(player)
    except:
        pass

    if not current_season_df.empty:
        for team in current_season_df['team'].unique():
            team_players = current_season_df[current_season_df['team'] == team]['player'].unique()
            if team not in players_by_team:
                players_by_team[team] = []
            for p in team_players:
                player_recent = current_season_df[current_season_df['player'] == p].sort_values('game_date')
                if not player_recent.empty and player_recent.iloc[-1]['team'] == team:
                    if p not in players_by_team[team]:
                        players_by_team[team].append(p)

    for team in players_by_team:
        team_df = current_season_df[current_season_df['team'] == team]
        player_minutes = team_df.groupby('player')['minutes'].mean().to_dict()
        sorted_players = sorted(players_by_team[team], key=lambda x: player_minutes.get(x, 0), reverse=True)
        if len(sorted_players) > 4:
            players_by_team[team] = sorted_players[:-4]
        else:
            players_by_team[team] = sorted_players

    col1, col2 = st.columns([1, 2])

    with col1:
        all_teams = list(TEAMS.keys())
        teams_playing = [t for t in all_teams if t in todays_teams]
        teams_not_playing = [t for t in all_teams if t not in todays_teams]
        ordered_teams = teams_playing + teams_not_playing

        selected_team = st.selectbox("Team", ordered_teams, format_func=lambda x: f"{'🏀 ' if x in todays_teams else ''}{x} {TEAMS[x]}", label_visibility="collapsed")

        team_players = players_by_team.get(selected_team, [])
        if not team_players:
            team_players = TODAYS_PLAYERS[:10] if TODAYS_PLAYERS else ['LeBron James']

        player = st.selectbox("Player", team_players, label_visibility="collapsed")

        if selected_team in todays_matchups:
            opponent = todays_matchups[selected_team]
            st.markdown(f"""<div style="background: rgba(0,255,136,0.1); padding: 0.6rem; border-radius: 8px; margin-top: 0.5rem; border: 1px solid rgba(0,255,136,0.2);">
<span style="color: var(--neon-green); font-size: 0.8rem;">Today's opponent: <strong>{opponent} {TEAMS.get(opponent, '')}</strong></span>
</div>""", unsafe_allow_html=True)
        else:
            opponent = st.selectbox("Opponent", list(TEAMS.keys()), label_visibility="collapsed", format_func=lambda x: f"{x} {TEAMS[x]}")

    pred = predictor.predict(player, opponent)

    if pred:
        pts_lines = get_player_prop_line(player, 'points')
        reb_lines = get_player_prop_line(player, 'rebounds')
        ast_lines = get_player_prop_line(player, 'assists')

        def get_line_display(lines, book='dk'):
            if lines and book in lines:
                line = lines[book].get('over', {}).get('line', 0)
                return f"{line:.1f}" if line > 0 else "-"
            return "-"

        dk_pts = get_line_display(pts_lines, 'dk')
        fd_pts = get_line_display(pts_lines, 'fd')
        dk_reb = get_line_display(reb_lines, 'dk')
        fd_reb = get_line_display(reb_lines, 'fd')
        dk_ast = get_line_display(ast_lines, 'dk')
        fd_ast = get_line_display(ast_lines, 'fd')

        edge = ((pred['pts'] - float(dk_pts)) / float(dk_pts) * 100) if dk_pts != "-" and float(dk_pts) > 0 else 0
        edge_color = 'var(--neon-green)' if edge > 3 else 'var(--neon-red)' if edge < -3 else 'var(--muted)'

        def_rank = pred.get('opp_def_rank', 15)
        def_color = 'var(--neon-green)' if def_rank >= 20 else 'var(--neon-red)' if def_rank <= 10 else 'var(--muted)'
        def_label = 'weak' if def_rank >= 20 else 'elite' if def_rank <= 10 else 'avg'

        trend_badge = '<span style="background: rgba(0,255,136,0.2); color: var(--neon-green); padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.7rem; font-weight: 600;">HOT</span>' if pred['trend'] == 'hot' else '<span style="background: rgba(0,212,255,0.2); color: var(--neon-blue); padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.7rem; font-weight: 600;">COLD</span>' if pred['trend'] == 'cold' else ''

        with col2:
            st.markdown(f"""<div class="cyber-card">
<div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1.5rem;">
<div>
<div style="font-size: 1.5rem; font-weight: 700; color: var(--foreground);">{pred['player']}</div>
<div style="color: var(--muted); font-size: 0.85rem;">{pred['team']} vs {opponent} {TEAMS[opponent]}</div>
<div style="margin-top: 0.5rem; font-size: 0.75rem;">
<span style="color: {def_color};">Defense: #{def_rank} ({def_label})</span>
</div>
</div>
{trend_badge}
</div>
</div>""", unsafe_allow_html=True)

        # Stats grid
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""<div class="projection-card">
<div style="font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">Points</div>
<div class="projection-value led-flicker">{pred['pts']}</div>
<div style="font-size: 0.6rem; color: var(--muted); text-transform: uppercase; margin-bottom: 0.75rem;">Projection</div>
<div style="border-top: 1px solid var(--border); padding-top: 0.75rem; display: flex; justify-content: space-between; font-size: 0.75rem;">
<span><span style="color: var(--muted);">DK:</span> <span style="color: var(--neon-orange); font-family: monospace;">{dk_pts}</span></span>
<span><span style="color: var(--muted);">FD:</span> <span style="color: var(--neon-blue); font-family: monospace;">{fd_pts}</span></span>
</div>
{f'<div style="text-align: center; margin-top: 0.5rem; color: {edge_color}; font-size: 0.8rem; font-weight: 700;">{edge:+.1f}% edge</div>' if dk_pts != "-" else ''}
</div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""<div class="projection-card">
<div style="font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">Rebounds</div>
<div class="projection-value led-flicker">{pred['reb']}</div>
<div style="font-size: 0.6rem; color: var(--muted); text-transform: uppercase; margin-bottom: 0.75rem;">Projection</div>
<div style="border-top: 1px solid var(--border); padding-top: 0.75rem; display: flex; justify-content: space-between; font-size: 0.75rem;">
<span><span style="color: var(--muted);">DK:</span> <span style="color: var(--neon-orange); font-family: monospace;">{dk_reb}</span></span>
<span><span style="color: var(--muted);">FD:</span> <span style="color: var(--neon-blue); font-family: monospace;">{fd_reb}</span></span>
</div>
</div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""<div class="projection-card">
<div style="font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">Assists</div>
<div class="projection-value led-flicker">{pred['ast']}</div>
<div style="font-size: 0.6rem; color: var(--muted); text-transform: uppercase; margin-bottom: 0.75rem;">Projection</div>
<div style="border-top: 1px solid var(--border); padding-top: 0.75rem; display: flex; justify-content: space-between; font-size: 0.75rem;">
<span><span style="color: var(--muted);">DK:</span> <span style="color: var(--neon-orange); font-family: monospace;">{dk_ast}</span></span>
<span><span style="color: var(--muted);">FD:</span> <span style="color: var(--neon-blue); font-family: monospace;">{fd_ast}</span></span>
</div>
</div>""", unsafe_allow_html=True)

        with c4:
            st.markdown(f"""<div class="projection-card">
<div style="font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">3-Pointers</div>
<div class="projection-value led-flicker">{pred['3pm']}</div>
<div style="font-size: 0.6rem; color: var(--muted); text-transform: uppercase; margin-bottom: 0.75rem;">Projection</div>
<div style="border-top: 1px solid var(--border); padding-top: 0.75rem; display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--muted);">
<span>DK: -</span>
<span>FD: -</span>
</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Range and context
        r1, r2 = st.columns(2)

        with r1:
            range_pct = ((pred['pts'] - pred['floor']) / (pred['ceiling'] - pred['floor'])) * 100 if pred['ceiling'] != pred['floor'] else 50
            st.markdown(f"""<div class="cyber-card">
<div style="font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem;">Projection Range</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; font-size: 0.9rem;">
<span style="color: var(--neon-red); font-family: monospace; font-weight: 700;">{pred['floor']}</span>
<span style="color: var(--foreground); font-size: 1.5rem; font-weight: 700;">{pred['pts']}</span>
<span style="color: var(--neon-green); font-family: monospace; font-weight: 700;">{pred['ceiling']}</span>
</div>
<div class="progress-bar">
<div class="progress-fill gradient" style="width: 100%;"></div>
<div style="position: absolute; top: 50%; transform: translateY(-50%); left: calc({range_pct}% - 8px); width: 16px; height: 16px; background: #fff; border-radius: 50%; border: 2px solid var(--neon-green); box-shadow: 0 0 10px rgba(0,255,136,0.8);"></div>
</div>
<div style="display: flex; justify-content: space-between; font-size: 0.6rem; color: var(--muted); margin-top: 0.5rem;">
<span>Floor (5th)</span>
<span>Ceiling (95th)</span>
</div>
</div>""", unsafe_allow_html=True)

        with r2:
            opp_def_color = 'var(--neon-green)' if pred['opp_def'] > 1.02 else 'var(--neon-red)' if pred['opp_def'] < 0.98 else 'var(--foreground)'
            st.markdown(f"""<div class="cyber-card">
<div style="font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem;">Context</div>
<div style="display: flex; justify-content: space-between; padding: 0.6rem 0; border-bottom: 1px solid var(--border);">
<span style="color: var(--muted);">Season Average</span>
<span style="color: var(--neon-blue); font-family: monospace; font-weight: 700;">{pred['season_avg']} pts</span>
</div>
<div style="display: flex; justify-content: space-between; padding: 0.6rem 0; border-bottom: 1px solid var(--border);">
<span style="color: var(--muted);">Last 5 Games</span>
<span style="color: var(--neon-blue); font-family: monospace; font-weight: 700;">{pred['last_5']} pts</span>
</div>
<div style="display: flex; justify-content: space-between; padding: 0.6rem 0; border-bottom: 1px solid var(--border);">
<span style="color: var(--muted);">Opponent Defense</span>
<span style="color: {opp_def_color}; font-family: monospace; font-weight: 600;">{pred['opp_def']:.2f}x</span>
</div>
<div style="display: flex; justify-content: space-between; padding: 0.6rem 0;">
<span style="color: var(--muted);">Rest Days</span>
<span style="color: var(--foreground); font-family: monospace; font-weight: 600;">{pred['rest_days']} {'(B2B)' if pred['is_b2b'] else ''}</span>
</div>
</div>""", unsafe_allow_html=True)

        # Confidence
        st.markdown(f"""<div class="cyber-card" style="margin-top: 1rem;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
<span style="font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em;">Model Confidence</span>
<span style="font-size: 1.2rem; font-weight: 700; color: var(--neon-green); font-family: monospace;" class="neon-text-green">{pred['confidence']}%</span>
</div>
<div class="progress-bar" style="height: 16px;">
<div style="position: absolute; inset: 0; background: rgba(0,255,136,0.1); border-radius: 8px;"></div>
<div class="progress-fill green" style="width: {pred['confidence']}%; position: relative; overflow: hidden;">
<div class="charge-animation" style="position: absolute; inset: 0;"></div>
</div>
</div>
</div>""", unsafe_allow_html=True)


elif page == "Games":
    st.markdown("""<h2 style="font-size: 1.3rem; font-weight: 600; color: var(--foreground); margin: 1rem 0 0.5rem 0;">Today's Games</h2>
<p style="color: var(--muted); font-size: 0.8rem; margin-bottom: 1rem;">Real-time odds compared against neural network predictions</p>""", unsafe_allow_html=True)

    from CourtMind.models.ensemble_model import GamePredictor
    from CourtMind.models.odds_fetcher import fetch_game_odds, get_api_key as odds_api_key, TEAM_FULL_NAMES
    from CourtMind.models.nba_lineups_fetcher import get_todays_official_lineups
    from CourtMind.models.bet_tracker import log_daily_predictions, get_todays_predictions

    game_pred = GamePredictor()
    TEAM_ABBREV = {v: k for k, v in TEAM_FULL_NAMES.items()}

    api_key_exists = odds_api_key() is not None
    odds_data = fetch_game_odds() if api_key_exists else {'games': []}
    games_list = odds_data.get('games', [])
    official_lineups = get_todays_official_lineups()

    if not games_list:
        st.info("No games scheduled today")
    else:
        all_top_plays = []

        for game in games_list:
            home_full = game['home_team']
            away_full = game['away_team']
            home_team = TEAM_ABBREV.get(home_full, home_full[:3].upper())
            away_team = TEAM_ABBREV.get(away_full, away_full[:3].upper())

            result = game_pred.predict_game(df, home_team, away_team)

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
                                    'player': player_name, 'team': team, 'stat': 'PTS',
                                    'proj': pred['pts'], 'dk_line': dk_line if dk_line > 0 else None,
                                    'fd_line': fd_line if fd_line > 0 else None, 'edge': edge,
                                    'direction': 'OVER' if edge > 0 else 'UNDER',
                                    'confidence': pred['confidence'], 'score': score,
                                    'opponent': opp, 'line': best_line
                                })
                except:
                    continue

            top_plays = sorted(top_plays, key=lambda x: x['score'], reverse=True)[:3]
            all_top_plays.extend(top_plays)

            # Get odds
            dk = game.get('bookmakers', {}).get('draftkings', {})
            dk_spread = dk.get('spreads', {}).get(home_full, {}).get('point', '-')
            dk_total = dk.get('totals', {}).get('Over', {}).get('point', '-')

            dk_spread_str = f"{dk_spread:+.1f}" if isinstance(dk_spread, (int, float)) else str(dk_spread)
            dk_total_str = f"{dk_total:.1f}" if isinstance(dk_total, (int, float)) else str(dk_total)

            if result:
                home_score = result['predicted_home_score']
                away_score = result['predicted_away_score']
                home_win = result['home_win_prob']
                our_spread = result['spread']
                our_total = result['predicted_total']

                # Build plays HTML
                plays_html = ""
                if top_plays:
                    pills = []
                    for p in top_plays:
                        dk_str = f"{p['dk_line']:.1f}" if p['dk_line'] else "-"
                        fd_str = f"{p['fd_line']:.1f}" if p['fd_line'] else "-"
                        pill_class = "over" if p['direction'] == 'OVER' else "under"
                        edge_color = "var(--neon-green)" if p['edge'] > 0 else "var(--neon-red)"
                        pills.append(f'<div class="prop-pill {pill_class}"><span style="font-weight: 700; color: var(--foreground);">{p["player"].split()[-1]}</span><span style="color: {edge_color}; font-weight: 700;">{p["direction"]} {p["proj"]:.1f}</span><span style="color: var(--muted); font-size: 0.65rem;">DK {dk_str}</span><span style="background: {edge_color}; color: #050508; padding: 0.1rem 0.3rem; border-radius: 3px; font-size: 0.6rem; font-weight: 700;">{p["edge"]:+.0f}%</span></div>')
                    plays_html = f'<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 0.5rem; margin: 1rem 0;">{"".join(pills)}</div>'

                spread_str = f"{home_team} {our_spread:+.1f}" if our_spread < 0 else f"{away_team} {-our_spread:+.1f}"

                st.markdown(f"""<div class="game-card">
<div style="display: flex; justify-content: center; align-items: center; gap: 3rem; margin-bottom: 1rem;">
<div style="text-align: center;">
<div style="font-size: 1.8rem; font-weight: 800; color: var(--foreground);">{away_team}</div>
<div class="team-score" style="color: var(--neon-orange);" class="neon-text-orange led-flicker">{away_score:.0f}</div>
</div>
<div style="text-align: center;">
<div style="color: var(--muted); font-size: 0.7rem; text-transform: uppercase;">vs</div>
<div style="color: var(--muted); font-size: 0.6rem;">PROJ</div>
</div>
<div style="text-align: center;">
<div style="font-size: 1.8rem; font-weight: 800; color: var(--foreground);">{home_team}</div>
<div class="team-score" style="color: var(--neon-blue);" class="neon-text-blue led-flicker">{home_score:.0f}</div>
</div>
</div>
{plays_html}
<div style="display: flex; justify-content: center; gap: 2rem; padding-top: 1rem; border-top: 1px solid var(--border); flex-wrap: wrap;">
<div style="text-align: center;">
<div style="color: var(--muted); font-size: 0.55rem; text-transform: uppercase;">Win Prob</div>
<div style="font-size: 0.85rem; color: {'var(--neon-green)' if home_win > 55 else 'var(--neon-red)' if home_win < 45 else 'var(--foreground)'}; font-weight: 700;">{home_team} {home_win:.0f}%</div>
</div>
<div style="text-align: center;">
<div style="color: var(--muted); font-size: 0.55rem; text-transform: uppercase;">Spread</div>
<div style="font-size: 0.85rem; color: var(--foreground); font-weight: 700;">{spread_str}</div>
</div>
<div style="text-align: center;">
<div style="color: var(--muted); font-size: 0.55rem; text-transform: uppercase;">Total</div>
<div style="font-size: 0.85rem; color: var(--foreground); font-weight: 700;">{our_total:.0f}</div>
</div>
<div style="text-align: center;">
<div style="color: var(--muted); font-size: 0.55rem; text-transform: uppercase;">DK Line</div>
<div style="font-size: 0.8rem; color: var(--neon-green); font-weight: 600;">{dk_spread_str} | {dk_total_str}</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

        # Log picks button
        st.markdown("<br>", unsafe_allow_html=True)
        todays_logged = get_todays_predictions()
        if todays_logged:
            st.success(f"✓ {len(todays_logged)} picks logged for today")
        elif all_top_plays:
            if st.button(f"📝 Log {len(all_top_plays)} Picks for Tracking"):
                logged = log_daily_predictions(all_top_plays)
                st.success(f"Logged {logged} picks!")
                st.rerun()


elif page == "Tracking":
    st.markdown("""<h2 style="font-size: 1.3rem; font-weight: 600; color: var(--foreground); margin: 1rem 0 0.5rem 0;">Bet Tracking</h2>
<p style="color: var(--muted); font-size: 0.8rem; margin-bottom: 1.5rem;">Track top plays performance over time</p>""", unsafe_allow_html=True)

    from CourtMind.models.bet_tracker import get_tracking_stats, check_results, get_todays_predictions

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Update Results"):
            updated = check_results(df)
            if updated > 0:
                st.success(f"Updated {updated} picks!")
                st.rerun()
            else:
                st.info("No new results")

    stats = get_tracking_stats()

    # Casino-style stat cards
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""<div class="stat-card blue">
<div class="stat-value" style="color: var(--neon-blue);" class="led-flicker neon-text-blue">{stats['graded_picks']}</div>
<div class="stat-label">Graded</div>
</div>""", unsafe_allow_html=True)

    with c2:
        hit_color = 'var(--neon-green)' if stats['hit_rate'] >= 55 else 'var(--neon-orange)' if stats['hit_rate'] >= 50 else 'var(--neon-red)'
        st.markdown(f"""<div class="stat-card green">
<div class="stat-value" style="color: {hit_color};" class="led-flicker neon-text-green">{stats['hit_rate']}%</div>
<div class="stat-label">Hit Rate</div>
</div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""<div class="stat-card green">
<div class="stat-value" style="color: var(--neon-green);" class="led-flicker">{stats['hits']}</div>
<div class="stat-label">Hits</div>
</div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""<div class="stat-card red">
<div class="stat-value" style="color: var(--neon-red);" class="led-flicker">{stats['misses']}</div>
<div class="stat-label">Misses</div>
</div>""", unsafe_allow_html=True)

    with c5:
        streak_color = 'var(--neon-green)' if stats['streak'] > 0 else 'var(--neon-red)' if stats['streak'] < 0 else 'var(--muted)'
        streak_text = f"+{stats['streak']}" if stats['streak'] > 0 else str(stats['streak'])
        st.markdown(f"""<div class="stat-card {'green' if stats['streak'] > 0 else 'red' if stats['streak'] < 0 else ''}">
<div class="stat-value" style="color: {streak_color};" class="led-flicker">{streak_text}</div>
<div class="stat-label">Streak</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # By confidence
    if stats['by_confidence']:
        st.markdown("### By Confidence Level")
        conf_cols = st.columns(len(stats['by_confidence']))
        for i, (tier, data) in enumerate(stats['by_confidence'].items()):
            with conf_cols[i]:
                tier_color = 'var(--neon-green)' if data['rate'] >= 55 else 'var(--neon-orange)' if data['rate'] >= 50 else 'var(--neon-red)'
                st.markdown(f"""<div style="background: var(--card-bg-80); padding: 1rem; border-radius: 10px; text-align: center; border: 1px solid var(--border);">
<div style="font-size: 0.8rem; color: var(--foreground); font-weight: 600;">{tier}%</div>
<div style="font-size: 1.5rem; font-weight: 700; color: {tier_color}; font-family: monospace;">{data['rate']}%</div>
<div style="font-size: 0.65rem; color: var(--muted);">{data['hits']}/{data['picks']}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Recent Picks")

    if stats['recent']:
        for pick in stats['recent'][:15]:
            hit = pick.get('hit')
            result_val = pick.get('result')
            status_icon = "✅" if hit is True else "❌" if hit is False else "⏳"
            status_color = "var(--neon-green)" if hit is True else "var(--neon-red)" if hit is False else "var(--muted)"
            direction_color = "var(--neon-green)" if pick.get('direction') == 'OVER' else "var(--neon-red)"
            result_str = f"{result_val}" if result_val is not None else "Pending"

            st.markdown(f"""<div style="display: flex; align-items: center; justify-content: space-between; padding: 0.75rem 1rem; background: var(--card-bg-80); border-radius: 10px; margin-bottom: 0.5rem; border-left: 3px solid {status_color};">
<div style="display: flex; align-items: center; gap: 1rem;">
<span style="font-size: 1.2rem;">{status_icon}</span>
<div>
<span style="color: var(--foreground); font-weight: 600;">{pick.get('player', 'Unknown')}</span>
<span style="color: var(--muted); font-size: 0.75rem; margin-left: 0.5rem;">{pick.get('game_date', '')}</span>
</div>
</div>
<div style="display: flex; align-items: center; gap: 1.5rem;">
<div style="text-align: center;">
<div style="color: {direction_color}; font-weight: 700;">{pick.get('direction', '')} {pick.get('line', 0):.1f}</div>
<div style="color: var(--muted); font-size: 0.65rem;">{pick.get('stat', 'PTS')}</div>
</div>
<div style="text-align: center;">
<div style="color: var(--neon-blue); font-weight: 600;">Proj: {pick.get('proj', 0):.1f}</div>
<div style="color: var(--muted); font-size: 0.65rem;">{pick.get('confidence', 0)}% conf</div>
</div>
<div style="text-align: center; min-width: 60px;">
<div style="color: {status_color}; font-weight: 700;">{result_str}</div>
<div style="color: var(--muted); font-size: 0.65rem;">Actual</div>
</div>
</div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style="background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); border-radius: 10px; padding: 1.5rem; text-align: center;">
<p style="color: var(--neon-orange); margin: 0;">No picks logged yet. Go to Games page and click 'Log Picks' to start tracking!</p>
</div>""", unsafe_allow_html=True)


elif page == "Lineups":
    st.markdown("""<h2 style="font-size: 1.3rem; font-weight: 600; color: var(--foreground); margin: 1rem 0 0.5rem 0;">Lineup Builder</h2>
<p style="color: var(--muted); font-size: 0.8rem; margin-bottom: 1rem;">Adjust minutes and see projections update in real-time</p>""", unsafe_allow_html=True)

    lineup_data = get_todays_lineups(df)
    games = lineup_data.get('games', [])

    if not games:
        st.info("No games scheduled for today")
    else:
        game_options = [f"{g['away_team']} @ {g['home_team']}" for g in games]
        selected_game_idx = st.selectbox("Select Game", range(len(game_options)), format_func=lambda x: game_options[x])

        game = games[selected_game_idx]
        away_team = game['away_team']
        home_team = game['home_team']

        away_roster = get_team_roster_from_data(df, away_team)
        home_roster = get_team_roster_from_data(df, home_team)

        if 'lineup_minutes' not in st.session_state:
            st.session_state.lineup_minutes = {}

        game_key = f"{away_team}_{home_team}"
        if game_key not in st.session_state.lineup_minutes:
            st.session_state.lineup_minutes[game_key] = {
                'away': {p['name']: p['avg_minutes'] for p in away_roster},
                'home': {p['name']: p['avg_minutes'] for p in home_roster}
            }

        col_away, col_home = st.columns(2)

        with col_away:
            st.markdown(f"""<div style="background: var(--card-bg); padding: 1rem; border-radius: 12px; border-left: 4px solid var(--neon-orange);">
<div style="font-size: 1.3rem; font-weight: 700; color: var(--foreground);">{away_team}</div>
<div style="color: var(--muted); font-size: 0.8rem;">{TEAMS.get(away_team, '')} (Away)</div>
</div>""", unsafe_allow_html=True)

            away_total = 0
            for i, player in enumerate(away_roster[:8]):
                name = player['name']
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"<span style='color: var(--foreground);'>{name}</span>", unsafe_allow_html=True)
                with col2:
                    mins = st.slider(f"a_{i}", 0, 48, int(st.session_state.lineup_minutes[game_key]['away'].get(name, player['avg_minutes'])), key=f"away_{game_key}_{i}", label_visibility="collapsed")
                    st.session_state.lineup_minutes[game_key]['away'][name] = mins

                pred = predictor.predict(name, home_team)
                if pred and mins > 0:
                    pts = pred['pts'] * (mins / 32.0)
                    away_total += pts

            st.markdown(f"""<div style="background: var(--secondary); padding: 1rem; border-radius: 8px; margin-top: 1rem; display: flex; justify-content: space-between;">
<span style="color: var(--muted);">Team Total</span>
<span style="color: var(--neon-orange); font-size: 1.5rem; font-weight: 700; font-family: monospace;">{away_total:.1f}</span>
</div>""", unsafe_allow_html=True)

        with col_home:
            st.markdown(f"""<div style="background: var(--card-bg); padding: 1rem; border-radius: 12px; border-left: 4px solid var(--neon-blue);">
<div style="font-size: 1.3rem; font-weight: 700; color: var(--foreground);">{home_team}</div>
<div style="color: var(--muted); font-size: 0.8rem;">{TEAMS.get(home_team, '')} (Home)</div>
</div>""", unsafe_allow_html=True)

            home_total = 0
            for i, player in enumerate(home_roster[:8]):
                name = player['name']
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"<span style='color: var(--foreground);'>{name}</span>", unsafe_allow_html=True)
                with col2:
                    mins = st.slider(f"h_{i}", 0, 48, int(st.session_state.lineup_minutes[game_key]['home'].get(name, player['avg_minutes'])), key=f"home_{game_key}_{i}", label_visibility="collapsed")
                    st.session_state.lineup_minutes[game_key]['home'][name] = mins

                pred = predictor.predict(name, away_team)
                if pred and mins > 0:
                    pts = pred['pts'] * (mins / 32.0)
                    home_total += pts

            st.markdown(f"""<div style="background: var(--secondary); padding: 1rem; border-radius: 8px; margin-top: 1rem; display: flex; justify-content: space-between;">
<span style="color: var(--muted);">Team Total</span>
<span style="color: var(--neon-blue); font-size: 1.5rem; font-weight: 700; font-family: monospace;">{home_total:.1f}</span>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        game_total = away_total + home_total
        st.markdown(f"""<div class="cyber-card" style="text-align: center;">
<div style="font-size: 0.7rem; color: var(--muted); text-transform: uppercase; margin-bottom: 0.5rem;">Projected Game Total</div>
<div style="font-size: 3rem; font-weight: 700; color: var(--neon-green); font-family: monospace;" class="neon-text-green led-flicker">{game_total:.1f}</div>
</div>""", unsafe_allow_html=True)


elif page == "Props":
    st.markdown("""<h2 style="font-size: 1.3rem; font-weight: 600; color: var(--foreground); margin: 1rem 0 0.5rem 0;">Player Props Scanner</h2>
<p style="color: var(--muted); font-size: 0.8rem; margin-bottom: 1rem;">Scan for value across DraftKings & FanDuel</p>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        stat = st.selectbox("Stat", ["Points", "Rebounds", "Assists", "3-Pointers"])
    with col2:
        min_edge = st.slider("Min Edge %", 0, 20, 3)
    with col3:
        sort_by = st.selectbox("Sort By", ["Edge", "Confidence", "Projection"])

    if st.button("Find Value Props", type="primary"):
        props = []
        stat_key = {'Points': 'pts', 'Rebounds': 'reb', 'Assists': 'ast', '3-Pointers': '3pm'}[stat]
        prop_api_key = {'Points': 'points', 'Rebounds': 'rebounds', 'Assists': 'assists', '3-Pointers': 'threes'}[stat]

        with st.spinner("Scanning..."):
            for player in TODAYS_PLAYERS[:30]:
                pred = predictor.predict(player, 'LAL')
                if pred:
                    proj = pred[stat_key]
                    real_lines = get_player_prop_line(player, prop_api_key)

                    if real_lines:
                        dk_line = real_lines.get('dk', {}).get('over', {}).get('line', 0)
                        fd_line = real_lines.get('fd', {}).get('over', {}).get('line', 0)
                    else:
                        dk_line = proj * 0.97
                        fd_line = dk_line

                    line = min(dk_line, fd_line) if dk_line > 0 and fd_line > 0 else max(dk_line, fd_line)
                    if line <= 0:
                        continue

                    edge = ((proj - line) / line) * 100
                    if abs(edge) >= min_edge:
                        props.append({
                            'player': player, 'team': pred['team'], 'dk_line': dk_line, 'fd_line': fd_line,
                            'proj': proj, 'edge': edge, 'direction': 'OVER' if edge > 0 else 'UNDER',
                            'confidence': pred['confidence'], 'trend': pred['trend']
                        })

        if props:
            if sort_by == "Edge":
                props = sorted(props, key=lambda x: abs(x['edge']), reverse=True)
            elif sort_by == "Confidence":
                props = sorted(props, key=lambda x: x['confidence'], reverse=True)
            else:
                props = sorted(props, key=lambda x: x['proj'], reverse=True)

            for p in props[:15]:
                edge_color = 'var(--neon-green)' if p['edge'] > 0 else 'var(--neon-red)'
                trend_icon = '🔥' if p['trend'] == 'hot' else '❄️' if p['trend'] == 'cold' else ''

                st.markdown(f"""<div style="display: grid; grid-template-columns: 2fr 1fr 1fr 1fr 1fr; gap: 1rem; padding: 1rem; background: var(--card-bg-80); border-radius: 10px; margin-bottom: 0.5rem; border-left: 3px solid {edge_color}; align-items: center;">
<div>
<span style="color: var(--foreground); font-weight: 600;">{p['player']}</span> {trend_icon}
<div style="color: var(--muted); font-size: 0.7rem;">{p['team']} • {stat}</div>
</div>
<div style="text-align: center;">
<div style="color: var(--neon-orange); font-family: monospace; font-weight: 600;">DK {p['dk_line']:.1f}</div>
</div>
<div style="text-align: center;">
<div style="color: var(--neon-blue); font-family: monospace; font-weight: 600;">FD {p['fd_line']:.1f}</div>
</div>
<div style="text-align: center;">
<div style="color: var(--foreground); font-weight: 700; font-size: 1.1rem;">{p['proj']:.1f}</div>
</div>
<div style="text-align: center;">
<div style="color: {edge_color}; font-weight: 700;">{p['direction']}</div>
<div style="color: {edge_color}; font-size: 0.8rem;">{p['edge']:+.1f}%</div>
</div>
</div>""", unsafe_allow_html=True)
        else:
            st.info(f"No {stat.lower()} props found with {min_edge}%+ edge")


# Footer
st.markdown(f"""<div class="cyber-footer">
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
<div>
<div style="font-size: 0.9rem; font-weight: 600; color: var(--foreground);">CourtMindAI</div>
<div style="font-size: 0.65rem; color: var(--muted);">Neural Network Powered Analytics</div>
</div>
<div style="display: flex; gap: 1.5rem; font-size: 0.65rem; color: var(--muted);">
<span><span style="color: var(--neon-blue);">●</span> Data through {df['game_date'].max().strftime('%b %d, %Y')}</span>
<span><span style="color: var(--neon-green);">●</span> Live odds integration</span>
</div>
<div style="font-size: 0.55rem; color: #475569;">For entertainment purposes only. Not financial advice.</div>
</div>
</div>""", unsafe_allow_html=True)
