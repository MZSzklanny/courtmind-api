# -*- coding: utf-8 -*-
"""
CourtMind AI - Production Dashboard
===================================
AI-powered NBA analytics platform for sports bettors.
"Built by degens. Powered by AI."
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import sys
sys.path.insert(0, 'C:/Users/user')

# Page config
st.set_page_config(
    page_title="CourtMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #12121f 0%, #1a1a2e 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #00ff88;
    }
    .stat-card {
        background: #1a1a2e;
        padding: 15px;
        border-radius: 8px;
        border-left: 3px solid #00ff88;
        margin: 5px 0;
    }
    .fatigue-high { color: #ff4444; }
    .fatigue-med { color: #ffaa00; }
    .fatigue-low { color: #00ff88; }
</style>
""", unsafe_allow_html=True)

# Constants
TOP_50_PLAYERS = [
    'Shai Gilgeous-Alexander', 'Luka Dončić', 'Giannis Antetokounmpo',
    'Anthony Edwards', 'Nikola Jokić', 'Tyrese Maxey', 'Jalen Brunson',
    'Kevin Durant', 'Cade Cunningham', 'Jayson Tatum', 'Devin Booker',
    'Joel Embiid', 'Stephen Curry', 'Donovan Mitchell', 'Paolo Banchero',
    'Kawhi Leonard', 'Jaylen Brown', 'Kyrie Irving', 'LeBron James',
    'Damian Lillard', 'Victor Wembanyama', 'Jamal Murray', 'Trae Young',
    'James Harden', 'Tyler Herro', 'Zion Williamson', 'Ja Morant',
    'Zach LaVine', 'Franz Wagner', 'Anthony Davis', 'Lauri Markkanen',
    'Karl-Anthony Towns', 'LaMelo Ball', 'Austin Reaves', 'Jalen Green',
    'Norman Powell', 'Trey Murphy III', 'Brandon Ingram', 'DeMar DeRozan',
    'Brandon Miller', 'Jalen Johnson', 'Pascal Siakam', 'Jaren Jackson Jr.',
    "De'Aaron Fox", 'Deni Avdija', 'RJ Barrett', 'CJ McCollum',
    'Coby White', 'Darius Garland', 'Jalen Williams'
]

TEAMS = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
         'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
         'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']

TEAM_NAMES = {
    'ATL': 'Hawks', 'BOS': 'Celtics', 'BKN': 'Nets', 'CHA': 'Hornets', 'CHI': 'Bulls',
    'CLE': 'Cavaliers', 'DAL': 'Mavericks', 'DEN': 'Nuggets', 'DET': 'Pistons', 'GSW': 'Warriors',
    'HOU': 'Rockets', 'IND': 'Pacers', 'LAC': 'Clippers', 'LAL': 'Lakers', 'MEM': 'Grizzlies',
    'MIA': 'Heat', 'MIL': 'Bucks', 'MIN': 'Timberwolves', 'NOP': 'Pelicans', 'NYK': 'Knicks',
    'OKC': 'Thunder', 'ORL': 'Magic', 'PHI': '76ers', 'PHX': 'Suns', 'POR': 'Trail Blazers',
    'SAC': 'Kings', 'SAS': 'Spurs', 'TOR': 'Raptors', 'UTA': 'Jazz', 'WAS': 'Wizards'
}

FEATURE_IMPORTANCE = {
    'pts_avg_10': 0.6906,
    'pts_avg_5': 0.0710,
    'minutes_avg_5': 0.0303,
    'rest_days': 0.0249,
    'ts_pct_5': 0.0233,
    'pts_trend': 0.0218,
    'ast_avg_5': 0.0217,
    'games_in_3days': 0.0191,
    'pts_std_10': 0.0191,
    'is_b2b': 0.0187,
    'games_in_week': 0.0186,
    'trb_avg_5': 0.0185,
    'travel_miles_week': 0.0117,
    'travel_fatigue': 0.0107,
}


@st.cache_data(ttl=3600)
def load_data():
    """Load production data."""
    return pd.read_parquet('C:/Users/user/NBA_PRODUCTION.parquet')


@st.cache_resource
def load_ensemble():
    """Load trained ensemble model."""
    try:
        with open('C:/Users/user/CourtMind/models/ensemble_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None


def get_player_features(df, player, opponent):
    """Get full features for a player prediction."""
    try:
        from CourtMind.models.feature_engineering import engineer_player_features
        return engineer_player_features(df, player, opponent, datetime.now().strftime('%Y-%m-%d'))
    except Exception as e:
        return None


def get_ensemble_prediction(df, player, opponent, ensemble_data):
    """Get prediction from ensemble model."""
    if ensemble_data is None:
        return None

    try:
        features = get_player_features(df, player, opponent)
        if features is None:
            return None

        xgb_model = ensemble_data.get('xgb')
        ridge_model = ensemble_data.get('ridge')

        if xgb_model is None:
            return None

        xgb_pred = xgb_model.predict(features)
        ridge_pred = ridge_model.predict(features) if ridge_model else xgb_pred

        ensemble_pred = 0.6 * xgb_pred + 0.4 * ridge_pred

        return {
            'ensemble': ensemble_pred,
            'xgb': xgb_pred,
            'ridge': ridge_pred,
            'features': features
        }
    except:
        return None


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.markdown("# 🧠 CourtMind AI")
st.sidebar.markdown("*Built by degens. Powered by AI.*")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏀 Player Projections",
    "📊 Prop Finder",
    "🎯 Game Predictions",
    "📈 Model Insights",
    "🏆 Standings"
])

# =============================================================================
# MAIN CONTENT
# =============================================================================

df = load_data()
ensemble_data = load_ensemble()

if page == "🏀 Player Projections":
    st.markdown("## 🏀 Player Projections")
    st.markdown("*AI-powered predictions with travel, fatigue, and matchup context*")

    col1, col2 = st.columns([2, 1])
    with col1:
        player = st.selectbox("Select Player", TOP_50_PLAYERS)
    with col2:
        opponent = st.selectbox("Opponent", TEAMS)

    if player and opponent:
        features = get_player_features(df, player, opponent)
        pred = get_ensemble_prediction(df, player, opponent, ensemble_data)

        if features:
            st.markdown("---")

            # Main projection
            col1, col2, col3, col4 = st.columns(4)

            pts_proj = pred['ensemble'] if pred else features['pts_avg_5']

            with col1:
                st.metric("📊 PTS Projection", f"{pts_proj:.1f}")
            with col2:
                st.metric("📈 Last 5 Avg", f"{features['pts_avg_5']:.1f}")
            with col3:
                st.metric("🎯 Ceiling (90th)", f"{features['ceiling_90']:.1f}")
            with col4:
                st.metric("📉 Floor (10th)", f"{features['floor_10']:.1f}")

            st.markdown("---")

            # Context factors
            st.markdown("### 🔍 Context Factors")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Schedule & Travel**")
                fatigue_class = "fatigue-high" if features['fatigue_index'] > 50 else "fatigue-med" if features['fatigue_index'] > 25 else "fatigue-low"
                st.markdown(f"- Games in Week: **{features['games_in_week']}**")
                st.markdown(f"- Back-to-Back: **{'Yes' if features['is_b2b'] else 'No'}**")
                st.markdown(f"- Travel Miles: **{features['travel_miles_week']:.0f} mi**")
                st.markdown(f"- <span class='{fatigue_class}'>Fatigue Index: **{features['fatigue_index']:.0f}**</span>", unsafe_allow_html=True)

            with col2:
                st.markdown("**Performance Metrics**")
                spm_emoji = "🔥" if features['spm'] > 5 else "❄️" if features['spm'] < -5 else "➡️"
                trend_emoji = "📈" if features['pts_trend'] > 0.05 else "📉" if features['pts_trend'] < -0.05 else "➡️"
                st.markdown(f"- SPM (Q4 Clutch): {spm_emoji} **{features['spm']:.1f}**")
                st.markdown(f"- Trend: {trend_emoji} **{features['pts_trend']*100:+.1f}%**")
                st.markdown(f"- TS%: **{features['ts_pct_5']:.1%}**")
                st.markdown(f"- Rest Days: **{features['rest_days']}**")

            with col3:
                st.markdown("**Opponent Matchup**")
                def_emoji = "🟢" if features['opp_def_rating'] > 1.05 else "🔴" if features['opp_def_rating'] < 0.95 else "🟡"
                st.markdown(f"- Defense Rating: {def_emoji} **{features['opp_def_rating']:.2f}x**")
                st.markdown(f"- Pace: **{features['opp_pace']:.1f}**")
                st.markdown(f"- vs Opp Factor: **{features['vs_opp_factor']:.2f}x**")

            # Model breakdown
            if pred:
                st.markdown("---")
                st.markdown("### 🤖 Model Predictions")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("XGBoost (45%)", f"{pred['xgb']:.1f}")
                with col2:
                    st.metric("Ridge (20%)", f"{pred['ridge']:.1f}")
                with col3:
                    st.metric("Ensemble", f"{pred['ensemble']:.1f}")


elif page == "📊 Prop Finder":
    st.markdown("## 📊 Prop Finder")
    st.markdown("*Find value in player props*")

    col1, col2 = st.columns(2)
    with col1:
        stat_type = st.selectbox("Stat", ['pts', 'trb', 'ast', 'fg3m'])
    with col2:
        min_edge = st.slider("Min Edge %", 0, 20, 5)

    if st.button("🔍 Find Props", type="primary"):
        props = []

        for player in TOP_50_PLAYERS[:30]:  # Top 30 for speed
            features = get_player_features(df, player, 'LAL')  # Generic opponent
            if features is None:
                continue

            stat_key = f'{stat_type}_avg_5' if stat_type != 'pts' else 'pts_avg_5'
            if stat_key not in features:
                continue

            avg = features.get(stat_key, features.get('pts_avg_5', 20))
            line = avg * 0.98

            pred = get_ensemble_prediction(df, player, 'LAL', ensemble_data)
            proj = pred['ensemble'] if pred and stat_type == 'pts' else avg

            edge = (proj - line) / line * 100

            if abs(edge) >= min_edge:
                props.append({
                    'Player': player,
                    'Stat': stat_type.upper(),
                    'Line': f"{line:.1f}",
                    'Projection': f"{proj:.1f}",
                    'Edge': f"{edge:+.1f}%",
                    'Direction': '🟢 OVER' if edge > 0 else '🔴 UNDER',
                    'Fatigue': f"{features.get('fatigue_index', 50):.0f}",
                    'Travel': f"{features.get('travel_miles_week', 0):.0f} mi"
                })

        if props:
            st.dataframe(pd.DataFrame(props), use_container_width=True, hide_index=True)
        else:
            st.info("No props found matching criteria")


elif page == "🎯 Game Predictions":
    st.markdown("## 🎯 Game Predictions")
    st.markdown("*Win probability using standings + stats*")

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", TEAMS, index=TEAMS.index('OKC'))
    with col2:
        away_team = st.selectbox("Away Team", [t for t in TEAMS if t != home_team], index=0)

    if home_team and away_team:
        from CourtMind.models.ensemble_model import GamePredictor
        game_pred = GamePredictor()
        result = game_pred.predict_game(df, home_team, away_team)

        if result:
            st.markdown("---")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"### {TEAM_NAMES[away_team]}")
                st.markdown(f"**Record: {result['away_record']}**")
                st.metric("Win Prob", f"{result['away_win_prob']:.0f}%")
                st.metric("Projected", f"{result['predicted_away_score']:.0f}")

            with col2:
                st.markdown("### 🏀 vs 🏀")
                st.metric("Total", f"{result['predicted_total']:.0f}")
                spread_team = home_team if result['spread'] < 0 else away_team
                st.metric("Spread", f"{spread_team} {abs(result['spread']):.1f}")

            with col3:
                st.markdown(f"### {TEAM_NAMES[home_team]}")
                st.markdown(f"**Record: {result['home_record']}**")
                st.metric("Win Prob", f"{result['home_win_prob']:.0f}%")
                st.metric("Projected", f"{result['predicted_home_score']:.0f}")


elif page == "📈 Model Insights":
    st.markdown("## 📈 Model Insights")
    st.markdown("*How the ensemble makes predictions*")

    st.markdown("### Feature Importance (XGBoost)")

    # Create bar chart
    importance_df = pd.DataFrame([
        {'Feature': k.replace('_', ' ').title(), 'Importance': v}
        for k, v in FEATURE_IMPORTANCE.items()
    ]).sort_values('Importance', ascending=True)

    st.bar_chart(importance_df.set_index('Feature'), horizontal=True)

    st.markdown("---")
    st.markdown("### Model Architecture")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**LSTM (35%)**")
        st.markdown("- Captures trends")
        st.markdown("- Hot/cold streaks")
        st.markdown("- Bidirectional + Attention")
    with col2:
        st.markdown("**XGBoost (45%)**")
        st.markdown("- Context features")
        st.markdown("- Travel, fatigue")
        st.markdown("- MAE: 4.82 pts")
    with col3:
        st.markdown("**Ridge (20%)**")
        st.markdown("- Baseline sanity")
        st.markdown("- Regularized")
        st.markdown("- MAE: 4.80 pts")

    st.markdown("---")
    st.markdown("### Features Used")
    st.markdown("""
    | Category | Features |
    |----------|----------|
    | **Performance** | pts_avg_5, pts_avg_10, pts_trend, ts_pct |
    | **Schedule** | games_in_week, games_in_3days, rest_days, is_b2b |
    | **Travel** | travel_miles_week, travel_fatigue |
    | **Opponent** | opp_def_rating, opp_pace, vs_opp_factor |
    | **Fatigue** | fatigue_index, SPM (Q4 clutch) |
    """)


elif page == "🏆 Standings":
    st.markdown("## 🏆 Current Standings")

    from CourtMind.models.ensemble_model import GamePredictor
    gp = GamePredictor()
    gp.calculate_team_records(df)

    if gp.team_records:
        records = []
        for team, rec in gp.team_records.items():
            records.append({
                'Team': f"{team} {TEAM_NAMES.get(team, '')}",
                'W': rec['wins'],
                'L': rec['losses'],
                'Win%': f"{rec['win_pct']:.3f}",
                'GB': 0  # Calculate later
            })

        records_df = pd.DataFrame(records).sort_values('Win%', ascending=False).reset_index(drop=True)
        records_df.index = records_df.index + 1

        # Calculate games back
        top_pct = float(records_df.iloc[0]['Win%'])
        records_df['GB'] = records_df.apply(
            lambda x: f"{((top_pct - float(x['Win%'])) * (x['W'] + x['L']) / 2):.1f}" if x.name > 1 else "-",
            axis=1
        )

        st.dataframe(records_df, use_container_width=True)


# Footer
st.markdown("---")
st.markdown(f"🧠 **CourtMind AI** | Data through {df['game_date'].max().strftime('%B %d, %Y')}")
