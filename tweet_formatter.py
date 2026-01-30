"""
CourtMind Tweet Formatter
=========================
Generates tweet text with hashtags for each content type.
"""

from CourtMind.config import HASHTAGS, TEAM_HASHTAGS, TWITTER_HANDLE


def get_team_hashtags(team, limit=2):
    """Get hashtags for a specific team."""
    tags = TEAM_HASHTAGS.get(team, [])
    return tags[:limit]


def get_base_hashtags():
    """Get hashtags to include on every post."""
    return HASHTAGS['always'] + ['#FantasyBasketball']


def format_player_prediction_tweet(
    player,
    team,
    opponent,
    proj_pts,
    pctile_90,
    pctile_90_odds,
    matchup_info="",
    is_hot=False,
    is_b2b=False,
    insight=""
):
    """
    Format tweet for player prediction card.

    Returns tweet text (under 280 chars).
    """
    lines = []

    # Header with status
    status = ""
    if is_hot:
        status = "HOT "
    elif is_b2b:
        status = "B2B "

    lines.append(f"CEILING WATCH: {status}{player} vs {opponent}")
    lines.append("")

    # Stats
    lines.append(f"Projected: {proj_pts:.1f} PTS")
    lines.append(f"90th Percentile: {pctile_90:.1f} PTS ({pctile_90_odds:.0f}% odds)")
    lines.append("")

    # Insight
    if matchup_info:
        lines.append(f"Matchup: {matchup_info}")
    if insight:
        lines.append(insight)

    # Hashtags
    hashtags = get_base_hashtags() + get_team_hashtags(team, 1)
    lines.append("")
    lines.append(" ".join(hashtags))

    tweet = "\n".join(lines)

    # Truncate if needed (rare)
    if len(tweet) > 280:
        # Remove insight line
        lines = [l for l in lines if l != insight]
        tweet = "\n".join(lines)

    return tweet


def format_top_picks_tweet(players, limit=5):
    """
    Format tweet for top picks graphic.

    Returns tweet text.
    """
    lines = []

    lines.append(f"TOP {min(len(players), limit)} CEILING GAMES TONIGHT")
    lines.append("")

    for i, p in enumerate(players[:limit]):
        name = p.get('player', 'Unknown')
        team = p.get('team', '')
        opp = p.get('opponent', '')
        pctile = p.get('pctile_90', 0)
        odds = p.get('pctile_90_odds', 0)

        lines.append(f"{i+1}. {name} ({team} vs {opp}) - {pctile:.0f}pts @ {odds:.0f}%")

    lines.append("")
    hashtags = get_base_hashtags() + ['#DFS']
    lines.append(" ".join(hashtags))

    return "\n".join(lines)


def format_fatigue_alert_tweet(players):
    """
    Format tweet for fatigue alert.

    Returns tweet text.
    """
    lines = []

    lines.append("B2B FATIGUE ALERT")
    lines.append("")
    lines.append("Players at elevated fatigue risk tonight:")
    lines.append("")

    for p in players[:3]:
        name = p.get('player', 'Unknown')
        team = p.get('team', '')
        sgdr = p.get('sgdr', 0)
        lines.append(f"{name} ({team}) - SGDR: {sgdr:.0f}")

    lines.append("")
    lines.append("Consider unders or reduced DFS exposure")
    lines.append("")

    hashtags = get_base_hashtags() + ['#NBABets', '#DFS']
    lines.append(" ".join(hashtags))

    return "\n".join(lines)


def format_hot_streak_tweet(player, team, opponent, hot_hand_mult, proj_pts, pctile_90):
    """
    Format tweet for hot streak player.

    Returns tweet text.
    """
    hot_pct = (hot_hand_mult - 1) * 100

    lines = []
    lines.append(f"HOT STREAK: {player}")
    lines.append("")
    lines.append(f"Scoring {hot_pct:.0f}% above season average last 3 games")
    lines.append(f"Tonight vs {opponent}")
    lines.append("")
    lines.append(f"Boosted Projection: {proj_pts:.1f} PTS")
    lines.append(f"Ceiling: {pctile_90:.1f} PTS")
    lines.append("")

    hashtags = get_base_hashtags() + get_team_hashtags(team, 1)
    lines.append(" ".join(hashtags))

    return "\n".join(lines)


def generate_insight_text(
    player,
    matchup_adj,
    sgdr,
    is_b2b,
    is_hot,
    opponent
):
    """
    Generate data-driven narrative insight for player.

    Returns short insight string.
    """
    insights = []

    # Matchup insight
    matchup_pct = (matchup_adj - 1) * 100
    if matchup_pct > 8:
        insights.append(f"Elite {matchup_pct:.0f}% matchup boost vs {opponent}")
    elif matchup_pct > 3:
        insights.append(f"Favorable matchup (+{matchup_pct:.0f}%) vs {opponent}")
    elif matchup_pct < -5:
        insights.append(f"Tough matchup ({matchup_pct:.0f}%) vs {opponent}")

    # Fatigue insight
    if is_b2b:
        if sgdr > 70:
            insights.append("High fatigue concern on B2B")
        else:
            insights.append("B2B but handles load well")
    elif sgdr < 40:
        insights.append("Fresh legs, low fatigue")

    # Hot/cold insight
    if is_hot:
        insights.append("Riding hot streak last 3 games")

    # Combine top 2 insights
    if len(insights) >= 2:
        return f"{insights[0]}. {insights[1]}."
    elif insights:
        return insights[0] + "."
    else:
        return ""


if __name__ == "__main__":
    # Test tweet formatting
    print("Testing tweet formatter...")

    tweet = format_player_prediction_tweet(
        player="Tyrese Maxey",
        team="PHI",
        opponent="ORL",
        proj_pts=24.5,
        pctile_90=31.2,
        pctile_90_odds=18,
        matchup_info="+8.2% vs ORL",
        is_hot=True,
        insight="Fresh legs after 2 days rest."
    )
    print("Player prediction tweet:")
    print(tweet)
    print(f"Length: {len(tweet)} chars\n")

    test_players = [
        {'player': 'Tyrese Maxey', 'team': 'PHI', 'opponent': 'ORL', 'pctile_90': 31.2, 'pctile_90_odds': 18},
        {'player': 'Luka Doncic', 'team': 'DAL', 'opponent': 'LAL', 'pctile_90': 38.1, 'pctile_90_odds': 21},
        {'player': 'Jayson Tatum', 'team': 'BOS', 'opponent': 'MIA', 'pctile_90': 35.4, 'pctile_90_odds': 16},
    ]

    top_tweet = format_top_picks_tweet(test_players)
    print("Top picks tweet:")
    print(top_tweet)
    print(f"Length: {len(top_tweet)} chars\n")

    fatigue_players = [
        {'player': 'Joel Embiid', 'team': 'PHI', 'sgdr': 72},
        {'player': 'LeBron James', 'team': 'LAL', 'sgdr': 68},
    ]
    fatigue_tweet = format_fatigue_alert_tweet(fatigue_players)
    print("Fatigue alert tweet:")
    print(fatigue_tweet)
    print(f"Length: {len(fatigue_tweet)} chars")
