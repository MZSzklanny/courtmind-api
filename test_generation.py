"""
Test CourtMind content generation.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CourtMind.infographic_generator import (
    create_player_prediction_card,
    create_top_picks_graphic,
    create_fatigue_alert_graphic
)
from CourtMind.tweet_formatter import (
    format_player_prediction_tweet,
    format_top_picks_tweet
)
from CourtMind.config import OUTPUT_DIR

def test_infographics():
    """Test infographic generation."""
    print("Testing infographic generation...")

    # Create test output directory
    test_dir = os.path.join(OUTPUT_DIR, "test")
    os.makedirs(test_dir, exist_ok=True)

    # Test player card
    player_path = create_player_prediction_card(
        player_name="Tyrese Maxey",
        team="PHI",
        opponent="ORL",
        proj_pts=24.5,
        pctile_90=31.2,
        pctile_90_odds=18,
        floor_pts=12.3,
        max_pts=42,
        sgdr=45,
        matchup_adj=1.08,
        is_hot=True,
        is_b2b=False,
        insight_text="Elite matchup vs ORL perimeter D.",
        game_time="7:00 PM ET",
        output_path=os.path.join(test_dir, "test_player_card.png")
    )
    print(f"  Player card: {player_path}")

    # Test top picks
    test_players = [
        {'player': 'Tyrese Maxey', 'team': 'PHI', 'opponent': 'ORL', 'proj_pts': 24.5, 'pctile_90': 31.2, 'pctile_90_odds': 18},
        {'player': 'Luka Doncic', 'team': 'DAL', 'opponent': 'LAL', 'proj_pts': 28.3, 'pctile_90': 38.1, 'pctile_90_odds': 21},
        {'player': 'Jayson Tatum', 'team': 'BOS', 'opponent': 'MIA', 'proj_pts': 26.8, 'pctile_90': 35.4, 'pctile_90_odds': 16},
        {'player': 'Stephen Curry', 'team': 'GSW', 'opponent': 'DEN', 'proj_pts': 25.1, 'pctile_90': 34.2, 'pctile_90_odds': 15},
        {'player': 'Anthony Edwards', 'team': 'MIN', 'opponent': 'PHX', 'proj_pts': 24.8, 'pctile_90': 33.5, 'pctile_90_odds': 17},
    ]
    top_path = create_top_picks_graphic(
        test_players,
        output_path=os.path.join(test_dir, "test_top_picks.png")
    )
    print(f"  Top picks: {top_path}")

    # Test fatigue alert
    fatigue_players = [
        {'player': 'Joel Embiid', 'team': 'PHI', 'sgdr': 72},
        {'player': 'LeBron James', 'team': 'LAL', 'sgdr': 68},
        {'player': 'Kevin Durant', 'team': 'PHX', 'sgdr': 65},
    ]
    fatigue_path = create_fatigue_alert_graphic(
        fatigue_players,
        output_path=os.path.join(test_dir, "test_fatigue.png")
    )
    print(f"  Fatigue alert: {fatigue_path}")

    print(f"\nAll test graphics saved to: {test_dir}")
    return True


def test_tweets():
    """Test tweet formatting."""
    print("\nTesting tweet formatting...")

    tweet = format_player_prediction_tweet(
        player="Tyrese Maxey",
        team="PHI",
        opponent="ORL",
        proj_pts=24.5,
        pctile_90=31.2,
        pctile_90_odds=18,
        matchup_info="+8% vs ORL",
        is_hot=True,
        insight="Fresh legs after 2 days rest."
    )
    print(f"  Player tweet ({len(tweet)} chars):")
    print("-" * 40)
    print(tweet)
    print("-" * 40)

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("CourtMind Test Suite")
    print("=" * 60)

    try:
        test_infographics()
        test_tweets()
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
