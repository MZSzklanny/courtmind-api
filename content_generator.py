"""
CourtMind Content Generator
===========================
Main orchestration script for generating daily Twitter content.
"""

import os
import json
from datetime import datetime

from CourtMind.config import OUTPUT_DIR, CONTENT_SETTINGS
from CourtMind.schedule_fetcher import get_todays_games, identify_b2b_teams
from CourtMind.prediction_engine import generate_all_predictions, select_top_storylines
from CourtMind.infographic_generator import (
    create_player_prediction_card,
    create_top_picks_graphic,
    create_fatigue_alert_graphic
)
from CourtMind.tweet_formatter import (
    format_player_prediction_tweet,
    format_top_picks_tweet,
    format_fatigue_alert_tweet,
    format_hot_streak_tweet,
    generate_insight_text
)


def generate_daily_content(date=None):
    """
    Main function to generate all daily content.

    Args:
        date: datetime object or None for today

    Returns:
        List of generated post dicts with image_path, tweet_text, metadata
    """
    if date is None:
        date = datetime.now()

    date_str = date.strftime('%Y-%m-%d')
    output_dir = os.path.join(OUTPUT_DIR, date_str)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"CourtMind Content Generation - {date_str}")
    print('='*60)

    # Step 1: Get today's games
    print("\n[1/5] Fetching today's games...")
    games = get_todays_games(date)

    if not games:
        print("No games today. Skipping content generation.")
        return []

    print(f"Found {len(games)} games:")
    for g in games:
        print(f"  - {g['away_team']} @ {g['home_team']}")

    # Step 2: Identify B2B teams
    print("\n[2/5] Identifying B2B situations...")
    b2b_teams = identify_b2b_teams(games)
    if b2b_teams:
        print(f"B2B teams: {', '.join(b2b_teams)}")
    else:
        print("No B2B teams tonight")

    # Step 3: Generate predictions
    print("\n[3/5] Generating player predictions...")
    predictions_df = generate_all_predictions(games, b2b_teams)

    if predictions_df.empty:
        print("No predictions generated. Check data availability.")
        return []

    print(f"Generated {len(predictions_df)} player predictions")

    # Step 4: Select storylines
    print("\n[4/5] Selecting top storylines...")
    storylines = select_top_storylines(predictions_df, n_posts=CONTENT_SETTINGS['morning_posts'] + 1)
    print(f"Selected {len(storylines)} storylines")

    # Step 5: Generate content
    print("\n[5/5] Generating infographics and tweets...")
    posts = []

    for i, storyline in enumerate(storylines):
        post_type = storyline['type']
        print(f"  Creating {post_type} content...")

        if post_type == 'ceiling_watch':
            data = storyline['data']
            player = storyline['player']
            team = storyline['team']
            opponent = storyline['opponent']

            # Generate insight
            insight = generate_insight_text(
                player=player,
                matchup_adj=data.get('matchup_adj', 1.0),
                sgdr=data.get('sgdr', 50),
                is_b2b=data.get('is_b2b', False),
                is_hot=data.get('is_hot', False),
                opponent=opponent
            )

            # Create image
            image_path = create_player_prediction_card(
                player_name=player,
                team=team,
                opponent=opponent,
                proj_pts=data.get('proj_pts', 0),
                pctile_90=data.get('pctile_90', 0),
                pctile_90_odds=data.get('pctile_90_odds', 10),
                floor_pts=data.get('floor_pts', 0),
                max_pts=data.get('max_pts', 0),
                sgdr=data.get('sgdr', 50),
                matchup_adj=data.get('matchup_adj', 1.0),
                is_hot=data.get('is_hot', False),
                is_b2b=data.get('is_b2b', False),
                insight_text=insight,
                output_path=os.path.join(output_dir, f"post_{i+1}_ceiling_{player.replace(' ', '_')}.png")
            )

            # Create tweet
            matchup_info = data.get('matchup_info', '')
            if not matchup_info and data.get('matchup_adj', 1.0) != 1.0:
                pct = (data.get('matchup_adj', 1.0) - 1) * 100
                matchup_info = f"{'+' if pct > 0 else ''}{pct:.0f}% vs {opponent}"

            tweet_text = format_player_prediction_tweet(
                player=player,
                team=team,
                opponent=opponent,
                proj_pts=data.get('proj_pts', 0),
                pctile_90=data.get('pctile_90', 0),
                pctile_90_odds=data.get('pctile_90_odds', 10),
                matchup_info=matchup_info,
                is_hot=data.get('is_hot', False),
                is_b2b=data.get('is_b2b', False),
                insight=insight
            )

            posts.append({
                'post_id': f"post_{i+1}_{post_type}",
                'type': post_type,
                'image_path': image_path,
                'tweet_text': tweet_text,
                'player': player,
                'team': team,
                'opponent': opponent,
                'status': 'pending',
                'created_at': datetime.now().isoformat(),
            })

        elif post_type == 'top_picks':
            players = storyline['players']

            # Create image
            image_path = create_top_picks_graphic(
                players=players,
                output_path=os.path.join(output_dir, f"post_{i+1}_top_picks.png")
            )

            # Create tweet
            tweet_text = format_top_picks_tweet(players)

            posts.append({
                'post_id': f"post_{i+1}_{post_type}",
                'type': post_type,
                'image_path': image_path,
                'tweet_text': tweet_text,
                'players': [p['player'] for p in players[:5]],
                'status': 'pending',
                'created_at': datetime.now().isoformat(),
            })

        elif post_type == 'fatigue_alert':
            players = storyline['players']

            # Create image
            image_path = create_fatigue_alert_graphic(
                players=players,
                output_path=os.path.join(output_dir, f"post_{i+1}_fatigue.png")
            )

            # Create tweet
            tweet_text = format_fatigue_alert_tweet(players)

            posts.append({
                'post_id': f"post_{i+1}_{post_type}",
                'type': post_type,
                'image_path': image_path,
                'tweet_text': tweet_text,
                'players': [p['player'] for p in players[:3]],
                'status': 'pending',
                'created_at': datetime.now().isoformat(),
            })

        elif post_type == 'hot_streak':
            data = storyline['data']
            player = storyline['player']
            team = storyline['team']
            opponent = storyline['opponent']

            # Use player card for hot streak (with HOT badge)
            insight = generate_insight_text(
                player=player,
                matchup_adj=data.get('matchup_adj', 1.0),
                sgdr=data.get('sgdr', 50),
                is_b2b=data.get('is_b2b', False),
                is_hot=True,
                opponent=opponent
            )

            image_path = create_player_prediction_card(
                player_name=player,
                team=team,
                opponent=opponent,
                proj_pts=data.get('proj_pts', 0),
                pctile_90=data.get('pctile_90', 0),
                pctile_90_odds=data.get('pctile_90_odds', 10),
                floor_pts=data.get('floor_pts', 0),
                max_pts=data.get('max_pts', 0),
                sgdr=data.get('sgdr', 50),
                matchup_adj=data.get('matchup_adj', 1.0),
                is_hot=True,
                is_b2b=data.get('is_b2b', False),
                insight_text=insight,
                output_path=os.path.join(output_dir, f"post_{i+1}_hot_{player.replace(' ', '_')}.png")
            )

            tweet_text = format_hot_streak_tweet(
                player=player,
                team=team,
                opponent=opponent,
                hot_hand_mult=data.get('hot_hand_mult', 1.0),
                proj_pts=data.get('proj_pts', 0),
                pctile_90=data.get('pctile_90', 0)
            )

            posts.append({
                'post_id': f"post_{i+1}_{post_type}",
                'type': post_type,
                'image_path': image_path,
                'tweet_text': tweet_text,
                'player': player,
                'team': team,
                'opponent': opponent,
                'status': 'pending',
                'created_at': datetime.now().isoformat(),
            })

    # Save posts metadata
    metadata_path = os.path.join(output_dir, 'posts.json')
    with open(metadata_path, 'w') as f:
        json.dump(posts, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Content generation complete!")
    print(f"Generated {len(posts)} posts")
    print(f"Output directory: {output_dir}")
    print('='*60)

    return posts


def get_pending_posts(date=None):
    """
    Get list of pending posts for a date.

    Returns list of post dicts.
    """
    if date is None:
        date = datetime.now()

    date_str = date.strftime('%Y-%m-%d')
    metadata_path = os.path.join(OUTPUT_DIR, date_str, 'posts.json')

    if not os.path.exists(metadata_path):
        return []

    with open(metadata_path, 'r') as f:
        posts = json.load(f)

    return [p for p in posts if p.get('status') == 'pending']


def mark_post_as_posted(post_id, tweet_url=None, date=None):
    """
    Mark a post as successfully posted.
    """
    if date is None:
        date = datetime.now()

    date_str = date.strftime('%Y-%m-%d')
    metadata_path = os.path.join(OUTPUT_DIR, date_str, 'posts.json')

    if not os.path.exists(metadata_path):
        return False

    with open(metadata_path, 'r') as f:
        posts = json.load(f)

    for post in posts:
        if post['post_id'] == post_id:
            post['status'] = 'posted'
            post['posted_at'] = datetime.now().isoformat()
            if tweet_url:
                post['tweet_url'] = tweet_url
            break

    with open(metadata_path, 'w') as f:
        json.dump(posts, f, indent=2)

    return True


def preview_posts(date=None):
    """
    Print preview of all pending posts.
    """
    posts = get_pending_posts(date)

    if not posts:
        print("No pending posts found for today.")
        return

    print(f"\n{'='*60}")
    print(f"PENDING POSTS ({len(posts)} total)")
    print('='*60)

    for i, post in enumerate(posts):
        print(f"\n--- Post {i+1}: {post['type'].upper()} ---")
        print(f"Image: {post['image_path']}")
        print(f"\nTweet text:")
        print("-" * 40)
        print(post['tweet_text'])
        print("-" * 40)
        print(f"Status: {post['status']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'preview':
        preview_posts()
    else:
        posts = generate_daily_content()
        if posts:
            print("\nTo preview posts, run: python -m CourtMind.content_generator preview")
