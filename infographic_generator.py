"""
CourtMind Infographic Generator - FIRE EDITION
===============================================
Fire/cosmic style graphics matching the template.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Wedge, Arc
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from datetime import datetime

from CourtMind.config import (
    COLORS, RATING_COLORS, TEAM_COLORS, FONTS,
    TWITTER_LANDSCAPE, TWITTER_HANDLE, OUTPUT_DIR
)


def setup_fire_style():
    """Configure matplotlib for fire/cosmic aesthetic."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial Black', 'Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.facecolor': '#0a0608',
        'figure.facecolor': '#0a0608',
        'text.color': '#ffffff',
    })


# Fire color palette
FIRE_COLORS = {
    'bg_dark': '#0a0608',
    'bg_red': '#1a0a0a',
    'fire_orange': '#ff6b00',
    'fire_red': '#ff3300',
    'fire_yellow': '#ffaa00',
    'neon_green': '#00ff66',
    'lime_green': '#7fff00',
    'white': '#ffffff',
    'cream': '#fff8e7',
    'gray': '#8a8a8a',
    'dark_gray': '#4a4a4a',
}


def create_fire_background(ax, fig):
    """Create fire/cosmic background with particles."""
    # Dark base with red tint
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor='#0a0608', transform=ax.transAxes))

    # Radial gradient from center-bottom (fire glow)
    n_layers = 40
    for i in range(n_layers):
        # Bottom fire glow
        alpha = 0.08 * (1 - i/n_layers)
        height = 0.3 + (i * 0.02)
        color_mix = i / n_layers
        # Blend from orange-red at bottom to transparent
        ax.add_patch(Rectangle((0, 0), 1, height,
                               facecolor=FIRE_COLORS['fire_red'], alpha=alpha * 0.5,
                               transform=ax.transAxes))

    # Add some particle/star effect
    np.random.seed(42)
    n_particles = 80
    for _ in range(n_particles):
        x = np.random.random()
        y = np.random.random() * 0.7  # More particles at bottom
        size = np.random.random() * 2 + 0.5
        alpha = np.random.random() * 0.4 + 0.1
        color = np.random.choice([FIRE_COLORS['fire_orange'], FIRE_COLORS['fire_yellow'], '#ff4400'])
        ax.scatter([x], [y], s=size, c=color, alpha=alpha, transform=ax.transAxes)


def add_glow_text(ax, x, y, text, fontsize, color, glow_color=None, glow_strength=3, **kwargs):
    """Add text with glow effect."""
    if glow_color is None:
        glow_color = color

    # Glow layers
    for offset in range(glow_strength, 0, -1):
        alpha = 0.15 / offset
        ax.text(x, y, text, fontsize=fontsize, color=glow_color, alpha=alpha,
                transform=ax.transAxes, **kwargs)

    # Main text
    ax.text(x, y, text, fontsize=fontsize, color=color,
            transform=ax.transAxes, **kwargs)


def draw_speedometer_gauge(ax, x, y, radius, percentage, label_text, sub_label):
    """Draw a speedometer-style gauge."""
    # Outer glow rings
    for r_mult, alpha in [(1.15, 0.1), (1.10, 0.15), (1.05, 0.2)]:
        circle = Circle((x, y), radius * r_mult, facecolor='none',
                        edgecolor=FIRE_COLORS['fire_orange'], linewidth=1, alpha=alpha,
                        transform=ax.transAxes)
        ax.add_patch(circle)

    # Main circle background
    circle_bg = Circle((x, y), radius, facecolor='#0a0a0a',
                       edgecolor=FIRE_COLORS['fire_orange'], linewidth=3, alpha=0.9,
                       transform=ax.transAxes)
    ax.add_patch(circle_bg)

    # Inner decorative ring
    inner_ring = Circle((x, y), radius * 0.85, facecolor='none',
                        edgecolor=FIRE_COLORS['dark_gray'], linewidth=1, alpha=0.5,
                        transform=ax.transAxes)
    ax.add_patch(inner_ring)

    # Tick marks around the gauge
    n_ticks = 20
    for i in range(n_ticks):
        angle = np.radians(225 - (i * 270 / (n_ticks - 1)))  # From 225 to -45 degrees
        inner_r = radius * 0.75
        outer_r = radius * 0.85

        # Tick position (transform to figure coordinates manually)
        # This is approximate since we're in axes coordinates
        tick_length = 0.008 if i % 5 == 0 else 0.005

        x1 = x + np.cos(angle) * (radius * 0.70)
        y1 = y + np.sin(angle) * (radius * 0.70) * (1200/675)  # Aspect ratio adjustment
        x2 = x + np.cos(angle) * (radius * 0.80)
        y2 = y + np.sin(angle) * (radius * 0.80) * (1200/675)

        tick_color = FIRE_COLORS['fire_orange'] if i < (percentage/100 * n_ticks) else FIRE_COLORS['dark_gray']
        ax.plot([x1, x2], [y1, y2], color=tick_color, linewidth=1.5, alpha=0.8, transform=ax.transAxes)

    # Percentage text
    add_glow_text(ax, x, y + 0.02, f"{percentage:.0f}%", fontsize=36,
                  color=FIRE_COLORS['fire_orange'], glow_color=FIRE_COLORS['fire_orange'],
                  fontweight='bold', ha='center', va='center')

    # Label below percentage
    ax.text(x, y - 0.06, label_text, fontsize=11, fontweight='bold',
            color=FIRE_COLORS['fire_orange'], ha='center', va='center', transform=ax.transAxes)

    # Sub-label
    ax.text(x, y - 0.11, sub_label, fontsize=8,
            color=FIRE_COLORS['gray'], ha='center', va='center', transform=ax.transAxes,
            style='italic')


def create_player_prediction_card(
    player_name,
    team,
    opponent,
    proj_pts,
    pctile_90,
    pctile_90_odds,
    floor_pts,
    max_pts,
    sgdr,
    matchup_adj,
    is_hot=False,
    is_b2b=False,
    insight_text="",
    game_time="Tonight",
    output_path=None,
    analysis_bullets=None
):
    """
    Create fire-style player prediction card matching the template.
    """
    setup_fire_style()

    fig_w, fig_h = TWITTER_LANDSCAPE
    fig = plt.figure(figsize=(fig_w/100, fig_h/100), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Fire background
    create_fire_background(ax, fig)

    # === HEADER SECTION ===
    # "CEILING SPOTLIGHT" header
    add_glow_text(ax, 0.30, 0.93, "CEILING", fontsize=28,
                  color=FIRE_COLORS['fire_orange'], glow_color=FIRE_COLORS['fire_orange'],
                  fontweight='bold', ha='right', va='center')
    ax.text(0.32, 0.93, "SPOTLIGHT", fontsize=28, fontweight='bold',
            color=FIRE_COLORS['cream'], ha='left', va='center', transform=ax.transAxes)

    # HOT badge
    if is_hot:
        badge_x = 0.88
        ax.text(badge_x, 0.93, "HOT", fontsize=13, fontweight='bold',
                color=FIRE_COLORS['fire_orange'], ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#ff330033',
                         edgecolor=FIRE_COLORS['fire_orange'], linewidth=2))

    # Player name
    add_glow_text(ax, 0.35, 0.83, player_name.upper(), fontsize=38,
                  color=FIRE_COLORS['cream'], glow_color=FIRE_COLORS['fire_orange'],
                  fontweight='bold', ha='center', va='center')

    # "vs CHI" in orange
    ax.text(0.35, 0.75, f"vs {opponent}", fontsize=16, fontweight='bold',
            color=FIRE_COLORS['fire_orange'], ha='center', va='center', transform=ax.transAxes)

    # === LEFT SECTION: CEILING GAME ===
    left_x = 0.22

    # "CEILING GAME" label
    ax.text(left_x, 0.65, "CEILING GAME", fontsize=14, fontweight='bold',
            color=FIRE_COLORS['neon_green'], ha='center', va='center', transform=ax.transAxes)

    # Large 90th percentile number
    add_glow_text(ax, left_x - 0.03, 0.50, f"{pctile_90:.1f}", fontsize=72,
                  color=FIRE_COLORS['neon_green'], glow_color=FIRE_COLORS['neon_green'],
                  fontweight='bold', ha='center', va='center')
    ax.text(left_x + 0.12, 0.52, "PTS", fontsize=24, fontweight='bold',
            color=FIRE_COLORS['lime_green'], ha='left', va='center', transform=ax.transAxes)

    # "(90TH PERCENTILE)" label
    ax.text(left_x, 0.36, "(90TH PERCENTILE)", fontsize=11,
            color=FIRE_COLORS['gray'], ha='center', va='center', transform=ax.transAxes)

    # === RIGHT SECTION: SPEEDOMETER GAUGE ===
    gauge_x = 0.75
    gauge_y = 0.52
    gauge_radius = 0.13

    # Calculate 30+ pts probability (use odds or adjust)
    prob_30_plus = pctile_90_odds + 5  # Approximate adjustment

    draw_speedometer_gauge(ax, gauge_x, gauge_y, gauge_radius,
                          pctile_90_odds, f"{int(pctile_90-5)}+ PTS PROBABILITY",
                          "Based on matchup +\nrecent form")

    # === ANALYSIS SECTION ===
    if analysis_bullets is None:
        # Default bullets based on matchup
        analysis_bullets = [
            ("*", f"Unfavorable defensive metrics vs wings", "white"),
            ("*", f"+{(matchup_adj-1)*100:.0f}% scoring vs elite scorers", FIRE_COLORS['fire_orange']),
            ("*", f"{player_name.split()[0]}'s ceiling is in play tonight", FIRE_COLORS['neon_green']),
        ]

    # Section header
    add_glow_text(ax, 0.06, 0.28, "ELITE WINGS", fontsize=14,
                  color=FIRE_COLORS['neon_green'], glow_color=FIRE_COLORS['neon_green'],
                  fontweight='bold', ha='left', va='center')
    ax.text(0.26, 0.28, f"TORCH THE {opponent}", fontsize=14, fontweight='bold',
            color=FIRE_COLORS['fire_orange'], ha='left', va='center', transform=ax.transAxes)

    # Bullet points
    bullet_y = 0.21
    bullet_spacing = 0.050

    for i, (icon, text, color) in enumerate(analysis_bullets):
        y = bullet_y - (i * bullet_spacing)
        # Colored bullet point
        bullet_color = FIRE_COLORS['fire_orange'] if i == 1 else (FIRE_COLORS['neon_green'] if i == 2 else FIRE_COLORS['cream'])
        ax.text(0.06, y, ">", fontsize=12, ha='left', va='center', color=bullet_color,
                transform=ax.transAxes, fontweight='bold')
        ax.text(0.09, y, text, fontsize=11, color=color if color != "white" else FIRE_COLORS['cream'],
                ha='left', va='center', transform=ax.transAxes)

    # === FOOTER ===
    # Tagline
    ax.text(0.06, 0.05, "Trust the model. Chase the ceiling.", fontsize=12,
            color=FIRE_COLORS['cream'], ha='left', va='center', transform=ax.transAxes,
            style='italic')

    # Handle
    ax.text(0.06, 0.015, TWITTER_HANDLE, fontsize=11, fontweight='bold',
            color=FIRE_COLORS['neon_green'], ha='left', va='center', transform=ax.transAxes)

    # Save
    if output_path is None:
        today = datetime.now().strftime('%Y-%m-%d')
        os.makedirs(os.path.join(OUTPUT_DIR, today), exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, today, f"player_{player_name.replace(' ', '_')}.png")

    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0,
                facecolor='#0a0608')
    plt.close()
    return output_path


def create_top_picks_graphic(players, title="TOP CEILING GAMES TONIGHT", output_path=None):
    """
    Create fire-style Top 5 Picks graphic.
    """
    setup_fire_style()

    fig_w, fig_h = TWITTER_LANDSCAPE
    fig = plt.figure(figsize=(fig_w/100, fig_h/100), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Fire background
    create_fire_background(ax, fig)

    # Header
    add_glow_text(ax, 0.5, 0.93, title, fontsize=28,
                  color=FIRE_COLORS['cream'], glow_color=FIRE_COLORS['fire_orange'],
                  fontweight='bold', ha='center', va='center')

    # Divider line
    ax.axhline(y=0.86, xmin=0.05, xmax=0.95, color=FIRE_COLORS['fire_orange'], linewidth=2, alpha=0.6)

    # Column headers
    header_y = 0.81
    headers = [("#", 0.06), ("PLAYER", 0.22), ("MATCHUP", 0.48),
               ("PROJ", 0.62), ("90th", 0.76), ("ODDS", 0.90)]
    for text, x in headers:
        ax.text(x, header_y, text, fontsize=11, fontweight='bold',
                color=FIRE_COLORS['fire_orange'], ha='center', va='center', transform=ax.transAxes)

    # Player rows
    row_height = 0.125
    start_y = 0.72

    for i, player in enumerate(players[:5]):
        y = start_y - (i * row_height)

        # Alternating row backgrounds
        if i % 2 == 0:
            ax.add_patch(Rectangle((0.03, y - 0.045), 0.94, 0.09,
                                   facecolor=FIRE_COLORS['bg_red'], alpha=0.4, transform=ax.transAxes))

        team = player.get('team', '')

        # Rank with fire glow
        add_glow_text(ax, 0.06, y, str(i + 1), fontsize=22,
                      color=FIRE_COLORS['fire_orange'], glow_color=FIRE_COLORS['fire_orange'],
                      fontweight='bold', ha='center', va='center')

        # Player name and team
        name = player.get('player', 'Unknown')
        ax.text(0.12, y + 0.015, name, fontsize=13, fontweight='bold',
                color=FIRE_COLORS['cream'], ha='left', va='center', transform=ax.transAxes)
        ax.text(0.12, y - 0.022, team, fontsize=9,
                color=FIRE_COLORS['fire_orange'], ha='left', va='center', transform=ax.transAxes)

        # Matchup
        opponent = player.get('opponent', '')
        ax.text(0.48, y, f"vs {opponent}", fontsize=11,
                color=FIRE_COLORS['gray'], ha='center', va='center', transform=ax.transAxes)

        # Projected
        proj = player.get('proj_pts', 0)
        ax.text(0.62, y, f"{proj:.1f}", fontsize=14, fontweight='bold',
                color=FIRE_COLORS['cream'], ha='center', va='center', transform=ax.transAxes)

        # 90th percentile (green)
        pctile = player.get('pctile_90', 0)
        add_glow_text(ax, 0.76, y, f"{pctile:.1f}", fontsize=14,
                      color=FIRE_COLORS['neon_green'], glow_color=FIRE_COLORS['neon_green'],
                      fontweight='bold', ha='center', va='center')

        # Odds with fire color
        odds = player.get('pctile_90_odds', 0)
        add_glow_text(ax, 0.90, y, f"{odds:.0f}%", fontsize=14,
                      color=FIRE_COLORS['fire_orange'], glow_color=FIRE_COLORS['fire_orange'],
                      fontweight='bold', ha='center', va='center')

    # Footer
    ax.text(0.06, 0.04, "Trust the model. Chase the ceiling.", fontsize=10,
            color=FIRE_COLORS['cream'], ha='left', va='center', transform=ax.transAxes, style='italic')
    ax.text(0.06, 0.015, TWITTER_HANDLE, fontsize=10, fontweight='bold',
            color=FIRE_COLORS['neon_green'], ha='left', va='center', transform=ax.transAxes)
    ax.text(0.94, 0.015, datetime.now().strftime('%B %d, %Y'), fontsize=9,
            color=FIRE_COLORS['fire_orange'], ha='right', va='center', transform=ax.transAxes)

    # Save
    if output_path is None:
        today = datetime.now().strftime('%Y-%m-%d')
        os.makedirs(os.path.join(OUTPUT_DIR, today), exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, today, "top_picks.png")

    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0,
                facecolor='#0a0608')
    plt.close()
    return output_path


def create_fatigue_alert_graphic(players, output_path=None):
    """Create fire-style B2B fatigue alert graphic."""
    setup_fire_style()

    fig_w, fig_h = TWITTER_LANDSCAPE
    fig = plt.figure(figsize=(fig_w/100, fig_h/100), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    create_fire_background(ax, fig)

    # Warning header
    ax.text(0.08, 0.91, "⚠️", fontsize=24, ha='center', va='center', transform=ax.transAxes)
    add_glow_text(ax, 0.5, 0.91, "B2B FATIGUE ALERT", fontsize=30,
                  color=FIRE_COLORS['fire_red'], glow_color=FIRE_COLORS['fire_red'],
                  fontweight='bold', ha='center', va='center')

    ax.text(0.5, 0.84, "Players at elevated fatigue risk tonight",
            fontsize=13, color=FIRE_COLORS['fire_orange'], ha='center', va='center', transform=ax.transAxes)

    ax.axhline(y=0.80, xmin=0.1, xmax=0.9, color=FIRE_COLORS['fire_red'], linewidth=2, alpha=0.5)

    # Player cards
    num_players = min(len(players), 3)
    card_width = 0.26
    card_height = 0.50
    total_width = num_players * card_width + (num_players - 1) * 0.04
    start_x = (1 - total_width) / 2

    for i, player in enumerate(players[:3]):
        x = start_x + i * (card_width + 0.04)
        y = 0.18

        sgdr = player.get('sgdr', 0)

        # Card background
        fancy_box = FancyBboxPatch(
            (x, y), card_width, card_height,
            boxstyle="round,pad=0,rounding_size=0.02",
            facecolor='#1a0a0a', edgecolor=FIRE_COLORS['fire_red'],
            linewidth=2, alpha=0.9, transform=ax.transAxes
        )
        ax.add_patch(fancy_box)

        name = player.get('player', 'Unknown')
        team = player.get('team', '')

        ax.text(x + card_width/2, y + card_height - 0.08, name,
                fontsize=12, fontweight='bold', color=FIRE_COLORS['cream'],
                ha='center', va='center', transform=ax.transAxes)
        ax.text(x + card_width/2, y + card_height - 0.15, f"({team})",
                fontsize=10, color=FIRE_COLORS['fire_orange'],
                ha='center', va='center', transform=ax.transAxes)

        # SGDR value
        add_glow_text(ax, x + card_width/2, y + 0.22, f"{sgdr:.0f}",
                      fontsize=44, color=FIRE_COLORS['fire_red'], glow_color=FIRE_COLORS['fire_red'],
                      fontweight='bold', ha='center', va='center')
        ax.text(x + card_width/2, y + 0.08, "SGDR",
                fontsize=11, fontweight='bold', color=FIRE_COLORS['gray'],
                ha='center', va='center', transform=ax.transAxes)

    # Footer
    ax.text(0.5, 0.08, "Consider unders or reduced exposure in DFS",
            fontsize=11, color=FIRE_COLORS['gray'], ha='center', va='center', transform=ax.transAxes)
    ax.text(0.06, 0.03, TWITTER_HANDLE, fontsize=10, fontweight='bold',
            color=FIRE_COLORS['neon_green'], ha='left', va='center', transform=ax.transAxes)

    if output_path is None:
        today = datetime.now().strftime('%Y-%m-%d')
        os.makedirs(os.path.join(OUTPUT_DIR, today), exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, today, "fatigue_alert.png")

    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0,
                facecolor='#0a0608')
    plt.close()
    return output_path


if __name__ == "__main__":
    print("Testing FIRE infographic generator...")
    path = create_player_prediction_card(
        player_name="Kevin Durant",
        team="HOU",
        opponent="CHI",
        proj_pts=24.0,
        pctile_90=35.5,
        pctile_90_odds=34,
        floor_pts=12.3,
        max_pts=60,
        sgdr=45,
        matchup_adj=1.05,
        is_hot=True,
        is_b2b=False,
        insight_text="Elite ceiling game vs CHI perimeter defense",
        game_time="Tonight"
    )
    print(f"Created: {path}")
