# CourtMind Daily Workflow

## Overview

CourtMind uses a calibrated prediction model with Odds API integration to generate daily betting picks. The system learns from historical performance to filter out losing edge ranges and prioritize profitable picks.

## Daily Update Process

### Automated (Scheduled Tasks)

**Morning (6:00 AM):**
- Grades yesterday's picks
- Generates today's games with fresh odds
- Creates TOP 10 picks

**Evening (5:00 PM):**
- Refreshes odds and lines
- Regenerates TOP 10 picks

### Manual Update

Run this anytime to refresh all data:

```bash
# Windows
C:\Users\user\CourtMind\daily_update.bat

# Or directly with Python
python C:\Users\user\CourtMind\run_daily_update.py
```

## What Gets Generated

### 1. **Lineups** (Rotowire)
- Starting lineups for all teams playing today
- Injury updates
- Updated every run

### 2. **Player Props** (Odds API)
- Points, Rebounds, Assists lines from DK + FD
- Cached for 6 hours
- ~30-50 players per day (starters in today's games)

### 3. **Game Odds** (Odds API)
- Spreads, Totals, Moneylines from DK + FD
- All games today
- Fresh on every run

### 4. **Top 10 Picks** (Model-Generated)
- **Calibrated** picks using historical feedback
- Only includes 20%+ edge (data shows <20% loses money)
- Filters out bad data (>80% edge = likely errors)
- Applies stat-specific scoring

## Model Calibration

Based on 100 graded historical picks:

### Edge Performance
| Edge Tier | Hit Rate | Status |
|-----------|----------|--------|
| <10% | 50.0% | Break even |
| 10-20% | 45.8% | **LOSES MONEY** |
| **20-30%** | **61.5%** | **PROFITABLE** ✓ |
| 30%+ | 46.2% | Overconfident |

### Stat Performance
- **Points**: 50.9% (28/55) - Standard
- **Rebounds**: 100% (3/3) - Prioritized ✓
- **Assists**: 100% (1/1) - Limited data
- **Spreads**: 52.6% - Good ✓
- **Totals**: 47.4% - Underperforms

### Applied Adjustments

```python
CALIBRATION = {
    'min_edge': 20%,              # Required minimum
    'confidence_multiplier': 0.85, # Model slightly overconfident
    'stat_multipliers': {
        'POINTS': 0.89,           # Underperforms
        'REBOUNDS': 1.25,         # Outperforms - prioritize
        'ASSISTS': 1.0,           # Neutral
        'SPREAD': 1.11,           # Slightly outperforms
        'TOTAL': 0.83             # Underperforms
    },
    'max_edge_filter': 80%        # Filter obvious bad data
}
```

## API Endpoints

After running the daily update, these endpoints serve cached data:

- `GET /api/games` - Today's games with predictions
- `GET /api/top-picks` - TOP 10 calibrated picks
- `GET /api/lineups` - Starting lineups
- `GET /api/props/{player}` - Player prop lines from Odds API
- `GET /api/tracking/today` - Today's logged picks
- `POST /api/daily-tracking/grade` - Grade yesterday's picks

## Files Generated

### Cached Data Files
- `games_cache.json` - Games with predictions and odds
- `top_picks_cache.json` - Top 10 picks of the day
- `props_cache.json` - Player props from Odds API
- `odds_cache.json` - Game odds from Odds API
- `rotowire_lineups.json` - Scraped lineups

### Tracking Files
- `predictions_log.json` - All historical predictions
- `daily_tracking.json` - Daily summary stats
- `model_feedback.json` - Performance analysis for calibration

## Important Notes

### ✅ DO Use
- TOP 10 picks from `/api/top-picks` endpoint
- 20%+ edge picks only
- Picks that pass calibration filters

### ❌ DON'T Use
- Raw player props with >80% edge (bad data)
- <20% edge picks (historically lose money)
- Unfiltered predictions

### Example Bad Data

**Jarrett Allen ASSISTS OVER 1.5** (293% edge)
- Line: 1.5 assists
- Projection: 5.9 assists
- Problem: Center with low assist rate, line artificially low
- Model extrapolates from small quarter samples
- Filtered out by >80% edge rule ✓

## Troubleshooting

**No player props showing?**
- Props require 20%+ edge to show
- Most props have <20% edge = filtered out
- This is intentional - lower edge loses money

**Odds API not working?**
- Check API key in `odds_api_key.txt`
- Verify `/api/health` shows `"odds_api": true`
- Props cache expires after 6 hours

**Predictions seem off?**
- Model uses quarter-level data (not per-game)
- Predictions normalized to per-36 minutes
- Some extrapolation for low-minute players
- Calibration filters reduce false positives

## Historical Performance

System tracks all predictions and updates calibration:
- 100+ graded picks
- 50.8% overall hit rate on player props
- 61.5% hit rate on 20-30% edge picks ✓
- Continuous feedback loop improves filtering
