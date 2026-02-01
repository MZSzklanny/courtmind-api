"""Grade yesterday's picks and update tracking"""
import pandas as pd
from datetime import datetime

TRACKING_FILE = 'C:/Users/user/CourtMind/predictions_tracking.xlsx'

# Load tracking
df = pd.read_excel(TRACKING_FILE)

print("Jan 31 picks to grade:")
print("=" * 60)
props = df[df['type'].isin(['PLAYER_PROP', 'TOP_PICK'])]
for _, r in props.iterrows():
    print(f"{r['player']} {r['direction']} {r['line']} {r['stat']} (proj: {r['projection']})")

print("\n" + "=" * 60)
print("Enter actual stats to grade (or 'skip' to skip):")
print("=" * 60)

# We'll update with actual results
# For now, let's just show what needs grading
