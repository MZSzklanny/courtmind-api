"""Update Jan 31 picks with actual results"""
import pandas as pd

TRACKING_FILE = 'C:/Users/user/CourtMind/predictions_tracking.xlsx'

# Actual results from Jan 31 games
RESULTS = {
    ('Matas Buzelis', 'POINTS'): {'actual': 21, 'hit': False},  # UNDER 19.5 - MISS
    ('Isaac Okoro', 'POINTS'): {'actual': 20, 'hit': False},  # UNDER 10.5 - MISS
    ('Saddiq Bey', 'POINTS'): {'actual': 34, 'hit': True},  # OVER 15.5 - HIT
    ('Anthony Edwards', 'POINTS'): {'actual': 33, 'hit': True},  # OVER 27.5 - HIT
    ('VJ Edgecombe', 'REBOUNDS'): {'actual': 3, 'hit': False},  # OVER 4.5 - MISS
    ('VJ Edgecombe', 'POINTS'): {'actual': 15, 'hit': True},  # OVER 13.5 - HIT
    ('Jalen Smith', 'REBOUNDS'): {'actual': None, 'hit': None},  # DNP
    ('Jalen Smith', 'POINTS'): {'actual': None, 'hit': None},  # DNP
    ('Donte DiVincenzo', 'POINTS'): {'actual': 11, 'hit': False},  # OVER 12.5 - MISS
    ('Cam Spencer', 'POINTS'): {'actual': 7, 'hit': True},  # UNDER 13.5 - HIT
    ("De'Aaron Fox", 'POINTS'): {'actual': 9, 'hit': False},  # OVER 17.5 - MISS
    ('Kon Knueppel', 'REBOUNDS'): {'actual': 6, 'hit': True},  # OVER 4.5 - HIT
}

# Load tracking
df = pd.read_excel(TRACKING_FILE)

# Update results
updated = 0
for idx, row in df.iterrows():
    key = (row['player'], row['stat'])
    if key in RESULTS:
        result = RESULTS[key]
        df.at[idx, 'actual'] = result['actual']
        df.at[idx, 'hit'] = result['hit']
        updated += 1
        status = "HIT" if result['hit'] else ("MISS" if result['hit'] is False else "DNP")
        print(f"{row['player']} {row['stat']}: {result['actual']} - {status}")

# Save
df.to_excel(TRACKING_FILE, index=False)
print(f"\nUpdated {updated} picks")

# Summary
graded = df[df['hit'].notna()]
if len(graded) > 0:
    hits = graded['hit'].sum()
    total = len(graded)
    print(f"\nJan 31 Results: {int(hits)}/{total} = {hits/total*100:.1f}% hit rate")
