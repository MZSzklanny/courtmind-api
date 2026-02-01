import pandas as pd
from datetime import datetime

TRACKING_FILE = 'C:/Users/user/CourtMind/predictions_tracking.xlsx'
today = datetime.now().strftime('%Y-%m-%d')

# Top 3 picks to track
top_picks = [
    {'date': today, 'type': 'TOP_PICK', 'matchup': 'CHI @ MIA', 'player': 'Jalen Smith', 'stat': 'REBOUNDS', 'line': 10.5, 'direction': 'UNDER', 'projection': 6.7, 'edge': -36.2, 'confidence': 71, 'book': 'DK', 'actual': None, 'hit': None, 'logged_at': datetime.now().isoformat()},
    {'date': today, 'type': 'TOP_PICK', 'matchup': 'SAS @ CHA', 'player': "De'Aaron Fox", 'stat': 'POINTS', 'line': 17.5, 'direction': 'OVER', 'projection': 22.5, 'edge': 28.6, 'confidence': 78, 'book': 'DK', 'actual': None, 'hit': None, 'logged_at': datetime.now().isoformat()},
    {'date': today, 'type': 'TOP_PICK', 'matchup': 'SAS @ CHA', 'player': 'Kon Knueppel', 'stat': 'REBOUNDS', 'line': 4.5, 'direction': 'OVER', 'projection': 5.9, 'edge': 31.1, 'confidence': 70, 'book': 'DK', 'actual': None, 'hit': None, 'logged_at': datetime.now().isoformat()},
]

new_df = pd.DataFrame(top_picks)

# Load existing
try:
    existing_df = pd.read_excel(TRACKING_FILE)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
except:
    combined_df = new_df

combined_df.to_excel(TRACKING_FILE, index=False)
print(f'Added 3 TOP_PICK entries to {TRACKING_FILE}')
