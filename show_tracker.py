import pandas as pd

df = pd.read_excel('C:/Users/user/CourtMind/predictions_tracking.xlsx')
print(f"Total: {len(df)} predictions")
print()
print("By Type:")
print(df['type'].value_counts().to_string())
print()
print("Player Props:")
props = df[df['type'] == 'PLAYER_PROP'][['player', 'stat', 'line', 'direction', 'edge']]
for _, r in props.iterrows():
    print(f"  {r['player']} {r['direction']} {r['line']} {r['stat']} ({r['edge']}%)")
print()
print("Top Picks:")
tops = df[df['type'] == 'TOP_PICK'][['player', 'stat', 'line', 'direction', 'edge']]
for _, r in tops.iterrows():
    print(f"  {r['player']} {r['direction']} {r['line']} {r['stat']} ({r['edge']}%)")
