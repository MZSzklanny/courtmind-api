# -*- coding: utf-8 -*-
"""
Migration: Add fg3m column to NBA_PRODUCTION.parquet

Run this ONCE on production server if player props fail with fg3m error.
This adds a placeholder fg3m column so the predictor can work.
"""
import pandas as pd
from pathlib import Path

# Try multiple locations for the parquet file
possible_paths = [
    Path('data/NBA_PRODUCTION.parquet'),
    Path('NBA_PRODUCTION.parquet'),
    Path('/app/data/NBA_PRODUCTION.parquet'),
    Path('../NBA_PRODUCTION.parquet'),
]

parquet_file = None
for path in possible_paths:
    if path.exists():
        parquet_file = path
        break

if not parquet_file:
    print("ERROR: Could not find NBA_PRODUCTION.parquet")
    print("Searched:")
    for p in possible_paths:
        print(f"  - {p.absolute()}")
    exit(1)

print(f"Found parquet: {parquet_file.absolute()}")

# Load
print("Loading...")
df = pd.read_parquet(parquet_file)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)[:10]}...")

# Check if fg3m exists
if 'fg3m' in df.columns:
    print("✓ fg3m column already exists - no migration needed")
    exit(0)

# Add fg3m column
print("Adding fg3m column (placeholder)...")
df['fg3m'] = 0

# Save
print(f"Saving to {parquet_file}...")
df.to_parquet(parquet_file, index=False)

print(f"✓ Migration complete - added fg3m column to {df.shape[0]} rows")
