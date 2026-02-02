# -*- coding: utf-8 -*-
"""
CourtMind Model Feedback
========================
Analyzes graded predictions to improve model performance.
Uses tracking results as feedback for model calibration.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
PREDICTIONS_FILE = BASE_DIR / 'predictions_log.json'
FEEDBACK_FILE = BASE_DIR / 'model_feedback.json'


def load_graded_predictions():
    """Load only graded predictions with results."""
    if not PREDICTIONS_FILE.exists():
        return []

    with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    return [p for p in predictions if p.get('hit') is not None]


def analyze_performance():
    """
    Analyze model performance across different dimensions.

    Returns:
        dict with performance metrics and recommendations
    """
    predictions = load_graded_predictions()

    if not predictions:
        return {'error': 'No graded predictions found'}

    df = pd.DataFrame(predictions)

    analysis = {
        'total_graded': len(df),
        'overall_hit_rate': round(df['hit'].mean() * 100, 1),
        'by_confidence': {},
        'by_stat': {},
        'by_type': {},
        'by_edge': {},
        'recommendations': []
    }

    # By confidence tier
    if 'confidence' in df.columns:
        df['conf_tier'] = pd.cut(df['confidence'],
                                  bins=[0, 60, 70, 80, 100],
                                  labels=['<60', '60-70', '70-80', '80+'])
        conf_stats = df.groupby('conf_tier')['hit'].agg(['sum', 'count', 'mean'])
        for tier, row in conf_stats.iterrows():
            if row['count'] > 0:
                analysis['by_confidence'][str(tier)] = {
                    'picks': int(row['count']),
                    'hits': int(row['sum']),
                    'rate': round(row['mean'] * 100, 1)
                }

    # By stat type
    if 'stat' in df.columns:
        stat_stats = df.groupby('stat')['hit'].agg(['sum', 'count', 'mean'])
        for stat, row in stat_stats.iterrows():
            if row['count'] > 0:
                analysis['by_stat'][str(stat)] = {
                    'picks': int(row['count']),
                    'hits': int(row['sum']),
                    'rate': round(row['mean'] * 100, 1)
                }

    # By prediction type
    if 'type' in df.columns:
        type_stats = df.groupby('type')['hit'].agg(['sum', 'count', 'mean'])
        for ptype, row in type_stats.iterrows():
            if row['count'] > 0:
                analysis['by_type'][str(ptype)] = {
                    'picks': int(row['count']),
                    'hits': int(row['sum']),
                    'rate': round(row['mean'] * 100, 1)
                }

    # By edge tier
    if 'edge' in df.columns:
        df['edge_abs'] = df['edge'].abs()
        df['edge_tier'] = pd.cut(df['edge_abs'],
                                  bins=[0, 10, 20, 30, 100],
                                  labels=['<10%', '10-20%', '20-30%', '30%+'])
        edge_stats = df.groupby('edge_tier')['hit'].agg(['sum', 'count', 'mean'])
        for tier, row in edge_stats.iterrows():
            if row['count'] > 0:
                analysis['by_edge'][str(tier)] = {
                    'picks': int(row['count']),
                    'hits': int(row['sum']),
                    'rate': round(row['mean'] * 100, 1)
                }

    # Generate recommendations
    analysis['recommendations'] = generate_recommendations(analysis)

    return analysis


def generate_recommendations(analysis):
    """Generate model tuning recommendations based on performance."""
    recommendations = []

    # Check confidence calibration
    conf = analysis.get('by_confidence', {})
    if conf:
        high_conf = conf.get('80+', {})
        low_conf = conf.get('<60', {})

        if high_conf.get('rate', 0) < 55:
            recommendations.append({
                'type': 'confidence_calibration',
                'message': f"High confidence picks (80+) only hitting {high_conf.get('rate')}%. Model may be overconfident.",
                'action': 'Lower confidence scores by 10-15%'
            })

        if low_conf.get('rate', 0) > 50 and low_conf.get('picks', 0) > 5:
            recommendations.append({
                'type': 'confidence_calibration',
                'message': f"Low confidence picks (<60) hitting {low_conf.get('rate')}%. Model may be underconfident.",
                'action': 'Consider raising confidence threshold'
            })

    # Check stat type performance
    stats = analysis.get('by_stat', {})
    for stat, data in stats.items():
        if data.get('picks', 0) >= 5:
            if data.get('rate', 0) < 40:
                recommendations.append({
                    'type': 'stat_filter',
                    'message': f"{stat} predictions only hitting {data.get('rate')}%",
                    'action': f"Increase minimum edge threshold for {stat} props"
                })
            elif data.get('rate', 0) > 60:
                recommendations.append({
                    'type': 'stat_boost',
                    'message': f"{stat} predictions hitting {data.get('rate')}%!",
                    'action': f"Prioritize {stat} props in picks"
                })

    # Check edge performance
    edges = analysis.get('by_edge', {})
    high_edge = edges.get('30%+', {})
    low_edge = edges.get('<10%', {})

    if high_edge.get('picks', 0) > 3 and high_edge.get('rate', 0) < 50:
        recommendations.append({
            'type': 'edge_calibration',
            'message': f"High edge (30%+) picks only hitting {high_edge.get('rate')}%",
            'action': 'Edge calculation may be inflated - review projection accuracy'
        })

    # Check prediction types
    types = analysis.get('by_type', {})
    for ptype, data in types.items():
        if data.get('picks', 0) >= 5 and data.get('rate', 0) < 40:
            recommendations.append({
                'type': 'type_filter',
                'message': f"{ptype} predictions only hitting {data.get('rate')}%",
                'action': f"Consider filtering out {ptype} or raising threshold"
            })

    return recommendations


def calculate_calibration_adjustments():
    """
    Calculate specific adjustments to improve model calibration.

    Returns confidence multipliers and edge adjustments based on actual performance.
    """
    predictions = load_graded_predictions()

    if len(predictions) < 20:
        return {'message': 'Need at least 20 graded predictions for calibration'}

    df = pd.DataFrame(predictions)

    adjustments = {
        'confidence_multiplier': 1.0,
        'edge_adjustments': {},
        'stat_multipliers': {},
        'min_edge_by_type': {}
    }

    # Calculate confidence calibration
    if 'confidence' in df.columns:
        # Group by confidence deciles
        df['conf_bin'] = pd.qcut(df['confidence'], q=4, duplicates='drop')
        calibration = df.groupby('conf_bin')['hit'].mean()

        # If high confidence picks aren't hitting proportionally higher, adjust
        if len(calibration) >= 2:
            expected_diff = 0.2  # High conf should hit ~20% more than low
            actual_diff = calibration.iloc[-1] - calibration.iloc[0]

            if actual_diff < expected_diff * 0.5:
                adjustments['confidence_multiplier'] = 0.85  # Reduce confidence
            elif actual_diff > expected_diff * 1.5:
                adjustments['confidence_multiplier'] = 1.1  # Boost confidence

    # Calculate stat-specific adjustments
    if 'stat' in df.columns:
        overall_rate = df['hit'].mean()
        stat_rates = df.groupby('stat')['hit'].mean()

        for stat, rate in stat_rates.items():
            if df[df['stat'] == stat].shape[0] >= 5:
                # Multiplier based on relative performance
                adjustments['stat_multipliers'][stat] = round(rate / overall_rate, 2) if overall_rate > 0 else 1.0

    # Calculate minimum edge by type
    if 'type' in df.columns and 'edge' in df.columns:
        for ptype in df['type'].unique():
            type_df = df[df['type'] == ptype]
            if len(type_df) >= 5:
                # Find edge threshold where hit rate > 50%
                type_df = type_df.copy()
                type_df['edge_abs'] = type_df['edge'].abs()
                type_df_sorted = type_df.sort_values('edge_abs')

                # Find breakeven edge
                cumsum = type_df_sorted['hit'].expanding().mean()
                above_50 = cumsum[cumsum >= 0.5]

                if len(above_50) > 0:
                    idx = above_50.index[0]
                    min_edge = type_df_sorted.loc[idx, 'edge_abs']
                    adjustments['min_edge_by_type'][ptype] = round(min_edge, 1)

    return adjustments


def save_feedback():
    """Save performance analysis and adjustments to file."""
    analysis = analyze_performance()
    adjustments = calculate_calibration_adjustments()

    feedback = {
        'generated_at': datetime.now().isoformat(),
        'analysis': analysis,
        'adjustments': adjustments
    }

    with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
        json.dump(feedback, f, indent=2)

    return feedback


def get_feedback():
    """Load saved feedback or generate new."""
    if FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return save_feedback()


if __name__ == "__main__":
    print("=== Model Performance Analysis ===\n")

    analysis = analyze_performance()

    print(f"Total Graded: {analysis.get('total_graded', 0)}")
    print(f"Overall Hit Rate: {analysis.get('overall_hit_rate', 0)}%\n")

    print("By Confidence:")
    for tier, data in analysis.get('by_confidence', {}).items():
        print(f"  {tier}: {data['hits']}/{data['picks']} = {data['rate']}%")

    print("\nBy Stat Type:")
    for stat, data in analysis.get('by_stat', {}).items():
        print(f"  {stat}: {data['hits']}/{data['picks']} = {data['rate']}%")

    print("\nBy Prediction Type:")
    for ptype, data in analysis.get('by_type', {}).items():
        print(f"  {ptype}: {data['hits']}/{data['picks']} = {data['rate']}%")

    print("\nBy Edge:")
    for edge, data in analysis.get('by_edge', {}).items():
        print(f"  {edge}: {data['hits']}/{data['picks']} = {data['rate']}%")

    print("\n=== Recommendations ===")
    for rec in analysis.get('recommendations', []):
        print(f"\n[{rec['type']}]")
        print(f"  {rec['message']}")
        print(f"  Action: {rec['action']}")

    print("\n=== Calibration Adjustments ===")
    adjustments = calculate_calibration_adjustments()
    print(f"Confidence Multiplier: {adjustments.get('confidence_multiplier', 1.0)}")
    print(f"Stat Multipliers: {adjustments.get('stat_multipliers', {})}")
    print(f"Min Edge by Type: {adjustments.get('min_edge_by_type', {})}")
