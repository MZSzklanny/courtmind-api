"""
CourtMind End-of-Game Comeback Simulation
==========================================
Monte Carlo simulation to determine optimal 3-point shooting aggressiveness
when trailing in end-of-game scenarios.

Scenario: Down 20 points with 10 minutes remaining
Exit criteria: Cut deficit to 4 points or less, OR less than 90 seconds remaining
"""

import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

from CourtMind.config import OUTPUT_DIR, COLORS

# =============================================================================
# SCENARIO CONFIGURATION
# =============================================================================
STARTING_DEFICIT = 20          # Points behind
STARTING_TIME = 600            # Seconds (10 minutes)
EXIT_TIME_THRESHOLD = 90       # Exit if < 90 seconds remain
EXIT_DEFICIT_THRESHOLD = 4     # Success if deficit <= 4 points

# =============================================================================
# POSSESSION PARAMETERS
# =============================================================================
AVG_POSSESSION_LENGTH = 15     # Seconds per possession
POSSESSION_STD = 3             # Standard deviation for possession length
MIN_POSSESSION_LENGTH = 8      # Minimum possession length
MAX_POSSESSION_LENGTH = 24     # Maximum possession length
TURNOVER_RATE = 0.13           # 13% of possessions end in turnover

# =============================================================================
# SHOT DISTRIBUTION BASELINE
# =============================================================================
BASELINE_3P_RATE = 0.37
BASELINE_RIM_RATE = 0.38
BASELINE_MID_RATE = 0.25

# =============================================================================
# FIELD GOAL PERCENTAGES
# =============================================================================
RIM_FG_PCT = 0.65              # 65% at rim
MID_FG_PCT = 0.48              # 48% midrange
THREE_FG_PCT = 0.36            # 36% 3P (from 55% TS)

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
N_SIMULATIONS = 10000
THREE_RATE_RANGE = np.arange(0.37, 0.76, 0.05)


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class ShotDistribution:
    """Defines shot selection strategy."""
    three_rate: float
    rim_rate: float
    mid_rate: float

    def __post_init__(self):
        total = self.three_rate + self.rim_rate + self.mid_rate
        assert abs(total - 1.0) < 0.01, f"Shot distribution must sum to 1.0, got {total}"


@dataclass
class PossessionResult:
    """Result of a single possession."""
    points_scored: int
    is_turnover: bool
    shot_type: str
    time_elapsed: float


@dataclass
class SimulationResult:
    """Result of a single simulation run."""
    success: bool
    final_deficit: int
    time_remaining: float
    exit_reason: str
    trailing_possessions: int
    leading_possessions: int
    trailing_points: int
    leading_points: int


# =============================================================================
# CORE SIMULATION FUNCTIONS
# =============================================================================
def simulate_possession(
    distribution: ShotDistribution,
    rng: np.random.Generator
) -> PossessionResult:
    """
    Simulate a single possession.

    Args:
        distribution: Shot selection percentages
        rng: NumPy random generator for reproducibility

    Returns:
        PossessionResult with points, shot type, and time used
    """
    # Possession length varies around average
    time_elapsed = np.clip(
        rng.normal(AVG_POSSESSION_LENGTH, POSSESSION_STD),
        MIN_POSSESSION_LENGTH,
        MAX_POSSESSION_LENGTH
    )

    # Check for turnover
    if rng.random() < TURNOVER_RATE:
        return PossessionResult(
            points_scored=0,
            is_turnover=True,
            shot_type='turnover',
            time_elapsed=time_elapsed
        )

    # Select shot type based on distribution
    shot_roll = rng.random()
    if shot_roll < distribution.three_rate:
        shot_type = 'three'
        make_prob = THREE_FG_PCT
        points_if_made = 3
    elif shot_roll < distribution.three_rate + distribution.rim_rate:
        shot_type = 'rim'
        make_prob = RIM_FG_PCT
        points_if_made = 2
    else:
        shot_type = 'mid'
        make_prob = MID_FG_PCT
        points_if_made = 2

    # Determine if shot is made
    points = points_if_made if rng.random() < make_prob else 0

    return PossessionResult(
        points_scored=points,
        is_turnover=False,
        shot_type=shot_type,
        time_elapsed=time_elapsed
    )


def simulate_game(
    trailing_distribution: ShotDistribution,
    leading_distribution: ShotDistribution,
    starting_deficit: int = STARTING_DEFICIT,
    starting_time: float = STARTING_TIME,
    seed: int = None
) -> SimulationResult:
    """
    Simulate one end-of-game scenario.

    Args:
        trailing_distribution: Shot distribution for trailing team
        leading_distribution: Shot distribution for leading team
        starting_deficit: Initial point deficit
        starting_time: Initial time remaining (seconds)
        seed: Random seed for reproducibility

    Returns:
        SimulationResult with outcome details
    """
    rng = np.random.default_rng(seed)

    deficit = starting_deficit
    time_remaining = starting_time
    trailing_possessions = 0
    leading_possessions = 0
    trailing_points = 0
    leading_points = 0

    trailing_turn = True  # Trailing team possesses first

    while time_remaining >= EXIT_TIME_THRESHOLD:
        # Check success condition
        if deficit <= EXIT_DEFICIT_THRESHOLD:
            return SimulationResult(
                success=True,
                final_deficit=deficit,
                time_remaining=time_remaining,
                exit_reason='success',
                trailing_possessions=trailing_possessions,
                leading_possessions=leading_possessions,
                trailing_points=trailing_points,
                leading_points=leading_points
            )

        # Early exit if blowout (deficit > 35)
        if deficit > 35:
            return SimulationResult(
                success=False,
                final_deficit=deficit,
                time_remaining=time_remaining,
                exit_reason='blowout',
                trailing_possessions=trailing_possessions,
                leading_possessions=leading_possessions,
                trailing_points=trailing_points,
                leading_points=leading_points
            )

        # Simulate possession
        if trailing_turn:
            result = simulate_possession(trailing_distribution, rng)
            trailing_possessions += 1
            trailing_points += result.points_scored
            deficit -= result.points_scored  # Scoring reduces deficit
        else:
            result = simulate_possession(leading_distribution, rng)
            leading_possessions += 1
            leading_points += result.points_scored
            deficit += result.points_scored  # Opponent scoring increases deficit

        time_remaining -= result.time_elapsed
        trailing_turn = not trailing_turn

    # Time threshold reached
    return SimulationResult(
        success=deficit <= EXIT_DEFICIT_THRESHOLD,
        final_deficit=deficit,
        time_remaining=time_remaining,
        exit_reason='time_threshold',
        trailing_possessions=trailing_possessions,
        leading_possessions=leading_possessions,
        trailing_points=trailing_points,
        leading_points=leading_points
    )


def create_distribution(three_rate: float) -> ShotDistribution:
    """
    Create shot distribution for a given 3P rate.
    Proportionally adjusts rim and mid rates.
    """
    if three_rate > BASELINE_3P_RATE:
        remaining = 1.0 - three_rate
        baseline_remaining = BASELINE_RIM_RATE + BASELINE_MID_RATE
        rim_rate = BASELINE_RIM_RATE * (remaining / baseline_remaining)
        mid_rate = BASELINE_MID_RATE * (remaining / baseline_remaining)
    else:
        rim_rate = BASELINE_RIM_RATE
        mid_rate = BASELINE_MID_RATE

    return ShotDistribution(
        three_rate=three_rate,
        rim_rate=rim_rate,
        mid_rate=mid_rate
    )


def run_strategy_simulations(
    three_rate: float,
    n_simulations: int = N_SIMULATIONS,
    seed: int = 42
) -> dict:
    """
    Run multiple simulations for a given 3P rate strategy.

    Args:
        three_rate: 3P rate for trailing team (0.0 to 1.0)
        n_simulations: Number of simulation iterations
        seed: Random seed for reproducibility

    Returns:
        Dictionary with aggregated statistics
    """
    trailing_dist = create_distribution(three_rate)
    leading_dist = create_distribution(BASELINE_3P_RATE)

    results = []
    for i in range(n_simulations):
        result = simulate_game(
            trailing_distribution=trailing_dist,
            leading_distribution=leading_dist,
            seed=seed + i
        )
        results.append(result)

    # Aggregate statistics
    successes = [r for r in results if r.success]
    deficits = np.array([r.final_deficit for r in results])
    success_times = np.array([r.time_remaining for r in successes]) if successes else np.array([])

    return {
        'three_rate': three_rate,
        'n_simulations': n_simulations,
        'success_rate': len(successes) / n_simulations * 100,
        'success_count': len(successes),
        'avg_final_deficit': deficits.mean(),
        'deficit_std': deficits.std(),
        'deficit_median': np.median(deficits),
        'deficit_p10': np.percentile(deficits, 10),
        'deficit_p25': np.percentile(deficits, 25),
        'deficit_p75': np.percentile(deficits, 75),
        'deficit_p90': np.percentile(deficits, 90),
        'avg_time_remaining_success': success_times.mean() if len(success_times) > 0 else 0,
        'deficit_distribution': deficits,
    }


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def run_full_analysis(
    three_rates: List[float] = None,
    n_simulations: int = N_SIMULATIONS
) -> pd.DataFrame:
    """
    Run simulations across all 3P rates and compile results.
    """
    if three_rates is None:
        three_rates = THREE_RATE_RANGE.tolist()

    all_results = []
    for rate in three_rates:
        print(f"[Simulation] Running {n_simulations:,} sims at {rate*100:.0f}% 3P rate...")
        result = run_strategy_simulations(rate, n_simulations)
        all_results.append(result)

    # Create DataFrame without the large array column
    df_data = [{k: v for k, v in r.items() if k != 'deficit_distribution'} for r in all_results]
    return pd.DataFrame(df_data)


def calculate_optimal_rate(results_df: pd.DataFrame) -> dict:
    """
    Determine optimal 3P rate with statistical analysis.
    """
    optimal_idx = results_df['success_rate'].idxmax()
    optimal = results_df.loc[optimal_idx]
    baseline = results_df[results_df['three_rate'] == BASELINE_3P_RATE].iloc[0]

    return {
        'optimal_rate': optimal['three_rate'],
        'peak_success_rate': optimal['success_rate'],
        'baseline_success_rate': baseline['success_rate'],
        'improvement_pp': optimal['success_rate'] - baseline['success_rate'],
    }


def test_statistical_significance(
    results_df: pd.DataFrame,
    baseline_rate: float = BASELINE_3P_RATE
) -> pd.DataFrame:
    """
    Test whether higher 3P rates produce statistically significant improvements.
    Uses two-proportion z-test.
    """
    baseline = results_df[results_df['three_rate'] == baseline_rate].iloc[0]

    significance_tests = []
    for _, row in results_df.iterrows():
        if row['three_rate'] == baseline_rate:
            significance_tests.append({
                'three_rate': row['three_rate'],
                'success_rate': row['success_rate'],
                'vs_baseline_diff': 0,
                'z_statistic': 0,
                'p_value': 1.0,
                'significant_at_05': False,
                'significant_at_01': False,
            })
            continue

        # Two-proportion z-test
        p1 = row['success_rate'] / 100
        p2 = baseline['success_rate'] / 100
        n1 = n2 = row['n_simulations']

        pooled_p = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2)) if pooled_p > 0 else 0.001
        z_stat = (p1 - p2) / se
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))

        significance_tests.append({
            'three_rate': row['three_rate'],
            'success_rate': row['success_rate'],
            'vs_baseline_diff': row['success_rate'] - baseline['success_rate'],
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant_at_05': p_value < 0.05,
            'significant_at_01': p_value < 0.01,
        })

    return pd.DataFrame(significance_tests)


# =============================================================================
# VISUALIZATION
# =============================================================================
def create_success_rate_chart(
    results_df: pd.DataFrame,
    output_path: str = None
) -> str:
    """
    Create line chart showing success rate by 3P attempt rate.
    """
    plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    ax.set_facecolor(COLORS['background'])
    fig.patch.set_facecolor(COLORS['background'])

    # Plot success rate line
    x = results_df['three_rate'] * 100
    y = results_df['success_rate']

    ax.plot(x, y, color=COLORS['neon_green'], linewidth=3, marker='o', markersize=10)
    ax.fill_between(x, y, alpha=0.2, color=COLORS['neon_green'])

    # Highlight optimal point
    optimal_idx = results_df['success_rate'].idxmax()
    optimal = results_df.loc[optimal_idx]
    ax.scatter([optimal['three_rate'] * 100], [optimal['success_rate']],
               s=300, color=COLORS['accent_neutral'], zorder=5, marker='*')

    # Baseline marker
    baseline_result = results_df[results_df['three_rate'] == BASELINE_3P_RATE].iloc[0]
    ax.axvline(x=BASELINE_3P_RATE * 100, color=COLORS['text_secondary'], linestyle='--', alpha=0.7)
    ax.text(BASELINE_3P_RATE * 100 + 1, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 5,
            'Baseline', color=COLORS['text_secondary'], fontsize=11)

    # Labels
    ax.set_xlabel('3-Point Attempt Rate (%)', color=COLORS['text_primary'], fontsize=14)
    ax.set_ylabel('Comeback Success Rate (%)', color=COLORS['text_primary'], fontsize=14)
    ax.set_title('COMEBACK PROBABILITY BY 3P AGGRESSIVENESS\nDown 20 with 10 Minutes Left | Exit: ≤4 pts or <90 sec',
                 color=COLORS['accent_neutral'], fontsize=16, fontweight='bold', pad=20)

    # Style axes
    ax.tick_params(colors=COLORS['text_primary'], labelsize=11)
    ax.spines['bottom'].set_color(COLORS['accent_neutral'])
    ax.spines['left'].set_color(COLORS['accent_neutral'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, color=COLORS['text_secondary'])

    # Annotation for optimal
    ax.annotate(
        f'Optimal: {optimal["three_rate"]*100:.0f}%\n({optimal["success_rate"]:.2f}% success)',
        xy=(optimal['three_rate'] * 100, optimal['success_rate']),
        xytext=(optimal['three_rate'] * 100 + 8, optimal['success_rate'] + 0.5),
        fontsize=12, color=COLORS['accent_neutral'], fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=COLORS['accent_neutral'], lw=2)
    )

    # Add baseline annotation
    ax.annotate(
        f'Baseline: {baseline_result["success_rate"]:.2f}%',
        xy=(BASELINE_3P_RATE * 100, baseline_result['success_rate']),
        xytext=(BASELINE_3P_RATE * 100 - 10, baseline_result['success_rate'] - 0.5),
        fontsize=11, color=COLORS['text_secondary'],
        arrowprops=dict(arrowstyle='->', color=COLORS['text_secondary'], lw=1.5)
    )

    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'comeback_3p_analysis.png')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor=COLORS['background'], bbox_inches='tight')
    plt.close()

    return output_path


def create_deficit_distribution_chart(
    results_by_rate: dict,
    rates_to_compare: List[float] = [0.37, 0.52, 0.67],
    output_path: str = None
) -> str:
    """
    Create histogram comparison of final deficit distributions at different 3P rates.
    """
    plt.style.use('dark_background')

    fig, axes = plt.subplots(1, len(rates_to_compare), figsize=(14, 5), dpi=150)
    fig.patch.set_facecolor(COLORS['background'])

    colors_list = [COLORS['text_secondary'], COLORS['accent_neutral'], COLORS['neon_green']]

    for i, rate in enumerate(rates_to_compare):
        ax = axes[i]
        ax.set_facecolor(COLORS['background'])

        if rate in results_by_rate:
            deficits = results_by_rate[rate]['deficit_distribution']
            success_rate = results_by_rate[rate]['success_rate']

            ax.hist(deficits, bins=30, color=colors_list[i], alpha=0.7, edgecolor='white', linewidth=0.5)
            ax.axvline(x=EXIT_DEFICIT_THRESHOLD, color=COLORS['accent_negative'], linestyle='--',
                      linewidth=2, label='Success threshold')

            ax.set_title(f'{rate*100:.0f}% 3P Rate\n{success_rate:.2f}% Success',
                        color=colors_list[i], fontsize=13, fontweight='bold')
            ax.set_xlabel('Final Deficit', color=COLORS['text_primary'], fontsize=11)
            ax.set_ylabel('Frequency', color=COLORS['text_primary'], fontsize=11)
            ax.tick_params(colors=COLORS['text_primary'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    fig.suptitle('FINAL DEFICIT DISTRIBUTIONS BY STRATEGY',
                 color=COLORS['accent_neutral'], fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'deficit_distributions.png')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor=COLORS['background'], bbox_inches='tight')
    plt.close()

    return output_path


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def run_comeback_analysis(n_simulations: int = N_SIMULATIONS) -> dict:
    """
    Main entry point for the comeback simulation analysis.
    """
    print("=" * 70)
    print("CourtMind End-of-Game Comeback Simulation")
    print("=" * 70)
    print(f"Scenario: Down {STARTING_DEFICIT} points with {STARTING_TIME//60} minutes remaining")
    print(f"Success criteria: Cut deficit to {EXIT_DEFICIT_THRESHOLD} pts or less")
    print(f"Exit criteria: Time < {EXIT_TIME_THRESHOLD} seconds remaining")
    print(f"Simulations per strategy: {n_simulations:,}")
    print("=" * 70)
    print()

    # Store full results with distributions for charts
    three_rates = THREE_RATE_RANGE.tolist()
    results_by_rate = {}

    for rate in three_rates:
        print(f"[Simulation] Running {n_simulations:,} sims at {rate*100:.0f}% 3P rate...")
        result = run_strategy_simulations(rate, n_simulations)
        results_by_rate[rate] = result

    # Create DataFrame for analysis
    df_data = [{k: v for k, v in r.items() if k != 'deficit_distribution'} for r in results_by_rate.values()]
    results_df = pd.DataFrame(df_data)

    # Find optimal strategy
    optimal = calculate_optimal_rate(results_df)

    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Optimal 3P Rate: {optimal['optimal_rate']*100:.0f}%")
    print(f"Peak Success Rate: {optimal['peak_success_rate']:.2f}%")
    print(f"Baseline (37%) Success Rate: {optimal['baseline_success_rate']:.2f}%")
    print(f"Improvement vs Baseline: +{optimal['improvement_pp']:.2f} percentage points")
    print()

    # Statistical significance
    sig_df = test_statistical_significance(results_df)
    print("Statistical Significance (vs 37% baseline):")
    print("-" * 50)
    for _, row in sig_df.iterrows():
        if row['three_rate'] == BASELINE_3P_RATE:
            continue
        sig_marker = "***" if row['significant_at_01'] else ("**" if row['significant_at_05'] else "")
        print(f"  {row['three_rate']*100:.0f}%: {row['vs_baseline_diff']:+.2f}pp (p={row['p_value']:.4f}) {sig_marker}")

    print()
    print("=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    chart1 = create_success_rate_chart(results_df)
    print(f"Created: {chart1}")

    chart2 = create_deficit_distribution_chart(results_by_rate)
    print(f"Created: {chart2}")

    print()
    print("=" * 70)
    print("FULL RESULTS TABLE")
    print("=" * 70)
    print(results_df[['three_rate', 'success_rate', 'avg_final_deficit', 'deficit_median',
                      'deficit_p10', 'deficit_p90']].to_string(index=False))

    return {
        'results_df': results_df,
        'results_by_rate': results_by_rate,
        'optimal': optimal,
        'significance': sig_df,
        'charts': [chart1, chart2]
    }


if __name__ == "__main__":
    results = run_comeback_analysis()
