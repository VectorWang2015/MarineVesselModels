# ALOS Parameter Ablation Experiment Report

*Generated on: 2026-02-03 14:21:36*

## Experiment Setup

### Parameter Grid
- **Reset mechanisms**: hard_reset (α=0.0), soft_reset (α=0.5), no_reset (α=1.0)
- **Leakage values**: 0.0, 0.1, 0.01, 0.005, 0.003, 0.001
- **Total parameter combos**: 18

### Environment Scenarios
- **Current speeds**: 0.1, 0.2, 0.3, 0.4 m/s
- **Directions**: 0°, 30°, 90°, 150°, 180°, 270°
- **Force magnitude**: (current_speed / 0.1) × 5 N
- **Total scenarios**: 24

### Fixed Configuration
- **Clamp**: ON (β_max = 30°)
- **Conditional integration**: ON (ψ_err_threshold = 15°)
- **Path**: Zigzag [(0,0), (100,100), (0,200), (100,300), (0,400)]
- **Total simulations**: 432

### Scoring
For each environment scenario:
1. Normalize CTE RMSE and CTE P95 across parameter combos
2. Combined score: `0.7 × norm(CTE RMSE) + 0.3 × norm(CTE P95)`
3. Lower score is better

## Results

### Top 3 Parameter Combinations

| Rank | Reset Mode | Leakage | Mean Score | Std Score | Win Rate |
|------|------------|---------|------------|-----------|----------|
| 1 | soft_reset | 0.0030 | 0.2155 | 0.0983 | 0.00% |
| 2 | soft_reset | 0.0010 | 0.2228 | 0.1378 | 0.00% |
| 3 | soft_reset | 0.0000 | 0.2296 | 0.1398 | 0.00% |

### Optimal Combination

- **Reset mechanism**: soft_reset
- **Leakage**: 0.0030
- **Soft reset coefficient**: 0.5
- **Mean score**: 0.2155
- **Score std**: 0.0983
- **Win rate**: 0.00% (wins in 0 of 24 scenarios)

### All Parameter Combinations (Sorted)

| Reset Mode | Leakage | Mean Score | Std Score | Win Rate |
|------------|---------|------------|-----------|----------|
| soft_reset | 0.0030 | 0.2155 | 0.0983 | 0.00% |
| soft_reset | 0.0010 | 0.2228 | 0.1378 | 0.00% |
| soft_reset | 0.0000 | 0.2296 | 0.1398 | 0.00% |
| soft_reset | 0.0050 | 0.2319 | 0.0946 | 4.17% |
| hard_reset | 0.0010 | 0.2700 | 0.1994 | 4.17% |
| hard_reset | 0.0000 | 0.2735 | 0.2001 | 29.17% |
| hard_reset | 0.0030 | 0.2783 | 0.1981 | 8.33% |
| hard_reset | 0.0050 | 0.2883 | 0.1888 | 0.00% |
| no_reset | 0.0030 | 0.3118 | 0.3408 | 16.67% |
| no_reset | 0.0050 | 0.3120 | 0.2987 | 0.00% |
| ... 8 more combinations ... |

## Output Files

- `results_summary.csv`: Raw simulation results (432 rows)
- `analysis_results.json`: Analysis data
- `heatmap_*.png`: Performance heatmaps for each parameter combo
- `experiment_config.json`: Experiment configuration
- `report.md`: This report

## Failure Analysis

**24** out of 432 runs failed or did not complete.

Top failed configurations:

| Reset Mode | Leakage | Current | Force | Direction | Fail Reason |
|------------|---------|---------|-------|-----------|-------------|
| hard_reset | 0.1 | 0.4 m/s | 20.0 N | 30° | diverged |
| hard_reset | 0.1 | 0.4 m/s | 20.0 N | 150° | diverged |
| soft_reset | 0.0 | 0.4 m/s | 20.0 N | 30° | diverged |
| soft_reset | 0.0 | 0.4 m/s | 20.0 N | 150° | diverged |
| soft_reset | 0.1 | 0.4 m/s | 20.0 N | 30° | diverged |

To visualize these failed cases, run:
```bash
python demos/alos/demo_alos_single_scenario.py --reset 0.0 --leakage 0.1 --force 20.0 --current 0.4 --direction 30 --max-steps 300000
python demos/alos/demo_alos_single_scenario.py --reset 0.0 --leakage 0.1 --force 20.0 --current 0.4 --direction 150 --max-steps 300000
python demos/alos/demo_alos_single_scenario.py --reset 0.5 --leakage 0.0 --force 20.0 --current 0.4 --direction 30 --max-steps 300000
python demos/alos/demo_alos_single_scenario.py --reset 0.5 --leakage 0.0 --force 20.0 --current 0.4 --direction 150 --max-steps 300000
python demos/alos/demo_alos_single_scenario.py --reset 0.5 --leakage 0.1 --force 20.0 --current 0.4 --direction 30 --max-steps 300000
```

## Conclusion

The optimal ALOS parameter combination for the tested zigzag path under varying current+force disturbances is **soft_reset with leakage=0.0030**.
This combination achieved the lowest mean score (0.2155) with reasonable robustness (std=0.0983).
