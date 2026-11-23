# GridPG simulator

This folder contains the Python simulator behind the paper *Potential-Game Multi-Hop Routing over Grid-Based Clusters for Energy-Balanced IoT Sensing in Smart-Grid Monitoring*.

## Requirements
Python 3.10+ with:

```
pip install numpy scipy pandas matplotlib tabulate seaborn openpyxl
```

## Files

| File | What it does |
|---|---|
| `sim.py` | **The simulator.** It contains the energy model (`etx`, `erx`), the network and harvesting model (`Net`), the protocols (`LEACH`, `LEACHC`, `SLEACH`, `PSOC`, `GridGWOGT`), the grid rule (`k_opt`, `grid_size_rule`), potential-game routing (`gt_route`), per-round energy accounting (`account`) and one full run (`run`). |
| `run_all.py` | Full experiment campaign: the main scenario with 30 seeds, plus sensitivity runs over density, DER, grid size and BS position. Writes `results/summary.csv` and `results/curves_main.npz`. |
| `analyze.py` | Computes the statistics (means, 95% CIs, Wilcoxon tests with Holm correction) and draws all paper figures into `figures/`. |
| `run_lambda.py` | Sensitivity to the congestion weight λ. Writes `results/lambda_sensitivity.csv`. |
| `validate_grid_rule.py` | Validates the k_opt grid rule on held-out seeds 501–510. |
| `pilot.py`, `pilot_der.py` | Early pilot runs on tuning seeds 1000–1005. |
| `user_analysis/analysis_user.py` | Author's analysis script: Excel summary and extra plots. |

## Main settings
All defaults are in `DEFAULT` at the top of `sim.py`: N, field size, BS position, E0, packet sizes, radio constants, grid size, λ, GWO and PSO parameters. Any of them can be overridden per run:

```python
import sim
r = sim.run('GridPG', seed=1, overrides=dict(N=200, lam=2.0))
print(r['FND'], r['HND'], r['LND'], r['PDR'])
```

Protocol names accepted by `sim.run` are LEACH, LEACH-C, S-LEACH, PSO-C, GridPG, GridPG-GWO, GridPG-Direct, GridPG-3x3, Grid-Greedy-GT, GWO-GT and Grid-GWO-Direct.

## Reproduce the paper
```
python run_all.py          # takes about 12 min on 16 cores
python run_lambda.py
python validate_grid_rule.py
python analyze.py          # writes the tables and figures
```

## Seeds
| Use | Seeds |
|---|---|
| Tuning | 1000–1005 |
| Grid-rule validation | 501–510 |
| Evaluation | 1–30 |

## Multi-hop baseline (EEUC), added after the internal audit

- `sim.py` class `EEUC`: energy-efficient unequal clustering (Li et al., IEEE MASS 2005) with multi-hop relaying.
- `tune_eeuc.py`: tunes EEUC on the tuning seeds 1000-1005 only (39 settings) -> `results/eeuc_tuning.csv`.
  Chosen: R0 = 120 m, c = 0.5, T = 0.2, relay range 2*Rc, direct threshold 150 m.
- `run_eeuc.py`: runs EEUC on exactly the evaluation seeds/experiments of `run_all.py` and merges it into
  `results/summary.csv` and `results/curves_main.npz` (other protocols are unaffected: every run is seeded).
- Re-run order: `python run_all.py`, `python run_eeuc.py`, `python run_lambda.py`, `python analyze.py`.

## GridPG-NJ variant (nearest-CH membership)

- `tune_unequal.py`: 19 candidate variants (distance-banded unequal partitions, with and without nearest-CH
  membership) evaluated on tuning seeds 1000-1005 only, with a selection rule fixed before running
  (highest HND among candidates whose FND >= 95% of GridPG) -> `results/unequal_tuning.csv`.
- Selected: `GridPG-NJ` (analytic grid for CH election + routing game; members join the nearest CH).
- Evaluated once on the evaluation seeds: `python run_eeuc.py GridPG-NJ`, then `python analyze.py`.
