"""Tune the EEUC baseline on the tuning seeds 1000-1005 only (never on evaluation seeds 1-30).
Grid: R0 in {60, 90, 120} m, k in {2, 3}, TD in {d0, 150 m}, T in {0.2, 0.4}; criterion: mean HND.
Output: results/eeuc_tuning.csv"""
import csv
import itertools
import os
from multiprocessing import Pool
import numpy as np
import sim

OUT = os.path.join(os.path.dirname(__file__), 'results')
GRID = (list(itertools.product((60.0, 90.0, 120.0), (2.0, 3.0), (sim.D0, 150.0), (0.2, 0.4)))
        # stage 2: the stage-1 optimum lay on the grid edge (R0 = 120 m, k = 2, T = 0.2), so the grid was extended
        + list(itertools.product((150.0, 180.0, 240.0), (1.5, 2.0), (150.0,), (0.1, 0.2)))
        + list(itertools.product((120.0,), (1.5,), (150.0,), (0.1, 0.2)))
        + list(itertools.product((120.0,), (2.0,), (150.0,), (0.1,))))
SEEDS = range(1000, 1006)


def job(a):
    (R0, k, TD, T), s = a
    r = sim.run('EEUC', s, dict(eeuc_R0=R0, eeuc_k=k, eeuc_TD=TD, eeuc_T=T))
    return dict(R0=R0, k=k, TD=round(TD, 1), T=T, seed=s, FND=r['FND'], HND=r['HND'], LND=r['LND'],
                PDR=r['PDR'], energy_per_reading_mJ=r['energy_per_reading_mJ'])


if __name__ == '__main__':
    tasks = [(g, s) for g in GRID for s in SEEDS]
    with Pool(16) as P:
        rows = P.map(job, tasks, chunksize=1)
    with open(os.path.join(OUT, 'eeuc_tuning.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    agg = {}
    for r in rows:
        agg.setdefault((r['R0'], r['k'], r['TD'], r['T']), []).append((r['FND'], r['HND'], r['PDR']))
    ranked = sorted(agg.items(), key=lambda kv: -np.mean([x[1] for x in kv[1]]))
    for key, v in ranked:
        v = np.array(v)
        print(key, 'FND %.1f HND %.1f PDR %.3f' % tuple(v.mean(0)))
