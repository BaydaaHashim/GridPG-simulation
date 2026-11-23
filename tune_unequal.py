"""Select an unequal-cell partition for GridPG on the TUNING seeds 1000-1005 only.
Pre-declared selection rule (fixed before running): among the candidate partitions, choose the one with the
highest mean HND whose mean FND is at least 95% of the FND of the analytic 2x2 GridPG on the same seeds.
Partition = list of segment counts per distance band from the BS, nearest band first.
Output: results/unequal_tuning.csv"""
import csv
import os
from multiprocessing import Pool
import numpy as np
import sim

OUT = os.path.join(os.path.dirname(__file__), 'results')
CANDS = [(1,), (1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (3, 2), (1, 1, 1), (2, 1, 1), (2, 2, 1), (3, 2, 1),
         (2, 2, 2), (3, 3, 2),
         # stage 2 (after stage 1 found no admissible gain): nearest-CH membership, as in EEUC
         ('near', None), ('near', (2, 1)), ('near', (3, 1)), ('near', (2, 1, 1)), ('near', (2, 2, 1)),
         ('near', (3, 2))]
SEEDS = range(1000, 1006)


def job(a):
    cand, s = a
    if cand == 'GridPG':
        r = sim.run('GridPG', s)
    elif cand == 'EEUC':
        r = sim.run('EEUC', s)
    elif cand[0] == 'near':                      # analytic m x m grid, members join the nearest CH
        r = sim.run('GridPG', s, dict(join_nearest=True, **({'partition': list(cand[1])} if cand[1] else {})))
    else:
        r = sim.run('Grid-Greedy-GT', s, dict(partition=list(cand)))
    return dict(cand=str(cand), seed=s, FND=r['FND'], HND=r['HND'], LND=r['LND'], PDR=r['PDR'],
                energy_per_reading_mJ=r['energy_per_reading_mJ'], latency_ms=r['latency_ms'])


if __name__ == '__main__':
    tasks = [(c, s) for c in ['GridPG', 'EEUC'] + CANDS for s in SEEDS]
    with Pool(16) as P:
        rows = P.map(job, tasks, chunksize=1)
    with open(os.path.join(OUT, 'unequal_tuning.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    agg = {}
    for r in rows:
        agg.setdefault(r['cand'], []).append([r[k] for k in ('FND', 'HND', 'LND', 'PDR', 'energy_per_reading_mJ',
                                                             'latency_ms')])
    ref_fnd = np.mean([x[0] for x in agg['GridPG']])
    print('cand            FND    HND    LND    PDR   mJ   delay')
    for k, v in sorted(agg.items(), key=lambda kv: -np.mean([x[1] for x in kv[1]])):
        v = np.array(v, float).mean(0)
        ok = 'ok' if v[0] >= 0.95 * ref_fnd else '--'
        print(f'{k:14s} {v[0]:6.1f} {v[1]:6.1f} {v[2]:6.1f} {v[3]:.3f} {v[4]:.3f} {v[5]:5.0f} {ok}')
