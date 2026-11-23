"""Sensitivity to the congestion weight lambda (lambda = 0 is plain minimum-energy next-hop routing).
Same evaluation seeds 1-30 as the main scenario."""
import csv
import os
from multiprocessing import Pool
import numpy as np
from scipy import stats
import sim

LAMS = (0.0, 0.5, 1.0, 2.0, 4.0)


def job(a):
    lam, s = a
    r = sim.run('GridPG', s, dict(lam=lam))
    return lam, s, r['FND'], r['HND'], r['LND'], r['PDR'], r['energy_per_reading_mJ'], r['latency_ms']


if __name__ == '__main__':
    with Pool(16) as P:
        res = P.map(job, [(l, s) for l in LAMS for s in range(1, 31)])
    out = os.path.join(os.path.dirname(__file__), 'results', 'lambda_sensitivity.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['lam', 'seed', 'FND', 'HND', 'LND', 'PDR', 'energy_per_reading_mJ', 'latency_ms'])
        w.writerows(res)
    A = np.array(res)
    for lam in LAMS:
        X = A[A[:, 0] == lam]
        print(f"lambda={lam}: FND {X[:,2].mean():.1f}  HND {X[:,3].mean():.1f}  LND {X[:,4].mean():.1f}  "
              f"PDR {X[:,5].mean():.3f}  E/read {X[:,6].mean():.3f}")
    base = A[A[:, 0] == 1.0]; zero = A[A[:, 0] == 0.0]
    for k, name in ((2, 'FND'), (3, 'HND'), (4, 'LND')):
        p = stats.wilcoxon(base[:, k], zero[:, k]).pvalue
        print(f"lambda=1 vs 0 [{name}]: {100*(base[:,k].mean()/zero[:,k].mean()-1):+.1f}%  p={p:.2e}")
