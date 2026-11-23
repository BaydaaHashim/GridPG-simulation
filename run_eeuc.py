"""Add the EEUC multi-hop baseline to an existing campaign without re-running the other protocols.
Runs EEUC on exactly the experiments/seeds of run_all.tasks() used for BASE protocols, then replaces any
previous EEUC rows in results/summary.csv and appends EEUC curves to results/curves_main.npz.
(All protocols are seeded per run, so adding a protocol does not change the results of the others.)"""
import csv
import os
import sys
import time
from multiprocessing import Pool
import numpy as np
import run_all

OUT = run_all.OUT

if __name__ == '__main__':
    NAME = sys.argv[1] if len(sys.argv) > 1 else 'EEUC'      # usage: python run_eeuc.py [EEUC|GridPG-NJ]
    T = [t[:3] + (NAME,) + t[4:] for t in run_all.tasks() if t[3] == 'LEACH']
    t0 = time.time()
    with Pool(16) as P:
        res = P.map(run_all.job, T, chunksize=1)
    path = os.path.join(OUT, 'summary.csv')
    with open(path, newline='') as f:
        old = [r for r in csv.DictReader(f) if r['proto'] != NAME]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=run_all.FIELDS)
        w.writeheader()
        w.writerows(old)
        for row, _, _ in res:
            w.writerow(row)
    cz = dict(np.load(os.path.join(OUT, 'curves_main.npz')))
    keep = cz['names'] != NAME
    cur = [(row, c) for row, c, _ in res if c is not None]
    R = cz['alive'].shape[1]
    for key, i in (('alive', 0), ('energy', 1), ('jain', 2)):
        new = np.array([c[i][:R] for _, c in cur])
        cz[key] = np.concatenate([cz[key][keep], new])
    cz['names'] = np.concatenate([cz['names'][keep], np.array([r['proto'] for r, _ in cur])])
    cz['seeds'] = np.concatenate([cz['seeds'][keep], np.array([r['seed'] for r, _ in cur])])
    np.savez_compressed(os.path.join(OUT, 'curves_main.npz'), **cz)
    print(f'{len(T)} {NAME} runs in {time.time()-t0:.0f}s')

