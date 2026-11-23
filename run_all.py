"""Full experiment campaign. Evaluation seeds 1-30 (main) / 1-10 (sensitivity); tuning used 1000-1005 only.
Outputs: results/summary.csv (one row per run), results/curves_main.npz (alive/energy vs rounds, main scenario)."""
import csv
import os
import time
from multiprocessing import Pool
import numpy as np
import sim

OUT = os.path.join(os.path.dirname(__file__), 'results')
BASE = ['LEACH', 'LEACH-C', 'S-LEACH', 'PSO-C', 'GridPG', 'GridPG-GWO']
ALL = BASE + ['GridPG-Direct', 'GridPG-3x3']
FIELDS = ['experiment', 'param', 'value', 'proto', 'seed', 'FND', 'HND', 'LND', 'PDR', 'latency_ms',
          'hops', 'energy_per_reading_mJ', 'delivered', 'opt_time_ms', 'gt_sweeps']


def tasks():
    t = []
    for p in ALL:
        for s in range(1, 31):
            t.append(('main', 'none', 0, p, s, {}))
    for n in (50, 150, 200):
        for p in BASE:
            for s in range(1, 11):
                t.append(('density', 'N', n, p, s, dict(N=n)))
    for f in (0.25, 0.5, 0.75):
        for p in BASE:
            for s in range(1, 11):
                t.append(('der', 'der_frac', f, p, s, dict(der_frac=f)))
    for g in (2, 3, 4, 5):
        for s in range(1, 11):
            t.append(('grid', 'grid', g, 'Grid-Greedy-GT', s, dict(grid=g)))
    for p in BASE:
        for s in range(1, 11):
            t.append(('bs_center', 'BS', '(125,125)', p, s, dict(BSY=125.0)))
    return t


def job(a):
    exp, par, val, p, s, ov = a
    snap = 20 if (exp == 'main' and s == 1 and p in ('GridPG', 'GridPG-3x3')) else None
    r = sim.run(p, s, ov, snapshot_round=snap)
    row = dict(experiment=exp, param=par, value=val, proto=p, seed=s)
    for k in FIELDS[5:]:
        row[k] = r.get(k, '')
    curves = (r['alive'], r['energy'], r['jain']) if exp == 'main' else None
    extra = {k: r[k] for k in ('snap', 'curves') if k in r and exp == 'main' and s == 1}
    return row, curves, extra


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    T = tasks()
    t0 = time.time()
    with Pool(16) as P:
        res = P.map(job, T, chunksize=1)
    with open(os.path.join(OUT, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row, _, _ in res:
            w.writerow(row)
    alive, energy, jain, names, seeds = [], [], [], [], []
    extras = {}
    for row, c, ex in res:
        if c is not None:
            alive.append(c[0]); energy.append(c[1]); jain.append(c[2])
            names.append(row['proto']); seeds.append(row['seed'])
        if ex:
            for k, v in ex.items():
                extras[f"{row['proto']}_{k}"] = v
    np.savez_compressed(os.path.join(OUT, 'curves_main.npz'), alive=np.array(alive), energy=np.array(energy),
                        jain=np.array(jain), names=np.array(names), seeds=np.array(seeds))
    np.save(os.path.join(OUT, 'extras.npy'), extras, allow_pickle=True)
    print(f'{len(T)} runs done in {time.time()-t0:.0f}s')
