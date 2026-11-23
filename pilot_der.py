"""Pilot (held-out seeds 1000-1005): GWO vs greedy CH selection when part of the nodes are DER-assisted."""
import collections
from multiprocessing import Pool
import numpy as np
import sim


def job(a):
    p, s, ov = a
    r = sim.run(p, s, ov)
    return p, ov['der_frac'], [r['FND'], r['HND'], r['LND'], r['PDR'], r['energy_per_reading_mJ']]


if __name__ == '__main__':
    seeds = range(1000, 1006)
    tasks = [(p, s, dict(der_frac=f)) for f in (0.25, 0.5, 0.75)
             for p in ('GWO-GT', 'Grid-Greedy-GT', 'PSO-C', 'S-LEACH') for s in seeds]
    with Pool(16) as P:
        res = P.map(job, tasks)
    d = collections.defaultdict(list)
    for p, f, v in res:
        d[(f, p)].append(v)
    print('DER | protocol | FND HND LND PDR E/reading')
    for k in sorted(d):
        print(k[0], k[1], np.round(np.mean(d[k], 0), 3))
