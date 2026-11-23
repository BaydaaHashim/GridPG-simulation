"""Pilot study (held-out seeds 1000-1005): does GWO add value over greedy max-energy CH selection?"""
import collections
from multiprocessing import Pool
import numpy as np
import sim


def job(a):
    p, s, ov = a
    r = sim.run(p, s, ov)
    return p, str(ov), [r['FND'], r['HND'], r['LND'], r['PDR'], r['energy_per_reading_mJ'], r['latency_ms']]


if __name__ == '__main__':
    seeds = range(1000, 1006)
    tasks = []
    for g in (3, 4, 5):
        for w in ((0.4, 0.2, 0.3, 0.1), (0.1, 0.1, 0.7, 0.1)):
            tasks += [('GWO-GT', s, dict(grid=g, weights=w)) for s in seeds]
        tasks += [('Grid-Greedy-GT', s, dict(grid=g)) for s in seeds]
    tasks += [('GWO-GT', s, dict(grid=3, wolves=20, iters=50)) for s in seeds]
    with Pool(16) as P:
        res = P.map(job, tasks)
    d = collections.defaultdict(list)
    for p, ov, v in res:
        d[(p, ov)].append(v)
    print('protocol | config | FND HND LND PDR E/reading(mJ) latency(ms)')
    for k, v in d.items():
        print(k[0], k[1], np.round(np.mean(v, 0), 3))
