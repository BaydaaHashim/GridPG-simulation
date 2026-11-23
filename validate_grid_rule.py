"""Validation of the k_opt grid-size rule on held-out seeds 501-510 (not used anywhere else).
For each scenario, run the greedy+potential-game scheme with fixed grids 2..5 and report
which grid is best and which grid the rule picks."""
import collections
from multiprocessing import Pool
import numpy as np
import sim

SCEN = {
    'far N=50': dict(N=50), 'far N=200': dict(N=200),
    'center N=50': dict(N=50, BSY=125.0), 'center N=100': dict(N=100, BSY=125.0),
    'center N=200': dict(N=200, BSY=125.0),
    'corner N=100': dict(N=100, BSX=0.0, BSY=0.0),
}


def job(a):
    name, ov, m, s = a
    r = sim.run('Grid-Greedy-GT', s, dict(ov, grid=m))
    return name, m, r['FND'], r['HND']


if __name__ == '__main__':
    T = [(n, ov, m, s) for n, ov in SCEN.items() for m in (2, 3, 4, 5) for s in range(501, 511)]
    with Pool(16) as P:
        res = P.map(job, T)
    d = collections.defaultdict(list)
    for n, m, f, h in res:
        d[(n, m)].append((f, h))
    for n, ov in SCEN.items():
        cfg = dict(sim.DEFAULT, **ov)
        net = sim.Net(cfg, 501)
        k = sim.k_opt(cfg, net)
        rule = sim.grid_size_rule(cfg, net)
        hnd = {m: np.mean([x[1] for x in d[(n, m)]]) for m in (2, 3, 4, 5)}
        fnd = {m: np.mean([x[0] for x in d[(n, m)]]) for m in (2, 3, 4, 5)}
        best = max(hnd, key=hnd.get)
        print(f"{n:14s} k_opt={k:5.2f} rule m={rule} best-HND m={best} | HND " +
              ' '.join(f"{m}:{hnd[m]:.0f}" for m in hnd) + ' | FND ' + ' '.join(f"{m}:{fnd[m]:.0f}" for m in fnd) +
              f" | rule/best HND = {hnd[rule]/hnd[best]:.3f}")
