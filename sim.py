"""
Discrete-round simulator for grid-based, energy-aware clustering in IoT/WSN.

Protocols implemented
  LEACH      - Heinzelman et al., distributed probabilistic CH election, single hop to BS
  LEACH-C    - centralized CH election (simulated annealing), single hop to BS
  S-LEACH    - sector-based LEACH: angular sectors around the BS, max-residual-energy CH per sector
  PSO-C      - centralized PSO clustering (Latiff et al.)
  GWO-GT     - proposed: grid cells + GWO joint CH selection + potential-game multi-hop routing
  ablations  - Grid-Greedy-GT (no GWO), Grid-GWO-Direct (no game-theoretic routing)

Energy model: first-order radio model (free space d^2 / multipath d^4, threshold d0).
All protocols share the same deployment, energy model, packet sizes and harvesting traces per seed.
"""
import math
import time
import numpy as np

EELEC = 50e-9          # J/bit
EFS = 10e-12           # J/bit/m^2
EMP = 0.0013e-12       # J/bit/m^4
EDA = 5e-9             # J/bit/signal
D0 = math.sqrt(EFS / EMP)

DEFAULT = dict(
    N=100, W=250.0, H=250.0, BSX=125.0, BSY=350.0, E0=0.5,
    L=6400, LC=64, p=0.1, RMAX=5000,
    rate=250e3,            # bit/s (IEEE 802.15.4)
    ell=750.0,             # link-reliability scale (m): PRR(d)=exp(-(d/ell)^2)
    der_frac=0.0, Hmax=0.75e-3, T_day=200,
    grid=3, lam=1.0, weights=(0.40, 0.25, 0.25, 0.10),
    wolves=12, iters=25,
    pso_particles=20, pso_iters=30, sa_iters=300,
    sectors=8,             # S-LEACH: 8 sectors as in Mohammed et al. (2022), Table 3
    eeuc_c=0.5, eeuc_R0=120.0, eeuc_T=0.2, eeuc_k=2.0, eeuc_TD=150.0,   # EEUC baseline (tuned on seeds 1000-1005)
)


def etx(bits, d):
    d = np.asarray(d, dtype=float)
    return bits * EELEC + np.where(d < D0, bits * EFS * d * d, bits * EMP * d ** 4)


def erx(bits):
    return bits * EELEC


def prr(d, ell):
    return np.exp(-(np.asarray(d, dtype=float) / ell) ** 2)


# --------------------------------------------------------------------------- network
class Net:
    def __init__(self, cfg, seed):
        rng = np.random.default_rng(seed)
        N = cfg['N']
        self.N = N
        self.pos = np.column_stack([rng.uniform(0, cfg['W'], N), rng.uniform(0, cfg['H'], N)])
        self.bs = np.array([cfg['BSX'], cfg['BSY']])
        self.dbs = np.hypot(self.pos[:, 0] - self.bs[0], self.pos[:, 1] - self.bs[1])
        dx = self.pos[:, None, 0] - self.pos[None, :, 0]
        dy = self.pos[:, None, 1] - self.pos[None, :, 1]
        self.D = np.hypot(dx, dy)
        self.E0 = cfg['E0']
        self.E = np.full(N, cfg['E0'])
        k = int(round(cfg['der_frac'] * N))
        self.der = np.zeros(N, bool)
        if k > 0:
            self.der[rng.choice(N, k, replace=False)] = True
        R, T = cfg['RMAX'], cfg['T_day']
        q = rng.uniform(0.5, 1.0, N)          # site/panel quality
        sig = rng.uniform(0.0, 0.6, N)        # cloud/volatility level
        phase = rng.uniform(0, 0.1 * T, N)
        r = np.arange(R)
        base = np.maximum(0.0, np.sin(2 * np.pi * (r[None, :] + phase[:, None]) / T))
        cloud = 1.0 - sig[:, None] * rng.random((N, R))
        self.harv = (cfg['Hmax'] * q[:, None] * base * cloud) * self.der[:, None]
        self.Hmax = cfg['Hmax']
        self.T = T

    def stability(self, r):
        """DER stability score S_i = w1*A_i + w2*R_i from the last T rounds of harvesting history."""
        S = np.zeros(self.N)
        if not self.der.any():
            return S
        lo = max(0, r - self.T)
        if r - lo < 5:
            S[self.der] = 0.5
            return S
        h = self.harv[self.der, lo:r]
        A = np.clip(h.mean(1) / (self.Hmax / math.pi), 0, 1)
        hp = np.where(h > 0, h, np.nan)
        with np.errstate(all='ignore'):
            cv = np.nanstd(hp, 1) / np.nanmean(hp, 1)
        Rl = np.clip(1 - np.nan_to_num(cv, nan=1.0), 0, 1)
        S[self.der] = 0.5 * A + 0.5 * Rl
        return S


def nearest(net, nodes, chs):
    """index (into chs) of nearest CH for each node in nodes"""
    return np.argmin(net.D[np.ix_(nodes, chs)], axis=1)


# --------------------------------------------------------------------------- protocols
class LEACH:
    name = 'LEACH'

    def __init__(self, cfg, net):
        self.p = cfg['p']
        self.last = np.full(net.N, -10 ** 9)

    def plan(self, net, r, alive, rng):
        p = self.p
        epoch = int(round(1 / p))
        G = alive & (r - self.last >= epoch)
        T = p / (1 - p * (r % epoch))
        chs = np.where(G & (rng.random(net.N) < T))[0]
        self.last[chs] = r
        if len(chs) == 0:
            chs = np.where(alive)[0]
        mem_of = np.full(net.N, -1)
        mem = np.setdiff1d(np.where(alive)[0], chs)
        if len(mem):
            mem_of[mem] = chs[nearest(net, mem, chs)]
        return chs, mem_of, np.full(len(chs), -1), dict(piggy=0, n_adv=len(chs), extra=None)


class SLEACH:
    name = 'S-LEACH'

    def __init__(self, cfg, net):
        th = np.arctan2(net.pos[:, 1] - net.bs[1], net.pos[:, 0] - net.bs[0])
        S = cfg['sectors']
        lo, hi = th.min(), th.max() + 1e-9
        self.sector = np.minimum(((th - lo) / (hi - lo) * S).astype(int), S - 1)
        self.S = S
        self.LC = cfg['LC']

    def plan(self, net, r, alive, rng):
        chs, extra = [], np.zeros(net.N)
        mem_of = np.full(net.N, -1)
        for s in range(self.S):
            idx = np.where(alive & (self.sector == s))[0]
            if len(idx) == 0:
                continue
            if r == 0:
                ch = rng.choice(idx)
            else:
                ch = idx[np.argmax(net.E[idx])]
            chs.append(ch)
            # every node broadcasts (energy, random number) inside its sector and hears its peers
            rad = net.D[np.ix_(idx, idx)].max(1)
            extra[idx] += etx(self.LC, rad) + (len(idx) - 1) * erx(self.LC)
            mem = idx[idx != ch]
            mem_of[mem] = ch
        chs = np.array(chs, dtype=int)
        return chs, mem_of, np.full(len(chs), -1), dict(piggy=0, n_adv=1, extra=extra)


def _assign_cost(net, alive_idx, chs):
    return (net.D[np.ix_(alive_idx, chs)].min(1) ** 2).sum()


class LEACHC:
    name = 'LEACH-C'

    def __init__(self, cfg, net):
        self.p = cfg['p']
        self.iters = cfg['sa_iters']
        self.LC = cfg['LC']

    def plan(self, net, r, alive, rng):
        a = np.where(alive)[0]
        elig = a[net.E[a] >= net.E[a].mean()]
        K = max(1, min(len(elig), int(round(self.p * len(a)))))
        cur = rng.choice(elig, K, replace=False)
        c_cost = _assign_cost(net, a, cur)
        best, b_cost = cur.copy(), c_cost
        T0 = c_cost * 0.05 + 1e-12
        for it in range(self.iters if len(elig) > K else 0):
            cand = cur.copy()
            out = rng.integers(K)
            pool = np.setdiff1d(elig, cur)
            cand[out] = rng.choice(pool)
            cc = _assign_cost(net, a, cand)
            Tt = T0 * (1 - it / self.iters) + 1e-12
            if cc < c_cost or rng.random() < math.exp(-(cc - c_cost) / Tt):
                cur, c_cost = cand, cc
                if cc < b_cost:
                    best, b_cost = cand.copy(), cc
        chs = np.sort(best)
        mem_of = np.full(net.N, -1)
        mem = np.setdiff1d(a, chs)
        if len(mem):
            mem_of[mem] = chs[nearest(net, mem, chs)]
        return chs, mem_of, np.full(len(chs), -1), dict(piggy=self.LC, n_adv=1, extra=None)


class PSOC:
    name = 'PSO-C'

    def __init__(self, cfg, net):
        self.p = cfg['p']
        self.P = cfg['pso_particles']
        self.T = cfg['pso_iters']
        self.LC = cfg['LC']
        self.W, self.H = cfg['W'], cfg['H']
        self.diag = math.hypot(cfg['W'], cfg['H'])

    def _decode(self, net, elig, X):
        # X: (K,2) coordinates -> nearest eligible nodes (distinct)
        d = np.hypot(X[:, None, 0] - net.pos[None, elig, 0], X[:, None, 1] - net.pos[None, elig, 1])
        chosen = []
        for k in range(len(X)):
            order = np.argsort(d[k])
            for o in order:
                if elig[o] not in chosen:
                    chosen.append(elig[o])
                    break
        return np.array(chosen)

    def _cost(self, net, a, chs, beta=0.5):
        dm = net.D[np.ix_(a, chs)]
        asg = dm.argmin(1)
        dmin = dm[np.arange(len(a)), asg]
        cnt = np.bincount(asg, minlength=len(chs))
        sm = np.bincount(asg, weights=dmin, minlength=len(chs))
        f1 = (sm / np.maximum(cnt, 1)).max() / self.diag
        f2 = net.E[a].sum() / net.E[chs].sum() / (len(a) / len(chs))
        return beta * f1 + (1 - beta) * f2

    def plan(self, net, r, alive, rng):
        a = np.where(alive)[0]
        elig = a[net.E[a] >= net.E[a].mean()]
        K = max(1, min(len(elig), int(round(self.p * len(a)))))
        lo, hi = np.array([0, 0]), np.array([self.W, self.H])
        X = rng.uniform(lo, hi, (self.P, K, 2))
        V = np.zeros_like(X)
        pb = X.copy()
        pbc = np.array([self._cost(net, a, self._decode(net, elig, x)) for x in X])
        g = pb[pbc.argmin()].copy()
        gc = pbc.min()
        vmax = 0.2 * hi
        for t in range(self.T):
            w = 0.9 - 0.5 * t / self.T
            r1, r2 = rng.random(X.shape), rng.random(X.shape)
            V = np.clip(w * V + 2 * r1 * (pb - X) + 2 * r2 * (g - X), -vmax, vmax)
            X = np.clip(X + V, lo, hi)
            for i in range(self.P):
                c = self._cost(net, a, self._decode(net, elig, X[i]))
                if c < pbc[i]:
                    pbc[i], pb[i] = c, X[i].copy()
                    if c < gc:
                        gc, g = c, X[i].copy()
        chs = np.sort(self._decode(net, elig, g))
        mem_of = np.full(net.N, -1)
        mem = np.setdiff1d(a, chs)
        if len(mem):
            mem_of[mem] = chs[nearest(net, mem, chs)]
        return chs, mem_of, np.full(len(chs), -1), dict(piggy=self.LC, n_adv=1, extra=None)


def gt_route(net, chs, cfg, rng, lam):
    """Best-response dynamics of the next-hop congestion (exact potential) game among CHs.
    Returns next hop (node id, -1 = BS) and number of BR sweeps until a pure Nash equilibrium."""
    n = len(chs)
    b = cfg['L'] + cfg['LC']
    E = np.maximum(net.E[chs], 1e-12)
    dbs = net.dbs[chs]
    D = net.D[np.ix_(chs, chs)]
    a_bs = etx(b, dbs) / E
    a = etx(b, D) / E[:, None]
    w = (erx(b) + etx(b, dbs)) / E
    allowed = dbs[None, :] < dbs[:, None]
    choice = np.full(n, -1)
    cnt = np.zeros(n)
    sweeps = 0
    for sweeps in range(1, 201):
        changed = False
        for i in rng.permutation(n):
            old = choice[i]
            if old >= 0:
                cnt[old] -= 1
            costs = np.where(allowed[i], a[i] + lam * w * (cnt + 1), np.inf)
            j = int(np.argmin(costs)) if n else -1
            best = -1 if (n == 0 or a_bs[i] <= costs[j]) else j
            choice[i] = best
            if best >= 0:
                cnt[best] += 1
            if best != old:
                changed = True
        if not changed:
            break
    nh = np.where(choice >= 0, chs[np.maximum(choice, 0)], -1)
    return nh, sweeps


def k_opt(cfg, net):
    """Analytic optimal number of clusters (Heinzelman et al., 2002), multipath BS links."""
    d2 = float(np.mean(net.dbs ** 2))
    M = math.sqrt(cfg['W'] * cfg['H'])
    return math.sqrt(net.N / (2 * math.pi)) * math.sqrt(EFS / EMP) * M / d2


def grid_size_rule(cfg, net):
    """m = ceil(sqrt(k_opt)), at least 2, so that the m x m grid offers >= k_opt cells."""
    return max(2, int(math.ceil(math.sqrt(k_opt(cfg, net)))))


def unequal_cells(net, part):
    """Static unequal partition: len(part) equal-width bands of distance to the BS (nearest band first);
    band b is split into part[b] segments of equal angular width (as seen from the BS) over its nodes."""
    d = net.dbs
    B = len(part)
    edges = np.linspace(d.min(), d.max() + 1e-9, B + 1)
    band = np.clip(np.searchsorted(edges, d, side='right') - 1, 0, B - 1)
    ang = np.arctan2(net.pos[:, 1] - net.bs[1], net.pos[:, 0] - net.bs[0])
    cell = np.zeros(net.N, dtype=int)
    off = 0
    for b in range(B):
        idx = np.where(band == b)[0]
        s = part[b]
        if len(idx):
            a = ang[idx]
            # unwrap around the band's angular range (robust when the BS lies inside the field)
            order = np.sort(a)
            gaps = np.diff(np.concatenate([order, order[:1] + 2 * np.pi]))
            start = order[(np.argmax(gaps) + 1) % len(order)]
            rel = (a - start) % (2 * np.pi)
            span = rel.max() + 1e-9
            cell[idx] = off + np.minimum((rel / span * s).astype(int), s - 1)
        off += s
    return cell


class GridGWOGT:
    name = 'GWO-GT'

    def __init__(self, cfg, net, use_gwo=True, use_gt=True):
        self.cfg = cfg
        self.use_gwo, self.use_gt = use_gwo, use_gt
        if not use_gwo:
            self.name = 'Grid-Greedy-GT'
        if not use_gt:
            self.name = 'Grid-GWO-Direct'
        part = cfg.get('partition')
        if part is not None:
            # unequal cells: distance bands around the BS (nearest first), each split into part[b] angular segments
            self.cell = unequal_cells(net, part)
            self.m_used = None
            m = int(np.ceil(np.sqrt(self.cell.max() + 1)))
            self.Gx = self.Gy = cfg['W'] / m
            ncell = self.cell.max() + 1
            cen = np.array([net.pos[self.cell == c].mean(0) if np.any(self.cell == c) else net.bs
                            for c in range(ncell)])
            self.ref = etx(cfg['L'] + cfg['LC'], np.hypot(cen[:, 0] - net.bs[0], cen[:, 1] - net.bs[1]))
        else:
            m = cfg['grid']
            if m == 'auto':
                m = grid_size_rule(cfg, net)
            self.m_used = m
            self.Gx, self.Gy = cfg['W'] / m, cfg['H'] / m
            cx = np.minimum((net.pos[:, 0] // self.Gx).astype(int), m - 1)
            cy = np.minimum((net.pos[:, 1] // self.Gy).astype(int), m - 1)
            self.cell = cx * m + cy
            cxs = (np.arange(m * m) // m + 0.5) * self.Gx
            cys = (np.arange(m * m) % m + 0.5) * self.Gy
            self.ref = etx(cfg['L'] + cfg['LC'], np.hypot(cxs - net.bs[0], cys - net.bs[1]))
        self.m = m
        self.nd2 = (self.Gx ** 2 + self.Gy ** 2) / 6.0
        self.b = cfg['L'] + cfg['LC']
        self.gwo_time = []
        self.gt_sweeps = []
        self.curves = {}

    # -- fitness, vectorised over wolves -------------------------------------
    def _prepare(self, net, r, alive):
        S = net.stability(r)
        cells = np.unique(self.cell[alive])
        b, LC, L = self.b, self.cfg['LC'], self.cfg['L']
        cands, fM, fS, base, Ec, refdrain = [], [], [], [], [], []
        for c in cells:
            idx = np.where(alive & (self.cell == c))[0]
            idx = idx[np.argsort(-net.E[idx])]
            cands.append(idx)
            n = len(idx)
            Dc = net.D[np.ix_(idx, idx)]
            # energy spent by the members of this cell if candidate k is the CH
            fM.append((etx(b, Dc) + etx(LC, Dc)).sum(1) - (etx(b, 0) + etx(LC, 0)))
            fS.append(1 - S[idx])
            # CH intra-cluster cost (reception, aggregation, advertisement): candidate independent
            base.append((n - 1) * (erx(b) + erx(LC)) + EDA * L * n + etx(LC, Dc.max(1)))
            Ec.append(net.E[idx])
            refdrain.append((base[-1].mean() + self.ref[c]) / net.E[idx].mean())
        C = len(cells)
        nmax = max(len(x) for x in cands)
        pad = lambda arr, v: np.array([np.pad(x, (0, nmax - len(x)), constant_values=v) for x in arr])
        self.P_ids = pad(cands, -1)
        self.P_fM, self.P_fS = pad(fM, 0.), pad(fS, 0.)
        self.P_base, self.P_E = pad(base, 0.), pad(Ec, 1.)
        self.n_c = np.array([len(x) for x in cands])
        self.cells = cells
        self.refdrain = float(np.mean(refdrain))
        mref = np.array([fm.mean() for fm in fM])
        self.refsum = self.ref[cells].sum() + np.array([bb.mean() for bb in base]).sum() + mref.sum()
        return C

    def _decode(self, X):
        k = np.minimum((X * self.n_c[None, :]).astype(int), self.n_c[None, :] - 1)
        return k

    def _fitness(self, net, K):
        """F = a*f_bottleneck + b*f_mean_drain + g*f_energy + d*f_stability   (lower is better)"""
        al, be, ga, de = self.cfg['weights']
        Wn, C = K.shape
        cidx = np.arange(C)[None, :]
        ch = self.P_ids[cidx, K]                      # (W,C) node ids
        E = self.P_E[cidx, K]
        own = self.P_base[cidx, K]
        mem = self.P_fM[cidx, K]
        fS = self.P_fS[cidx, K].mean(1)
        dbs = net.dbs[ch]
        direct = etx(self.b, dbs)
        relay = np.zeros((Wn, C))
        if self.use_gt and C > 1:
            D = net.D[ch[:, :, None], ch[:, None, :]]
            two = np.where(dbs[:, None, :] < dbs[:, :, None], etx(self.b, D), np.inf)
            two_tot = two + erx(self.b) + direct[:, None, :]
            j = two_tot.argmin(2)
            use = two_tot[np.arange(Wn)[:, None], cidx, j] < direct
            tx = np.where(use, two[np.arange(Wn)[:, None], cidx, j], direct)
            add = np.where(use, erx(self.b) + direct[np.arange(Wn)[:, None], j], 0.0)
            np.add.at(relay, (np.repeat(np.arange(Wn), C), j.ravel()), add.ravel())
        else:
            tx = direct
        drain = (own + tx + relay) / E / self.refdrain
        f_bot = drain.max(1)
        f_mean = drain.mean(1)
        f_en = (own + tx + relay + mem).sum(1) / self.refsum
        return al * f_bot + be * f_mean + ga * f_en + de * fS

    def _gwo(self, net, rng, C, record=None):
        Wn, T = self.cfg['wolves'], self.cfg['iters']
        X = rng.random((Wn, C))
        X[0] = 0.5 / self.n_c                      # warm start: greedy max-energy CH per cell
        F = self._fitness(net, self._decode(X))
        curve = []
        for t in range(T):
            o = np.argsort(F)
            Xa, Xb, Xd = X[o[0]], X[o[1]], X[o[2]]
            curve.append(F[o[0]])
            a = 2 - 2 * t / T
            newX = np.zeros_like(X)
            for L in (Xa, Xb, Xd):
                A = 2 * a * rng.random(X.shape) - a
                Cc = 2 * rng.random(X.shape)
                newX += L - A * np.abs(Cc * L - X)
            Xn = np.clip(newX / 3, 0, 1 - 1e-9)
            Fn = self._fitness(net, self._decode(Xn))
            keep = Fn < F                            # greedy replacement keeps the population elite
            X[keep], F[keep] = Xn[keep], Fn[keep]
        curve.append(F.min())
        if record is not None:
            self.curves[record] = np.array(curve)
        return X[np.argmin(F)]

    def plan(self, net, r, alive, rng):
        C = self._prepare(net, r, alive)
        t0 = time.perf_counter()
        if self.use_gwo:
            rec = r if r in (0, 300, 600) else None
            x = self._gwo(net, rng, C, record=rec)
            k = self._decode(x[None, :])[0]
        else:
            k = np.zeros(C, dtype=int)                   # max residual energy in cell
        self.gwo_time.append(time.perf_counter() - t0)
        chs = self.P_ids[np.arange(C), k]
        mem_of = np.full(net.N, -1)
        if self.cfg.get('join_nearest'):
            mem = np.setdiff1d(np.where(alive)[0], chs)
            if len(mem):
                mem_of[mem] = chs[nearest(net, mem, chs)]
        else:
            for c_i, c in enumerate(self.cells):
                idx = np.where(alive & (self.cell == c))[0]
                mem_of[idx[idx != chs[c_i]]] = chs[c_i]
        if self.use_gt:
            nh, sw = gt_route(net, chs, self.cfg, rng, self.cfg['lam'])
            self.gt_sweeps.append(sw)
        else:
            nh = np.full(len(chs), -1)
        return chs, mem_of, nh, dict(piggy=self.cfg['LC'], n_adv=1, extra=None)


class EEUC:
    """Energy-efficient unequal clustering (Li, Ye, Chen, Wu, IEEE MASS 2005), multi-hop baseline.
    Tentative CHs (probability T) compete with radius Rc = (1 - c (dmax - d_iBS)/(dmax - dmin)) R0; the
    tentative CH with the most residual energy wins inside max(Rc_i, Rc_j). Members join the nearest CH.
    A CH farther than TD from the BS relays through the adjacent CH (within k*Rc, closer to the BS) that has
    more residual energy among the two with the smallest d(i,j)^2 + d(j,BS)^2; otherwise it sends directly."""
    name = 'EEUC'

    def __init__(self, cfg, net):
        self.c, self.R0, self.T = cfg['eeuc_c'], cfg['eeuc_R0'], cfg['eeuc_T']
        self.k, self.TD = cfg['eeuc_k'], cfg['eeuc_TD']
        self.LC = cfg['LC']

    def plan(self, net, r, alive, rng):
        a = np.where(alive)[0]
        d = net.dbs[a]
        span = max(d.max() - d.min(), 1e-9)
        Rc = np.zeros(net.N)
        Rc[a] = (1 - self.c * (d.max() - d) / span) * self.R0
        extra = np.zeros(net.N)
        tent = a[rng.random(len(a)) < self.T]
        if len(tent) == 0:
            tent = a[[np.argmax(net.E[a])]]
        for t in tent:                                   # COMPETE_HEAD_MSG within Rc
            extra[t] += etx(self.LC, Rc[t])
            nb = a[(net.D[t, a] <= Rc[t]) & (a != t)]
            extra[nb] += erx(self.LC)
        chs = []
        for t in tent[np.argsort(-net.E[tent], kind='stable')]:
            if all(net.D[t, c] > max(Rc[t], Rc[c]) for c in chs):
                chs.append(t)
        chs = np.array(chs, dtype=int)
        mem_of = np.full(net.N, -1)
        mem = np.setdiff1d(a, chs)
        if len(mem):
            mem_of[mem] = chs[nearest(net, mem, chs)]
        # inter-cluster routing: CHs broadcast (energy, distance to BS) within k*Rc
        rng_k = self.k * Rc[chs]
        Dcc = net.D[np.ix_(chs, chs)]
        extra[chs] += etx(self.LC, rng_k)
        extra[chs] += ((Dcc <= rng_k[:, None]) & ~np.eye(len(chs), dtype=bool)).sum(0) * erx(self.LC)
        dch, Ech = net.dbs[chs], net.E[chs]
        nh = np.full(len(chs), -1)
        for i in range(len(chs)):
            if dch[i] <= self.TD:
                continue
            cand = np.where((dch < dch[i]) & (Dcc[i] <= rng_k[i]))[0]
            if len(cand) == 0:
                continue
            cost = Dcc[i, cand] ** 2 + dch[cand] ** 2
            two = cand[np.argsort(cost, kind='stable')[:2]]
            nh[i] = chs[two[np.argmax(Ech[two])]]
        return chs, mem_of, nh, dict(piggy=0, n_adv=1, extra=extra)


PROTOCOLS = {
    'EEUC': lambda cfg, net: EEUC(cfg, net),
    'LEACH': lambda cfg, net: LEACH(cfg, net),
    'LEACH-C': lambda cfg, net: LEACHC(cfg, net),
    'S-LEACH': lambda cfg, net: SLEACH(cfg, net),
    'PSO-C': lambda cfg, net: PSOC(cfg, net),
    'GWO-GT': lambda cfg, net: GridGWOGT(cfg, net),
    'Grid-Greedy-GT': lambda cfg, net: GridGWOGT(cfg, net, use_gwo=False),
    'Grid-GWO-Direct': lambda cfg, net: GridGWOGT(cfg, net, use_gt=False),
    # final method (approved framing): analytically sized grid + max-residual-energy CH + potential-game routing
    'GridPG': lambda cfg, net: GridGWOGT(dict(cfg, grid='auto'), net, use_gwo=False),
    'GridPG-GWO': lambda cfg, net: GridGWOGT(dict(cfg, grid='auto'), net, use_gwo=True),
    'GridPG-Direct': lambda cfg, net: GridGWOGT(dict(cfg, grid='auto'), net, use_gwo=False, use_gt=False),
    'GridPG-3x3': lambda cfg, net: GridGWOGT(dict(cfg, grid=3), net, use_gwo=False),
    # variant selected on tuning seeds (tune_unequal.py): cell-wise CH election, members join the nearest CH
    'GridPG-NJ': lambda cfg, net: GridGWOGT(dict(cfg, grid='auto', join_nearest=True), net, use_gwo=False),
}


# --------------------------------------------------------------------------- energy accounting
def account(net, cfg, alive, chs, mem_of, nh, piggy, n_adv, extra):
    N, L, LC, ell = net.N, cfg['L'], cfg['LC'], cfg['ell']
    cons = np.zeros(N) if extra is None else extra.copy()
    bm = L + piggy
    members = np.where(alive & (mem_of >= 0))[0]
    chm = mem_of[members]
    dm = net.D[members, chm]
    cons[members] += etx(bm, dm) + etx(LC, dm) + n_adv * erx(LC)
    idx = np.full(N, -1)
    idx[chs] = np.arange(len(chs))
    ci = idx[chm]
    nC = len(chs)
    nmem = np.bincount(ci, minlength=nC).astype(float)
    rad = np.zeros(nC)
    np.maximum.at(rad, ci, dm)
    cons[chs] += etx(LC, rad) + nmem * (erx(bm) + erx(LC)) + EDA * L * (nmem + 1)
    nhi = np.where(nh >= 0, idx[np.maximum(nh, 0)], -1)
    order = np.argsort(-net.dbs[chs])               # farthest from BS first
    load = np.ones(nC)
    for k in order:
        if nhi[k] >= 0:
            load[nhi[k]] += load[k]
    dnext = np.where(nhi >= 0, net.D[chs, np.maximum(nh, 0)], net.dbs[chs])
    cons[chs] += (load - 1) * erx(bm) + load * etx(bm, dnext)
    ok = alive & (cons <= net.E + 1e-15)
    pathp = np.zeros(nC)
    hops = np.zeros(nC)
    tpkt = bm / cfg['rate']
    lat = np.zeros(nC)
    for k in order[::-1]:                            # closest to BS first
        pk = prr(dnext[k], ell) * ok[chs[k]]
        w = (load[k] + 1) / 2 * tpkt
        if nhi[k] >= 0:
            pk *= pathp[nhi[k]]
            hops[k] = 1 + hops[nhi[k]]
            lat[k] = w + lat[nhi[k]]
        else:
            hops[k] = 1
            lat[k] = w
        pathp[k] = pk
    m_ok = prr(dm, ell) * ok[members]
    dcl = pathp * (1 + np.bincount(ci, weights=m_ok, minlength=nC))
    clat = nmem * tpkt + lat
    return dict(cons=cons, ok=ok, deliv=dcl.sum(), gen=float(alive.sum()),
                lat=(dcl * clat).sum(), hops=(dcl * hops).sum(), nch=nC)


# --------------------------------------------------------------------------- run
def run(proto, seed, overrides=None, snapshot_round=None):
    cfg = dict(DEFAULT)
    if overrides:
        cfg.update(overrides)
    net = Net(cfg, seed)
    P = PROTOCOLS[proto](cfg, net)
    rng = np.random.default_rng(seed * 7919 + 17)
    R = cfg['RMAX']
    alive_h = np.zeros(R, np.int16)
    en_h = np.zeros(R, np.float32)
    jain_h = np.zeros(R, np.float32)
    death = np.full(net.N, R)
    tot = dict(deliv=0., gen=0., lat=0., hops=0., nch=0.)
    consumed = 0.
    snap = None
    last = R
    for r in range(R):
        alive = net.E > 0
        na = alive.sum()
        alive_h[r] = na
        en_h[r] = net.E.sum()
        e = net.E
        jain_h[r] = e.sum() ** 2 / (net.N * (e ** 2).sum() + 1e-30)
        if na == 0:
            last = r
            break
        chs, mem_of, nh, ctl = P.plan(net, r, alive, rng)
        res = account(net, cfg, alive, chs, mem_of, nh, ctl['piggy'], ctl['n_adv'], ctl['extra'])
        if snapshot_round is not None and r == snapshot_round:
            snap = dict(pos=net.pos.copy(), chs=chs.copy(), mem_of=mem_of.copy(), nh=nh.copy(),
                        bs=net.bs.copy(), E=net.E.copy(), m=getattr(P, 'm_used', None))
        for k in tot:
            tot[k] += res[k]
        used = np.where(res['ok'], res['cons'], net.E * alive)
        consumed += used.sum()
        net.E = np.where(res['ok'], net.E - res['cons'], 0.0)
        net.E = np.maximum(net.E, 0.0)
        live = net.E > 0
        if net.der.any():
            net.E = np.where(live, np.minimum(net.E0, net.E + net.harv[:, r]), net.E)
        newly = alive & ~live
        death[newly] = r + 1
    if last < R:
        alive_h[last:] = 0
        en_h[last:] = 0
        jain_h[last:] = 0
    ds = np.sort(death)
    out = dict(
        proto=proto, seed=seed,
        FND=int(ds[0]), HND=int(ds[net.N // 2 - 1]), LND=int(ds[-1]),
        delivered=tot['deliv'], generated=tot['gen'],
        PDR=tot['deliv'] / max(tot['gen'], 1),
        latency_ms=1000 * tot['lat'] / max(tot['deliv'], 1e-9),
        hops=tot['hops'] / max(tot['deliv'], 1e-9),
        energy_per_reading_mJ=1000 * consumed / max(tot['deliv'], 1e-9),
        alive=alive_h, energy=en_h, jain=jain_h,
    )
    if isinstance(P, GridGWOGT):
        out['opt_time_ms'] = 1000 * float(np.mean(P.gwo_time)) if P.gwo_time else 0.
        out['gt_sweeps'] = float(np.mean(P.gt_sweeps)) if P.gt_sweeps else 0.
        out['curves'] = P.curves
    if snap is not None:
        out['snap'] = snap
    return out


if __name__ == '__main__':
    for p in PROTOCOLS:
        t = time.time()
        o = run(p, 1)
        print(f"{p:16s} FND={o['FND']:5d} HND={o['HND']:5d} LND={o['LND']:5d} PDR={o['PDR']:.3f} "
              f"lat={o['latency_ms']:.0f}ms hops={o['hops']:.2f} E/read={o['energy_per_reading_mJ']:.3f} "
              f"t={time.time()-t:.1f}s")

