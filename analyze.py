"""Statistics and figures from results/summary.csv and results/curves_main.npz."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

ORDER = ['LEACH', 'LEACH-C', 'S-LEACH', 'PSO-C', 'EEUC', 'GridPG-GWO', 'GridPG', 'GridPG-NJ', 'GridPG-Direct', 'GridPG-3x3']
COLOR = {'LEACH': '#2a78d6', 'LEACH-C': '#eb6834', 'S-LEACH': '#1baf7a', 'PSO-C': '#eda100', 'EEUC': '#8a5a2b',
         'GridPG-GWO': '#e87ba4', 'GridPG': '#008300', 'GridPG-NJ': '#5fb05f', 'GridPG-Direct': '#6250d6', 'GridPG-3x3': '#e34948',
         'Grid-Greedy-GT': '#008300'}
MARK = {'LEACH': 'o', 'LEACH-C': 's', 'S-LEACH': '^', 'PSO-C': 'D', 'EEUC': 'h', 'GridPG-GWO': 'v', 'GridPG': '*', 'GridPG-NJ': 'p',
        'GridPG-Direct': 'x', 'GridPG-3x3': 'P', 'Grid-Greedy-GT': '*'}
LS = {'LEACH': ':', 'LEACH-C': '--', 'S-LEACH': '-.', 'PSO-C': (0, (5, 1, 1, 1)), 'EEUC': (0, (2, 2)), 'GridPG-GWO': (0, (3, 1)),
      'GridPG': '-', 'GridPG-NJ': (0, (4, 1)), 'GridPG-Direct': (0, (1, 1)), 'GridPG-3x3': (0, (6, 2)), 'Grid-Greedy-GT': '-'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e1e0d9'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 8,
    'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 6.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.edgecolor': '#898781', 'axes.linewidth': 0.6, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.5, 'savefig.dpi': 600, 'savefig.bbox': 'tight', 'lines.linewidth': 1.2,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
})
W1 = 3.5   # IEEE single column (in)


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def holm(pvals):
    p = np.asarray(pvals)
    o = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1, run)
    return adj


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.png'))
    fig.savefig(os.path.join(FIG, name + '.pdf'))
    plt.close(fig)


df = pd.read_csv(os.path.join(RES, 'summary.csv'))
metrics = ['FND', 'HND', 'LND', 'PDR', 'latency_ms', 'hops', 'energy_per_reading_mJ']
lines = []

# ---------------------------------------------------------------- main table + tests
main = df[df.experiment == 'main']
rows = []
for p in ORDER:
    d = main[main.proto == p]
    r = {'Protocol': p}
    for m in metrics:
        r[m] = f"{d[m].mean():.3f} ± {ci95(d[m]):.3f}" if m == 'PDR' else f"{d[m].mean():.1f} ± {ci95(d[m]):.1f}"
    rows.append(r)
tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(RES, 'table_main.csv'), index=False)
lines.append('## Main scenario (N=100, battery-only, 30 seeds): mean ± 95% CI\n')
lines.append(tab.to_markdown(index=False))

lines.append('\n\n## Paired Wilcoxon signed-rank tests (by seed), Holm-corrected within each reference\n')
for ref in ('GridPG',):
    others = [p for p in ORDER if p != ref]
    for m in ('FND', 'HND', 'LND', 'energy_per_reading_mJ', 'PDR'):
        a = main[main.proto == ref].sort_values('seed')[m].values
        ps, eff = [], []
        for o in others:
            b = main[main.proto == o].sort_values('seed')[m].values
            dlt = a - b
            p = 1.0 if np.allclose(dlt, 0) else stats.wilcoxon(a, b).pvalue
            ps.append(p)
            eff.append(100 * (a.mean() - b.mean()) / b.mean())
        adj = holm(ps)
        for o, e, pa in zip(others, eff, adj):
            lines.append(f"- {ref} vs {o} [{m}]: {e:+.1f}% , Holm p = {pa:.2e}")
lines.append(f"\nGWO optimisation time per round: {main[main.proto=='GridPG-GWO'].opt_time_ms.mean():.2f} ms; "
             f"best-response sweeps to NE: {main[main.proto=='GridPG'].gt_sweeps.mean():.2f} (max over seeds "
             f"{main[main.proto=='GridPG'].gt_sweeps.max():.2f})")

# ---------------------------------------------------------------- sensitivity tables
for exp, par in (('density', 'N'), ('der', 'der_frac'), ('grid', 'grid'), ('bs_center', 'BS')):
    d = df[df.experiment == exp]
    if exp in ('der', 'density', 'bs_center'):
        d = pd.concat([d, main[main.proto.isin(d.proto.unique())].assign(
            value={'density': 100, 'der': 0.0, 'bs_center': '(125,350)'}[exp])])
    g = d.groupby(['value', 'proto'])[['FND', 'HND', 'LND', 'PDR', 'energy_per_reading_mJ']].mean().round(3)
    g.to_csv(os.path.join(RES, f'table_{exp}.csv'))
    lines.append(f"\n\n## Sensitivity: {exp} ({par})\n")
    lines.append(g.to_markdown())

with open(os.path.join(RES, 'stats_report.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

# ---------------------------------------------------------------- figures
C = np.load(os.path.join(RES, 'curves_main.npz'))
names = C['names']


def curve_fig(key, ylabel, fname, scale=1.0, xmax=None):
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    for p in ORDER:
        A = C[key][names == p].astype(float) * scale
        mu = A.mean(0)
        last = np.nonzero(mu > 0)[0]
        end = (last[-1] + 2) if len(last) else 10
        x = np.arange(end)
        ax.plot(x, mu[:end], color=COLOR[p], ls=LS[p], lw=1.6 if p == 'GridPG' else 1.1, label=p,
                marker=MARK[p], markevery=max(1, end // 6), ms=4, mfc='white' if p != 'GridPG' else COLOR[p])
    ax.set_xlabel('Round')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, xmax or ax.get_xlim()[1])
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=1, loc='center left', bbox_to_anchor=(1.01, 0.5))
    save(fig, fname)


curve_fig('alive', 'Alive nodes (mean of 30 runs)', 'fig_alive_nodes', xmax=550)
curve_fig('energy', 'Residual network energy (J)', 'fig_residual_energy', xmax=550)

# bar charts FND/HND/LND
fig, ax = plt.subplots(figsize=(W1, 2.3))
bw = 0.11
for i, p in enumerate(ORDER):
    d = main[main.proto == p]
    for j, m in enumerate(('FND', 'HND', 'LND')):
        ax.bar(j + (i - (len(ORDER) - 1) / 2) * bw, d[m].mean(), bw * 0.9, yerr=ci95(d[m]), color=COLOR[p],
               label=p if j == 0 else None, error_kw=dict(lw=0.6, capsize=1.2, ecolor=INK),
               hatch='////' if p == 'GridPG' else None, edgecolor='white', linewidth=0.3)
ax.set_xticks(range(3), ['FND', 'HND', 'LND'])
ax.set_ylabel('Round (mean ± 95% CI)')
ax.legend(frameon=False, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=6)
ax.grid(axis='x', visible=False)
save(fig, 'fig_lifetime_bars')


# metric bars (one panel per metric, no dual axes)
def metric_bar(m, ylabel, fname, fmt):
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    for i, p in enumerate(ORDER):
        d = main[main.proto == p][m]
        ax.bar(i, d.mean(), 0.7, yerr=ci95(d), color=COLOR[p], edgecolor='white', linewidth=0.3,
               error_kw=dict(lw=0.6, capsize=1.5, ecolor=INK), hatch='////' if p == 'GridPG' else None)
        ax.text(i, d.mean() + ci95(d), fmt.format(d.mean()), ha='center', va='bottom', fontsize=6, color=MUTED)
    ax.set_xticks(range(len(ORDER)), ORDER, rotation=30, ha='right')
    ax.set_ylabel(ylabel)
    ax.grid(axis='x', visible=False)
    save(fig, fname)


metric_bar('energy_per_reading_mJ', 'Energy per delivered reading (mJ)', 'fig_energy_per_reading', '{:.2f}')
metric_bar('PDR', 'Packet delivery ratio', 'fig_pdr', '{:.3f}')
metric_bar('latency_ms', 'Mean end-to-end delay (ms)', 'fig_latency', '{:.0f}')


# sensitivity line plots
def sens_fig(exp, xvals, xlabel, metric, fname, base_value):
    d = df[df.experiment == exp]
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    for p in [q for q in ORDER + ['Grid-Greedy-GT'] if q in d.proto.unique()]:
        ys, es = [], []
        for v in xvals:
            if v == base_value:
                s = main[main.proto == p][metric]
            else:
                s = d[(d.proto == p) & (d.value.astype(float) == float(v))][metric]
            ys.append(s.mean()); es.append(ci95(s))
        ax.errorbar(xvals, ys, yerr=es, color=COLOR[p], ls=LS[p], marker=MARK[p], ms=4, capsize=1.5, lw=1.1,
                    mfc='white' if p != 'GridPG' else COLOR[p], label=p)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f'{metric} (round)')
    ax.legend(frameon=False, ncol=1, fontsize=6, loc='center left', bbox_to_anchor=(1.01, 0.5))
    save(fig, fname)


sens_fig('density', [50, 100, 150, 200], 'Number of nodes N', 'HND', 'fig_sens_density_hnd', 100)
sens_fig('density', [50, 100, 150, 200], 'Number of nodes N', 'FND', 'fig_sens_density_fnd', 100)
sens_fig('der', [0.0, 0.25, 0.5, 0.75], 'Fraction of DER-assisted nodes', 'FND', 'fig_sens_der_fnd', 0.0)
sens_fig('der', [0.0, 0.25, 0.5, 0.75], 'Fraction of DER-assisted nodes', 'HND', 'fig_sens_der_hnd', 0.0)
sens_fig('grid', [2, 3, 4, 5], 'Grid size m (m x m cells)', 'HND', 'fig_sens_grid_hnd', None)

# GWO convergence + deployment snapshot
ex = np.load(os.path.join(RES, 'extras.npy'), allow_pickle=True).item()
if 'GridPG-GWO_curves' in ex:
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    for i, (r, c) in enumerate(sorted(ex['GridPG-GWO_curves'].items())):
        ax.plot(np.arange(len(c)), c, color=['#2a78d6', '#eb6834', '#1baf7a'][i % 3], marker='os^'[i % 3],
                ms=3, markevery=5, label=f'round {r}')
    ax.set_xlabel('GWO iteration')
    ax.set_ylabel('Best fitness F')
    ax.legend(frameon=False)
    save(fig, 'fig_gwo_convergence')
for key, fname in (('GridPG_snap', 'fig_deployment_snapshot'), ('GridPG-3x3_snap', 'fig_deployment_snapshot_3x3')):
    if key not in ex:
        continue
    s = ex[key]
    m = s.get('m') or (3 if '3x3' in key else 2)
    fig, ax = plt.subplots(figsize=(W1, 3.6))
    pos, chs, mem, nh, bs = s['pos'], s['chs'], s['mem_of'], s['nh'], s['bs']
    for k in range(1, m):
        ax.axvline(250 * k / m, color='#c3c2b7', lw=0.6, ls='--')
        ax.axhline(250 * k / m, color='#c3c2b7', lw=0.6, ls='--')
    for i in np.where(mem >= 0)[0]:
        ax.plot([pos[i, 0], pos[mem[i], 0]], [pos[i, 1], pos[mem[i], 1]], color='#c3c2b7', lw=0.4, zorder=1)
    for c, n in zip(chs, nh):
        tgt = bs if n < 0 else pos[n]
        ax.annotate('', xy=tgt, xytext=pos[c], arrowprops=dict(arrowstyle='->', color='#2a78d6', lw=1.0), zorder=2)
    ax.scatter(pos[:, 0], pos[:, 1], s=8, c='#898781', zorder=3, label='Sensor node')
    ax.scatter(pos[chs, 0], pos[chs, 1], s=40, marker='s', c='#008300', edgecolor='white', lw=0.6, zorder=4,
               label='Cluster head')
    ax.scatter([bs[0]], [bs[1]], s=90, marker='^', c='#eb6834', edgecolor='white', zorder=5, label='Base station')
    ax.plot([], [], color='#2a78d6', lw=1.0, label='CH next hop (Nash eq.)')
    ax.set_xlim(-5, 255); ax.set_ylim(-5, 365)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.legend(frameon=False, loc='upper left', fontsize=6)
    ax.grid(False)
    save(fig, fname)

print(open(os.path.join(RES, 'stats_report.md'), encoding='utf-8').read())


