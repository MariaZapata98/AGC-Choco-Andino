r"""
07d_plot_uncertainty.py
Figure of the uncertainty of the AGC totals (reads the tables of 07b, does NOT rerun the bootstrap).

  A. Forest AGC total per map date (selected scenario) with
       - 95 % CI bootstrap only (dark band)
       - 95 % CI bootstrap + systematic field error (light band)
     (the sensitivity to the 16 scenarios is reported in a table, not in the figure; set
     SHOW_SCENARIOS = True to draw them as thin lines).
  B. How much total forest AGC changed between consecutive dates and over 2000-2022 (Tg C),
     with its 95 % CI. Dark bar = the CI does not include zero (the change is larger than the
     uncertainty); white bar = the CI includes zero (the change cannot be distinguished from 0).

Mask: forest of each year (forest_own, FCD >= 30 %), as in the paper.
Inputs   RESULTADOS\07_incertidumbre\totals_by_year.csv, change.csv, scenario_sensitivity_totals.csv
Figure   FIGURAS\fig_agc_uncertainty_sensitivity.tif (300 dpi)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 11

ROOT    = r"D:\PAPER-ECOLIN"
IN_DIR  = os.path.join(ROOT, "RESULTADOS", "07_incertidumbre")
FIG_DIR = os.path.join(ROOT, "FIGURAS")
os.makedirs(FIG_DIR, exist_ok=True)
MASK = 'forest_own'
SHOW_SCENARIOS = False
SHOW_REPLICATES = True     # draw the 50 bootstrap totals as thin lines (instead of the dark band)

tot = pd.read_csv(os.path.join(IN_DIR, 'totals_by_year.csv'))
chg = pd.read_csv(os.path.join(IN_DIR, 'change.csv'))
sen_path = os.path.join(IN_DIR, 'scenario_sensitivity_totals.csv')
sen = pd.read_csv(sen_path) if os.path.exists(sen_path) else None
rep_path = os.path.join(IN_DIR, 'bootstrap_replicates.csv')
reps = pd.read_csv(rep_path) if os.path.exists(rep_path) else None

t = tot[tot['mask'] == MASK].sort_values('year')
c = chg[chg['mask'] == MASK].copy()
sel = t.scenario.iloc[0]


def style(ax):
    ax.tick_params(direction='in', length=4)
    ax.grid(True, axis='y', color='#e0e0e0', linewidth=0.5, zorder=0)
    for s in ax.spines.values():
        s.set_linewidth(0.8); s.set_color('black')


fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios': [1.15, 1]})

# ── A. totals + CI + scenarios ───────────────────────────────────────────────
axA.fill_between(t.year, t.CI95_low, t.CI95_high, color='#e3e3e3', zorder=1)
if SHOW_REPLICATES and reps is not None:
    cols = [f'{y}|{MASK}' for y in t.year]
    for _, rr in reps.iterrows():
        axA.plot(t.year, rr[cols].values, color='#7a7a7a', lw=0.5, alpha=0.6, zorder=2)
else:
    axA.fill_between(t.year, t.boot_p2_5, t.boot_p97_5, color='#a9a9a9', zorder=2)
if sen is not None and SHOW_SCENARIOS:
    s = sen[(sen['mask'] == MASK) & (sen.scenario != sel)]
    for sc, g in s.groupby('scenario'):
        g = g.sort_values('year')
        ls = '-' if str(g.fcd.iloc[0]).lower() in ('yes', 'true', '1') else (0, (4, 2))
        axA.plot(g.year, g.AGC_TgC, ls=ls, color='#5a5a5a', lw=0.7, alpha=0.8, zorder=3)
axA.plot(t.year, t.AGC_TgC, '-o', color='black', mfc='white', mec='black', lw=1.8, ms=6, zorder=4)
axA.set_xticks(t.year)
axA.set_xlabel('Year'); axA.set_ylabel('Forest AGC (Tg C)')
axA.set_title('A. Forest AGC and uncertainty', loc='left', fontweight='bold')
style(axA)
hA = [Line2D([], [], color='black', marker='o', mfc='white', lw=1.8, label=f'Selected model ({sel})'),
      (Line2D([], [], color='#7a7a7a', lw=0.8, label=f'Bootstrap replicates (n = {len(reps)})')
       if SHOW_REPLICATES and reps is not None else Patch(fc='#a9a9a9', label='95% CI, bootstrap')),
      Patch(fc='#e3e3e3', label='95% CI, bootstrap + field error')]
if sen is not None and SHOW_SCENARIOS:
    hA += [Line2D([], [], color='#5a5a5a', lw=0.7, label='Other scenarios, with FCD'),
           Line2D([], [], color='#5a5a5a', lw=0.7, ls=(0, (4, 2)), label='Other scenarios, without FCD')]
axA.legend(handles=hA, frameon=False, fontsize=9, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.15))

# ── B. change with CI ────────────────────────────────────────────────────────
c['start'] = c.period.str.split('-').str[0].astype(int)
cons = c[c.years < c.years.max()].sort_values('start')
full = c[c.years == c.years.max()]
cc = pd.concat([cons, full])
x = np.arange(len(cc)) + np.r_[np.zeros(len(cons)), np.full(len(full), 0.6)]   # gap before 2000-2022
sig = cc.significant.astype(str).str.lower().isin(['true', '1'])
axB.bar(x, cc.change_TgC, 0.6, color=np.where(sig, '#4d4d4d', 'white'), edgecolor='black',
        linewidth=0.8, zorder=3)
axB.errorbar(x, cc.change_TgC, yerr=[cc.change_TgC - cc.CI95_low, cc.CI95_high - cc.change_TgC],
             fmt='none', ecolor='black', elinewidth=0.9, capsize=4, zorder=4)
axB.axhline(0, color='black', lw=0.8)
axB.set_xticks(x); axB.set_xticklabels([p.replace('-', '–') for p in cc.period], fontsize=10)
axB.set_xlabel('Period'); axB.set_ylabel('Change in forest AGC (Tg C)')
axB.set_title('B. Change in forest AGC', loc='left', fontweight='bold')
off = 0.03 * (cc.CI95_high.max() - min(cc.CI95_low.min(), 0))
for xi, v, hi in zip(x, cc.change_TgC, cc.CI95_high):
    axB.text(xi, hi + off, f'{v:+.2f}', ha='center', va='bottom', fontsize=9)
axB.set_ylim(top=cc.CI95_high.max() + 5 * off)
style(axB)
hB = [Patch(fc='#4d4d4d', ec='black', label='Significant change'),
      Patch(fc='white', ec='black', label='Not significant (95% CI includes 0)'),
      Line2D([], [], color='black', marker='|', ms=10, ls='', label='95% CI')]
axB.legend(handles=hB, frameon=False, fontsize=9, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.15))

fig.tight_layout()
out = os.path.join(FIG_DIR, 'fig_agc_uncertainty_sensitivity.tif')
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
plt.close(fig)

print(t[['year', 'AGC_TgC', 'boot_p2_5', 'boot_p97_5', 'CI95_low', 'CI95_high']].round(2).to_string(index=False))
print(cc[['period', 'change_TgC', 'CI95_low', 'CI95_high', 'significant']].round(2).to_string(index=False))
if sen is not None:
    r = sen[sen['mask'] == MASK].groupby('year').AGC_TgC.agg(['min', 'max']).round(2)
    print('\nRange of the 16 scenarios (Tg C):\n' + r.to_string())
print('saved', out)
