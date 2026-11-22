r"""
04b_fcd_records_and_pixels.py
(1) FCD of every training record, by data source:
      Field carbon plots (USFQ) | Avitabile (CAM2, RBA, SAM3, SAM4, SAM5) | GEDI
    -> how many records fall in FCD 10-30 % (and < 10 %), and how many of those are
       weak-labels (AGB == 15).
(2) Number of FCD pixels (and area) in each FCD range for every map year
    (< 10 | 10-30 | 30-70 | >= 70 %), on the native FCD raster, INSIDE the RBCA polygon
    (panel B). Panel A (training records) is not clipped: it describes the training data.

Outputs  RESULTADOS\00_exploratorio\fcd_records_by_source.csv
         RESULTADOS\00_exploratorio\fcd_pixels_by_class.csv
Figure   FIGURAS\fig_fcd_training_records.tif
"""
import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 11

ROOT    = r"D:\PAPER-ECOLIN"
CSV_IN  = os.path.join(ROOT, "DATA", "interim", "training_oct2025_predictors.csv")
FCD_DIR = os.path.join(ROOT, "DATA", "raw", "FCD")
RBCA_SHP = os.path.join(ROOT, "DATA", "raw", "RBCA_shp", "RBiosfera_ChocoAndino.shp")
OUT_DIR = os.path.join(ROOT, "RESULTADOS", "00_exploratorio")
FIG_DIR = os.path.join(ROOT, "FIGURAS")
for d in [OUT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

YEARS = [2000, 2006, 2012, 2016, 2022]
SOURCE = {'USFQ': 'Field plots',
          'CAM2': 'Avitabile', 'RBA': 'Avitabile', 'SAM3': 'Avitabile',
          'SAM4': 'Avitabile', 'SAM5': 'Avitabile', 'GEDI': 'GEDI'}
ORDER = ['Field plots', 'Avitabile', 'GEDI']
BINS = [(-np.inf, 10, '<10%'), (10, 30, '10-30%'), (30, 70, '30-70%'), (70, np.inf, '>=70%')]

# ── (1) Training records ─────────────────────────────────────────────────────
df = pd.read_csv(CSV_IN)
df['source'] = df.code.map(SOURCE)
df['weak_label'] = np.isclose(df.AGB, 15.0)
df['fcd_class'] = pd.cut(df.fcd_average_period, [b[0] for b in BINS] + [np.inf],
                         right=False, labels=[b[2] for b in BINS])

rows = []
for src in ORDER + ['All']:
    g = df if src == 'All' else df[df.source == src]
    r = dict(source=src, codes=', '.join(sorted(g.code.unique())), n_records=len(g))
    for _, _, lab in BINS:
        m = g.fcd_class == lab
        r[f'n_{lab}'] = int(m.sum())
        r[f'pct_{lab}'] = 100 * m.mean()
    m1030 = g.fcd_class == '10-30%'
    r['n_10-30%_weak_label'] = int((m1030 & g.weak_label).sum())
    r['n_10-30%_real_AGB'] = int((m1030 & ~g.weak_label).sum())
    rows.append(r)
rec = pd.DataFrame(rows)
rec.to_csv(os.path.join(OUT_DIR, 'fcd_records_by_source.csv'), index=False)
print("Training records by FCD class:")
print(rec[['source', 'n_records', 'n_<10%', 'n_10-30%', 'n_10-30%_weak_label',
           'n_10-30%_real_AGB', 'n_30-70%', 'n_>=70%']].to_string(index=False))
print("\nRecords with FCD 10-30 % and real AGB (not weak-label):")
print(df[(df.fcd_class == '10-30%') & ~df.weak_label][['code', 'AGB', 'fcd_average_period', 'x', 'y']]
      .to_string(index=False))

# ── (2) FCD pixels by class, per year ────────────────────────────────────────
pix = []
for yr in YEARS:
    p = os.path.join(FCD_DIR, f'landtrendr_fitted_prediction_{yr}.tif')
    if not os.path.exists(p):
        print(f"  (missing {p})"); continue
    with rasterio.open(p) as src:
        a = src.read(1).astype('float32')
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
        if src.crs.is_geographic:      # pixel area varies with latitude
            lat = src.transform.f + (np.arange(src.height) + 0.5) * src.transform.e
            ha_row = (abs(src.transform.a) * 111320 * np.cos(np.radians(lat))) * \
                     (abs(src.transform.e) * 110574) / 1e4
            ha = np.repeat(ha_row[:, None], src.width, axis=1)
        else:
            ha = np.full(a.shape, abs(src.transform.a * src.transform.e) / 1e4)
        rbca = gpd.read_file(RBCA_SHP).to_crs(src.crs)
        inside = rasterize([(g, 1) for g in rbca.geometry], out_shape=a.shape,
                           transform=src.transform, fill=0, dtype='uint8').astype(bool)
    if np.nanmax(a) > 200:
        a = a / 100.0
    valid = np.isfinite(a) & inside
    print(f"  {yr}: RBCA {ha[inside].sum():,.0f} ha | valid FCD {ha[valid].sum():,.0f} ha")
    r = dict(year=yr, n_pixels_valid=int(valid.sum()), area_valid_ha=ha[valid].sum())
    for lo, hi, lab in BINS:
        m = valid & (a >= lo) & (a < hi)
        r[f'n_pixels_{lab}'] = int(m.sum())
        r[f'area_ha_{lab}'] = ha[m].sum()
        r[f'pct_{lab}'] = 100 * m.sum() / valid.sum()
    pix.append(r)
pix = pd.DataFrame(pix)
pix.to_csv(os.path.join(OUT_DIR, 'fcd_pixels_by_class.csv'), index=False)
print("\nFCD pixels by class (inside the RBCA):")
print(pix[['year', 'n_pixels_10-30%', 'area_ha_10-30%', 'pct_10-30%',
           'pct_<10%', 'pct_30-70%', 'pct_>=70%']].round(1).to_string(index=False))

# ── Figure ───────────────────────────────────────────────────────────────────
def style(ax):
    ax.tick_params(direction='in', length=4)
    ax.grid(True, axis='y', color='#e0e0e0', linewidth=0.5, zorder=0)
    for s in ax.spines.values():
        s.set_linewidth(0.8); s.set_color('black')

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios': [1.3, 1]})
ax = axes[0]
rng = np.random.default_rng(0)
ax.axhspan(10, 30, color='#e6e6e6', zorder=0, label='FCD 10–30 %')
ax.axhline(30, color='black', linewidth=0.8, linestyle=(0, (4, 3)))
for i, s in enumerate(ORDER):
    g = df[df.source == s]
    ax.boxplot(g.fcd_average_period, positions=[i], widths=0.5, showfliers=False,
               patch_artist=True, boxprops=dict(facecolor='white', linewidth=0.8),
               medianprops=dict(color='black', linewidth=1.2), zorder=2)
    jit = rng.uniform(-0.18, 0.18, len(g))
    real = ~g.weak_label.values
    ax.scatter(i + jit[real], g.fcd_average_period[real], s=10, color='#808080',
               edgecolors='none', alpha=0.7, zorder=3)
    ax.scatter(i + jit[~real], g.fcd_average_period[~real], s=18, marker='x', color='black',
               linewidths=0.8, zorder=4)
    n1030 = int(((g.fcd_average_period >= 10) & (g.fcd_average_period < 30)).sum())
    ax.text(i, 104, f"n = {len(g)}\n10–30 %: {n1030}", ha='center', va='bottom', fontsize=9)
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels([f"{s}\n({', '.join(sorted(df[df.source == s].code.unique()))})" for s in ORDER],
                   fontsize=9)
ax.set_ylim(-3, 122); ax.set_yticks(range(0, 101, 20))
ax.set_ylabel('FCD of the record (%)')
ax.set_title('A. FCD of training records by source', loc='left', fontweight='bold')
h = [plt.Line2D([], [], marker='o', ls='', color='#808080', label='Measured AGB'),
     plt.Line2D([], [], marker='x', ls='', color='black', label='Weak-label (AGB = 15)'),
     plt.Rectangle((0, 0), 1, 1, fc='#e6e6e6', label='FCD 10–30 %')]
ax.legend(handles=h, frameon=True, edgecolor='black', fontsize=9, loc='lower left')
style(ax)

ax = axes[1]
if len(pix):
    labs = [b[2] for b in BINS]
    x = np.arange(len(pix))
    wbar = 0.2
    shades = ['#d9d9d9', '#a6a6a6', '#636363', '#252525']
    for k, (lab, c) in enumerate(zip(labs, shades)):
        ax.bar(x + (k - 1.5) * wbar, pix[f'area_ha_{lab}'] / 1e3, wbar, color=c,
               edgecolor='black', linewidth=0.6, label=lab.replace('-', '–').replace('>=', '≥'), zorder=3)
    for xi, (v, pc) in enumerate(zip(pix['area_ha_10-30%'] / 1e3, pix['pct_10-30%'])):
        ax.text(xi - 0.5 * wbar, v + 4, f"{pc:.1f}%", ha='center', fontsize=8, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(pix.year)
    ax.set_xlabel('Year'); ax.set_ylabel('Area (thousand ha)')
    ax.legend(title='FCD class', frameon=False, fontsize=9, title_fontsize=9, ncol=4,
              loc='upper center', bbox_to_anchor=(0.5, -0.15))      # below the axes
ax.set_title('B. FCD classes in the RBCA', loc='left', fontweight='bold')
style(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_fcd_training_records.tif'), dpi=300,
            bbox_inches='tight', facecolor='white', format='tiff')
plt.close(fig)
print(f"\nSaved: {OUT_DIR} and {FIG_DIR}\\fig_fcd_training_records.tif")
