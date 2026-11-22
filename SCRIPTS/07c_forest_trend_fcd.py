r"""
07c_forest_trend_fcd.py
How does the forest change from 2000 onwards? Follows the SAME pixels through
2000, 2006, 2012, 2016 and 2022, grouped by their FCD class in 2000 (cohorts):

    Non-forest (FCD < 10 %) | Degraded (10-30 %) | Intermediate (30-70 %) | Dense (>= 70 %)

For each cohort and year: area, mean / median / IQR of FCD, % of the cohort that is
forest (FCD >= 30 %), and mean AGC (from the 07 maps of the selected scenario; non-forest
pixels, FCD < 30 %, have AGC = 0 in those maps).
Per pixel: linear trend (slope) of FCD and of AGC over the 5 dates -> mean slope per
cohort and % of pixels with increasing / decreasing trend. Slope maps are saved.

Grid: the AGC maps of 07 (UTM 17S, 30 m); FCD is resampled to it (bilinear), as in 07.
Pixels with true NoData in ANY year (mask codes 2-5) are excluded from everything.
Optional: restricted to AOI_SHP (e.g. RBCA) if the file exists.

Note: FCD comes from LandTrendr-fitted trajectories (temporally smoothed), so trends
are smooth by construction; interpret slopes as direction and magnitude of change,
not as year-to-year variability.

Outputs  RESULTADOS\07c_tendencia_fcd\
   cohort_trajectory.csv, cohort_trends.csv, forest_area_by_year.csv,
   fcd_slope.tif, agc_slope.tif, cohort_2000.tif
Figures  FIGURAS\fig_fcd_cohort_trend.tif, fig_forest_area_trend.tif
"""

import os
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 11
warnings.filterwarnings('ignore', message='Mean of empty slice')

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT     = r"D:\PAPER-ECOLIN"
FCD_DIR  = os.path.join(ROOT, "DATA", "raw", "FCD")
MAPS_DIR = os.path.join(ROOT, "RESULTADOS", "05_mapas")
OUT_DIR  = os.path.join(ROOT, "RESULTADOS", "07c_tendencia_fcd")
FIG_DIR  = os.path.join(ROOT, "FIGURAS")
AOI_SHP  = os.path.join(ROOT, "DATA", "raw", "AREA_STUDIO", "RBCA.shp")   # optional
for d in [OUT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

YEARS      = [2000, 2006, 2012, 2016, 2022]
FCD_FOREST = 30.0
NODATA     = -9999.0
COHORTS = [  # (code, name, low, high)  on FCD 2000
    (1, 'Non-forest (<10%)', -np.inf, 10),
    (2, 'Degraded (10-30%)', 10, 30),
    (3, 'Intermediate (30-70%)', 30, 70),
    (4, 'Dense (>=70%)', 70, np.inf),
]
STYLE = {1: dict(color='#b0b0b0', ls=':', marker='v'),
         2: dict(color='#808080', ls='-.', marker='s'),
         3: dict(color='#505050', ls='--', marker='^'),
         4: dict(color='black', ls='-', marker='o')}

# ── Reference grid = AGC maps of 07 ──────────────────────────────────────────
with rasterio.open(os.path.join(MAPS_DIR, f'agc_{YEARS[0]}.tif')) as src:
    grid = dict(crs=src.crs, transform=src.transform, width=src.width, height=src.height)
    prof = src.profile.copy()
H, W = grid['height'], grid['width']
px_ha = abs(grid['transform'].a * grid['transform'].e) / 1e4


def fcd_on_grid(yr):
    out = np.full((H, W), np.nan, dtype='float32')
    with rasterio.open(os.path.join(FCD_DIR, f'landtrendr_fitted_prediction_{yr}.tif')) as src:
        reproject(source=rasterio.band(src, 1), destination=out, src_transform=src.transform,
                  src_crs=src.crs, src_nodata=src.nodata, dst_transform=grid['transform'],
                  dst_crs=grid['crs'], dst_nodata=np.nan, resampling=Resampling.bilinear)
    if np.nanmax(out) > 200:
        out = out / 100.0
    return out


# ── Load stacks ──────────────────────────────────────────────────────────────
fcd = np.stack([fcd_on_grid(y) for y in YEARS])                     # (T, H, W)
agc = np.full((len(YEARS), H, W), np.nan, dtype='float32')
universe = np.ones((H, W), dtype=bool)
for i, yr in enumerate(YEARS):
    with rasterio.open(os.path.join(MAPS_DIR, f'agc_{yr}.tif')) as src:
        a = src.read(1)
    agc[i] = np.where(a == NODATA, np.nan, a)
    with rasterio.open(os.path.join(MAPS_DIR, f'mask_{yr}.tif')) as src:
        m = src.read(1)
    universe &= ~np.isin(m, [2, 3, 4, 5])
    agc[i][m == 6] = 0.0          # non-forest (FCD < 30 %): AGC = 0 (already 0 in the 07 maps)
universe &= np.all(np.isfinite(fcd), axis=0)
if os.path.exists(AOI_SHP):
    import geopandas as gpd
    from rasterio.features import rasterize
    shp = gpd.read_file(AOI_SHP).to_crs(grid['crs'])
    universe &= rasterize([(g, 1) for g in shp.geometry], out_shape=(H, W),
                          transform=grid['transform'], fill=0, dtype='uint8').astype(bool)
    print(f"AOI applied: {AOI_SHP}")
print(f"Pixels analysed: {universe.sum():,} ({universe.sum() * px_ha:,.0f} ha)")

# cohorts by FCD 2000
cohort = np.zeros((H, W), dtype='uint8')
for code, _, lo, hi in COHORTS:
    cohort[universe & (fcd[0] >= lo) & (fcd[0] < hi)] = code

# ── Per-pixel linear trend (slope per year) ──────────────────────────────────
t = np.array(YEARS, dtype='float32')
tc = (t - t.mean())[:, None, None]
den = (tc ** 2).sum()
fcd_slope = ((fcd - fcd.mean(axis=0)) * tc).sum(axis=0) / den            # FCD %-points / yr
agc_ok = np.all(np.isfinite(agc), axis=0)
agc_slope = np.where(agc_ok, ((agc - np.nanmean(agc, axis=0)) * tc).sum(axis=0) / den, np.nan)
for arr in (fcd_slope, agc_slope):
    arr[~universe] = np.nan

# ── Tables ───────────────────────────────────────────────────────────────────
traj, trends = [], []
for code, name, _, _ in COHORTS:
    sel = cohort == code
    n = sel.sum()
    if n == 0:
        continue
    for i, yr in enumerate(YEARS):
        f = fcd[i][sel]
        a = agc[i][sel]
        traj.append(dict(cohort=name, year=yr, area_ha=n * px_ha,
                         FCD_mean=f.mean(), FCD_median=np.median(f),
                         FCD_p25=np.percentile(f, 25), FCD_p75=np.percentile(f, 75),
                         pct_forest=100 * np.mean(f >= FCD_FOREST),
                         AGC_mean_MgC_ha=np.nanmean(a), AGC_total_TgC=np.nansum(a) * px_ha / 1e6))
    fs, as_ = fcd_slope[sel], agc_slope[sel]
    trends.append(dict(cohort=name, area_ha=n * px_ha,
                       FCD_2000_mean=fcd[0][sel].mean(), FCD_2022_mean=fcd[-1][sel].mean(),
                       FCD_change_2000_2022=fcd[-1][sel].mean() - fcd[0][sel].mean(),
                       FCD_slope_mean_pct_per_yr=np.nanmean(fs),
                       pct_pixels_FCD_increasing=100 * np.mean(fs > 0.1),
                       pct_pixels_FCD_decreasing=100 * np.mean(fs < -0.1),
                       AGC_slope_mean_MgC_ha_yr=np.nanmean(as_),
                       pct_pixels_AGC_increasing=100 * np.nanmean(as_ > 0),
                       pct_now_forest_2022=100 * np.mean(fcd[-1][sel] >= FCD_FOREST)))
traj = pd.DataFrame(traj)
trends = pd.DataFrame(trends)
traj.to_csv(os.path.join(OUT_DIR, 'cohort_trajectory.csv'), index=False)
trends.to_csv(os.path.join(OUT_DIR, 'cohort_trends.csv'), index=False)

area = pd.DataFrame([dict(year=yr,
                          forest_area_ha=np.sum(universe & (fcd[i] >= FCD_FOREST)) * px_ha,
                          dense_area_ha=np.sum(universe & (fcd[i] >= 70)) * px_ha,
                          intermediate_area_ha=np.sum(universe & (fcd[i] >= 30) & (fcd[i] < 70)) * px_ha,
                          FCD_mean_all=fcd[i][universe].mean())
                     for i, yr in enumerate(YEARS)])
area.to_csv(os.path.join(OUT_DIR, 'forest_area_by_year.csv'), index=False)

print("\nTrend by 2000 cohort:")
print(trends.round(2).to_string(index=False))
print("\nForest area by year:")
print(area.round(0).to_string(index=False))

# ── Rasters ──────────────────────────────────────────────────────────────────
prof.update(count=1, dtype='float32', nodata=NODATA, compress='deflate', tiled=True)
for name, arr, desc in [('fcd_slope', fcd_slope, 'FCD slope (%-points per year)'),
                        ('agc_slope', agc_slope, 'AGC slope (Mg C ha-1 per year)')]:
    with rasterio.open(os.path.join(OUT_DIR, f'{name}.tif'), 'w', **prof) as dst:
        dst.write(np.where(np.isfinite(arr), arr, NODATA).astype('float32'), 1)
        dst.set_band_description(1, desc)
p2 = prof.copy(); p2.update(dtype='uint8', nodata=0)
with rasterio.open(os.path.join(OUT_DIR, 'cohort_2000.tif'), 'w', **p2) as dst:
    dst.write(cohort, 1)
    dst.set_band_description(1, '1 <10 | 2 10-30 | 3 30-70 | 4 >=70 (FCD 2000)')


# ── Figures ──────────────────────────────────────────────────────────────────
def style(ax):
    ax.tick_params(direction='in', length=4)
    ax.grid(True, color='#e0e0e0', linewidth=0.5, zorder=0)
    for s in ax.spines.values():
        s.set_linewidth(0.8); s.set_color('black')

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for code, name, _, _ in COHORTS:
    g = traj[traj.cohort == name]
    if g.empty:
        continue
    st = STYLE[code]
    axes[0].fill_between(g.year, g.FCD_p25, g.FCD_p75, color=st['color'], alpha=0.15, zorder=1)
    axes[0].plot(g.year, g.FCD_mean, color=st['color'], ls=st['ls'], marker=st['marker'],
                 mfc='white', label=name, zorder=3)
    axes[1].plot(g.year, g.AGC_mean_MgC_ha, color=st['color'], ls=st['ls'], marker=st['marker'],
                 mfc='white', label=name, zorder=3)
axes[0].axhline(FCD_FOREST, color='black', linewidth=0.8, linestyle=(0, (4, 3)))
axes[0].set_ylabel('FCD (%)'); axes[0].set_ylim(0, 100)
axes[0].set_title('A. Canopy density of the 2000 cohorts', loc='left', fontweight='bold')
axes[1].set_ylabel('AGC (Mg C ha$^{-1}$)')
axes[1].set_title('B. AGC of the 2000 cohorts', loc='left', fontweight='bold')
for ax in axes:
    ax.set_xticks(YEARS); ax.set_xlabel('Year'); style(ax)
axes[1].legend(title='FCD class in 2000', frameon=True, edgecolor='black', fontsize=9, title_fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_fcd_cohort_trend.tif'), dpi=300, bbox_inches='tight',
            facecolor='white', format='tiff')
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.8, 4))
ax.plot(area.year, area.forest_area_ha / 1e3, '-o', color='black', mfc='#606060', label='Forest (FCD ≥ 30%)')
ax.plot(area.year, area.dense_area_ha / 1e3, '--^', color='#505050', mfc='white', label='Dense (FCD ≥ 70%)')
ax.plot(area.year, area.intermediate_area_ha / 1e3, ':s', color='#808080', mfc='white', label='Intermediate (30–70%)')
ax.set_xticks(YEARS); ax.set_xlabel('Year'); ax.set_ylabel('Area (thousand ha)')
ax.set_title('A. Forest area by canopy density', loc='left', fontweight='bold')
ax.legend(frameon=True, edgecolor='black', fontsize=9)
style(ax)
fig.savefig(os.path.join(FIG_DIR, 'fig_forest_area_trend.tif'), dpi=300, bbox_inches='tight',
            facecolor='white', format='tiff')
plt.close(fig)

# Trend per cohort (bars): does each 2000 cohort go up or down?
tr = trends.copy()
labels = [c.replace(' (', '\n(') for c in tr.cohort]
codes = [c[0] for c in COHORTS if c[1] in set(tr.cohort)]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, col, pct, ylab, ttl in [
        (axes[0], 'FCD_slope_mean_pct_per_yr', 'pct_pixels_FCD_increasing',
         'FCD trend (%-points yr$^{-1}$)', 'A. Canopy density trend 2000–2022'),
        (axes[1], 'AGC_slope_mean_MgC_ha_yr', 'pct_pixels_AGC_increasing',
         'AGC trend (Mg C ha$^{-1}$ yr$^{-1}$)', 'B. AGC trend 2000–2022')]:
    colors = [STYLE[c]['color'] for c in codes]
    bars = ax.bar(range(len(tr)), tr[col], color=colors, edgecolor='black', linewidth=0.8, zorder=3)
    ax.axhline(0, color='black', linewidth=0.8)
    span = max(abs(tr[col]).max(), 1e-6)
    for b, v, p in zip(bars, tr[col], tr[pct]):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.04 * span if v >= 0 else -0.04 * span),
                f"{p:.0f}% ↑", ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
    ax.set_xticks(range(len(tr))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel('FCD class in 2000'); ax.set_ylabel(ylab)
    ax.set_title(ttl, loc='left', fontweight='bold')
    ax.set_ylim(min(0, tr[col].min()) - 0.15 * span, max(0, tr[col].max()) + 0.2 * span)
    style(ax)
fig.text(0.01, -0.02, 'Labels: % of pixels with increasing trend.', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_cohort_trend_bars.tif'), dpi=300, bbox_inches='tight',
            facecolor='white', format='tiff')
plt.close(fig)

print(f"\nResults: {OUT_DIR}\nDone.")
