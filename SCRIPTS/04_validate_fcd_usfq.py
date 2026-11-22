"""
04_validate_fcd_usfq.py
FCD temporal stability at the USFQ field sites (93 plot-visits, one value per site)

Inputs:
  - D:/PAPER-ECOLIN/DATA/interim/training_oct2025_predictors.csv  (USFQ = 93 records)
  - D:/PAPER-ECOLIN/DATA/raw/FCD/landtrendr_fitted_prediction_XXXX.tif

Outputs:
  - D:/PAPER-ECOLIN/DATA/interim/fcd_validation_usfq.csv   (one row per site)
  - D:/PAPER-ECOLIN/FIGURAS/fig_fcd_temporal_stability.tif  (300 dpi)
FCD is a pixel value: repeated visits of the same site share it, so the
stability is assessed per SITE (unique x, y), not per visit.
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 11

# ── Paths ────────────────────────────────────────────────────────────────────
CSV_IN   = r"D:\PAPER-ECOLIN\DATA\interim\training_oct2025_predictors.csv"
FCD_DIR  = r"D:\PAPER-ECOLIN\DATA\raw\FCD"
OUT_DIR  = r"D:\PAPER-ECOLIN\DATA\interim"
FIG_DIR  = r"D:\PAPER-ECOLIN\FIGURAS"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

FCD_YEARS = [2000, 2006, 2012, 2016, 2022]
FCD_FILES = {
    yr: os.path.join(FCD_DIR, f"landtrendr_fitted_prediction_{yr}.tif")
    for yr in FCD_YEARS
}

# ── 1. Load and filter USFQ plots ────────────────────────────────────────────
df = pd.read_csv(CSV_IN)
usfq_all = df[df['code'] == 'USFQ'].copy()
usfq_all['Year(s)_Avitabile'] = usfq_all['Year(s)_Avitabile'].astype(str)
visits = (usfq_all.groupby(['x', 'y'])['Year(s)_Avitabile']
          .apply(lambda v: '/'.join(sorted(v))).rename('visits'))
usfq = (usfq_all.sort_values('Year(s)_Avitabile').groupby(['x', 'y'], as_index=False)
        .agg(unique_id_combined=('unique_id_combined', 'first'), code=('code', 'first'),
             n_visits=('AGB', 'size'), AGB_first=('AGB', 'first'), AGB_last=('AGB', 'last')))
usfq = usfq.merge(visits.reset_index(), on=['x', 'y'])
usfq.insert(0, 'site_id', [f"USFQ_{i:02d}" for i in range(len(usfq))])
N_SITES = len(usfq)
print(f"USFQ records: {len(usfq_all)} | sites: {N_SITES}")
print(f"Visits per site: {usfq.n_visits.value_counts().sort_index().to_dict()}")

# ── 2. Verify CRS of first FCD raster ────────────────────────────────────────
with rasterio.open(FCD_FILES[2022]) as src:
    crs = src.crs
    print(f"\nFCD 2022 CRS: {crs}")
    print(f"FCD 2022 bounds: {src.bounds}")
    print(f"FCD 2022 resolution: {src.res}")
    assert crs.to_epsg() == 4326, \
        f"Expected WGS84 (EPSG:4326) but got {crs}. Reproject FCD rasters first."

# ── 3. Extract FCD values for all 5 periods ──────────────────────────────────
def extract_values(raster_path, xs, ys):
    """Sample raster at (x, y) coordinates. Returns array of values."""
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        coords = list(zip(xs, ys))
        vals = np.array([v[0] for v in src.sample(coords)], dtype=float)
        if nodata is not None:
            vals[vals == nodata] = np.nan
        # FCD scale: if stored as 0-10000 rescale to 0-100
        if np.nanmax(vals) > 200:
            vals = vals / 100.0
    return vals

xs = usfq['x'].values
ys = usfq['y'].values

for yr in FCD_YEARS:
    print(f"Extracting FCD {yr}...")
    usfq[f'fcd_{yr}'] = extract_values(FCD_FILES[yr], xs, ys)
    print(f"  mean={usfq[f'fcd_{yr}'].mean():.1f}  "
          f"min={usfq[f'fcd_{yr}'].min():.1f}  "
          f"max={usfq[f'fcd_{yr}'].max():.1f}  "
          f"NaN={usfq[f'fcd_{yr}'].isna().sum()}")

# ── 4. Save interim CSV ───────────────────────────────────────────────────────
cols_out = ['site_id', 'unique_id_combined', 'code', 'x', 'y', 'n_visits', 'visits',
            'AGB_first', 'AGB_last'] + [f'fcd_{yr}' for yr in FCD_YEARS]
usfq[cols_out].to_csv(os.path.join(OUT_DIR, 'fcd_validation_usfq.csv'), index=False)
print("\nSaved: fcd_validation_usfq.csv")

# ── 5. Figure — Temporal stability (violin + boxplot) ────────────────────────
fcd_cols = [f'fcd_{yr}' for yr in FCD_YEARS]
fcd_data = [usfq[c].dropna().values for c in fcd_cols]

fig2, ax = plt.subplots(figsize=(8, 4.5))

positions = np.arange(1, len(FCD_YEARS) + 1)

# Violin
parts = ax.violinplot(fcd_data, positions=positions,
                      showmedians=False, showextrema=False, widths=0.7)
for pc in parts['bodies']:
    pc.set_facecolor('#d0d0d0')
    pc.set_edgecolor('#606060')
    pc.set_linewidth(0.8)
    pc.set_alpha(0.8)

# Boxplot overlay
bp = ax.boxplot(fcd_data, positions=positions, widths=0.25,
                patch_artist=True, manage_ticks=False,
                medianprops=dict(color='white', linewidth=2),
                boxprops=dict(facecolor='#606060', linewidth=0.8),
                whiskerprops=dict(color='#404040', linewidth=0.8),
                capprops=dict(color='#404040', linewidth=0.8),
                flierprops=dict(marker='o', markersize=3,
                                markerfacecolor='#808080',
                                markeredgecolor='#404040', linewidth=0.5))

# Forest threshold line
ax.axhline(y=30, color='black', linestyle='--', linewidth=1.0,
           label='Forest threshold (FCD = 30%)', zorder=5)

ax.set_xticks(positions)
ax.set_xticklabels([str(yr) for yr in FCD_YEARS], fontsize=11)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("FCD (%)", fontsize=11)
ax.set_title(f"A. Temporal stability of FCD at the USFQ field sites (n = {N_SITES})",
             fontsize=11, fontweight='bold', loc='left')
ax.set_ylim(0, 110)
ax.tick_params(direction='in', length=4)
ax.grid(True, axis='y', color='#e0e0e0', linewidth=0.5, zorder=0)
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color('black')

ax.legend(fontsize=10, frameon=True, edgecolor='black',
          loc='lower right', framealpha=1.0)

out_violin = os.path.join(FIG_DIR, 'fig_fcd_temporal_stability.tif')
fig2.savefig(out_violin, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
plt.close(fig2)
print(f"Saved: fig_fcd_temporal_stability.tif")

print("\n=== Done ===")
print(f"Outputs in: {OUT_DIR}")
