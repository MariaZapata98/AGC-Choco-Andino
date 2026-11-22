r"""
08_ledger_figures.py
AGC change, transitions and carbon ledger 2000-2022 from the 07 maps (selected scenario).
Replaces the original Figures 5-8 and answers R1-5, R1-6, R2-3, R2-4:

  Fig 5  fig_fcd_alluvial.tif        FCD class transitions 2000 -> 2022 (<10, 10-30, 30-70, >=70 %)
  Fig 6  fig_agc_change_maps.tif     A. annualized AGC stock change 2000-2022
                                     B. AGC 2022   C. FCD 2022
  Fig 7  fig_bivariate_elev_agc.tif  bivariate map: elevation (SRTM) x annualized AGC change
         fig_si_heatmap_fcd_elev.tif (SI) mean annualized change by FCD-2000 class x elevation
  Fig 8  fig_agc_ledger.tif          carbon ledger with forest transitions

Definitions
  forest              FCD >= 30 % (same as 07); non-forest has AGC = 0
  annualized change   (AGC_2022 - AGC_2000) / 22   [Mg C ha-1 yr-1]  ("annualized AGC stock change")
  change threshold    a pixel GAINS or LOSES only if |AGC_2022 - AGC_2000| > 1.96 x RMSE_change,
                      RMSE_change = RMSE of (map change - field change) at the re-measured USFQ
                      sites (map 2022 - map 2016 vs visit 2021 - visit 2017; 07 validation points).
                      Replaces the +-0.5 sigma rule. Fallback if not available: 1.96 x sqrt(2) x RMSE_stock.
  ledger (4 classes; closes in area and carbon by construction)
      1 Non-forest        non-forest in 2000 and 2022
      2 Stable reservoir  forest in both years and |change| <= threshold
      3 Active sink       forest in both years and change > threshold, or new forest (non-forest -> forest)
      4 Net source        forest in both years and change < -threshold, or forest lost (forest -> non-forest)
  Pixels with NoData (mask codes 2-5) in any of the five map years are excluded (same area as 07b).
  Restricted to the RBCA polygon (AOI_SHP) if it exists.

Outputs  RESULTADOS\08_ledger\
   ledger_2000_2022.tif, agc_change_rate_2000_2022.tif
   ledger_table.csv            area, AGC 2000/2022, change (Tg C), rate per category + total
   fcd_transitions_ha.csv      4 x 4 transition matrix (ha)
   elevation_summary.csv       AGC 2022, change and ledger shares by elevation band
   heatmap_fcd_elevation.csv   mean / SD / area per FCD-2000 class x elevation band
   change_threshold.csv
Usage    python 08_ledger_figures.py
"""

import os
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling, transform as warp_transform
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 11

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT     = r"D:\PAPER-ECOLIN"
FCD_DIR  = os.path.join(ROOT, "DATA", "raw", "FCD")
SRTM     = os.path.join(ROOT, "DATA", "interim", "srtm-dem", "srtm_dem.tif")
RES_DIR  = os.path.join(ROOT, "RESULTADOS")
MAPS_DIR = os.path.join(RES_DIR, "05_mapas")
VAL_PTS  = os.path.join(RES_DIR, "06_validacion_mapas", "map_validation_points.csv")
MODEL_JS = os.path.join(RES_DIR, "03_modelo_final", "rf_selected_model.json")
OUT_DIR  = os.path.join(RES_DIR, "08_ledger")
FIG_DIR  = os.path.join(ROOT, "FIGURAS")
AOI_SHP  = os.path.join(ROOT, "DATA", "raw", "AREA_STUDIO", "RBCA.shp")
for d in [OUT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Settings ─────────────────────────────────────────────────────────────────
Y0, Y1     = 2000, 2022
C_FACTOR   = 0.456
FCD_FOREST = 30.0
NODATA     = -9999.0
FCD_CLASSES = [(-np.inf, 10, '<10%'), (10, 30, '10–30%'), (30, 70, '30–70%'), (70, np.inf, '≥70%')]
ELEV_BANDS  = [(-np.inf, 500, '<500'), (500, 1000, '500–1000'), (1000, 2000, '1000–2000'),
               (2000, 3000, '2000–3000'), (3000, np.inf, '>3000')]
BIV_ELEV    = [(-np.inf, 1000, '<1000 m'), (1000, 2000, '1000–2000 m'), (2000, np.inf, '≥2000 m')]
LEDGER = {1: 'Non-forest', 2: 'Stable reservoir', 3: 'Active sink', 4: 'Net source'}
# ledger in colour: light grey = non-forest, yellow-green = stable reservoir, blue = sink, dark red = source
LEDGER_STYLE = {1: ('#efefef', None), 2: ('#c3df62', None), 3: ('#3554d1', None), 4: ('#990000', None)}
FCD_SHADES = ['#e0e0e0', '#b0b0b0', '#707070', '#303030']
# The SIGN of change (gain vs loss) cannot be read in a single grey ramp, so the change map (Fig 6A)
# and the SI heatmap use one restrained diverging ramp; everything else stays grayscale.
# Set CHANGE_CMAP = 'Greys' to force grayscale (then gain and loss differ only by the colorbar).
CHANGE_CMAP = 'RdBu'
# bivariate (two hues, 3 x 3): rows = AGC change (loss, stable, gain), cols = elevation (low -> high)
BIV_COLORS = [['#e8e8e8', '#ace4e4', '#5ac8c8'],
              ['#dfb0d6', '#a5add3', '#5698b9'],
              ['#be64ac', '#8c62aa', '#3b4994']]


def style(ax, grid=False):
    ax.tick_params(direction='in', length=4)
    if grid:
        ax.grid(True, color='#e0e0e0', linewidth=0.5, zorder=0)
    for s in ax.spines.values():
        s.set_linewidth(0.8); s.set_color('black')


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
    plt.close(fig)
    print(f"  figure: {name}")


# ── Grid and layers ──────────────────────────────────────────────────────────
def read_band(path, b=1):
    with rasterio.open(path) as s:
        a = s.read(b).astype('float32')
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
    return a


with rasterio.open(os.path.join(MAPS_DIR, f'agc_{Y1}.tif')) as src:
    grid = dict(crs=src.crs, transform=src.transform, width=src.width, height=src.height)
    prof = src.profile.copy()
H, W = grid['height'], grid['width']
px_ha = abs(grid['transform'].a * grid['transform'].e) / 1e4


def to_grid(path, b=1, resampling=Resampling.bilinear):
    out = np.full((H, W), np.nan, dtype='float32')
    with rasterio.open(path) as s:
        reproject(source=rasterio.band(s, b), destination=out, src_transform=s.transform, src_crs=s.crs,
                  src_nodata=s.nodata, dst_transform=grid['transform'], dst_crs=grid['crs'],
                  dst_nodata=np.nan, resampling=resampling)
    return out


agc0, agc1 = read_band(os.path.join(MAPS_DIR, f'agc_{Y0}.tif')), read_band(os.path.join(MAPS_DIR, f'agc_{Y1}.tif'))
m0, m1 = read_band(os.path.join(MAPS_DIR, f'mask_{Y0}.tif')), read_band(os.path.join(MAPS_DIR, f'mask_{Y1}.tif'))
fcd = {}
for y in (Y0, Y1):
    f = to_grid(os.path.join(FCD_DIR, f'landtrendr_fitted_prediction_{y}.tif'))
    fcd[y] = f / 100.0 if np.nanmax(f) > 200 else f
with rasterio.open(SRTM) as s:
    d = [x.lower() if x else '' for x in s.descriptions]
    eb = d.index('elevation') + 1 if 'elevation' in d else 1
elev = to_grid(SRTM, eb)

universe = (np.isin(m0, [1, 6]) & np.isin(m1, [1, 6]) & np.isfinite(agc0) & np.isfinite(agc1)
            & np.isfinite(fcd[Y0]) & np.isfinite(fcd[Y1]) & np.isfinite(elev))
# same universe as 07b: drop pixels that are NoData (codes 2-5) in ANY of the five map years,
# so ledger totals and the uncertainty totals refer to the same area
ALL_YEARS = [2000, 2006, 2012, 2016, 2022]
for y in ALL_YEARS:
    pm = os.path.join(MAPS_DIR, f'mask_{y}.tif')
    if os.path.exists(pm):
        universe &= np.isin(read_band(pm), [1, 6])
aoi_geoms = None
if os.path.exists(AOI_SHP):
    import geopandas as gpd
    from rasterio.features import rasterize
    shp = gpd.read_file(AOI_SHP).to_crs(grid['crs'])
    aoi_geoms = shp.geometry
    universe &= rasterize([(g, 1) for g in shp.geometry], out_shape=(H, W), transform=grid['transform'],
                          fill=0, dtype='uint8').astype(bool)
    AREA_NAME = 'RBCA'
else:
    print(f"WARNING: {AOI_SHP} not found -> full map extent is used (totals not comparable with the paper)")
    AREA_NAME = 'map extent'
print(f"Pixels analysed: {universe.sum():,} ({universe.sum() * px_ha:,.0f} ha, {AREA_NAME})")

forest0 = (m0 == 1) & universe
forest1 = (m1 == 1) & universe
delta = np.where(universe, agc1 - agc0, np.nan)                 # Mg C ha-1 over 22 yr
rate = delta / (Y1 - Y0)                                         # Mg C ha-1 yr-1

# ── Change threshold from the field validation ───────────────────────────────
thr_src = ''
THR = np.nan
if os.path.exists(VAL_PTS):
    vp = pd.read_csv(VAL_PTS).dropna(subset=['AGB_map_pixel'])
    yrs = sorted(vp.map_year.unique())
    if len(yrs) >= 2:
        w_map = vp.pivot_table(index='site_id', columns='map_year', values='AGB_map_pixel')
        w_fld = vp.pivot_table(index='site_id', columns='map_year', values='AGB_field')
        e = ((w_map[yrs[-1]] - w_map[yrs[0]]) - (w_fld[yrs[-1]] - w_fld[yrs[0]])).dropna() * C_FACTOR
        if len(e) >= 3:
            rmse_chg = float(np.sqrt(np.mean(e ** 2)))
            THR = 1.96 * rmse_chg
            thr_src = (f'1.96 x RMSE of map change vs field change at {len(e)} re-measured sites '
                       f'(maps {yrs[0]}-{yrs[-1]}); RMSE_change = {rmse_chg:.2f} Mg C ha-1')
    if not np.isfinite(THR):
        rmse_stock = float(np.sqrt(np.mean(((vp.AGB_map_pixel - vp.AGB_field) * C_FACTOR) ** 2)))
        THR = 1.96 * np.sqrt(2) * rmse_stock
        thr_src = f'fallback 1.96 x sqrt(2) x RMSE_stock ({rmse_stock:.2f} Mg C ha-1)'
else:
    raise FileNotFoundError(f"{VAL_PTS} missing: run 07 first")
pd.DataFrame([dict(threshold_MgC_ha=THR, threshold_MgC_ha_yr=THR / (Y1 - Y0), source=thr_src)]).to_csv(
    os.path.join(OUT_DIR, 'change_threshold.csv'), index=False)
print(f"Change threshold: |dAGC| > {THR:.1f} Mg C/ha ({THR / (Y1 - Y0):.2f} Mg C/ha/yr) — {thr_src}")

# ── Ledger ───────────────────────────────────────────────────────────────────
ledger = np.zeros((H, W), dtype='uint8')
ledger[universe & ~forest0 & ~forest1] = 1
pers = forest0 & forest1
ledger[pers & (np.abs(delta) <= THR)] = 2                         # stable reservoir
ledger[(pers & (delta > THR)) | (universe & ~forest0 & forest1)] = 3   # active sink (gain + new forest)
ledger[(pers & (delta < -THR)) | (universe & forest0 & ~forest1)] = 4  # net source (loss + forest lost)

rows = []
for k, name in LEDGER.items():
    s = ledger == k
    a0, a1 = agc0[s].sum() * px_ha / 1e6, agc1[s].sum() * px_ha / 1e6
    rows.append(dict(code=k, category=name, area_ha=s.sum() * px_ha, pct_area=100 * s.sum() / universe.sum(),
                     AGC_2000_TgC=a0, AGC_2022_TgC=a1, change_TgC=a1 - a0,
                     share_of_net_change_pct=np.nan,
                     rate_mean_MgC_ha_yr=np.nanmean(rate[s]) if s.any() else np.nan))
led = pd.DataFrame(rows)
net = led.change_TgC.sum()
led['share_of_net_change_pct'] = 100 * led.change_TgC / net if net != 0 else np.nan
tot = dict(code=0, category='Total', area_ha=led.area_ha.sum(), pct_area=led.pct_area.sum(),
           AGC_2000_TgC=led.AGC_2000_TgC.sum(), AGC_2022_TgC=led.AGC_2022_TgC.sum(), change_TgC=net,
           share_of_net_change_pct=100.0, rate_mean_MgC_ha_yr=np.nanmean(rate[universe]))
led = pd.concat([led, pd.DataFrame([tot])], ignore_index=True)
# closure checks
assert abs(led.area_ha.iloc[:-1].sum() - universe.sum() * px_ha) < 1e-3 * universe.sum() * px_ha
led['forest_2000_ha'] = np.nan; led['forest_2022_ha'] = np.nan
led.loc[led.code == 0, 'forest_2000_ha'] = forest0.sum() * px_ha
led.loc[led.code == 0, 'forest_2022_ha'] = forest1.sum() * px_ha
led.to_csv(os.path.join(OUT_DIR, 'ledger_table.csv'), index=False)
print("\nLedger (closes in area and carbon):\n" + led.round(3).to_string(index=False))

p2 = prof.copy(); p2.update(count=1, dtype='uint8', nodata=0, compress='deflate', tiled=True)
with rasterio.open(os.path.join(OUT_DIR, f'ledger_{Y0}_{Y1}.tif'), 'w', **p2) as dst:
    dst.write(ledger, 1)
    dst.set_band_description(1, ' | '.join(f'{k} {v}' for k, v in LEDGER.items()))
p3 = prof.copy(); p3.update(count=1, dtype='float32', nodata=NODATA, compress='deflate', tiled=True)
with rasterio.open(os.path.join(OUT_DIR, f'agc_change_rate_{Y0}_{Y1}.tif'), 'w', **p3) as dst:
    dst.write(np.where(np.isfinite(rate), rate, NODATA).astype('float32'), 1)
    dst.set_band_description(1, 'Annualized AGC stock change (Mg C ha-1 yr-1)')

# ── FCD transitions (alluvial) ───────────────────────────────────────────────
def fcd_class(a):
    c = np.full(a.shape, -1, dtype='int8')
    for i, (lo, hi, _) in enumerate(FCD_CLASSES):
        c[(a >= lo) & (a < hi)] = i
    return c


c0, c1 = fcd_class(fcd[Y0]), fcd_class(fcd[Y1])
labs = [c[2] for c in FCD_CLASSES]
T = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        T[i, j] = np.sum(universe & (c0 == i) & (c1 == j)) * px_ha
trans = pd.DataFrame(T, index=[f'{Y0} {l}' for l in labs], columns=[f'{Y1} {l}' for l in labs])
trans.to_csv(os.path.join(OUT_DIR, 'fcd_transitions_ha.csv'))
print("\nFCD transitions (ha):\n" + trans.round(0).to_string())

# ── Elevation summary ────────────────────────────────────────────────────────
erows = []
for lo, hi, lab in ELEV_BANDS:
    s = universe & (elev >= lo) & (elev < hi)
    f1 = s & forest1
    r = dict(elevation_band=lab, area_ha=s.sum() * px_ha, forest_2022_ha=f1.sum() * px_ha,
             AGC_2022_mean_forest_MgC_ha=agc1[f1].mean() if f1.any() else np.nan,
             AGC_2022_TgC=agc1[s].sum() * px_ha / 1e6,
             change_TgC=np.nansum(delta[s]) * px_ha / 1e6,
             rate_mean_MgC_ha_yr=np.nanmean(rate[s]) if s.any() else np.nan)
    for k, name in LEDGER.items():
        r[f'pct_{name}'] = 100 * np.sum(s & (ledger == k)) / max(s.sum(), 1)
    erows.append(r)
pd.DataFrame(erows).to_csv(os.path.join(OUT_DIR, 'elevation_summary.csv'), index=False)

# heatmap table: FCD-2000 class (forest-relevant classes) x elevation band
hrows = []
for i, (_, _, fl) in enumerate(FCD_CLASSES[1:], start=1):
    for lo, hi, el in ELEV_BANDS:
        s = universe & (c0 == i) & (elev >= lo) & (elev < hi)
        v = rate[s]
        hrows.append(dict(fcd_2000=fl, elevation_band=el, area_ha=s.sum() * px_ha,
                          mean=np.nanmean(v) if v.size else np.nan, sd=np.nanstd(v) if v.size else np.nan))
heat = pd.DataFrame(hrows)
heat.to_csv(os.path.join(OUT_DIR, 'heatmap_fcd_elevation.csv'), index=False)

# ══ Figures ══════════════════════════════════════════════════════════════════
DEC = max(1, int(max(H, W) / 1500))               # decimation for plotting
sl = (slice(None, None, DEC), slice(None, None, DEC))
ext = [grid['transform'].c, grid['transform'].c + W * grid['transform'].a,
       grid['transform'].f + H * grid['transform'].e, grid['transform'].f]


def geo_axes(ax):
    """Lon / lat tick labels on the UTM map (paper style)."""
    xc, yc = (ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2
    lon0, lat0 = warp_transform(grid['crs'], 'EPSG:4326', [ext[0], ext[1]], [yc, yc])
    lon1, lat1 = warp_transform(grid['crs'], 'EPSG:4326', [xc, xc], [ext[2], ext[3]])
    lons = np.arange(np.ceil(lon0[0] * 10) / 10, lon0[1], 0.2)
    lats = np.arange(np.ceil(lat1[0] * 10) / 10, lat1[1], 0.2)
    xs, _ = warp_transform('EPSG:4326', grid['crs'], list(lons), [lat0[0]] * len(lons))
    _, ys = warp_transform('EPSG:4326', grid['crs'], [lon1[0]] * len(lats), list(lats))
    ax.set_xticks(xs); ax.set_xticklabels([f'{v:.1f}°' for v in lons], fontsize=9)
    ax.set_yticks(ys); ax.set_yticklabels([f'{v:.1f}°' for v in lats], fontsize=9)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    if aoi_geoms is not None:
        aoi_geoms.boundary.plot(ax=ax, color='black', linewidth=0.8, zorder=5)
    style(ax)


def show(ax, arr, **kw):
    return ax.imshow(np.where(universe, arr, np.nan)[sl], extent=ext, interpolation='nearest', **kw)


# Fig 5 — alluvial FCD transitions
fig, ax = plt.subplots(figsize=(8, 6.2))
total = T.sum(); gap = 0.015 * total; xL, xR, wn = 0.0, 1.0, 0.07


def stack(v):
    y, out = 0.0, []
    for a in v[::-1]:                      # first class on top
        out.append((y, y + a)); y += a + gap
    return out[::-1]


left, right = stack(T.sum(1)), stack(T.sum(0))
lo_ptr = [l[0] for l in left]; ro_ptr = [r[0] for r in right]
order = [(i, j) for i in range(4) for j in range(4)]
off_l = {i: left[i][1] for i in range(4)}; off_r = {j: right[j][1] for j in range(4)}
t = np.linspace(0, 1, 100); sm = 3 * t ** 2 - 2 * t ** 3
for i in range(4):
    for j in range(4):
        v = T[i, j]
        if v <= 0:
            continue
        la1 = off_l[i]; la0 = la1 - v; off_l[i] = la0
        ra1 = off_r[j]; ra0 = ra1 - v; off_r[j] = ra0
        x = xL + wn + (xR - xL - 2 * wn) * t
        ax.fill_between(x, la0 + (ra0 - la0) * sm, la1 + (ra1 - la1) * sm, color=FCD_SHADES[i],
                        alpha=0.45, linewidth=0, zorder=1)
for k, (side, xs) in enumerate([(left, xL), (right, xR - wn)]):
    vals = T.sum(1) if k == 0 else T.sum(0)
    for i, (a, b) in enumerate(side):
        ax.add_patch(Rectangle((xs, a), wn, b - a, facecolor=FCD_SHADES[i], edgecolor='black', linewidth=0.6, zorder=3))
        tx = xs - 0.01 if k == 0 else xs + wn + 0.01
        ax.text(tx, (a + b) / 2, f"{labs[i]}\n{vals[i]:,.0f} ha", ha='right' if k == 0 else 'left',
                va='center', fontsize=9)
pers_dense = T[3, 3]
ax.text(0.5, left[3][0] + 0.45 * (left[3][1] - left[3][0]), f"Persistent ≥70%\n{pers_dense:,.0f} ha",
        ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=0.6), zorder=4)
ax.text(xL + wn / 2, total + 4 * gap, str(Y0), ha='center', fontweight='bold')
ax.text(xR - wn / 2, total + 4 * gap, str(Y1), ha='center', fontweight='bold')
ax.set_xlim(-0.22, 1.22); ax.set_ylim(-gap, total + 7 * gap)
ax.axis('off')
ax.set_title(f'A. Canopy density transitions {Y0}–{Y1} ({AREA_NAME})', loc='left', fontweight='bold')
fig.text(0.5, 0.04, 'Ribbon width proportional to area. Forest = FCD ≥ 30 %.', ha='center', fontsize=9)
save(fig, 'fig_fcd_alluvial.tif')

# Fig 6 — A annualized change, B AGC 2022, C FCD 2022 (one row, paper style)
fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
fig.subplots_adjust(wspace=0.5)
lim = np.nanpercentile(np.abs(rate[universe]), 98)
panels = [(axes[0], rate, dict(cmap=CHANGE_CMAP, vmin=-lim, vmax=lim),
           'Annualized AGC stock change\n(Mg C ha$^{-1}$ yr$^{-1}$)', f'A. AGC change {Y0}–{Y1}'),
          (axes[1], np.where(forest1, agc1, 0), dict(cmap='Greys', vmin=0, vmax=np.nanpercentile(agc1[forest1], 99)),
           'AGC (Mg C ha$^{-1}$)', f'B. AGC {Y1}'),
          (axes[2], fcd[Y1], dict(cmap='Greys', vmin=0, vmax=100), 'FCD (%)', f'C. FCD {Y1}')]
for ax, arr, kw, cl, ttl in panels:
    im = show(ax, arr, **kw)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.85)
    cb.set_label(cl); cb.ax.tick_params(direction='in')
    ax.set_title(ttl, loc='left', fontweight='bold'); geo_axes(ax)
save(fig, 'fig_agc_change_maps.tif')

# Fig 7 — bivariate map elevation x AGC change (two hues)
chg_cls = np.full((H, W), -1, dtype='int8')
dom = universe & (forest0 | forest1)
chg_cls[dom & (delta < -THR)] = 0
chg_cls[dom & (np.abs(delta) <= THR)] = 1
chg_cls[dom & (delta > THR)] = 2
el_cls = np.full((H, W), -1, dtype='int8')
for i, (lo, hi, _) in enumerate(BIV_ELEV):
    el_cls[(elev >= lo) & (elev < hi)] = i
biv = np.where((chg_cls >= 0) & (el_cls >= 0), chg_cls * 3 + el_cls, -1)
flat = [BIV_COLORS[r][c] for r in range(3) for c in range(3)]
cmap_b = ListedColormap(['white'] + flat)
fig = plt.figure(figsize=(10, 7.5))
ax = fig.add_axes([0.05, 0.08, 0.68, 0.86])
ax.imshow(np.where(universe, biv + 1, np.nan)[sl], extent=ext, cmap=cmap_b,
          norm=BoundaryNorm(np.arange(-0.5, 10.5), 10), interpolation='nearest')
ax.set_title(f'A. Elevation and AGC change {Y0}–{Y1}', loc='left', fontweight='bold'); geo_axes(ax)
lg = fig.add_axes([0.80, 0.55, 0.15, 0.2])
for r in range(3):
    for c in range(3):
        lg.add_patch(Rectangle((c, r), 1, 1, facecolor=BIV_COLORS[r][c], edgecolor='white', linewidth=1))
lg.set_xlim(0, 3); lg.set_ylim(0, 3)
lg.set_xticks([0.5, 1.5, 2.5]); lg.set_xticklabels([b[2] for b in BIV_ELEV], fontsize=8, rotation=30, ha='right')
lg.set_yticks([0.5, 1.5, 2.5]); lg.set_yticklabels(['Loss', 'Stable', 'Gain'], fontsize=8)
lg.set_xlabel('Elevation (SRTM) →', fontsize=9, labelpad=2); lg.set_ylabel('AGC change →', fontsize=9, labelpad=2)
for s in lg.spines.values():
    s.set_visible(False)
lg.tick_params(length=0)
fig.text(0.76, 0.40, f'Gain / loss: |ΔAGC| > {THR:.0f} Mg C ha$^{{-1}}$\n(1.96 × RMSE of change,\nre-measured plots)\n'
                     'White: non-forest both years', fontsize=8, va='top')
save(fig, 'fig_bivariate_elev_agc.tif')

# SI — heatmap FCD 2000 x elevation (grayscale, values annotated)
hm = heat.pivot(index='fcd_2000', columns='elevation_band', values='mean').reindex(
    index=[c[2] for c in FCD_CLASSES[1:]][::-1], columns=[b[2] for b in ELEV_BANDS])
hs = heat.pivot(index='fcd_2000', columns='elevation_band', values='sd').reindex_like(hm)
ha_ = heat.pivot(index='fcd_2000', columns='elevation_band', values='area_ha').reindex_like(hm)
fig, ax = plt.subplots(figsize=(8.5, 4.4))
vlim = np.nanmax(np.abs(hm.values))
im = ax.imshow(hm.values, cmap=CHANGE_CMAP, vmin=-vlim, vmax=vlim, aspect='auto')
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        v = hm.values[i, j]
        if np.isfinite(v):
            dark = abs(v) > 0.5 * vlim
            ax.text(j, i, f"{v:.2f}\n({hs.values[i, j]:.2f})\n{ha_.values[i, j] / 1e3:.1f} kha", ha='center',
                    va='center', fontsize=8, color='white' if dark else 'black')
ax.set_xticks(range(hm.shape[1])); ax.set_xticklabels(hm.columns)
ax.set_yticks(range(hm.shape[0])); ax.set_yticklabels(hm.index)
ax.set_xlabel('Elevation (m a.s.l.)'); ax.set_ylabel(f'FCD class in {Y0}')
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label('Mean annualized AGC change\n(Mg C ha$^{-1}$ yr$^{-1}$)')
ax.set_title('A. AGC change by canopy density and elevation', loc='left', fontweight='bold')
style(ax)
fig.text(0.01, -0.03, 'Cells: mean (SD) and area (thousand ha). Empty: no pixels.', fontsize=9)
save(fig, 'fig_si_heatmap_fcd_elev.tif')

# Fig 8 — ledger map (grayscale + hatching) with area / carbon table
fig = plt.figure(figsize=(11.5, 7.5))
ax = fig.add_axes([0.04, 0.08, 0.62, 0.86])
cm = ListedColormap([LEDGER_STYLE[k][0] for k in LEDGER])
ax.imshow(np.where(universe, ledger, np.nan)[sl], extent=ext, cmap=cm,
          norm=BoundaryNorm(np.arange(0.5, len(LEDGER) + 1.5), len(LEDGER)), interpolation='nearest')
xx = np.linspace(ext[0], ext[1], ledger[sl].shape[1]); yy = np.linspace(ext[3], ext[2], ledger[sl].shape[0])
for k, (_, h) in LEDGER_STYLE.items():
    if h:
        ax.contourf(xx, yy, np.where((ledger == k) & universe, 1.0, np.nan)[sl], levels=[0.5, 1.5],
                    colors='none', hatches=[h], zorder=3)
ax.set_title(f'A. AGC ledger {Y0}–{Y1}', loc='left', fontweight='bold'); geo_axes(ax)
lax = fig.add_axes([0.69, 0.3, 0.3, 0.6]); lax.axis('off')
for n, k in enumerate(LEDGER):
    r = led[led.code == k].iloc[0]
    y = 1 - n * 0.15
    lax.add_patch(Rectangle((0, y - 0.06), 0.1, 0.08, facecolor=LEDGER_STYLE[k][0], edgecolor='black',
                            linewidth=0.6, hatch=LEDGER_STYLE[k][1]))
    lax.text(0.14, y - 0.02, f"{LEDGER[k]}\n{r.area_ha:,.0f} ha · {r.change_TgC:+.2f} Tg C",
             va='center', fontsize=9)
lax.set_xlim(0, 1); lax.set_ylim(0.05, 1.08)
tt = led[led.code == 0].iloc[0]
fig.text(0.69, 0.24, f"Net change {tt.change_TgC:+.2f} Tg C\n({tt.AGC_2000_TgC:.2f} → {tt.AGC_2022_TgC:.2f} Tg C)\n"
                     f"Gain / loss: |ΔAGC| > {THR:.0f} Mg C ha$^{{-1}}$", fontsize=9, va='top')
save(fig, 'fig_agc_ledger.tif')

if os.path.exists(MODEL_JS):
    with open(MODEL_JS) as f:
        print(f"\nScenario: {json.load(f)['scenario']}")
print(f"Results: {OUT_DIR}\nDone.")
