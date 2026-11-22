r"""
09_rbca_recortes.py
Reporting inside the Chocó Andino Biosphere Reserve (RBCA). Reads what 07 / 08 produced
(prediction and training are NOT re-run), clips it to the RBCA polygon, extracts the
statistics and redraws the result figures for the RBCA.

Inputs
  DATA\raw\RBCA_shp\RBiosfera_ChocoAndino.shp          RBCA boundary
  DATA\interim\srtm-dem\srtm_dem.tif                    elevation
  DATA\interim\training_oct2025_predictors.csv          USFQ plots
  DATA\interim\split_temporal_validation.csv            re-measured sites (05)
  DATA\raw\FCD\landtrendr_fitted_prediction_<yr>.tif
  RESULTADOS\05_mapas\agc_<yr>.tif, mask_<yr>.tif       (07)
  RESULTADOS\08_ledger\change_threshold.csv             gain / loss threshold (08; recomputed if missing)

Outputs  RESULTADOS\recortes\
  rasters\  agc, mask, fcd, elevation, ledger_2000_2022, agc_change_rate_2000_2022 (+07c/08 rasters)
  rbca_area_check.csv, rbca_agc_by_year.csv, rbca_agc_by_elevation.csv, rbca_agc_by_fcd_class.csv,
  rbca_ledger_2000_2022.csv, rbca_fcd_transitions_ha.csv, rbca_heatmap_fcd_elevation.csv,
  rbca_cohort_trajectory.csv, rbca_cohort_trends.csv, rbca_forest_area_by_year.csv,
  rbca_fcd_classes_by_year.csv, rbca_plots.csv
Figures  FIGURAS\recortes\
  fig_agc_maps_rbca, fig_fcd_maps_rbca, fig_agc_change_maps_rbca (Fig 6), fig_fcd_alluvial_rbca (Fig 5),
  fig_bivariate_elev_agc_rbca (Fig 7), fig_si_heatmap_fcd_elev_rbca, fig_agc_ledger_rbca (Fig 8),
  fig_fcd_cohort_trend_rbca, fig_cohort_trend_bars_rbca, fig_forest_area_trend_rbca,
  fig_fcd_classes_rbca, fig_agc_trajectory_rbca
Usage   python 09_rbca_recortes.py
"""

import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling, transform as warp_transform
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 11

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = r"D:\PAPER-ECOLIN"
RBCA_SHP  = os.path.join(ROOT, "DATA", "raw", "RBCA_shp", "RBiosfera_ChocoAndino.shp")
SRTM      = os.path.join(ROOT, "DATA", "interim", "srtm-dem", "srtm_dem.tif")
CSV_PTS   = os.path.join(ROOT, "DATA", "interim", "training_oct2025_predictors.csv")
CSV_TEMP  = os.path.join(ROOT, "DATA", "interim", "split_temporal_validation.csv")
FCD_DIR   = os.path.join(ROOT, "DATA", "raw", "FCD")
RES_DIR   = os.path.join(ROOT, "RESULTADOS")
MAPS_DIR  = os.path.join(RES_DIR, "05_mapas")
OPT_DIRS  = [os.path.join(RES_DIR, "07c_tendencia_fcd"), os.path.join(RES_DIR, "08_ledger")]
OUT_DIR   = os.path.join(RES_DIR, "recortes")
OUT_RAS   = os.path.join(OUT_DIR, "rasters")
FIG_DIR   = os.path.join(ROOT, "FIGURAS", "recortes")
for d in [OUT_DIR, OUT_RAS, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Settings ─────────────────────────────────────────────────────────────────
YEARS      = [2000, 2006, 2012, 2016, 2022]
NODATA     = -9999.0
ELEV_BANDS = [(-np.inf, 500, '<500'), (500, 1000, '500–1000'), (1000, 2000, '1000–2000'),
              (2000, 3000, '2000–3000'), (3000, np.inf, '>3000')]
FCD_CLASSES = [(-np.inf, 10, '<10%'), (10, 30, '10–30%'), (30, 70, '30–70%'), (70, np.inf, '≥70%')]
# hypsometric classes and colours of the study-area map (as in the original Figure 2)
HYPSO_BREAKS = [0, 1200, 1800, 2500, 3000, 3500, 4000, 6000]
HYPSO_COLORS = ['#6fa8b8', '#a7cdbf', '#d6e8cf', '#f7f3c6', '#ecd9b0', '#d9b996', '#c49a82']
LABELS_GEO = [(-78.70, 0.245, 'Guayllabamba river'), (-78.60, -0.17, 'Pichincha\nvolcano')]
SCALE_KM   = 14


def style(ax):
    ax.tick_params(direction='in', length=4)
    for s in ax.spines.values():
        s.set_linewidth(0.8); s.set_color('black')


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
    plt.close(fig)
    print(f"  figure: FIGURAS\\recortes\\{name}")


# ── Reference grid (07 maps) and RBCA mask ───────────────────────────────────
ref = os.path.join(MAPS_DIR, f'agc_{YEARS[-1]}.tif')
if not os.path.exists(ref):
    ref = SRTM
with rasterio.open(ref) as s:
    G = dict(crs=s.crs, transform=s.transform, width=s.width, height=s.height)
rbca = gpd.read_file(RBCA_SHP).to_crs(G['crs'])
full = rasterize([(g, 1) for g in rbca.geometry], out_shape=(G['height'], G['width']),
                 transform=G['transform'], fill=0, dtype='uint8').astype(bool)
rr, cc = np.where(full)
win = (slice(rr.min(), rr.max() + 1), slice(cc.min(), cc.max() + 1))
M = full[win]                                             # RBCA mask on the cropped window
TR = G['transform'] * rasterio.Affine.translation(win[1].start, win[0].start)
H, W = M.shape
px_ha = abs(TR.a * TR.e) / 1e4
EXT = [TR.c, TR.c + W * TR.a, TR.f + H * TR.e, TR.f]
poly_ha = rbca.to_crs(rbca.estimate_utm_crs()).area.sum() / 1e4
pd.DataFrame([dict(polygon_area_ha=poly_ha, pixel_area_ha=M.sum() * px_ha, n_pixels=int(M.sum()),
                   pixel_size_m=abs(TR.a))]).to_csv(os.path.join(OUT_DIR, 'rbca_area_check.csv'), index=False)
print(f"RBCA: polygon {poly_ha:,.0f} ha | pixels {M.sum() * px_ha:,.0f} ha ({M.sum():,} px of {abs(TR.a):.0f} m)")


def read_clip(path, band=1, resampling=Resampling.bilinear):
    """Any raster -> cropped RBCA window of the reference grid; NaN outside the polygon / NoData."""
    out = np.full((H, W), np.nan, dtype='float32')
    with rasterio.open(path) as s:
        reproject(source=rasterio.band(s, band), destination=out, src_transform=s.transform,
                  src_crs=s.crs, src_nodata=s.nodata, dst_transform=TR, dst_crs=G['crs'],
                  dst_nodata=np.nan, resampling=resampling)
    out[~M] = np.nan
    return out


def write_clip(name, arrays, descs, dtype='float32', nodata=NODATA):
    prof = dict(driver='GTiff', height=H, width=W, count=len(arrays), dtype=dtype, crs=G['crs'],
                transform=TR, nodata=nodata, compress='deflate', tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(os.path.join(OUT_RAS, name), 'w', **prof) as dst:
        for i, (a, d) in enumerate(zip(arrays, descs), 1):
            dst.write(np.where(np.isfinite(a), a, nodata).astype(dtype), i)
            dst.set_band_description(i, d)


def geo_axes(ax, step=0.15, outline=True, pad=0.0):
    dx, dy = pad * (EXT[1] - EXT[0]), pad * (EXT[3] - EXT[2])
    E = [EXT[0] - dx, EXT[1] + dx, EXT[2] - dy, EXT[3] + dy]
    xc, yc = (E[0] + E[1]) / 2, (E[2] + E[3]) / 2
    lon, _ = warp_transform(G['crs'], 'EPSG:4326', [E[0], E[1]], [yc, yc])
    _, lat = warp_transform(G['crs'], 'EPSG:4326', [xc, xc], [E[2], E[3]])
    lons = np.arange(np.ceil(lon[0] / step) * step, lon[1], step)
    lats = np.arange(np.ceil(lat[0] / step) * step, lat[1], step)
    xs, _ = warp_transform('EPSG:4326', G['crs'], list(lons), [lat[0]] * len(lons))
    _, ys = warp_transform('EPSG:4326', G['crs'], [lon[0]] * len(lats), list(lats))
    ax.set_xticks(xs); ax.set_xticklabels([f'{v:.2f}°' for v in lons], fontsize=9)
    ax.set_yticks(ys); ax.set_yticklabels([f'{v:.2f}°' for v in lats], fontsize=9)
    ax.set_xlim(E[0], E[1]); ax.set_ylim(E[2], E[3])
    if outline:
        rbca.boundary.plot(ax=ax, color='black', linewidth=0.8, zorder=6)
    ax.set_xlabel(''); ax.set_ylabel('')
    style(ax)


# ── 1. Clip rasters ──────────────────────────────────────────────────────────
print("\nClipping rasters ...")
with rasterio.open(SRTM) as s:
    d = [x.lower() if x else '' for x in s.descriptions]
    eb = d.index('elevation') + 1 if 'elevation' in d else 1
elev = read_clip(SRTM, eb)
write_clip('elevation.tif', [elev], ['Elevation SRTM (m)'])
agc, sd, mask, fcd = {}, {}, {}, {}
for yr in YEARS:
    f = read_clip(os.path.join(FCD_DIR, f'landtrendr_fitted_prediction_{yr}.tif'))
    fcd[yr] = f / 100.0 if np.nanmax(f) > 200 else f
    write_clip(f'fcd_{yr}.tif', [fcd[yr]], [f'FCD {yr} (%)'])
    pa, pm = os.path.join(MAPS_DIR, f'agc_{yr}.tif'), os.path.join(MAPS_DIR, f'mask_{yr}.tif')
    if os.path.exists(pa):
        agc[yr] = read_clip(pa, 1, Resampling.nearest); sd[yr] = read_clip(pa, 2, Resampling.nearest)
        mask[yr] = read_clip(pm, 1, Resampling.nearest)
        mask[yr][mask[yr] == 0] = np.nan
        write_clip(f'agc_{yr}.tif', [agc[yr], sd[yr]], ['AGC_MgC_ha', 'AGC_SD_MgC_ha'])
        write_clip(f'mask_{yr}.tif', [np.nan_to_num(mask[yr], nan=0)],
                   ['1 forest|2 noLandsat|3 noPALSAR|4 noSRTM|5 noFCD|6 nonforest FCD<30'], dtype='uint8', nodata=0)
for dd in OPT_DIRS:
    for p in sorted(glob.glob(os.path.join(dd, '*.tif'))):
        with rasterio.open(p) as s:
            cat = s.dtypes[0] == 'uint8'
            descs = list(s.descriptions); n = s.count
        arrs = [read_clip(p, b, Resampling.nearest if cat else Resampling.bilinear) for b in range(1, n + 1)]
        write_clip(os.path.basename(p), [np.nan_to_num(a, nan=0) if cat else a for a in arrs],
                   [x or '' for x in descs], dtype='uint8' if cat else 'float32', nodata=0 if cat else NODATA)
print(f"  rasters: {OUT_RAS}")

# ── Common pixel universe ────────────────────────────────────────────────────
# All RBCA statistics (stocks by year, elevation, FCD classes, cohorts, ledger) use the SAME
# pixels: inside the RBCA and valid (forest or non-forest, not NoData) in ALL map years.
# This is the universe of 07b (uncertainty) and of the ledger, so every table adds up.
U = M.copy()
for y in agc:
    U &= np.isin(mask[y], [1, 6])
for y in YEARS:
    U &= np.isfinite(fcd[y])
U &= np.isfinite(elev)
print(f"  common universe: {U.sum() * px_ha:,.0f} ha of {M.sum() * px_ha:,.0f} ha "
      f"(NoData in at least one year: {(M & ~U).sum() * px_ha:,.0f} ha)")

# ── 2. Statistics inside the RBCA ────────────────────────────────────────────
yrows, erows, frows = [], [], []
for yr in agc:
    m = mask[yr]
    forest, nonf, nod = U & (m == 1), U & (m == 6), M & ~U
    yrows.append(dict(year=yr, area_rbca_ha=M.sum() * px_ha, area_analysed_ha=U.sum() * px_ha,
                      area_forest_ha=forest.sum() * px_ha,
                      area_nonforest_ha=nonf.sum() * px_ha, area_nodata_ha=nod.sum() * px_ha,
                      pct_forest=100 * forest.sum() / U.sum(),
                      AGC_forest_TgC=np.nansum(agc[yr][forest]) * px_ha / 1e6,
                      AGC_mean_forest_MgC_ha=np.nanmean(agc[yr][forest]),
                      AGC_median_forest_MgC_ha=np.nanmedian(agc[yr][forest]),
                      AGC_SD_spatial_MgC_ha=np.nanstd(agc[yr][forest]),
                      AGC_pixelSD_mean_MgC_ha=np.nanmean(sd[yr][forest]),
                      FCD_mean=np.nanmean(fcd[yr][U])))
    for lo, hi, lab in ELEV_BANDS:
        b = U & (elev >= lo) & (elev < hi)
        fb = b & forest
        erows.append(dict(year=yr, elevation_band=lab, area_ha=b.sum() * px_ha, forest_ha=fb.sum() * px_ha,
                          pct_forest=100 * fb.sum() / max(b.sum(), 1),
                          AGC_TgC=np.nansum(agc[yr][fb]) * px_ha / 1e6,
                          AGC_mean_forest_MgC_ha=np.nanmean(agc[yr][fb]) if fb.any() else np.nan,
                          FCD_mean=np.nanmean(fcd[yr][b]) if b.any() else np.nan))
    for lo, hi, lab in FCD_CLASSES:
        b = U & (fcd[yr] >= lo) & (fcd[yr] < hi)
        frows.append(dict(year=yr, fcd_class=lab, area_ha=b.sum() * px_ha,
                          AGC_TgC=np.nansum(agc[yr][b]) * px_ha / 1e6,
                          AGC_mean_MgC_ha=np.nanmean(agc[yr][b]) if b.any() else np.nan,
                          elevation_mean_m=np.nanmean(elev[b]) if b.any() else np.nan))
if yrows:
    by_year = pd.DataFrame(yrows)
    by_year.to_csv(os.path.join(OUT_DIR, 'rbca_agc_by_year.csv'), index=False)
    pd.DataFrame(erows).to_csv(os.path.join(OUT_DIR, 'rbca_agc_by_elevation.csv'), index=False)
    pd.DataFrame(frows).to_csv(os.path.join(OUT_DIR, 'rbca_agc_by_fcd_class.csv'), index=False)
    print("\nRBCA by year:\n" + by_year.round(2).to_string(index=False))
else:
    print("  (no 07 maps found: AGC tables skipped)")

# USFQ plots inside the RBCA
pts = pd.read_csv(CSV_PTS)
pts = pts[pts.code == 'USFQ'].copy()
pts['Year(s)_Avitabile'] = pts['Year(s)_Avitabile'].astype(str)
sites = (pts.groupby(['x', 'y']).agg(n_visits=('AGB', 'size'),
                                     visits=('Year(s)_Avitabile', lambda v: '/'.join(sorted(v))))
         .reset_index())
remeas = set()
if os.path.exists(CSV_TEMP):
    tv = pd.read_csv(CSV_TEMP)
    remeas = set(zip(tv.x.round(6), tv.y.round(6)))
sites['temporal_validation'] = [(round(a, 6), round(b, 6)) in remeas for a, b in zip(sites.x, sites.y)]
X, Y = warp_transform('EPSG:4326', G['crs'], sites.x.tolist(), sites.y.tolist())
sites['X'], sites['Y'] = X, Y
inv = ~TR
col_row = [inv * (a, b) for a, b in zip(X, Y)]
sites['col'] = [int(np.floor(c)) for c, _ in col_row]; sites['row'] = [int(np.floor(r)) for _, r in col_row]
ok = sites.col.between(0, W - 1) & sites.row.between(0, H - 1)
sites['in_RBCA'] = False
sites.loc[ok, 'in_RBCA'] = M[sites.loc[ok, 'row'], sites.loc[ok, 'col']]
sites['elevation_m'] = np.where(sites.in_RBCA, elev[sites.row.clip(0, H - 1), sites.col.clip(0, W - 1)], np.nan)
for yr in YEARS:
    sites[f'FCD_{yr}'] = np.where(sites.in_RBCA, fcd[yr][sites.row.clip(0, H - 1), sites.col.clip(0, W - 1)], np.nan)
    if yr in agc:
        sites[f'AGC_map_{yr}'] = np.where(sites.in_RBCA, agc[yr][sites.row.clip(0, H - 1), sites.col.clip(0, W - 1)], np.nan)
sites.drop(columns=['X', 'Y', 'col', 'row']).to_csv(os.path.join(OUT_DIR, 'rbca_plots.csv'), index=False)
print(f"\nUSFQ sites: {len(sites)} | inside RBCA {sites.in_RBCA.sum()} | re-measured (temporal validation) "
      f"{sites.temporal_validation.sum()}")


# ── 3. AGC change 2000-2022 and ledger inside the RBCA ───────────────────────
Y0, Y1 = YEARS[0], YEARS[-1]
NY = Y1 - Y0
C_FACTOR = 0.456
HAVE_CHANGE = Y0 in agc and Y1 in agc
if HAVE_CHANGE:
    # U (common universe, defined above) = same pixels as 07b / 08
    f0, f1 = U & (mask[Y0] == 1), U & (mask[Y1] == 1)
    delta = np.where(U, agc[Y1] - agc[Y0], np.nan)
    rate = delta / NY
    # gain / loss threshold = the one of 08 (1.96 x RMSE of change at the re-measured sites)
    thr_csv = os.path.join(RES_DIR, '08_ledger', 'change_threshold.csv')
    if os.path.exists(thr_csv):
        THR = float(pd.read_csv(thr_csv).threshold_MgC_ha.iloc[0])
    else:
        vp = pd.read_csv(os.path.join(RES_DIR, '06_validacion_mapas', 'map_validation_points.csv')).dropna(
            subset=['AGB_map_pixel'])
        yv = sorted(vp.map_year.unique())
        wm = vp.pivot_table(index='site_id', columns='map_year', values='AGB_map_pixel')
        wf = vp.pivot_table(index='site_id', columns='map_year', values='AGB_field')
        e = ((wm[yv[-1]] - wm[yv[0]]) - (wf[yv[-1]] - wf[yv[0]])).dropna() * C_FACTOR
        THR = 1.96 * float(np.sqrt(np.mean(e ** 2)))
    print(f"\nGain / loss threshold: |dAGC| > {THR:.1f} Mg C/ha")
    LEDGER = {1: 'Non-forest', 2: 'Stable reservoir', 3: 'Active sink', 4: 'Net source'}   # as 08
    LEDGER_COL = {1: '#efefef', 2: '#c3df62', 3: '#3554d1', 4: '#990000'}
    led = np.zeros((H, W), dtype='uint8')
    led[U & ~f0 & ~f1] = 1
    pers = f0 & f1
    led[pers & (np.abs(delta) <= THR)] = 2                       # stable reservoir
    led[(pers & (delta > THR)) | (U & ~f0 & f1)] = 3             # active sink (gain + new forest)
    led[(pers & (delta < -THR)) | (U & f0 & ~f1)] = 4            # net source (loss + forest lost)
    write_clip('ledger_2000_2022.tif', [led], [' | '.join(f'{k} {v}' for k, v in LEDGER.items())],
               dtype='uint8', nodata=0)
    write_clip('agc_change_rate_2000_2022.tif', [rate], ['Annualized AGC stock change (Mg C ha-1 yr-1)'])
    rows = []
    for k, name in LEDGER.items():
        b = led == k
        a0, a1 = np.nansum(agc[Y0][b]) * px_ha / 1e6, np.nansum(agc[Y1][b]) * px_ha / 1e6
        rows.append(dict(code=k, category=name, area_ha=b.sum() * px_ha, pct_area=100 * b.sum() / U.sum(),
                         AGC_2000_TgC=a0, AGC_2022_TgC=a1, change_TgC=a1 - a0,
                         rate_mean_MgC_ha_yr=np.nanmean(rate[b]) if b.any() else np.nan))
    lt = pd.DataFrame(rows)
    net = lt.change_TgC.sum()
    lt['share_of_net_change_pct'] = 100 * lt.change_TgC / net if net else np.nan
    lt.loc[len(lt)] = dict(code=0, category='Total', area_ha=lt.area_ha.sum(), pct_area=100.0,
                           AGC_2000_TgC=lt.AGC_2000_TgC.sum(), AGC_2022_TgC=lt.AGC_2022_TgC.sum(),
                           change_TgC=net, rate_mean_MgC_ha_yr=np.nanmean(rate[U]), share_of_net_change_pct=100.0)
    lt.to_csv(os.path.join(OUT_DIR, 'rbca_ledger_2000_2022.csv'), index=False)
    print("Ledger inside the RBCA:\n" + lt.round(3).to_string(index=False))

    # FCD transitions 2000 -> 2022
    def fcd_cls(a):
        c = np.full(a.shape, -1, dtype='int8')
        for i, (lo, hi, _) in enumerate(FCD_CLASSES):
            c[(a >= lo) & (a < hi)] = i
        return c
    c0, c1 = fcd_cls(fcd[Y0]), fcd_cls(fcd[Y1])
    TT = np.array([[np.sum(U & (c0 == i) & (c1 == j)) * px_ha for j in range(4)] for i in range(4)])
    labs = [c[2] for c in FCD_CLASSES]
    pd.DataFrame(TT, index=[f'{Y0} {l}' for l in labs], columns=[f'{Y1} {l}' for l in labs]).to_csv(
        os.path.join(OUT_DIR, 'rbca_fcd_transitions_ha.csv'))

    # heatmap table: FCD 2000 class x elevation band
    hrows = []
    for i, (_, _, fl) in enumerate(FCD_CLASSES[1:], start=1):
        for lo, hi, el in ELEV_BANDS:
            b = U & (c0 == i) & (elev >= lo) & (elev < hi)
            hrows.append(dict(fcd_2000=fl, elevation_band=el, area_ha=b.sum() * px_ha,
                              mean=np.nanmean(rate[b]) if b.any() else np.nan,
                              sd=np.nanstd(rate[b]) if b.any() else np.nan))
    heat = pd.DataFrame(hrows)
    heat.to_csv(os.path.join(OUT_DIR, 'rbca_heatmap_fcd_elevation.csv'), index=False)

# Cohorts by FCD 2000 (as 07c) and forest area by year
COH = [(1, 'Non-forest (<10%)', -np.inf, 10), (2, 'Degraded (10-30%)', 10, 30),
       (3, 'Intermediate (30-70%)', 30, 70), (4, 'Dense (>=70%)', 70, np.inf)]
COH_STYLE = {1: dict(color='#b0b0b0', ls=':', marker='v'), 2: dict(color='#808080', ls='-.', marker='s'),
             3: dict(color='#505050', ls='--', marker='^'), 4: dict(color='black', ls='-', marker='o')}
yrs_a = [y for y in YEARS if y in agc]
UC = U.copy()                                        # common universe
t = np.array(YEARS, float); tc = (t - t.mean())[:, None]; den = (tc ** 2).sum()
F = np.stack([fcd[y][UC] for y in YEARS])
A = np.stack([np.where(mask[y][UC] == 6, 0, agc[y][UC]) for y in yrs_a]) if len(yrs_a) == len(YEARS) else None
fs = ((F - F.mean(0)) * tc).sum(0) / den
as_ = ((A - A.mean(0)) * tc).sum(0) / den if A is not None else None
traj, trends = [], []
for code, name, lo, hi in COH:
    sel = (F[0] >= lo) & (F[0] < hi)
    if not sel.any():
        continue
    for i, y in enumerate(YEARS):
        traj.append(dict(cohort=name, code=code, year=y, area_ha=sel.sum() * px_ha, FCD_mean=F[i][sel].mean(),
                         FCD_p25=np.percentile(F[i][sel], 25), FCD_p75=np.percentile(F[i][sel], 75),
                         pct_forest=100 * np.mean(F[i][sel] >= 30),
                         AGC_mean_MgC_ha=A[i][sel].mean() if A is not None else np.nan))
    trends.append(dict(cohort=name, code=code, area_ha=sel.sum() * px_ha, FCD_slope=fs[sel].mean(),
                       pct_FCD_up=100 * np.mean(fs[sel] > 0.1),
                       AGC_slope=as_[sel].mean() if as_ is not None else np.nan,
                       pct_AGC_up=100 * np.mean(as_[sel] > 0) if as_ is not None else np.nan))
traj, trends = pd.DataFrame(traj), pd.DataFrame(trends)
traj.to_csv(os.path.join(OUT_DIR, 'rbca_cohort_trajectory.csv'), index=False)
trends.to_csv(os.path.join(OUT_DIR, 'rbca_cohort_trends.csv'), index=False)
area = pd.DataFrame([dict(year=y, forest_ha=np.sum(UC & (fcd[y] >= 30)) * px_ha,
                          dense_ha=np.sum(UC & (fcd[y] >= 70)) * px_ha,
                          intermediate_ha=np.sum(UC & (fcd[y] >= 30) & (fcd[y] < 70)) * px_ha) for y in YEARS])
area.to_csv(os.path.join(OUT_DIR, 'rbca_forest_area_by_year.csv'), index=False)

# FCD classes per year (common universe)
fcls = pd.DataFrame([dict(year=y, **{lab: np.sum(U & (fcd[y] >= lo) & (fcd[y] < hi)) * px_ha
                                     for lo, hi, lab in FCD_CLASSES}) for y in YEARS])
fcls.to_csv(os.path.join(OUT_DIR, 'rbca_fcd_classes_by_year.csv'), index=False)

# ══ 4. Figures (paper style), all inside the RBCA ════════════════════════════
DEC = max(1, int(max(H, W) / 1400))
sl = (slice(None, None, DEC), slice(None, None, DEC))
ND_COL = '#cfe0f0'


def show(ax, arr, **kw):
    return ax.imshow(arr[sl], extent=EXT, interpolation='nearest', **kw)


def cbar(fig, im, ax, label, **kw):
    cb = fig.colorbar(im, ax=ax, **{**dict(fraction=0.046, pad=0.03), **kw})
    cb.set_label(label); cb.ax.tick_params(direction='in')
    return cb


def year_panels(data, vmax, cl, name, nd_mask=None):
    yrs = list(data)
    fig, axes = plt.subplots(1, len(yrs), figsize=(3.3 * len(yrs), 4.0), squeeze=False)
    for ax, yr, letter in zip(axes[0], yrs, 'ABCDE'):
        if nd_mask is not None:
            show(ax, np.where(nd_mask[yr], 1, np.nan), cmap=ListedColormap([ND_COL]), vmin=0, vmax=1)
        im = show(ax, data[yr], cmap='Greys', vmin=0, vmax=vmax)
        ax.set_title(f'{letter}. {yr}', loc='left', fontweight='bold')
        geo_axes(ax, step=0.3)
        if ax is not axes[0][0]:
            ax.set_yticklabels([])
    cbar(fig, im, axes[0].tolist(), cl, fraction=0.015, pad=0.01)
    if nd_mask is not None:                              # legend centred below the panels
        fig.legend(handles=[Patch(fc='white', ec='#808080'), Patch(fc=ND_COL, ec='#808080')],
                   labels=['Non-forest (FCD < 30 %)', 'NoData'], loc='lower center', ncol=2, fontsize=9,
                   frameon=True, edgecolor='black', bbox_to_anchor=(0.45, 0.04))
    save(fig, name)


print("\nFigures ...")
if agc:
    vmax = np.nanpercentile(np.concatenate([agc[y][M & (mask[y] == 1)] for y in agc]), 99)
    year_panels({y: np.where(U, agc[y], np.nan) for y in agc}, vmax,
                'AGC (Mg C ha$^{-1}$)', 'fig_agc_maps_rbca.tif',
                nd_mask={y: M & ~U for y in agc})
year_panels(fcd, 100, 'FCD (%)', 'fig_fcd_maps_rbca.tif')

if HAVE_CHANGE:
    # Figure 6: three equal panels in one row, horizontal colour bars underneath
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2))
    fig.subplots_adjust(wspace=0.12)
    lim = np.nanpercentile(np.abs(rate[U]), 98)
    show(axes[0], np.where(M & ~U, 1, np.nan), cmap=ListedColormap([ND_COL]), vmin=0, vmax=1)
    panels = [(axes[0], np.where(U, rate, np.nan), dict(cmap='RdBu', vmin=-lim, vmax=lim),
               'Annualized AGC stock change (Mg C ha$^{-1}$ yr$^{-1}$)', f'A. AGC change {Y0}–{Y1}'),
              (axes[1], np.where(U, agc[Y1], np.nan),
               dict(cmap='Greys', vmin=0, vmax=vmax), 'AGC (Mg C ha$^{-1}$)', f'B. AGC {Y1}'),
              (axes[2], fcd[Y1], dict(cmap='Greys', vmin=0, vmax=100), 'FCD (%)', f'C. FCD {Y1}')]
    for k, (ax, arr, kw, cl, ttl) in enumerate(panels):
        im = show(ax, arr, **kw)
        cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.08, aspect=30)
        cb.set_label(cl); cb.ax.tick_params(direction='in')
        ax.set_title(ttl, loc='left', fontweight='bold'); geo_axes(ax, step=0.2)
        if k:
            ax.set_yticklabels([])
    save(fig, 'fig_agc_change_maps_rbca.tif')

    # Figure 5: alluvial
    fig, ax = plt.subplots(figsize=(8, 6.2))
    SH = ['#e0e0e0', '#b0b0b0', '#707070', '#303030']
    tot = TT.sum(); gap = 0.015 * tot; wn = 0.07

    def stack(v):
        y, out = 0.0, []
        for a in v[::-1]:
            out.append((y, y + a)); y += a + gap
        return out[::-1]
    left, right = stack(TT.sum(1)), stack(TT.sum(0))
    ol = {i: left[i][1] for i in range(4)}; orr = {j: right[j][1] for j in range(4)}
    mid = {}                                   # centre line of every ribbon (to place flow labels)
    tt = np.linspace(0, 1, 100); sm = 3 * tt ** 2 - 2 * tt ** 3
    for i in range(4):
        for j in range(4):
            v = TT[i, j]
            if v <= 0:
                continue
            l1 = ol[i]; l0 = l1 - v; ol[i] = l0
            r1 = orr[j]; r0 = r1 - v; orr[j] = r0
            x = wn + (1 - 2 * wn) * tt
            ax.fill_between(x, l0 + (r0 - l0) * sm, l1 + (r1 - l1) * sm, color=SH[i], alpha=0.45, lw=0, zorder=1)
            mid[(i, j)] = lambda xf, l0=l0, l1=l1, r0=r0, r1=r1: (
                (l0 + l1) / 2 + ((r0 + r1) / 2 - (l0 + l1) / 2) * (3 * xf ** 2 - 2 * xf ** 3))
    for k, (side, xs) in enumerate([(left, 0.0), (right, 1 - wn)]):
        vals = TT.sum(1) if k == 0 else TT.sum(0)
        for i, (a, b) in enumerate(side):
            ax.add_patch(Rectangle((xs, a), wn, b - a, facecolor=SH[i], edgecolor='black', linewidth=0.6, zorder=3))
            ax.text(xs - 0.01 if k == 0 else xs + wn + 0.01, (a + b) / 2, f"{labs[i]}\n{vals[i]:,.0f} ha",
                    ha='right' if k == 0 else 'left', va='center', fontsize=9)
    box = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=0.6)
    ax.text(0.5, left[3][0] + 0.45 * (left[3][1] - left[3][0]), f"Persistent dense forest (≥70%)\n{TT[3, 3]:,.0f} ha",
            ha='center', va='center', fontsize=9, zorder=4, bbox=box)
    # main transitions: open land becoming forest, and intermediate forest becoming dense
    j_big = 2 if TT[0, 2] >= TT[0, 3] else 3
    if (0, j_big) in mid:
        xf = 0.30
        ax.text(wn + (1 - 2 * wn) * xf, mid[(0, j_big)](xf), f"<10% → forest (≥30%)\n{TT[0, 2] + TT[0, 3]:,.0f} ha",
                ha='center', va='center', fontsize=9, zorder=4, bbox=box)
    if (2, 3) in mid:
        xf = 0.62
        ax.text(wn + (1 - 2 * wn) * xf, mid[(2, 3)](xf), f"30–70% → ≥70%\n{TT[2, 3]:,.0f} ha",
                ha='center', va='center', fontsize=9, zorder=4, bbox=box)
    ax.text(wn / 2, tot + 4 * gap, str(Y0), ha='center', fontweight='bold')
    ax.text(1 - wn / 2, tot + 4 * gap, str(Y1), ha='center', fontweight='bold')
    ax.set_xlim(-0.22, 1.22); ax.set_ylim(-gap, tot + 7 * gap); ax.axis('off')
    ax.set_title(f'A. Canopy density transitions {Y0}–{Y1} (RBCA)', loc='left', fontweight='bold')
    fig.text(0.5, 0.04, 'Ribbon width proportional to area. Forest = FCD ≥ 30 %.', ha='center', fontsize=9)
    save(fig, 'fig_fcd_alluvial_rbca.tif')

    # Figure 7: bivariate elevation x AGC change (two hues)
    BIV_ELEV = [(-np.inf, 1000, '<1000 m'), (1000, 2000, '1000–2000 m'), (2000, np.inf, '≥2000 m')]
    BIV = [['#e8e8e8', '#ace4e4', '#5ac8c8'], ['#dfb0d6', '#a5add3', '#5698b9'], ['#be64ac', '#8c62aa', '#3b4994']]
    dom = U & (f0 | f1)
    cc_ = np.full((H, W), -1, 'int8')
    cc_[dom & (delta < -THR)] = 0; cc_[dom & (np.abs(delta) <= THR)] = 1; cc_[dom & (delta > THR)] = 2
    ee = np.full((H, W), -1, 'int8')
    for i, (lo, hi, _) in enumerate(BIV_ELEV):
        ee[(elev >= lo) & (elev < hi)] = i
    biv = np.where((cc_ >= 0) & (ee >= 0), cc_ * 3 + ee, -1)
    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_axes([0.05, 0.08, 0.68, 0.86])
    show(ax, np.where(U, biv + 1, np.nan), cmap=ListedColormap(['white'] + [BIV[r][c] for r in range(3) for c in range(3)]),
         norm=BoundaryNorm(np.arange(-0.5, 10.5), 10))
    ax.set_title(f'A. Elevation and AGC change {Y0}–{Y1}', loc='left', fontweight='bold'); geo_axes(ax)
    lg = fig.add_axes([0.80, 0.55, 0.15, 0.2])
    for r in range(3):
        for c in range(3):
            lg.add_patch(Rectangle((c, r), 1, 1, facecolor=BIV[r][c], edgecolor='white', linewidth=1))
    lg.set_xlim(0, 3); lg.set_ylim(0, 3)
    lg.set_xticks([0.5, 1.5, 2.5]); lg.set_xticklabels([b[2] for b in BIV_ELEV], fontsize=8, rotation=30, ha='right')
    lg.set_yticks([0.5, 1.5, 2.5]); lg.set_yticklabels(['Loss', 'Stable', 'Gain'], fontsize=8)
    lg.set_xlabel('Elevation (SRTM) →', fontsize=9); lg.set_ylabel('AGC change →', fontsize=9)
    lg.tick_params(length=0)
    for s_ in lg.spines.values():
        s_.set_visible(False)
    fig.text(0.76, 0.40, f'Gain / loss: |ΔAGC| > {THR:.0f} Mg C ha$^{{-1}}$\n(1.96 × RMSE of change,\n'
                         're-measured plots)\nWhite: non-forest both years', fontsize=8, va='top')
    save(fig, 'fig_bivariate_elev_agc_rbca.tif')

    # SI heatmap
    hm = heat.pivot(index='fcd_2000', columns='elevation_band', values='mean').reindex(
        index=[c[2] for c in FCD_CLASSES[1:]][::-1], columns=[b[2] for b in ELEV_BANDS])
    hs = heat.pivot(index='fcd_2000', columns='elevation_band', values='sd').reindex_like(hm)
    ha_ = heat.pivot(index='fcd_2000', columns='elevation_band', values='area_ha').reindex_like(hm)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    vl = np.nanmax(np.abs(hm.values))
    im = ax.imshow(hm.values, cmap='RdBu', vmin=-vl, vmax=vl, aspect='auto')
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            v = hm.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}\n({hs.values[i, j]:.2f})\n{ha_.values[i, j] / 1e3:.1f} kha", ha='center',
                        va='center', fontsize=8, color='white' if abs(v) > 0.6 * vl else 'black')
    ax.set_xticks(range(hm.shape[1])); ax.set_xticklabels(hm.columns)
    ax.set_yticks(range(hm.shape[0])); ax.set_yticklabels(hm.index)
    ax.set_xlabel('Elevation (m a.s.l.)'); ax.set_ylabel(f'FCD class in {Y0}')
    cbar(fig, im, ax, 'Mean annualized AGC change\n(Mg C ha$^{-1}$ yr$^{-1}$)', fraction=0.04, pad=0.02)
    ax.set_title('A. AGC change by canopy density and elevation (RBCA)', loc='left', fontweight='bold')
    style(ax)
    fig.text(0.01, -0.03, 'Cells: mean (SD) and area (thousand ha). Empty: no pixels.', fontsize=9)
    save(fig, 'fig_si_heatmap_fcd_elev_rbca.tif')

    # Figure 8: ledger (colour)
    fig = plt.figure(figsize=(11.5, 7.5))
    ax = fig.add_axes([0.04, 0.08, 0.62, 0.86])
    ND_LED = '#8c8c8c'          # NoData: mid grey (distinct from the light grey of non-forest)
    show(ax, np.where(M & ~U, 1, np.nan), cmap=ListedColormap([ND_LED]), vmin=0, vmax=1)
    show(ax, np.where(U, led, np.nan), cmap=ListedColormap([LEDGER_COL[k] for k in LEDGER]),
         norm=BoundaryNorm(np.arange(0.5, len(LEDGER) + 1.5), len(LEDGER)))
    ax.set_title(f'A. AGC ledger {Y0}–{Y1} (RBCA)', loc='left', fontweight='bold'); geo_axes(ax)
    lax = fig.add_axes([0.69, 0.3, 0.3, 0.6]); lax.axis('off')
    for n, k in enumerate(LEDGER):
        r = lt[lt.code == k].iloc[0]; y = 1 - n * 0.13
        lax.add_patch(Rectangle((0, y - 0.05), 0.1, 0.07, facecolor=LEDGER_COL[k], edgecolor='black', linewidth=0.6))
        lax.text(0.14, y - 0.015, f"{LEDGER[k]}\n{r.area_ha:,.0f} ha · {r.change_TgC:+.2f} Tg C", va='center', fontsize=9)
    y = 1 - len(LEDGER) * 0.13                       # NoData (no image data in at least one map year)
    lax.add_patch(Rectangle((0, y - 0.05), 0.1, 0.07, facecolor=ND_LED, edgecolor='black', linewidth=0.6))
    lax.text(0.14, y - 0.015, f"NoData\n{(M & ~U).sum() * px_ha:,.0f} ha", va='center', fontsize=9)
    lax.set_xlim(0, 1); lax.set_ylim(0.0, 1.08)
    tr_ = lt[lt.code == 0].iloc[0]
    fig.text(0.69, 0.24, f"Net change {tr_.change_TgC:+.2f} Tg C\n({tr_.AGC_2000_TgC:.2f} → {tr_.AGC_2022_TgC:.2f} Tg C)\n"
                         f"Gain / loss: |ΔAGC| > {THR:.0f} Mg C ha$^{{-1}}$", fontsize=9, va='top')
    save(fig, 'fig_agc_ledger_rbca.tif')

# Cohort trajectories (07c, RBCA)
if len(traj):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for code, name, _, _ in COH:
        g = traj[traj.code == code]
        if g.empty:
            continue
        st = COH_STYLE[code]
        # interquartile range (p25-p75) of the pixels of the cohort as vertical bars, slightly
        # offset per cohort so the four ranges do not overlap
        dx = (code - 2.5) * 0.45
        mid, half = (g.FCD_p25 + g.FCD_p75) / 2, (g.FCD_p75 - g.FCD_p25) / 2   # bar = p25..p75
        axes[0].errorbar(g.year + dx, mid, yerr=half, fmt='none', ecolor=st['color'],
                         elinewidth=1.0, capsize=3, zorder=2)
        axes[0].plot(g.year + dx, g.FCD_mean, color=st['color'], ls=st['ls'], marker=st['marker'], mfc='white',
                     zorder=3, label=name)
        axes[1].plot(g.year, g.AGC_mean_MgC_ha, color=st['color'], ls=st['ls'], marker=st['marker'], mfc='white', label=name)
    axes[0].axhline(30, color='black', linewidth=0.8, linestyle=(0, (4, 3)))
    axes[0].set_ylabel('FCD (%)'); axes[0].set_ylim(0, 100)
    axes[0].set_title('A. Canopy density of the 2000 cohorts', loc='left', fontweight='bold')
    axes[0].text(0.99, 0.02, 'Bars: interquartile range of pixels', transform=axes[0].transAxes,
                 ha='right', va='bottom', fontsize=8)
    axes[1].set_ylabel('AGC (Mg C ha$^{-1}$)')
    axes[1].set_title('B. AGC of the 2000 cohorts', loc='left', fontweight='bold')
    for ax in axes:
        ax.set_xticks(YEARS); ax.set_xlabel('Year'); style(ax); ax.grid(True, color='#e0e0e0', linewidth=0.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, title='FCD class in 2000', loc='lower center', ncol=4, fontsize=9, title_fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.08), handlelength=2.5)
    save(fig, 'fig_fcd_cohort_trend_rbca.tif')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    lbl = [c.replace(' (', '\n(') for c in trends.cohort]
    for ax, col, pct, yl, ttl in [(axes[0], 'FCD_slope', 'pct_FCD_up', 'FCD trend (%-points yr$^{-1}$)',
                                   'A. Canopy density trend 2000–2022'),
                                  (axes[1], 'AGC_slope', 'pct_AGC_up', 'AGC trend (Mg C ha$^{-1}$ yr$^{-1}$)',
                                   'B. AGC trend 2000–2022')]:
        if trends[col].isna().all():
            ax.axis('off'); continue
        bars = ax.bar(range(len(trends)), trends[col], color=[COH_STYLE[c]['color'] for c in trends.code],
                      edgecolor='black', linewidth=0.8, zorder=3)
        ax.axhline(0, color='black', linewidth=0.8)
        span = max(np.nanmax(np.abs(trends[col])), 1e-6)
        for b, v, p in zip(bars, trends[col], trends[pct]):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.04 * span if v >= 0 else -0.04 * span), f"{p:.0f}% ↑",
                    ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
        ax.set_xticks(range(len(trends))); ax.set_xticklabels(lbl, fontsize=9)
        ax.set_xlabel('FCD class in 2000'); ax.set_ylabel(yl)
        ax.set_title(ttl, loc='left', fontweight='bold')
        ax.set_ylim(min(0, np.nanmin(trends[col])) - 0.15 * span, max(0, np.nanmax(trends[col])) + 0.2 * span)
        style(ax); ax.grid(True, axis='y', color='#e0e0e0', linewidth=0.5)
    fig.text(0.01, -0.02, 'Labels: % of pixels with increasing trend.', fontsize=9)
    save(fig, 'fig_cohort_trend_bars_rbca.tif')

# Forest area by canopy density (RBCA)
fig, ax = plt.subplots(figsize=(5.8, 4))
ax.plot(area.year, area.forest_ha / 1e3, '-o', color='black', mfc='#606060', label='Forest (FCD ≥ 30%)')
ax.plot(area.year, area.dense_ha / 1e3, '--^', color='#505050', mfc='white', label='Dense (FCD ≥ 70%)')
ax.plot(area.year, area.intermediate_ha / 1e3, ':s', color='#808080', mfc='white', label='Intermediate (30–70%)')
ax.set_xticks(YEARS); ax.set_xlabel('Year'); ax.set_ylabel('Area (thousand ha)')
ax.set_title('A. Forest area by canopy density (RBCA)', loc='left', fontweight='bold')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=9)
style(ax); ax.grid(True, color='#e0e0e0', linewidth=0.5)
save(fig, 'fig_forest_area_trend_rbca.tif')

# FCD classes per year (RBCA) = panel B of fig_fcd_training_records
fig, ax = plt.subplots(figsize=(6, 4.4))
shades = ['#d9d9d9', '#a6a6a6', '#636363', '#252525']; wb = 0.2; x = np.arange(len(fcls))
for k, ((_, _, lab), c) in enumerate(zip(FCD_CLASSES, shades)):
    ax.bar(x + (k - 1.5) * wb, fcls[lab] / 1e3, wb, color=c, edgecolor='black', linewidth=0.6, label=lab, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(fcls.year); ax.set_xlabel('Year'); ax.set_ylabel('Area (thousand ha)')
ax.set_title('B. FCD classes in the RBCA', loc='left', fontweight='bold')
ax.legend(title='FCD class', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False,
          fontsize=9, title_fontsize=9)
style(ax); ax.grid(True, axis='y', color='#e0e0e0', linewidth=0.5)
save(fig, 'fig_fcd_classes_rbca.tif')

# Forest AGC trajectory (RBCA)
if yrows:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(by_year.year, by_year.AGC_forest_TgC, '-o', color='#404040', mfc='#808080', mec='black')
    ax.set_xticks(by_year.year); ax.set_xlabel('Year'); ax.set_ylabel('Forest AGC (Tg C)')
    ax.set_title('A. Forest AGC in the RBCA (FCD ≥ 30 %)', loc='left', fontweight='bold')
    style(ax); ax.grid(True, color='#e0e0e0', linewidth=0.5)
    save(fig, 'fig_agc_trajectory_rbca.tif')

# Forest AGC with uncertainty: only if 07b was run with the RBCA polygon (bootstrap cannot be clipped)
tb = os.path.join(RES_DIR, '07_incertidumbre', 'totals_by_year.csv')
print("\nNote: fig_agc_totals_uncertainty needs the bootstrap of 07b run inside the RBCA "
      f"(AOI_SHP = {RBCA_SHP} in 07b); the clipped maps cannot give its confidence intervals.")

print(f"\nTables: {OUT_DIR}\nFigures: {FIG_DIR}\nDone.")
