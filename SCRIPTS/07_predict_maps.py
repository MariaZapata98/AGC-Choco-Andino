r"""
07_predict_maps.py
Applies the SELECTED RF model (lowest internal-test RMSE, chosen in 06) to every pixel
and produces the AGB / AGC maps for 2000, 2006, 2012, 2016 and 2022, the per-pixel
uncertainty, NoData masks, a domain check against the training data, the landscape
totals and the map validation against the re-measured USFQ sites.

Everything follows the selected scenario, read from 03_modelo_final\rf_selected_model.json:
  * predictors and their order (features)
  * PCA of the 6 Landsat bands (pca_landsat.joblib from 05: same scaler + components
    fitted on train) -> PC1 (+ PC2 ... if PC1 < 80 %)
  * PALSAR is read only if the scenario uses it (HH_mean / HV_mean in features)

Grid: the Landsat export of each year (UTM 17S, 30 m, FCD extent; 02 v2).
      PALSAR, SRTM and FCD are resampled to that grid (bilinear).
PALSAR per map (same rule as training, script 03):
      2000 / 2006 / 2012 -> 2015 | 2016 -> mean 2015-2016 | 2022 -> mean 2015-2020
Prediction domain = FOREST: FCD >= 30 % and all predictors valid.
      FCD < 30 %            -> non-forest: AGB = AGC = 0 (mask code 6)
      missing image data    -> NoData -9999 (mask codes 2-5, reason stored)
Uncertainty: SD across the trees of the RF (per pixel).
Carbon: AGC = AGB x 0.456.
Map validation: split_temporal_validation.csv (05) -> column map_year
      (same 10 re-measured USFQ sites: visit 2017 -> map 2016, visit 2021 -> map 2022).

Outputs
  RESULTADOS\05_mapas\
      agb_<yr>.tif   band 1 AGB (Mg ha-1), band 2 AGB SD
      agc_<yr>.tif   band 1 AGC (Mg C ha-1), band 2 AGC SD
      mask_<yr>.tif  1 forest, predicted | 2 no Landsat | 3 no PALSAR | 4 no SRTM |
                     5 no FCD | 6 non-forest (FCD < 30 %, AGC = 0)
      map_summary.csv          area, NoData by reason, AGC totals per year
      domain_check.csv         % of predicted pixels outside the training range, per predictor
  RESULTADOS\06_validacion_mapas\
      map_validation_points.csv, map_validation_metrics.csv,
      map_change_validation.csv (map change vs field change at the re-measured sites)
  FIGURAS\
      fig_agc_maps.tif, fig_agc_trajectory.tif, fig_map_validation.tif

Usage: python 07_predict_maps.py              (all years)
       python 07_predict_maps.py 2000 2022    (only those years)
"""

import os
import sys
import glob
import json
import time
import numpy as np
import pandas as pd
import joblib
import rasterio
from rasterio.warp import reproject, Resampling, transform as warp_transform
from rasterio.merge import merge
from joblib import Parallel, delayed
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 11

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = r"D:\PAPER-ECOLIN"
LANDSAT_DIR = os.path.join(ROOT, "DATA", "interim", "landsat")
PALSAR_DIR  = os.path.join(ROOT, "DATA", "interim", "palsar-radar")
SRTM_DIR    = os.path.join(ROOT, "DATA", "interim", "srtm-dem")
FCD_DIR     = os.path.join(ROOT, "DATA", "raw", "FCD")
INTERIM     = os.path.join(ROOT, "DATA", "interim")
RES_DIR     = os.path.join(ROOT, "RESULTADOS")
FIG_DIR     = os.path.join(ROOT, "FIGURAS")
MODEL_DIR   = os.path.join(RES_DIR, "03_modelo_final")
OUT_MAPS    = os.path.join(RES_DIR, "05_mapas")
OUT_VAL     = os.path.join(RES_DIR, "06_validacion_mapas")
for d in [OUT_MAPS, OUT_VAL, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Settings ─────────────────────────────────────────────────────────────────
YEARS      = [2000, 2006, 2012, 2016, 2022]
if len(sys.argv) > 1:
    YEARS = [int(a) for a in sys.argv[1:]]
PALSAR_FOR = {2000: [2015], 2006: [2015], 2012: [2015],
              2016: [2015, 2016], 2022: [2015, 2016, 2017, 2018, 2019, 2020]}
C_FACTOR   = 0.456
FCD_FOREST = 30.0          # forest = FCD >= 30 %  -> prediction domain
NODATA     = -9999.0
CHUNK      = 200_000       # pixels per prediction chunk

# raster band name in the Landsat export  <-  training column
LANDSAT_RASTER = {'B_mean': 'Blue', 'G_mean': 'Green', 'R_mean': 'Red', 'NIR_mean': 'NIR',
                  'SWIR1_mean': 'SWIR1', 'SWIR2_mean': 'SWIR2', 'NDVI_mean': 'NDVI',
                  'GNDVI_mean': 'GNDVI', 'EVI_mean': 'EVI'}
INDICES    = ['NDVI_mean', 'GNDVI_mean', 'EVI_mean']
PALSAR     = ['HH_mean', 'HV_mean']
MASK_CODES = {1: 'forest predicted', 2: 'no Landsat', 3: 'no PALSAR', 4: 'no SRTM',
              5: 'no FCD', 6: 'non-forest (FCD < 30 %)'}

# ── Selected model + PCA ─────────────────────────────────────────────────────
model = joblib.load(os.path.join(MODEL_DIR, 'rf_selected_model.joblib'))
with open(os.path.join(MODEL_DIR, 'rf_selected_model.json')) as f:
    meta = json.load(f)
FEATURES = meta['features']
assert list(getattr(model, 'feature_names_in_', FEATURES)) == FEATURES, "Feature order mismatch"
pca_path = os.path.join(INTERIM, 'pca_landsat.joblib')
if not os.path.exists(pca_path):
    pca_path = meta.get('pca_file', pca_path)
PCA_INFO = joblib.load(pca_path)
PCA_BANDS, PCS = PCA_INFO['bands'], PCA_INFO['pcs']
assert PCS == meta.get('pca_components', PCS), "PCA file does not match the selected model"
USE_PALSAR = all(p in FEATURES for p in PALSAR)
print(f"Selected scenario {meta['scenario']} | WL {meta['weak_label']} | "
      f"PALSAR {'yes' if USE_PALSAR else 'no'} | {len(model.estimators_)} trees")
print(f"Predictors ({len(FEATURES)}): {FEATURES}")
print(f"PCA: {PCA_BANDS} -> {PCS}")


def to_pcs(bands_stack):
    """(6, n) raw Landsat bands -> (n_pc, n) components, with the scaler + PCA of 05."""
    sc, pca = PCA_INFO['scaler'], PCA_INFO['pca']
    Z = (bands_stack.T - sc.mean_) / sc.scale_
    return ((Z - pca.mean_) @ pca.components_.T).T.astype('float32')


# ── Helpers ──────────────────────────────────────────────────────────────────
def find_tif(folder, stem):
    """Returns path to <stem>.tif. If GEE split the export into tiles
    (<stem>-0000000000-0000000000.tif ...), merges them once into <stem>.tif."""
    p = os.path.join(folder, f"{stem}.tif")
    if os.path.exists(p):
        return p
    tiles = sorted(glob.glob(os.path.join(folder, f"{stem}-*.tif")))
    if not tiles:
        raise FileNotFoundError(f"Missing {p} (and no GEE tiles {stem}-*.tif)")
    print(f"  merging {len(tiles)} GEE tiles -> {p}")
    srcs = [rasterio.open(t) for t in tiles]
    arr, tr = merge(srcs)
    prof = srcs[0].profile.copy()
    descs = srcs[0].descriptions
    for s in srcs:
        s.close()
    prof.update(height=arr.shape[1], width=arr.shape[2], transform=tr,
                compress='deflate', tiled=True, BIGTIFF='IF_SAFER')
    with rasterio.open(p, 'w', **prof) as dst:
        dst.write(arr)
        for i, dsc in enumerate(descs, 1):
            if dsc:
                dst.set_band_description(i, dsc)
    return p


def band_index(src, name, fallback):
    d = [x.lower() if x else '' for x in src.descriptions]
    return d.index(name.lower()) + 1 if name.lower() in d else fallback


def read_to_grid(path, bands, grid):
    """Reads the requested bands (1-based) of `path` onto the reference grid.
    Same grid -> direct read; otherwise bilinear reprojection. NaN = no data."""
    out = np.full((len(bands), grid['height'], grid['width']), np.nan, dtype='float32')
    with rasterio.open(path) as src:
        same = (src.crs == grid['crs'] and src.transform == grid['transform']
                and src.width == grid['width'] and src.height == grid['height'])
        for i, b in enumerate(bands):
            if same:
                a = src.read(b).astype('float32')
                if src.nodata is not None:
                    a[a == src.nodata] = np.nan
                out[i] = a
            else:
                reproject(source=rasterio.band(src, b), destination=out[i],
                          src_transform=src.transform, src_crs=src.crs,
                          src_nodata=src.nodata,
                          dst_transform=grid['transform'], dst_crs=grid['crs'],
                          dst_nodata=np.nan, resampling=Resampling.bilinear)
    return out


def predict_mean_sd(X):
    """Mean and SD of the tree predictions, accumulated in parallel without storing
    all trees (mean = the RF prediction)."""
    trees = model.estimators_
    n_jobs = max(1, os.cpu_count() or 1)
    groups = np.array_split(np.arange(len(trees)), n_jobs)

    def part(idx):
        s = np.zeros(len(X)); s2 = np.zeros(len(X))
        for i in idx:
            p = trees[i].predict(X)
            s += p; s2 += p * p
        return s, s2

    res = Parallel(n_jobs=n_jobs, prefer='threads')(delayed(part)(g) for g in groups)
    s = sum(r[0] for r in res); s2 = sum(r[1] for r in res)
    n = len(trees)
    mean = s / n
    sd = np.sqrt(np.maximum(s2 / n - mean ** 2, 0))
    return mean.astype('float32'), sd.astype('float32')


def write_tif(path, arrays, grid, descs, dtype='float32', nodata=NODATA):
    prof = dict(driver='GTiff', height=grid['height'], width=grid['width'], count=len(arrays),
                dtype=dtype, crs=grid['crs'], transform=grid['transform'], nodata=nodata,
                compress='deflate', tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, 'w', **prof) as dst:
        for i, (a, d) in enumerate(zip(arrays, descs), 1):
            dst.write(a.astype(dtype), i)
            dst.set_band_description(i, d)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    rmse = np.sqrt(mean_squared_error(y, p))
    return dict(n=len(y), mean_obs=y.mean(), mean_pred=p.mean(), RMSE=rmse, rRMSE_pct=100 * rmse / y.mean(), bias=np.mean(p - y),
                MAE=mean_absolute_error(y, p), R2=r2_score(y, p) if len(y) > 2 else np.nan)


# ── Training ranges for the domain check (real-AGB rows of the modelling pool) ──
pool = pd.read_csv(os.path.join(INTERIM, 'split_train_internal.csv'))
pool = pool[~pool.is_weak_label.astype(bool)]
TRAIN_RANGE = {f: (pool[f].quantile(0.01), pool[f].quantile(0.99)) for f in FEATURES}

# ── Field validation sites (05): map_year -> visit ───────────────────────────
tv_all = pd.read_csv(os.path.join(INTERIM, 'split_temporal_validation.csv'))
tv_all['Year(s)_Avitabile'] = tv_all['Year(s)_Avitabile'].astype(str)
MAP_VALIDATION = (tv_all.groupby('map_year')['Year(s)_Avitabile'].first().to_dict()
                  if 'map_year' in tv_all else {})
print(f"Map validation: {MAP_VALIDATION}  ({tv_all.site_id.nunique()} sites)")

# ── SRTM (same for every year) ───────────────────────────────────────────────
srtm_path = find_tif(SRTM_DIR, 'srtm_dem')

summary, domain, preview, val_rows = [], [], {}, []
for yr in YEARS:
    t0 = time.time()
    print(f"\n=== {yr} ===")
    ls_path = find_tif(LANDSAT_DIR, f'landsat_{yr}')
    ls_cols = PCA_BANDS + INDICES
    with rasterio.open(ls_path) as src:
        grid = dict(crs=src.crs, transform=src.transform, width=src.width, height=src.height)
        ls_idx = [band_index(src, LANDSAT_RASTER[c], list(LANDSAT_RASTER).index(c) + 1) for c in ls_cols]
        cov_idx = band_index(src, 'coverage_main', None) or band_index(src, 'coverage', None)
    H, W = grid['height'], grid['width']
    px_ha = abs(grid['transform'].a * grid['transform'].e) / 1e4

    # Landsat: 6 bands (-> PCA) + 3 indices
    ls = read_to_grid(ls_path, ls_idx, grid)
    no_ls = ~np.all(np.isfinite(ls), axis=0)
    no_ls |= np.all(ls[:len(PCA_BANDS)] == 0, axis=0)          # GEE masked pixels exported as 0
    if cov_idx:
        cov = read_to_grid(ls_path, [cov_idx], grid)[0]
        no_ls |= ~np.isfinite(cov) | (cov == 0)
        u, c = np.unique(cov[np.isfinite(cov)].round().astype(int), return_counts=True)
        print("  Landsat source of each pixel: " +
              ", ".join(f"{ {0: 'none', 1: 'main year', 2: 'S2 previous year', 3: 'S1 next year'}.get(k, k)}"
                        f" {100 * n / (H * W):.1f}%" for k, n in zip(u, c)))
    layers = {c: ls[i] for i, c in enumerate(ls_cols)}
    del ls

    # PALSAR (only if the selected scenario uses it)
    pal_years = PALSAR_FOR[yr] if USE_PALSAR else []
    if USE_PALSAR:
        pal = np.zeros((2, H, W), dtype='float32')
        npal = np.zeros((2, H, W), dtype='float32')   # years with data per pixel
        for py in pal_years:
            p = find_tif(PALSAR_DIR, f'palsar_{py}')
            with rasterio.open(p) as src:
                idx = [band_index(src, 'HH', 1), band_index(src, 'HV', 2)]
            a = read_to_grid(p, idx, grid)
            ok = np.isfinite(a) & (a > 0)          # DN > 0; 0 = GEE gap
            pal += np.where(ok, a, 0); npal += ok
            print(f"    palsar_{py}: data in {100 * ok[0].mean():.1f} % of the grid")
        # mean of the years WITH data (a yearly mosaic with a gap does not erase the pixel)
        pal = np.where(npal > 0, pal / np.maximum(npal, 1), np.nan).astype('float32')
        layers['HH_mean'], layers['HV_mean'] = pal[0], pal[1]
        del pal
        print(f"  PALSAR: {'mean ' if len(pal_years) > 1 else ''}{pal_years}")
    else:
        print("  PALSAR: not used by the selected scenario")
    # SRTM (2)
    with rasterio.open(srtm_path) as src:
        s_idx = [band_index(src, 'elevation', 1), band_index(src, 'slope', 2)]
    srtm = read_to_grid(srtm_path, s_idx, grid)
    layers['elevation_srtm'], layers['slope_srtm'] = srtm[0], srtm[1]
    del srtm
    # FCD (1)
    fcd = read_to_grid(os.path.join(FCD_DIR, f'landtrendr_fitted_prediction_{yr}.tif'), [1], grid)[0]
    if np.nanmax(fcd) > 200:      # 0-10000 storage -> 0-100
        fcd = fcd / 100.0
    layers['fcd_average_period'] = fcd

    # Mask with reason (later assignments win: NoData reasons override non-forest)
    mask = np.ones((H, W), dtype='uint8')
    no_pal = (~(np.isfinite(layers['HH_mean']) & np.isfinite(layers['HV_mean']))
              if USE_PALSAR else np.zeros((H, W), bool))
    no_srtm = ~(np.isfinite(layers['elevation_srtm']) & np.isfinite(layers['slope_srtm']))
    no_fcd = ~np.isfinite(fcd)
    nonforest = np.isfinite(fcd) & (fcd < FCD_FOREST)
    for code, m in [(6, nonforest), (5, no_fcd), (4, no_srtm), (3, no_pal), (2, no_ls)]:
        mask[m] = code
    valid = mask == 1
    n_valid = int(valid.sum())
    print(f"  forest pixels predicted: {n_valid:,} of {H * W:,} ({100 * n_valid / (H * W):.1f} %) | "
          f"non-forest {100 * (mask == 6).mean():.1f} % | NoData {100 * np.isin(mask, [2, 3, 4, 5]).mean():.1f} %")

    # PCs of the Landsat bands (forest pixels only)
    pcs = to_pcs(np.stack([layers[b][valid] for b in PCA_BANDS]))
    vals = {**{p: pcs[i] for i, p in enumerate(PCS)},
            **{f: layers[f][valid] for f in FEATURES if f not in PCS}}
    del pcs

    # Domain check: share of predicted pixels outside the training p1-p99 range
    for f in FEATURES:
        v = vals[f]
        lo, hi = TRAIN_RANGE[f]
        domain.append(dict(year=yr, predictor=f, train_p1=lo, train_p99=hi,
                           map_p1=np.percentile(v, 1), map_p50=np.median(v),
                           map_p99=np.percentile(v, 99),
                           pct_below=100 * np.mean(v < lo), pct_above=100 * np.mean(v > hi)))

    # Prediction (chunks of forest pixels)
    X_all = np.stack([vals[f] for f in FEATURES], axis=1).astype('float32')
    del vals
    agb = np.full(n_valid, np.nan, dtype='float32')
    sd = np.full(n_valid, np.nan, dtype='float32')
    for s in range(0, n_valid, CHUNK):
        agb[s:s + CHUNK], sd[s:s + CHUNK] = predict_mean_sd(X_all[s:s + CHUNK])
        print(f"    {min(s + CHUNK, n_valid):,}/{n_valid:,}", end='\r')
    del X_all

    agb_map = np.full((H, W), NODATA, dtype='float32')
    sd_map = np.full((H, W), NODATA, dtype='float32')
    agb_map[mask == 6] = 0.0; sd_map[mask == 6] = 0.0            # non-forest
    agb_map[valid] = agb; sd_map[valid] = sd
    data = agb_map != NODATA
    agc_map = np.where(data, agb_map * C_FACTOR, NODATA).astype('float32')
    agcsd_map = np.where(data, sd_map * C_FACTOR, NODATA).astype('float32')
    write_tif(os.path.join(OUT_MAPS, f'agb_{yr}.tif'), [agb_map, sd_map], grid,
              ['AGB_Mg_ha', 'AGB_SD_Mg_ha'])
    write_tif(os.path.join(OUT_MAPS, f'agc_{yr}.tif'), [agc_map, agcsd_map], grid,
              ['AGC_MgC_ha', 'AGC_SD_MgC_ha'])
    write_tif(os.path.join(OUT_MAPS, f'mask_{yr}.tif'), [mask], grid,
              ['1 forest|2 noLandsat|3 noPALSAR|4 noSRTM|5 noFCD|6 nonforest FCD<30'],
              dtype='uint8', nodata=0)

    # Summary
    row = dict(year=yr, scenario=meta['scenario'],
               palsar_years='-'.join(map(str, pal_years)) if pal_years else 'not used',
               area_total_ha=H * W * px_ha, area_with_data_ha=data.sum() * px_ha,
               area_forest_ha=n_valid * px_ha, area_nonforest_ha=(mask == 6).sum() * px_ha,
               pct_forest=100 * n_valid / (H * W))
    for code, name in MASK_CODES.items():
        if 1 < code < 6:
            row[f'nodata_{name.replace(" ", "_")}_ha'] = (mask == code).sum() * px_ha
    row.update(AGC_forest_TgC=agc_map[valid].sum() * px_ha / 1e6,
               AGC_mean_forest_MgC_ha=agc_map[valid].mean() if n_valid else np.nan,
               AGC_SD_mean_forest_MgC_ha=agcsd_map[valid].mean() if n_valid else np.nan,
               AGB_mean_forest_Mg_ha=agb_map[valid].mean() if n_valid else np.nan)
    summary.append(row)
    print(f"  forest {row['area_forest_ha']:,.0f} ha | AGC forest {row['AGC_forest_TgC']:.2f} Tg C | "
          f"mean {row['AGC_mean_forest_MgC_ha']:.1f} Mg C/ha  [{time.time() - t0:.0f}s]")

    # Preview for the figure (decimated): AGC (0 in non-forest), NoData flag
    f_ = max(1, int(max(H, W) / 1200))
    preview[yr] = (np.where(data, agc_map, np.nan)[::f_, ::f_], (~data)[::f_, ::f_])

    # Map validation against the re-measured USFQ sites
    if yr in MAP_VALIDATION:
        tv = tv_all[tv_all.map_year == yr]
        xs, ys = warp_transform('EPSG:4326', grid['crs'], tv.x.tolist(), tv.y.tolist())
        inv = ~grid['transform']
        for (_, r), X, Y in zip(tv.iterrows(), xs, ys):
            c, rr = inv * (X, Y)
            c, rr = int(np.floor(c)), int(np.floor(rr))
            inside = 0 <= rr < H and 0 <= c < W
            px = agb_map[rr, c] if inside else NODATA
            win = agb_map[max(rr - 1, 0):rr + 2, max(c - 1, 0):c + 2] if inside else np.array([NODATA])
            win = win[win != NODATA]
            val_rows.append(dict(map_year=yr, visit=r['Year(s)_Avitabile'], site_id=r.site_id,
                                 x=r.x, y=r.y, AGB_field=r.AGB,
                                 AGB_map_pixel=np.nan if px == NODATA else px,
                                 AGB_map_3x3=win.mean() if win.size else np.nan,
                                 AGB_SD_pixel=np.nan if not inside or sd_map[rr, c] == NODATA else sd_map[rr, c],
                                 FCD_pixel=fcd[rr, c] if inside else np.nan,
                                 mask_code=int(mask[rr, c]) if inside else -1))
    del layers, agb_map, sd_map, agc_map, agcsd_map, mask, fcd

# ── Tables ───────────────────────────────────────────────────────────────────
summ = pd.DataFrame(summary)
summ.to_csv(os.path.join(OUT_MAPS, 'map_summary.csv'), index=False)
dom = pd.DataFrame(domain)
dom.to_csv(os.path.join(OUT_MAPS, 'domain_check.csv'), index=False)
print("\nMap summary:\n" + summ.round(2).to_string(index=False))
flag = dom[(dom.pct_below + dom.pct_above) > 5]
if len(flag):
    print("\nWARNING - predictors with > 5 % of forest pixels outside the training p1-p99 range:")
    print(flag[['year', 'predictor', 'train_p1', 'train_p99', 'map_p1', 'map_p99',
                'pct_below', 'pct_above']].round(3).to_string(index=False))

vm = None
if val_rows:
    val = pd.DataFrame(val_rows)
    val.to_csv(os.path.join(OUT_VAL, 'map_validation_points.csv'), index=False)
    vrows = []
    for yr, g in val.groupby('map_year'):
        for col in ['AGB_map_pixel', 'AGB_map_3x3']:
            gg = g.dropna(subset=[col])
            if len(gg):
                vrows.append(dict(map_year=yr, visit=g.visit.iloc[0], extraction=col,
                                  n_sites_total=len(g), n_nonforest_pixel=int((g.mask_code == 6).sum()),
                                  **metrics(gg.AGB_field, gg[col])))
    vm = pd.DataFrame(vrows)
    vm.to_csv(os.path.join(OUT_VAL, 'map_validation_metrics.csv'), index=False)
    print("\nMap validation (re-measured USFQ sites):\n" + vm.round(2).to_string(index=False))

    # Change between the two validated maps (e.g. map 2022 - map 2016) vs field change
    # (visit 2021 - visit 2017) at the same sites
    yrs_v = sorted(val.map_year.unique())
    if len(yrs_v) >= 2:
        y0, y1 = yrs_v[0], yrs_v[-1]
        crow = []
        for col in ['AGB_map_pixel', 'AGB_map_3x3']:
            a = val[val.map_year == y0].set_index('site_id')
            b = val[val.map_year == y1].set_index('site_id')
            j = a[['AGB_field', col]].join(b[['AGB_field', col]], lsuffix='_0', rsuffix='_1').dropna()
            if len(j) < 2:
                continue
            obs = j['AGB_field_1'] - j['AGB_field_0']
            pred = j[f'{col}_1'] - j[f'{col}_0']
            rmse = np.sqrt(np.mean((pred - obs) ** 2))
            crow.append(dict(maps=f'{y1}-{y0}', visits=f"{b.visit.iloc[0]}-{a.visit.iloc[0]}",
                             extraction=col, n_sites=len(j), mean_obs_change=obs.mean(),
                             mean_map_change=pred.mean(), RMSE_change=rmse, bias_change=np.mean(pred - obs),
                             direction_agreement_pct=100 * np.mean(np.sign(pred) == np.sign(obs))))
        vc = pd.DataFrame(crow)
        vc.to_csv(os.path.join(OUT_VAL, 'map_change_validation.csv'), index=False)
        print("\nMap change validation (AGB, Mg ha-1):\n" + vc.round(2).to_string(index=False))
    nf = val[val.mask_code == 6]
    if len(nf):
        print(f"  note: {len(nf)} site-visits fall on non-forest pixels (FCD < 30 %, map AGB = 0): "
              f"{nf[['map_year', 'site_id', 'FCD_pixel', 'AGB_field']].round(1).to_dict('records')}")


# ── Figures ──────────────────────────────────────────────────────────────────
def style(ax):
    ax.tick_params(direction='in', length=4)
    for s in ax.spines.values():
        s.set_linewidth(0.8); s.set_color('black')

# AGC maps (white = non-forest, AGC 0; light blue = NoData)
yrs = list(preview)
vmax = np.nanpercentile(np.concatenate([preview[y][0].ravel() for y in yrs]), 99)
fig, axes = plt.subplots(1, len(yrs), figsize=(3.4 * len(yrs), 3.9), squeeze=False)
for ax, yr, letter in zip(axes[0], yrs, 'ABCDE'):
    a, nd = preview[yr]
    ax.imshow(np.where(nd, 1, np.nan), cmap=ListedColormap(['#cfe0f0']), vmin=0, vmax=1,
              interpolation='nearest')
    im = ax.imshow(a, cmap='Greys', vmin=0, vmax=vmax, interpolation='nearest')
    ax.set_title(f'{letter}. {yr}', loc='left', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    style(ax)
cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.015, pad=0.01)
cb.set_label('AGC (Mg C ha$^{-1}$)')
axes[0][0].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc='white', ec='#808080'),
                           plt.Rectangle((0, 0), 1, 1, fc='#cfe0f0', ec='#808080')],
                  labels=['Non-forest (FCD < 30 %)', 'NoData'], loc='lower left', fontsize=8,
                  frameon=True, edgecolor='black')
fig.savefig(os.path.join(FIG_DIR, 'fig_agc_maps.tif'), dpi=300, bbox_inches='tight',
            facecolor='white', format='tiff')
plt.close(fig)

# Trajectory of forest AGC
if len(summ) > 1:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(summ.year, summ.AGC_forest_TgC, '-o', color='#404040', mfc='#808080', mec='black')
    ax.set_xticks(summ.year)
    ax.set_xlabel('Year'); ax.set_ylabel('Forest AGC (Tg C)')
    ax.set_title(f'A. Forest AGC (FCD ≥ {FCD_FOREST:.0f} %)', loc='left', fontweight='bold')
    ax.grid(True, color='#e0e0e0', linewidth=0.5)
    style(ax)
    fig.savefig(os.path.join(FIG_DIR, 'fig_agc_trajectory.tif'), dpi=300, bbox_inches='tight',
                facecolor='white', format='tiff')
    plt.close(fig)

# Map validation scatter
if vm is not None and len(vm):
    val = pd.DataFrame(val_rows).dropna(subset=['AGB_map_pixel'])
    yrs_v = sorted(val.map_year.unique())
    fig, axes = plt.subplots(1, len(yrs_v), figsize=(4.8 * len(yrs_v), 4.6), squeeze=False)
    for ax, yr, letter in zip(axes[0], yrs_v, 'AB'):
        g = val[val.map_year == yr]
        ax.errorbar(g.AGB_field, g.AGB_map_pixel, yerr=g.AGB_SD_pixel, fmt='o', color='#808080',
                    mec='#404040', ecolor='#a0a0a0', elinewidth=0.7, capsize=2, zorder=3)
        lo = 0; hi = max(g.AGB_field.max(), g.AGB_map_pixel.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.0); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        m = metrics(g.AGB_field, g.AGB_map_pixel)
        ax.text(0.04, 0.97, f"R$^2$ = {m['R2']:.2f}\nRMSE = {m['RMSE']:.1f}\nBias = {m['bias']:.1f}\n"
                            f"Mean field = {m['mean_obs']:.1f}\nMean map = {m['mean_pred']:.1f}\n"
                            f"n = {m['n']}", transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=0.8))
        ax.set_xlabel(f'Field AGB, USFQ visit {MAP_VALIDATION[yr]} (Mg ha$^{{-1}}$)')
        ax.set_ylabel(f'Map AGB {yr} (Mg ha$^{{-1}}$)')
        ax.set_title(f'{letter}. Map {yr}', loc='left', fontweight='bold')
        ax.grid(True, color='#e0e0e0', linewidth=0.5)
        style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_map_validation.tif'), dpi=300, bbox_inches='tight',
                facecolor='white', format='tiff')
    plt.close(fig)

print(f"\nMaps: {OUT_MAPS}\nValidation: {OUT_VAL}\nDone.")
