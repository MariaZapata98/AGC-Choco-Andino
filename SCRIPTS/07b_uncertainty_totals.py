r"""
07b_uncertainty_totals.py
Uncertainty of the landscape AGC totals (one number per map) and of the change
between maps, for the SELECTED scenario (rf_selected_model.json from 06: predictors,
weak-label value, PALSAR yes/no, hyperparameters, PCA of the Landsat bands).

  1. Model / training-sample uncertainty  -> spatial BLOCK bootstrap of the training
     set (blocks = the 10 km block_id of 05). Each bootstrap model predicts every map;
     totals and changes are computed inside each replicate.
  2. Systematic error from the field validation (07: the same 10 re-measured USFQ
     sites, visit 2017 vs map 2016 and visit 2021 vs map 2022) -> SE of the mean bias,
     applied to every map (stationarity assumption, stated in the paper).
     For CHANGE, the shared bias cancels; the remaining systematic term is the SE of
     the paired drift of the residuals between the two validated maps (same sites).

     sigma_total^2  = sigma_bootstrap^2 + (SE_bias  x forest area)^2
     sigma_change^2 = sigma_bootstrap^2 + (SE_drift x forest area)^2

  2b. Carbon ledger 2000-2022 (4 classes, same rules and threshold as 08) recomputed in every
     bootstrap replicate -> 95 % CI of the area and AGC change of each category (reviewer 1,
     comment 1), and point ledger for alternative gain/loss thresholds (1.0, 1.96, 2.58 x RMSE of
     change; reviewer 1, comment 6).
  3. Scenario sensitivity of the totals (S1-S16: S9-S16 without FCD), each scenario refitted on the full
     pool with its tuned hyperparameters.

Forest = FCD >= 30 % (mask code 1 of 07). Non-forest (code 6, FCD < 30 %) has AGC = 0,
as in the maps. Pixels with true NoData (codes 2-5) in ANY year are excluded from all
years (common universe), so gaps do not create artificial changes.
Totals are computed on a systematic pixel sample (every STEP-th row and column).
Two forest masks: own year (forest_own) and fixed forest-2022 (forest_2022_mask, as in
the original paper). Optional: restricted to AOI_SHP (e.g. RBCA) if the file exists.

Outputs  RESULTADOS\07_incertidumbre\
   totals_by_year.csv, change.csv, systematic_component.csv, nodata_excluded.csv,
   bootstrap_replicates.csv, scenario_sensitivity_totals.csv,
   ledger_uncertainty.csv, ledger_threshold_sensitivity.csv,
   Table_S_AGC_totals_uncertainty.csv (SI)
Figure   FIGURAS\fig_agc_totals_uncertainty.tif
Usage    python 07b_uncertainty_totals.py
"""

import os
import json
import time
import numpy as np
import pandas as pd
import joblib
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.ensemble import RandomForestRegressor
import matplotlib
import matplotlib.pyplot as plt

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
MAPS_DIR    = os.path.join(RES_DIR, "05_mapas")
VAL_DIR     = os.path.join(RES_DIR, "06_validacion_mapas")
SCEN_CSV    = os.path.join(RES_DIR, "02_test_interno", "rf_scenarios_internal_test.csv")
OUT_DIR     = os.path.join(RES_DIR, "07_incertidumbre")
AOI_SHP     = os.path.join(ROOT, "DATA", "raw", "RBCA_shp", "RBiosfera_ChocoAndino.shp")   # RBCA: totals reported in the paper
for d in [OUT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Settings ─────────────────────────────────────────────────────────────────
YEARS       = [2000, 2006, 2012, 2016, 2022]
PALSAR_FOR  = {2000: [2015], 2006: [2015], 2012: [2015],
               2016: [2015, 2016], 2022: [2015, 2016, 2017, 2018, 2019, 2020]}
C_FACTOR    = 0.456
FCD_FOREST  = 30.0
STEP        = 3            # pixel sampling step for totals
N_BOOT      = 50           # bootstrap replicates
LEDGER_Y0, LEDGER_Y1 = 2000, 2022
THR_FACTORS = [1.0, 1.96, 2.58]   # sensitivity of the ledger to the gain/loss threshold (x RMSE_change)
BOOT_TREES  = 200          # trees per bootstrap model (totals converge well below 500)
SEED        = 42
DO_SCENARIOS = True
CHANGE_PAIRS = [(2000, 2006), (2006, 2012), (2012, 2016), (2016, 2022), (2000, 2022)]
LANDSAT_RASTER = {'B_mean': 'Blue', 'G_mean': 'Green', 'R_mean': 'Red', 'NIR_mean': 'NIR',
                  'SWIR1_mean': 'SWIR1', 'SWIR2_mean': 'SWIR2', 'NDVI_mean': 'NDVI',
                  'GNDVI_mean': 'GNDVI', 'EVI_mean': 'EVI'}
INDICES = ['NDVI_mean', 'GNDVI_mean', 'EVI_mean']
PALSAR  = ['HH_mean', 'HV_mean']
NODATA_CODES = [2, 3, 4, 5]


# ── Raster helpers (same logic as 07) ────────────────────────────────────────
def find_tif(folder, stem):
    p = os.path.join(folder, f"{stem}.tif")
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"Missing {p} (run 07 first; it merges GEE tiles)")


def band_index(src, name, fallback):
    d = [x.lower() if x else '' for x in src.descriptions]
    return d.index(name.lower()) + 1 if name.lower() in d else fallback


def read_to_grid(path, bands, grid):
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
                          src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                          dst_transform=grid['transform'], dst_crs=grid['crs'],
                          dst_nodata=np.nan, resampling=Resampling.bilinear)
    return out


# ── Selected model, PCA and training pool ────────────────────────────────────
with open(os.path.join(MODEL_DIR, 'rf_selected_model.json')) as f:
    meta = json.load(f)
FEATURES = meta['features']
final_model = joblib.load(os.path.join(MODEL_DIR, 'rf_selected_model.joblib'))
pca_path = os.path.join(INTERIM, 'pca_landsat.joblib')
if not os.path.exists(pca_path):
    pca_path = meta.get('pca_file', pca_path)
PCA_INFO = joblib.load(pca_path)
PCA_BANDS, PCS = PCA_INFO['bands'], PCA_INFO['pcs']
CORE = PCS + INDICES + ['elevation_srtm', 'slope_srtm']                      # as in 06
FCD = ['fcd_average_period']
ALL_COLS = CORE + FCD + PALSAR
pool = pd.read_csv(os.path.join(INTERIM, 'split_train_internal.csv'))
pool['is_weak_label'] = pool['is_weak_label'].astype(bool)
print(f"Selected scenario {meta['scenario']} | WL {meta['weak_label']} | PALSAR {meta['palsar']} | "
      f"predictors {FEATURES}")
print(f"Pool {len(pool)} rows | blocks {pool.block_id.nunique()}")


def to_pcs(bands_stack):
    sc, pca = PCA_INFO['scaler'], PCA_INFO['pca']
    Z = (bands_stack.T - sc.mean_) / sc.scale_
    return ((Z - pca.mean_) @ pca.components_.T).T.astype('float32')


def scenario_data(wl):
    d = pool.copy()
    if wl is None:
        d = d[~d.is_weak_label]
    else:
        d.loc[d.is_weak_label, 'AGB'] = float(wl)
    return d.reset_index(drop=True)


# ── 1. Common universe from the 07 masks ─────────────────────────────────────
mask_codes = {}
for yr in YEARS:
    with rasterio.open(os.path.join(MAPS_DIR, f'mask_{yr}.tif')) as m:
        mask_codes[yr] = m.read(1)
shapes = {a.shape for a in mask_codes.values()}
if len(shapes) != 1:
    raise ValueError(f"Mask grids differ between years {shapes}; re-run 07 with the 02 v2 rasters.")
nodata_any = np.zeros(next(iter(shapes)), dtype=bool)
for yr, a in mask_codes.items():
    nodata_any |= np.isin(a, NODATA_CODES)
universe = ~nodata_any
excl = pd.DataFrame([dict(year=yr, pct_true_nodata_this_year=100 * np.isin(a, NODATA_CODES).mean(),
                          pct_forest=100 * (a == 1).mean(), pct_nonforest=100 * (a == 6).mean())
                     for yr, a in mask_codes.items()])
excl.loc[len(excl)] = dict(year='any year (excluded from all totals)',
                           pct_true_nodata_this_year=100 * nodata_any.mean(),
                           pct_forest=np.nan, pct_nonforest=np.nan)
excl.to_csv(os.path.join(OUT_DIR, 'nodata_excluded.csv'), index=False)
print("Mask shares (% of extent):\n" + excl.round(2).to_string(index=False))

# ── 2. Pixel sample of every map: predictors of the FOREST pixels ────────────
srtm_path = find_tif(SRTM_DIR, 'srtm_dem')
samples, aoi_cache = {}, {}
for yr in YEARS:
    t0 = time.time()
    ls_path = find_tif(LANDSAT_DIR, f'landsat_{yr}')
    ls_cols = PCA_BANDS + INDICES
    with rasterio.open(ls_path) as src:
        grid = dict(crs=src.crs, transform=src.transform, width=src.width, height=src.height)
        ls_idx = [band_index(src, LANDSAT_RASTER[c], list(LANDSAT_RASTER).index(c) + 1) for c in ls_cols]
    px_ha = abs(grid['transform'].a * grid['transform'].e) / 1e4
    sl = (slice(None, None, STEP), slice(None, None, STEP))

    inside = universe.copy()
    key = (grid['crs'].to_string(), tuple(grid['transform']), grid['width'], grid['height'])
    if os.path.exists(AOI_SHP):
        if key not in aoi_cache:
            import geopandas as gpd
            from rasterio.features import rasterize
            shp = gpd.read_file(AOI_SHP).to_crs(grid['crs'])
            aoi_cache[key] = rasterize([(g, 1) for g in shp.geometry],
                                       out_shape=(grid['height'], grid['width']),
                                       transform=grid['transform'], fill=0, dtype='uint8').astype(bool)
            print(f"AOI applied: {AOI_SHP}")
        inside &= aoi_cache[key]
    inside = inside[sl]                                   # sampled pixels of the universe
    forest = (mask_codes[yr][sl] == 1)[inside]           # forest flag of those pixels

    ls = read_to_grid(ls_path, ls_idx, grid)
    lay = {c: ls[i][sl][inside] for i, c in enumerate(ls_cols)}
    del ls
    pal = np.zeros((2, grid['height'], grid['width']), dtype='float32')
    npal = np.zeros((2, grid['height'], grid['width']), dtype='float32')   # years with data per pixel
    for py in PALSAR_FOR[yr]:            # always read: needed by the PALSAR scenarios (sensitivity)
        p = find_tif(PALSAR_DIR, f'palsar_{py}')
        with rasterio.open(p) as src:
            idx = [band_index(src, 'HH', 1), band_index(src, 'HV', 2)]
        a = read_to_grid(p, idx, grid)
        ok = np.isfinite(a) & (a > 0)          # DN > 0; 0 = GEE gap
        pal += np.where(ok, a, 0); npal += ok
    # mean of the years WITH data (a yearly mosaic with a gap does not erase the pixel)
    pal = np.where(npal > 0, pal / np.maximum(npal, 1), np.nan).astype('float32')
    lay['HH_mean'], lay['HV_mean'] = pal[0][sl][inside], pal[1][sl][inside]
    del pal
    with rasterio.open(srtm_path) as src:
        s_idx = [band_index(src, 'elevation', 1), band_index(src, 'slope', 2)]
    srtm = read_to_grid(srtm_path, s_idx, grid)
    lay['elevation_srtm'], lay['slope_srtm'] = srtm[0][sl][inside], srtm[1][sl][inside]
    del srtm
    fcd = read_to_grid(os.path.join(FCD_DIR, f'landtrendr_fitted_prediction_{yr}.tif'), [1], grid)[0]
    if np.nanmax(fcd) > 200:
        fcd = fcd / 100.0
    lay['fcd_average_period'] = fcd[sl][inside]
    del fcd
    pcs = to_pcs(np.stack([lay[b][forest] for b in PCA_BANDS]))
    X = pd.DataFrame({**{p: pcs[i] for i, p in enumerate(PCS)},
                      **{c: lay[c][forest] for c in ALL_COLS if c not in PCS}})[ALL_COLS]
    samples[yr] = dict(X=X, forest=forest, n=int(inside.sum()), area=px_ha * STEP * STEP,
                       fcd=lay['fcd_average_period'])
    print(f"{yr}: {inside.sum():,} sampled pixels, {forest.sum():,} forest "
          f"({100 / STEP ** 2:.1f} % sample) [{time.time() - t0:.0f}s]")

# fixed forest-2022 mask (paper definition), on the same sampled pixels
ref = 2022 if 2022 in samples else YEARS[-1]
for yr in samples:
    samples[yr]['forest_ref'] = samples[ref]['forest']


def totals(model, feats, keep=None):
    """AGC totals (Tg C) per year: own-year forest mask and fixed forest-2022 mask.
    Non-forest pixels of a year have AGC = 0 (as in the maps).
    keep = list of years whose pixel AGC (Mg C/ha, sampled pixels) is also returned."""
    out, px = {}, {}
    for yr, s in samples.items():
        agc = np.zeros(s['n'])
        Xf = s['X'][feats]
        ok = np.all(np.isfinite(Xf.values), axis=1)
        pred = np.full(len(Xf), np.nan)
        pred[ok] = model.predict(Xf[ok]) * C_FACTOR
        agc[s['forest']] = pred
        out[(yr, 'forest_own')] = np.nansum(agc[s['forest']]) * s['area'] / 1e6
        out[(yr, 'forest_2022_mask')] = np.nansum(agc[s['forest_ref']]) * s['area'] / 1e6
        if keep and yr in keep:
            px[yr] = agc
    return (out, px) if keep else out


# ── 3. Point estimate (final model) ──────────────────────────────────────────
point, px_point = totals(final_model, FEATURES, keep=[LEDGER_Y0, LEDGER_Y1])

# ── 4. Block bootstrap (selected scenario) ───────────────────────────────────
best = meta['hyperparameters']
d_sel = scenario_data(meta['weak_label'])
blocks = d_sel.block_id.unique()
by_block = {b: idx for b, idx in d_sel.groupby('block_id').indices.items()}
rng = np.random.default_rng(SEED)
boot, boot_px = [], []
t0 = time.time()
for b in range(N_BOOT):
    pick = rng.choice(blocks, size=len(blocks), replace=True)
    rows = np.concatenate([by_block[k] for k in pick])
    d = d_sel.iloc[rows]
    m = RandomForestRegressor(**{**best, 'n_estimators': BOOT_TREES},
                              random_state=SEED + b, n_jobs=-1).fit(d[FEATURES], d.AGB)
    tb_, px_ = totals(m, FEATURES, keep=[LEDGER_Y0, LEDGER_Y1])
    boot.append({'replicate': b, **{f'{k[0]}|{k[1]}': v for k, v in tb_.items()}})
    boot_px.append(px_)
    print(f"  bootstrap {b + 1}/{N_BOOT}  [{time.time() - t0:.0f}s]", end='\r')
boot = pd.DataFrame(boot)
boot.to_csv(os.path.join(OUT_DIR, 'bootstrap_replicates.csv'), index=False)
print()

# ── 5. Systematic component from the field validation (07) ───────────────────
vp = pd.read_csv(os.path.join(VAL_DIR, 'map_validation_points.csv')).dropna(subset=['AGB_map_pixel'])
vp['resid_C'] = (vp.AGB_map_pixel - vp.AGB_field) * C_FACTOR
sys_rows = []
for yr, g in vp.groupby('map_year'):
    sys_rows.append(dict(component='bias', map_year=yr, visit=g.visit.iloc[0], n=len(g),
                         mean_Mg_C_ha=g.resid_C.mean(), SE_Mg_C_ha=g.resid_C.std(ddof=1) / np.sqrt(len(g))))
pooled = vp.resid_C
SE_BIAS = pooled.std(ddof=1) / np.sqrt(len(pooled))
sys_rows.append(dict(component='bias', map_year='pooled', visit='-', n=len(pooled),
                     mean_Mg_C_ha=pooled.mean(), SE_Mg_C_ha=SE_BIAS))
yrs_v = sorted(vp.map_year.unique())
SE_DRIFT = np.nan
if len(yrs_v) >= 2:
    w = vp.pivot_table(index='site_id', columns='map_year', values='resid_C').dropna()
    drift = w[yrs_v[-1]] - w[yrs_v[0]]
    if len(drift) >= 3:
        SE_DRIFT = drift.std(ddof=1) / np.sqrt(len(drift))
        sys_rows.append(dict(component=f'drift {yrs_v[0]}->{yrs_v[-1]} (paired, same sites)', map_year='-',
                             visit='-', n=len(drift), mean_Mg_C_ha=drift.mean(), SE_Mg_C_ha=SE_DRIFT))
if not np.isfinite(SE_DRIFT):
    SE_DRIFT = SE_BIAS * np.sqrt(2)   # conservative fallback: independent biases
    sys_rows.append(dict(component='drift fallback (sqrt(2) x SE_bias)', map_year='-', visit='-',
                         n=np.nan, mean_Mg_C_ha=np.nan, SE_Mg_C_ha=SE_DRIFT))
sysdf = pd.DataFrame(sys_rows)
sysdf.to_csv(os.path.join(OUT_DIR, 'systematic_component.csv'), index=False)
print("\nSystematic component (Mg C/ha):\n" + sysdf.round(2).to_string(index=False))


# ── 6. Totals and change with combined uncertainty ───────────────────────────
def forest_area_ha(yr, mask):
    s = samples[yr]
    return (s['forest'] if mask == 'forest_own' else s['forest_ref']).sum() * s['area']


masks = ['forest_own', 'forest_2022_mask']
tot_rows, chg_rows = [], []
for mask in masks:
    for yr in samples:
        col = f'{yr}|{mask}'
        sb = boot[col].std(ddof=1)
        area = forest_area_ha(yr, mask)
        ss = SE_BIAS * area / 1e6
        sc = np.sqrt(sb ** 2 + ss ** 2)
        pt = point[(yr, mask)]
        tot_rows.append(dict(scenario=meta['scenario'], mask=mask, year=yr, forest_area_ha=area,
                             AGC_TgC=pt, AGC_mean_MgC_ha=1e6 * pt / area if area else np.nan,
                             boot_p2_5=boot[col].quantile(0.025), boot_p97_5=boot[col].quantile(0.975),
                             sd_bootstrap=sb, sd_systematic=ss, sd_combined=sc,
                             CI95_low=pt - 1.96 * sc, CI95_high=pt + 1.96 * sc,
                             CI95_pct=100 * 1.96 * sc / pt))
    for a, b in CHANGE_PAIRS:
        if a in samples and b in samples:
            diff = boot[f'{b}|{mask}'] - boot[f'{a}|{mask}']
            pt = point[(b, mask)] - point[(a, mask)]
            sb = diff.std(ddof=1)
            area = np.mean([forest_area_ha(a, mask), forest_area_ha(b, mask)])
            ss = SE_DRIFT * area / 1e6
            sc = np.sqrt(sb ** 2 + ss ** 2)
            chg_rows.append(dict(scenario=meta['scenario'], mask=mask, period=f'{a}-{b}', years=b - a,
                                 change_TgC=pt, change_TgC_per_yr=pt / (b - a),
                                 boot_p2_5=diff.quantile(0.025), boot_p97_5=diff.quantile(0.975),
                                 pct_replicates_positive=100 * (diff > 0).mean(),
                                 sd_bootstrap=sb, sd_systematic=ss, sd_combined=sc,
                                 CI95_low=pt - 1.96 * sc, CI95_high=pt + 1.96 * sc,
                                 significant=(pt - 1.96 * sc > 0) or (pt + 1.96 * sc < 0)))
tot = pd.DataFrame(tot_rows)
chg = pd.DataFrame(chg_rows)
tot.to_csv(os.path.join(OUT_DIR, 'totals_by_year.csv'), index=False)
chg.to_csv(os.path.join(OUT_DIR, 'change.csv'), index=False)
print("\nTotals (Tg C):\n" + tot.round(3).to_string(index=False))
print("\nChange (Tg C):\n" + chg.round(3).to_string(index=False))

# SI table (own forest mask)
si = tot[tot['mask'] == 'forest_own'].assign(
    **{'Forest area (ha)': lambda d: d.forest_area_ha.round(0).astype(int),
       'AGC (Tg C)': lambda d: d.AGC_TgC.round(2),
       '95% CI (Tg C)': lambda d: d.CI95_low.round(2).astype(str) + ' – ' + d.CI95_high.round(2).astype(str),
       'SD bootstrap': lambda d: d.sd_bootstrap.round(3),
       'SD systematic': lambda d: d.sd_systematic.round(3)})
si = si[['year', 'Forest area (ha)', 'AGC (Tg C)', '95% CI (Tg C)', 'SD bootstrap', 'SD systematic']]
si.to_csv(os.path.join(OUT_DIR, 'Table_S_AGC_totals_uncertainty.csv'), index=False, encoding='utf-8-sig')

# ── 6b. Carbon ledger 2000-2022 with uncertainty (same rules as 08) ──────────
LEDGER_CAT = {1: 'Non-forest', 2: 'Stable reservoir', 3: 'Active sink', 4: 'Net source'}
# 4 classes from 6 forest-status/change cases: 1 non-forest | 2 persistent stable |
# 3 persistent gain + new forest | 4 persistent loss + forest lost
SUB2CAT = {1: 1, 2: 2, 3: 3, 5: 3, 4: 4, 6: 4}
w_map = vp.pivot_table(index='site_id', columns='map_year', values='AGB_map_pixel')
w_fld = vp.pivot_table(index='site_id', columns='map_year', values='AGB_field')
e_chg = ((w_map[yrs_v[-1]] - w_map[yrs_v[0]]) - (w_fld[yrs_v[-1]] - w_fld[yrs_v[0]])).dropna() * C_FACTOR
RMSE_CHG = float(np.sqrt(np.mean(e_chg ** 2)))
THR = 1.96 * RMSE_CHG                                   # identical to 08 (change_threshold.csv)
f0, f1 = samples[LEDGER_Y0]['forest'], samples[LEDGER_Y1]['forest']
A_PX = samples[LEDGER_Y0]['area']                       # ha represented by one sampled pixel


def ledger(px, thr):
    """Per ledger class: area (ha), AGC change (Tg C) and systematic SE (Tg C).
    Systematic term per pixel case: persistent forest -> paired drift of residuals (SE_DRIFT);
    new forest / forest lost -> bias of the one map where the pixel is forest (SE_BIAS); non-forest -> 0.
    Summed within a class (fully correlated within the class: conservative)."""
    a0, a1 = px[LEDGER_Y0], px[LEDGER_Y1]
    d = a1 - a0
    cat = np.ones(len(a0), dtype='uint8')
    pers = f0 & f1
    cat[pers & (np.abs(d) <= thr)] = 2
    cat[pers & (d > thr)] = 3
    cat[pers & (d < -thr)] = 4
    cat[~f0 & f1] = 5
    cat[f0 & ~f1] = 6
    se_case = {1: 0.0, 2: SE_DRIFT, 3: SE_DRIFT, 4: SE_DRIFT, 5: SE_BIAS, 6: SE_BIAS}
    res = {c: dict(area=0.0, chg=0.0, sys=0.0) for c in LEDGER_CAT}
    for sub, c in SUB2CAT.items():
        m = cat == sub
        res[c]['area'] += m.sum() * A_PX
        res[c]['chg'] += np.nansum(d[m]) * A_PX / 1e6
        res[c]['sys'] += se_case[sub] * m.sum() * A_PX / 1e6
    return res


pt_led = ledger(px_point, THR)
bt_led = [ledger(p_, THR) for p_ in boot_px]
net_pt = sum(v['chg'] for v in pt_led.values())
lrows = []
for c, name in LEDGER_CAT.items():
    areas = np.array([r[c]['area'] for r in bt_led]); chgs = np.array([r[c]['chg'] for r in bt_led])
    area_c = pt_led[c]['area']
    se_sys = pt_led[c]['sys']
    sb = chgs.std(ddof=1)
    sc_ = np.sqrt(sb ** 2 + se_sys ** 2)
    lrows.append(dict(code=c, category=name, area_ha=area_c,
                      area_CI95_low=np.quantile(areas, 0.025), area_CI95_high=np.quantile(areas, 0.975),
                      change_TgC=pt_led[c]['chg'], sd_bootstrap=sb, sd_systematic=se_sys, sd_combined=sc_,
                      change_CI95_low=pt_led[c]['chg'] - 1.96 * sc_, change_CI95_high=pt_led[c]['chg'] + 1.96 * sc_,
                      share_of_net_change_pct=100 * pt_led[c]['chg'] / net_pt if net_pt else np.nan))
led_u = pd.DataFrame(lrows)
led_u.loc[len(led_u)] = dict(code=0, category='Total', area_ha=led_u.area_ha.sum(), change_TgC=net_pt,
                             share_of_net_change_pct=100.0)
led_u['threshold_MgC_ha'] = THR
led_u.to_csv(os.path.join(OUT_DIR, 'ledger_uncertainty.csv'), index=False)
print(f"\nLedger {LEDGER_Y0}-{LEDGER_Y1} with 95 % CI (threshold {THR:.1f} Mg C/ha, "
      f"sampled pixels, 1/{STEP ** 2}):\n" + led_u.round(3).to_string(index=False))

srows = []
for k in THR_FACTORS:
    r_ = ledger(px_point, k * RMSE_CHG)
    for c, name in LEDGER_CAT.items():
        srows.append(dict(threshold_factor=k, threshold_MgC_ha=k * RMSE_CHG, code=c, category=name,
                          area_ha=r_[c]['area'], change_TgC=r_[c]['chg'],
                          share_of_net_change_pct=100 * r_[c]['chg'] / net_pt if net_pt else np.nan))
sens_thr = pd.DataFrame(srows)
sens_thr.to_csv(os.path.join(OUT_DIR, 'ledger_threshold_sensitivity.csv'), index=False)
print("\nLedger sensitivity to the threshold (area ha):\n" +
      sens_thr.pivot_table(index='category', columns='threshold_factor', values='area_ha').round(0).to_string())

# ── 7. Scenario sensitivity of the totals ────────────────────────────────────
if DO_SCENARIOS and os.path.exists(SCEN_CSV):
    sc_tab = pd.read_csv(SCEN_CSV)
    rows = []
    for _, r in sc_tab.iterrows():
        wl = None if str(r.weak_label) == 'none' else float(r.weak_label)
        use_f = str(r.get('fcd', 'yes')) == 'yes'
        feats = CORE + (FCD if use_f else []) + (PALSAR if r.palsar == 'yes' else [])
        prm = json.loads(r.best_params)
        d = scenario_data(wl)
        m = RandomForestRegressor(**prm, random_state=SEED, n_jobs=-1).fit(d[feats], d.AGB)
        for (yr, mask), v in totals(m, feats).items():
            rows.append(dict(scenario=r.scenario, weak_label=r.weak_label, palsar=r.palsar,
                             fcd='yes' if use_f else 'no',
                             year=yr, mask=mask, AGC_TgC=v))
        print(f"  scenario {r.scenario} done")
    sens = pd.DataFrame(rows)
    s_sel = sens[sens.scenario == meta['scenario']][['year', 'mask', 'AGC_TgC']].rename(
        columns={'AGC_TgC': 'AGC_selected'})
    sens = sens.merge(s_sel, on=['year', 'mask'])
    sens['diff_vs_selected_TgC'] = sens.AGC_TgC - sens.AGC_selected
    sens['diff_vs_selected_pct'] = 100 * sens.diff_vs_selected_TgC / sens.AGC_selected
    sens.to_csv(os.path.join(OUT_DIR, 'scenario_sensitivity_totals.csv'), index=False)
    print(f"\nScenario sensitivity (own forest mask, % vs selected {meta['scenario']}):")
    print(sens[sens['mask'] == 'forest_own'].pivot_table(
        index='scenario', columns='year', values='diff_vs_selected_pct').round(1).to_string())

# ── 8. Figure: trajectory with uncertainty ───────────────────────────────────
t = tot[tot['mask'] == 'forest_own'].sort_values('year')
if len(t) > 1:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.fill_between(t.year, t.CI95_low, t.CI95_high, color='#e0e0e0', zorder=1,
                    label='95% CI (bootstrap + field bias)')
    ax.fill_between(t.year, t.boot_p2_5, t.boot_p97_5, color='#a8a8a8', zorder=2,
                    label='95% CI (bootstrap only)')
    ax.plot(t.year, t.AGC_TgC, '-o', color='black', mfc='#606060', zorder=3, label='Estimate')
    ax.set_xticks(t.year)
    ax.set_xlabel('Year'); ax.set_ylabel('Forest AGC (Tg C)')
    ax.set_title(f'A. Forest AGC (FCD ≥ {FCD_FOREST:.0f} %)', loc='left', fontweight='bold')
    ax.grid(True, color='#e0e0e0', linewidth=0.5, zorder=0)
    ax.tick_params(direction='in', length=4)
    for s in ax.spines.values():
        s.set_linewidth(0.8); s.set_color('black')
    ax.legend(frameon=True, edgecolor='black', fontsize=9, loc='best')
    fig.savefig(os.path.join(FIG_DIR, 'fig_agc_totals_uncertainty.tif'), dpi=300,
                bbox_inches='tight', facecolor='white', format='tiff')
    plt.close(fig)

print(f"\nResults: {OUT_DIR}\nDone.")
