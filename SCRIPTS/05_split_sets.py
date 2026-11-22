r"""
05_split_sets.py  –  builds train / test / temporal-val sets for the AGB RF.
See inline comments for the full workflow description.
Usage:  python 05_split_sets.py  [csv_in interim_dir fig_dir results_dir]
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.model_selection import GroupKFold

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 11

# ── Paths ────────────────────────────────────────────────────────────────────
CSV_IN  = r"D:\PAPER-ECOLIN\DATA\interim\training_oct2025_predictors.csv"
OUT_DIR = r"D:\PAPER-ECOLIN\DATA\interim"
FIG_DIR = r"D:\PAPER-ECOLIN\FIGURAS"
AOI_SHP = r"D:\PAPER-ECOLIN\DATA\raw\study_area_chocho_andino.shp"
RBCA_SHP = r"D:\PAPER-ECOLIN\DATA\raw\RBCA_shp\RBiosfera_ChocoAndino.shp"
RES_DIR = r"D:\PAPER-ECOLIN\RESULTADOS"

# Local Natural Earth countries shapefile (ne_50m or ne_110m)
WORLD_SHP = r"D:\PAPER-ECOLIN\DATA\raw\ne_50m_admin_0_countries\ne_50m_admin_0_countries.shp"

# Panel A extent: Latin America + Mexico
LATAM_BBOX = (-92, -34, -56, 14)   # (lon_min, lon_max, lat_min, lat_max)

if len(sys.argv) == 5:
    CSV_IN, OUT_DIR, FIG_DIR, RES_DIR = sys.argv[1:5]
D_EXPL = os.path.join(RES_DIR, '00_exploratorio')
for _d in [OUT_DIR, FIG_DIR, D_EXPL]:
    os.makedirs(_d, exist_ok=True)

# ── Settings ─────────────────────────────────────────────────────────────────
SEED       = 42
WL_VALUE   = 15.0
TEST_FRAC  = 0.20
N_FOLDS    = 5
BLOCK_KM   = 10.0
BUFFER_KM  = 5.0
VAL_VISITS = {'2017': 2016, '2021': 2022}
PCA_VAR    = 0.80

BANDS    = ['B_mean','G_mean','R_mean','NIR_mean','SWIR1_mean','SWIR2_mean']
FEATURES = BANDS + ['NDVI_mean','GNDVI_mean','EVI_mean','HH_mean','HV_mean',
                    'elevation_srtm','slope_srtm','fcd_average_period']

log = []
def say(msg=''):
    print(msg); log.append(str(msg))

# ── 1. Load + drop NA ────────────────────────────────────────────────────────
df = pd.read_csv(CSV_IN)
df['Year(s)_Avitabile'] = df['Year(s)_Avitabile'].astype(str)
n0 = len(df)
na = df[FEATURES + ['AGB']].isna().any(axis=1)
say(f"Rows read: {n0}")
say(f"Rows dropped (NA): {na.sum()}  by code: {df.loc[na,'code'].value_counts().to_dict()}")
df = df[~na].reset_index(drop=True)
# Flag RBCA membership for Panel A visualization (does NOT filter the pipeline)
try:
    import geopandas as gpd
    _rbca_flag = gpd.read_file(RBCA_SHP).to_crs(4326)
    _rbca_union_flag = _rbca_flag.geometry.unary_union
    _pts_flag = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['x'], df['y']), crs='EPSG:4326')
    df['in_rbca'] = _pts_flag.within(_rbca_union_flag).values
    say(f"RBCA flag: {df['in_rbca'].sum()} inside / {(~df['in_rbca']).sum()} outside "
        f"(all used in model)")
except Exception as _e:
    say(f"WARNING: RBCA flag failed ({_e}), all points flagged inside")
    df['in_rbca'] = True

df['is_weak_label'] = np.isclose(df['AGB'], WL_VALUE)
lat0 = df['y'].mean()
df['x_km'] = df['x'] * 111.32 * np.cos(np.radians(lat0))
df['y_km'] = df['y'] * 110.57

# ── 2. Temporal validation set ───────────────────────────────────────────────
usfq_all = pd.read_csv(CSV_IN)
usfq_all = usfq_all[usfq_all.code == 'USFQ']
usfq_all['Year(s)_Avitabile'] = usfq_all['Year(s)_Avitabile'].astype(str)
site_codes = {xy: f"USFQ_{i:02d}" for i,xy in enumerate(sorted(usfq_all.groupby(['x','y']).groups))}
df['site_id'] = ''
usfq_mask = df.code == 'USFQ'
df.loc[usfq_mask,'site_id'] = [site_codes[(a,b)] for a,b in zip(df.loc[usfq_mask,'x'],df.loc[usfq_mask,'y'])]
u = df[usfq_mask]
paired_sites = set.intersection(*[set(u.loc[u['Year(s)_Avitabile']==v,'site_id']) for v in VAL_VISITS])
is_temporal = usfq_mask & df.site_id.isin(paired_sites) & df['Year(s)_Avitabile'].isin(VAL_VISITS)
temporal = df[is_temporal].copy()
temporal['map_year'] = temporal['Year(s)_Avitabile'].map(VAL_VISITS)
v0, v1 = sorted(VAL_VISITS)
agb_v0 = temporal[temporal['Year(s)_Avitabile']==v0].set_index('site_id')['AGB']
temporal[f'AGB_{v0}'] = temporal.site_id.map(agb_v0)
temporal['delta_AGB_obs'] = np.where(temporal['Year(s)_Avitabile']==v1,
                                     temporal.AGB - temporal[f'AGB_{v0}'], np.nan)
pool = df[~is_temporal].copy()
val_xy = temporal.drop_duplicates('site_id')[['x_km','y_km']].values
dmin = np.sqrt(((pool[['x_km']].values - val_xy[:,0])**2 +
                (pool[['y_km']].values - val_xy[:,1])**2)).min(axis=1)
pool['dist_val_km'] = dmin
pool['in_buffer'] = dmin <= BUFFER_KM
say(f"\nTemporal validation (paired): {len(paired_sites)} sites x {len(VAL_VISITS)} visits = "
    f"{len(temporal)} rows  {temporal.groupby('Year(s)_Avitabile').size().to_dict()}  -> maps {VAL_VISITS}")
say(f"Change pairs ({v1} - {v0}): {temporal.delta_AGB_obs.notna().sum()}")
say(f"Pool records within {BUFFER_KM:.0f} km of validation sites: "
    f"{pool.in_buffer.sum()}  {pool[pool.in_buffer].code.value_counts().to_dict()}")

# ── 3. Semivariogram of OOB RF residuals → block size ────────────────────────
from sklearn.ensemble import RandomForestRegressor
real = pool[~pool.is_weak_label]
rf_v = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, oob_score=True,
                             random_state=SEED, n_jobs=-1).fit(real[FEATURES], real['AGB'])
resid = (real['AGB'] - rf_v.oob_prediction_).values
h = pdist(real[['x_km','y_km']].values)
g = 0.5 * pdist(resid[:,None], 'sqeuclidean')
edges = [0,2,5,10,15,20,30,45,70,150,600]
lag, gam, npair = [], [], []
for lo,hi in zip(edges[:-1], edges[1:]):
    m = (h>=lo) & (h<hi)
    if m.sum() >= 30:
        lag.append(h[m].mean()); gam.append(g[m].mean()); npair.append(m.sum())
lag, gam = np.array(lag), np.array(gam)
say(f"\nOOB R2: {rf_v.oob_score_:.2f}")
block_km = BLOCK_KM
pool['block_id'] = (np.floor(pool.x_km/block_km).astype(int).astype(str) + '_' +
                    np.floor(pool.y_km/block_km).astype(int).astype(str))

# ── 4. Spatial 80/20 split ───────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
real = ~pool.is_weak_label
n_real = real.sum()
forced_blocks = set(pool.loc[pool.in_buffer,'block_id'])
forced = pool.block_id.isin(forced_blocks) | pool.in_buffer
n_test = int((real & forced).sum())
free = pool[real & ~forced]
real_counts = free.groupby('block_id').size()
blocks = real_counts.index.values.copy()
rng.shuffle(blocks)
test_blocks = list(forced_blocks)
for b in blocks:
    if n_test/n_real >= TEST_FRAC: break
    if (n_test + real_counts[b])/n_real <= TEST_FRAC + 0.05:
        test_blocks.append(b); n_test += real_counts[b]
to_test = pool.block_id.isin(test_blocks) | pool.in_buffer
pool['set'] = 'train'
pool.loc[to_test & real,'set'] = 'test_interno'
pool.loc[to_test & pool.is_weak_label,'set'] = 'excluded_wl'

# Carry set labels back to df for Panel A.
# Use DataFrame index (NOT x,y tuples) to avoid collisions from repeated-visit
# sites that share the same coordinates across years.
df['set_label'] = 'outside_rbca'
_pool_lbl = pool['set'].copy()                               # 'train','test_interno','excluded_wl'
_pool_lbl[pool['is_weak_label'] & (pool['set'] == 'train')] = 'train_wl'
# 'excluded_wl' rows keep their label and are silently not plotted in either panel
df.loc[pool.index, 'set_label'] = _pool_lbl.values
df.loc[temporal.index, 'set_label'] = 'temporal_val'

# ── 5. 5-fold GroupKFold ─────────────────────────────────────────────────────
tr_real = pool[(pool.set=='train') & ~pool.is_weak_label]
fold_of_block = {}
for k,(_,te) in enumerate(GroupKFold(n_splits=N_FOLDS).split(tr_real, groups=tr_real.block_id)):
    for b in tr_real.iloc[te].block_id.unique(): fold_of_block[b] = k
pool['fold'] = -1
trm = pool.set == 'train'
pool.loc[trm,'fold'] = pool.loc[trm,'block_id'].map(fold_of_block).fillna(-1).astype(int)

# ── 6. PCA ────────────────────────────────────────────────────────────────────
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
tr_rows = pool.set == 'train'
scaler = StandardScaler().fit(pool.loc[tr_rows, BANDS])
pca_full = PCA(random_state=SEED).fit(scaler.transform(pool.loc[tr_rows, BANDS]))
cum = np.cumsum(pca_full.explained_variance_ratio_)
n_pc = int(np.searchsorted(cum, PCA_VAR) + 1)
pca = PCA(n_components=n_pc, random_state=SEED).fit(scaler.transform(pool.loc[tr_rows, BANDS]))
PCS = [f'PC{i+1}' for i in range(n_pc)]
for d_ in (pool, temporal):
    d_[PCS] = pca.transform(scaler.transform(d_[BANDS]))
joblib.dump(dict(scaler=scaler, pca=pca, bands=BANDS, pcs=PCS),
            os.path.join(OUT_DIR, 'pca_landsat.joblib'))
ev = pd.DataFrame({'component':[f'PC{i+1}' for i in range(len(cum))],
                   'explained_variance_ratio': pca_full.explained_variance_ratio_,
                   'cumulative': cum,
                   'eigenvalue': pca_full.explained_variance_ratio_*len(BANDS),
                   'kaiser_gt1': pca_full.explained_variance_ratio_*len(BANDS) > 1,
                   'kept': np.arange(len(cum)) < n_pc})
ev.to_csv(os.path.join(D_EXPL, 'pca_explained_variance.csv'), index=False)
load = pd.DataFrame(pca.components_.T, index=BANDS, columns=PCS)
load.to_csv(os.path.join(D_EXPL, 'pca_loadings.csv'))
say(f"\nPCA: kept {n_pc} PCs = {100*cum[n_pc-1]:.1f}% of variance (train n={tr_rows.sum()})")
say(ev.round(3).to_string(index=False))
say("Loadings:\n" + load.round(2).to_string())

# ── Summary ──────────────────────────────────────────────────────────────────
say("\nSets (modelling pool):")
say(pool.groupby(['set','is_weak_label']).size().unstack(fill_value=0).to_string())
say(f"\nBlocks: total {pool.block_id.nunique()}, test {len(test_blocks)}")
say(f"Real-AGB share in test_interno: {(pool.set=='test_interno').sum()/n_real:.1%}")
say("\nFolds inside train (real AGB points):")
say(pool[trm & ~pool.is_weak_label].fold.value_counts().sort_index().to_string())
for s in ['train','test_interno']:
    a = pool.loc[(pool.set==s) & ~pool.is_weak_label,'AGB']
    say(f"AGB {s:12s}: mean {a.mean():6.1f}  sd {a.std():6.1f}  min {a.min():6.1f}  max {a.max():6.1f}")
say("\nUSFQ in pool by set: " + str(pool[pool.code=='USFQ'].set.value_counts().to_dict()))

# ── Save ─────────────────────────────────────────────────────────────────────
keep = ['unique_id_combined','code','site_id','x','y','Year(s)_Avitabile',
        'Year_end_even','AGB','is_weak_label'] + FEATURES + PCS
pool[keep + ['block_id','set','fold','in_buffer','dist_val_km']].to_csv(
    os.path.join(OUT_DIR, 'split_train_internal.csv'), index=False)
temporal[keep + ['map_year', f'AGB_{v0}', 'delta_AGB_obs']].to_csv(
    os.path.join(OUT_DIR, 'split_temporal_validation.csv'), index=False)
pd.DataFrame({'lag_km':lag,'semivariance':gam,'n_pairs':npair}).to_csv(
    os.path.join(D_EXPL, 'residual_semivariogram.csv'), index=False)
with open(os.path.join(OUT_DIR, 'split_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(log))

# ════════════════════════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════════════════════════
def style(ax):
    ax.tick_params(direction='in', length=4)
    for s in ax.spines.values(): s.set_linewidth(0.8); s.set_color('black')

def legend(ax, **kw):
    leg = ax.legend(frameon=True, edgecolor='black', facecolor='white', framealpha=1.0, **kw)
    leg.get_frame().set_linewidth(0.8)
    return leg

# ── Embedded LatAm polygons (last-resort fallback, no internet needed) ────────
def _embedded_latam():
    from shapely.geometry import shape
    import geopandas as gpd
    _D = [
      ("Ecuador",  [[-80.0,1.4],[-77.0,0.9],[-75.5,0.0],[-75.3,-0.1],[-76.6,-0.3],
                    [-77.5,-0.6],[-78.1,-1.3],[-79.0,-2.4],[-79.8,-3.4],[-80.3,-3.5],
                    [-80.5,-2.5],[-80.5,-0.6],[-80.0,1.4]]),
      ("Colombia", [[-80.0,1.4],[-77.5,8.7],[-75.6,10.7],[-74.1,11.1],[-72.2,11.6],
                    [-71.0,11.8],[-67.5,11.2],[-65.7,10.6],[-63.1,9.1],[-62.0,8.5],
                    [-60.7,5.2],[-67.3,2.0],[-70.0,-1.0],[-73.5,-0.3],[-75.5,0.0],
                    [-77.0,0.9],[-80.0,1.4]]),
      ("Venezuela",[[-73.4,10.0],[-71.0,11.8],[-62.0,11.0],[-60.5,5.0],[-67.3,2.0],
                    [-67.3,4.0],[-70.1,6.9],[-73.4,10.0]]),
      ("Peru",     [[-80.5,-3.5],[-78.1,-1.3],[-75.5,0.0],[-73.5,-0.3],[-73.0,-6.0],
                    [-70.5,-9.0],[-69.5,-11.0],[-69.5,-17.0],[-70.5,-18.0],
                    [-75.5,-15.0],[-80.5,-8.0],[-80.5,-3.5]]),
      ("Bolivia",  [[-69.5,-11.0],[-63.5,-11.0],[-63.5,-17.0],[-67.5,-22.5],
                    [-69.5,-17.0],[-69.5,-11.0]]),
      ("Brazil",   [[-60.7,5.2],[-59.0,3.0],[-60.0,-1.0],[-65.0,-11.0],[-69.5,-11.0],
                    [-70.5,-9.0],[-67.3,2.0],[-60.7,5.2]]),
      ("Panama",   [[-77.5,8.7],[-80.0,8.1],[-82.9,8.2],[-83.6,8.0],
                    [-82.5,7.5],[-77.2,7.2],[-77.5,8.7]]),
      ("Guyana",   [[-60.7,5.2],[-59.0,6.8],[-58.5,7.5],[-60.0,8.5],[-62.0,8.5],[-60.7,5.2]]),
    ]
    recs = [{'NAME': n, 'geometry': shape({'type':'Polygon','coordinates':[c]})} for n,c in _D]
    return gpd.GeoDataFrame(recs, crs='EPSG:4326')

# ── Natural Earth loader ───────────────────────────────────────────────────────
def _load_naturalearth():
    import geopandas as gpd, pathlib
    tried = []

    if WORLD_SHP:
        p = pathlib.Path(WORLD_SHP)
        if p.exists():
            print(f"  [NE] WORLD_SHP: {p}"); return gpd.read_file(str(p)).to_crs(4326)
        tried.append(f'WORLD_SHP missing: {WORLD_SHP}')

    try:
        gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).to_crs(4326)
        print("  [NE] gpd.datasets OK"); return gdf
    except Exception as e: tried.append(f'gpd.datasets:{type(e).__name__}')

    try:
        for sub in ['naturalearth_lowres.shp',
                    os.path.join('naturalearth_lowres','naturalearth_lowres.shp')]:
            p = pathlib.Path(gpd.__file__).parent / 'datasets' / sub
            if p.exists():
                print(f"  [NE] pkg dir: {p}"); return gpd.read_file(str(p)).to_crs(4326)
        tried.append('gpd pkg dir: absent')
    except Exception as e: tried.append(f'gpd pkg dir:{type(e).__name__}')

    try:
        import geodatasets
        gdf = gpd.read_file(geodatasets.get_path('naturalearth lowres')).to_crs(4326)
        print("  [NE] geodatasets OK"); return gdf
    except ImportError: tried.append('geodatasets: not installed')
    except Exception as e: tried.append(f'geodatasets:{type(e).__name__}:{e}')

    try:
        import glob as _glob
        for _base in [r'C:\Program Files\QGIS*', r'C:\OSGeo4W*', r'C:\Program Files\OSGeo4W*']:
            for _shp in _glob.glob(_base+r'\**\ne_110m_admin_0_countries.shp', recursive=True):
                _gdf = gpd.read_file(_shp).to_crs(4326)
                if any(c in _gdf.columns for c in ('NAME','name','ADMIN')):
                    print(f"  [NE] QGIS: {_shp}"); return _gdf
        tried.append('QGIS: not found')
    except Exception as e: tried.append(f'QGIS:{type(e).__name__}')

    try:
        import cartopy.io.shapereader as shpreader
        shp = shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries')
        gdf = gpd.read_file(shp).to_crs(4326)
        print("  [NE] cartopy OK"); return gdf
    except ImportError: tried.append('cartopy: not installed')
    except Exception as e: tried.append(f'cartopy:{type(e).__name__}')

    print(f"  [NE] using embedded LatAm polygons (tried: {'; '.join(tried)})")
    return _embedded_latam()

# ── Variogram figure ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.plot(lag, gam, '-', color='#808080', linewidth=0.8, zorder=2)
ax.scatter(lag, gam, color='#808080', edgecolors='#404040', s=35, zorder=3)
ax.axvline(block_km, color='black', linestyle='--', linewidth=0.8)
ax.text(block_km, ax.get_ylim()[1], f'  block = {block_km:.0f} km', va='top', fontsize=9)
ax.set_xscale('log')
ax.set_xlabel('Distance (km, log scale)'); ax.set_ylabel('Semivariance (Mg² ha⁻²)')
ax.set_title('A. Semivariogram of RF out-of-bag residuals', loc='left', fontweight='bold')
ax.grid(True, color='#e0e0e0', linewidth=0.5, zorder=0); style(ax)
fig.savefig(os.path.join(FIG_DIR,'fig_agb_variogram.tif'), dpi=300, bbox_inches='tight',
            facecolor='white', format='tiff')
plt.close(fig)

# ── Location figure: Panel A (regional) + Panel B (study area) ───────────────
STUDY_BBOX = (-79.2, -78.2, -0.5, 0.4)
kx = 111.32 * np.cos(np.radians(lat0))
fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), gridspec_kw={'width_ratios':[1,1.15]})

def draw_sets(ax, s_scale=1.0):
    ax.scatter(pool.loc[(pool.set=='train')&~pool.is_weak_label,'x'],
               pool.loc[(pool.set=='train')&~pool.is_weak_label,'y'],
               marker='o', s=22*s_scale, color='#a0a0a0', edgecolors='#505050',
               linewidths=0.4, zorder=2,
               label=f"Train (n = {((pool.set=='train')&~pool.is_weak_label).sum()})")
    ax.scatter(pool.loc[(pool.set=='train')&pool.is_weak_label,'x'],
               pool.loc[(pool.set=='train')&pool.is_weak_label,'y'],
               marker='x', s=18*s_scale, color='#505050', linewidths=0.8, zorder=2,
               label=f"Train weak-labels (n = {((pool.set=='train')&pool.is_weak_label).sum()})")
    ax.scatter(pool.loc[pool.set=='test_interno','x'],
               pool.loc[pool.set=='test_interno','y'],
               marker='^', s=30*s_scale, color='black', zorder=3,
               label=f"Internal test (n = {(pool.set=='test_interno').sum()})")
    usfq_pts = temporal.drop_duplicates('site_id')
    ax.scatter(usfq_pts.x, usfq_pts.y, marker='s', s=55*s_scale, color='white',
               edgecolors='black', linewidths=1.1, zorder=4,
               label=f"Temporal validation (USFQ, {temporal.site_id.nunique()} sites)")

# ── PANEL A ───────────────────────────────────────────────────────────────────
ax = axes[0]

lo0, lo1, la0, la1 = STUDY_BBOX

# Extent driven by actual data, not fixed to Ecuador
_pad = 1.0
_ex0 = df['x'].min() - _pad
_ex1 = df['x'].max() + _pad
_ey0 = df['y'].min() - _pad
_ey1 = df['y'].max() + _pad

# Country borders clipped to data extent (drawn first, behind points)
try:
    import geopandas as gpd
    _world = _load_naturalearth()
    _nc = next((c for c in ('NAME','name','ADMIN','admin','SOVEREIGNT') if c in _world.columns), None)
    _world.cx[_ex0:_ex1, _ey0:_ey1].boundary.plot(
        ax=ax, color='#555555', linewidth=0.5, zorder=1, alpha=0.7)
    _world[_world[_nc] == 'Ecuador'].boundary.plot(
        ax=ax, color='black', linewidth=1.2, zorder=2)
except Exception as _e:
    print(f"WARNING Panel A borders: {_e}")

# All points with their split labels — same symbols as Panel B
_sl = df['set_label']
ax.scatter(df.loc[_sl=='train','x'], df.loc[_sl=='train','y'],
           marker='o', s=22, color='#a0a0a0', edgecolors='#505050',
           linewidths=0.4, zorder=3,
           label=f"Train (n = {(_sl=='train').sum()})")
ax.scatter(df.loc[_sl=='train_wl','x'], df.loc[_sl=='train_wl','y'],
           marker='x', s=20, color='#505050', linewidths=0.9, zorder=3,
           label=f"Train weak-labels (n = {(_sl=='train_wl').sum()})")
ax.scatter(df.loc[_sl=='test_interno','x'], df.loc[_sl=='test_interno','y'],
           marker='^', s=30, color='black', zorder=4,
           label=f"Internal test (n = {(_sl=='test_interno').sum()})")
_temporal_uniq_a = temporal.drop_duplicates('site_id')
ax.scatter(_temporal_uniq_a['x'], _temporal_uniq_a['y'],
           marker='s', s=52, color='white', edgecolors='black', linewidths=1.1, zorder=5,
           label=f"Temporal val – USFQ ({temporal.site_id.nunique()} sites)")

# RBCA boundary
try:
    import geopandas as gpd
    gpd.read_file(RBCA_SHP).to_crs(4326).boundary.plot(
        ax=ax, color='black', linewidth=1.0, zorder=5)
except Exception as _e:
    print(f"WARNING Panel A RBCA: {_e}")

# Study bbox rectangle (links to Panel B)
ax.add_patch(plt.Rectangle((lo0,la0), lo1-lo0, la1-la0, fill=False,
                            edgecolor='black', linewidth=1.2, linestyle='--', zorder=6))

ax.set_xlim(_ex0, _ex1)
ax.set_ylim(_ey0, _ey1)
ax.set_xlabel('Longitude (°)'); ax.set_ylabel('Latitude (°)')
ax.set_title('A. All plots', loc='left', fontweight='bold')
ax.set_aspect('equal'); ax.grid(True, color='#e6e6e6', linewidth=0.5, zorder=0)
legend(ax, fontsize=9, loc='lower left')
style(ax)

# ── PANEL B ───────────────────────────────────────────────────────────────────
ax = axes[1]
for v in np.arange(np.floor(lo0*kx/block_km), np.ceil(lo1*kx/block_km)+1)*block_km:
    ax.axvline(v/kx, color='#d9d9d9', linewidth=0.5, zorder=0)
for v in np.arange(np.floor(la0*110.57/block_km), np.ceil(la1*110.57/block_km)+1)*block_km:
    ax.axhline(v/110.57, color='#d9d9d9', linewidth=0.5, zorder=0)
if os.path.exists(AOI_SHP):
    try:
        import geopandas as gpd
        gpd.read_file(AOI_SHP).to_crs(4326).boundary.plot(
            ax=ax, color='black', linewidth=0.9, zorder=1)
    except Exception as e: print(f"AOI: {e}")

draw_sets(ax, 1.3)

_rbca_ok = False
try:
    import geopandas as gpd
    gpd.read_file(RBCA_SHP).to_crs(4326).boundary.plot(
        ax=ax, color='black', linewidth=1.2, zorder=5)
    _rbca_ok = True
    print("  [Panel B] RBCA drawn OK")
except Exception as _e:
    print(f"WARNING Panel B RBCA: {_e}")

for _, r in temporal.drop_duplicates('site_id').iterrows():
    ax.add_patch(matplotlib.patches.Ellipse(
        (r.x, r.y), 2*BUFFER_KM/kx, 2*BUFFER_KM/110.57,
        fill=False, edgecolor='black', linewidth=0.5, linestyle=':', zorder=1))

from matplotlib.lines import Line2D
_handles, _labels = ax.get_legend_handles_labels()
if _rbca_ok:
    _rbca_line = Line2D([0],[0], color='black', linewidth=1.2, linestyle='-')
    _handles = [_rbca_line] + _handles
    _labels  = ['RBCA'] + _labels
_buf_line = Line2D([0],[0], color='black', linewidth=0.8, linestyle=':')
_handles.append(_buf_line)
_labels.append(f'{BUFFER_KM:.0f} km buffer (not in train)')
legend(ax, handles=_handles, labels=_labels, fontsize=8, loc='lower left')

ax.set_xlim(lo0, lo1); ax.set_ylim(la0, la1)
ax.set_xlabel('Longitude (°)'); ax.set_ylabel('Latitude (°)')
ax.set_title(f'B. Study area (grid = {block_km:.0f} km spatial blocks)',
             loc='left', fontweight='bold')
ax.set_aspect('equal'); style(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR,'fig_sets_location.tif'), dpi=300,
            bbox_inches='tight', facecolor='white', format='tiff')
plt.close(fig)

# ── Predictor correlation matrix ─────────────────────────────────────────────
LABELS = {'B_mean':'Blue','G_mean':'Green','R_mean':'Red','NIR_mean':'NIR',
          'SWIR1_mean':'SWIR1','SWIR2_mean':'SWIR2','NDVI_mean':'NDVI',
          'GNDVI_mean':'GNDVI','EVI_mean':'EVI','HH_mean':'HH','HV_mean':'HV',
          'elevation_srtm':'Elevation','slope_srtm':'Slope',
          'fcd_average_period':'FCD','AGB':'AGB'}
LABELS.update({p:p for p in PCS})
FEAT_PCA = PCS + FEATURES[6:]
corr     = pool.loc[~pool.is_weak_label, FEATURES+['AGB']].corr(method='spearman')
corr_pca = pool.loc[~pool.is_weak_label, FEAT_PCA+['AGB']].corr(method='spearman')
corr.to_csv(os.path.join(D_EXPL,'predictor_correlation_spearman.csv'))
corr_pca.to_csv(os.path.join(D_EXPL,'predictor_correlation_spearman_pca.csv'))

def corr_panel(ax, c, title):
    n = len(c.columns); m = np.triu(np.ones((n,n),bool),k=1)
    im = ax.imshow(np.ma.masked_where(m, c.abs().values), cmap='Greys', vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if j<=i:
                v = c.values[i,j]
                ax.text(j,i,f"{v:.2f}",ha='center',va='center',fontsize=7,
                        color='white' if abs(v)>0.6 else 'black')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([LABELS[k] for k in c.columns],rotation=90)
    ax.set_yticklabels([LABELS[k] for k in c.columns])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(title, loc='left', fontweight='bold'); return im

fig, axes = plt.subplots(1,2,figsize=(16,7.5))
corr_panel(axes[0], corr,     f'A. Original predictors (n = {(~pool.is_weak_label).sum()})')
im = corr_panel(axes[1], corr_pca, f'B. After PCA of Landsat bands ({len(PCS)} PCs)')
fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02).set_label('|Spearman ρ|')
fig.savefig(os.path.join(FIG_DIR,'fig_predictor_correlation.tif'), dpi=300,
            bbox_inches='tight', facecolor='white', format='tiff')
plt.close(fig)

# ── PCA figure ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(9,4.2), gridspec_kw={'width_ratios':[1.5,0.7]})
ax = axes[0]; k = np.arange(1, len(cum)+1)
ax.bar(k, 100*pca_full.explained_variance_ratio_,
       color=['#606060' if i<n_pc else '#d0d0d0' for i in range(len(k))],
       edgecolor='black', linewidth=0.6, zorder=3)
ax.plot(k, 100*cum, '-o', color='black', mfc='white', zorder=4)
ax.axhline(100*PCA_VAR, color='black', linestyle=(0,(4,3)), linewidth=0.8)
ax.set_xticks(k); ax.set_xticklabels([f'PC{i}' for i in k])
ax.set_ylabel('Explained variance (%)'); ax.set_ylim(0,105)
ax.set_title('A. Explained variance (train)', loc='left', fontweight='bold')
ax.grid(True, axis='y', color='#e0e0e0', linewidth=0.5, zorder=0); style(ax)
ax = axes[1]; L = load.values
im = ax.imshow(np.abs(L), cmap='Greys', vmin=0, vmax=1)
for i in range(L.shape[0]):
    for j in range(L.shape[1]):
        ax.text(j,i,f"{L[i,j]:.2f}",ha='center',va='center',fontsize=9,
                color='white' if abs(L[i,j])>0.6 else 'black')
ax.set_xticks(range(len(PCS))); ax.set_xticklabels(PCS)
ax.set_yticks(range(len(BANDS))); ax.set_yticklabels([LABELS[b] for b in BANDS])
ax.set_title('B. Loadings', loc='left', fontweight='bold')
fig.colorbar(im, ax=ax, fraction=0.08, pad=0.04).set_label('|Loading|')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR,'fig_pca_landsat.tif'), dpi=300,
            bbox_inches='tight', facecolor='white', format='tiff')
plt.close(fig)

hi = [(a,b,corr.loc[a,b]) for i,a in enumerate(FEATURES) for b in FEATURES[i+1:]
      if abs(corr.loc[a,b]) >= 0.9]
say(f"\nPredictor pairs |rho|>=0.9: {len(hi)}")
for a,b,v in hi: say(f"  {LABELS[a]:9s} - {LABELS[b]:9s}  {v:+.2f}")
say("\nSaved: " + OUT_DIR + ", " + FIG_DIR)
