r"""
06_rf_scenarios.py
16 scenarios (weak-label AGB x PALSAR x FCD) -> spatial K-fold tuning -> internal test
-> scenario selection -> final model -> permutation importance -> temporal validation.

Inputs (from 05_split_sets.py, interim):
  split_train_internal.csv      set = train / test_interno / excluded_wl, fold, block_id
  split_temporal_validation.csv paired USFQ re-measurement visits (2017, 2021)
  pca_landsat.joblib            scaler + PCA of the 6 Landsat bands (from 05)

Rules
  * Weak-labels (is_weak_label) are only ever used for training. Never scored.
  * Hyperparameters: GridSearch with the 5 spatial folds of 'train'.
      - training part of each fold drops every row in the held-out blocks (incl. WL)
      - held-out part = real-AGB rows only
      - metric = RMSE
  * Landsat: the 6 reflectance bands enter as their principal component(s) (PC1..),
    fitted on 'train' in 05 (pca_landsat.joblib); NDVI, GNDVI, EVI enter as such.
  * Scenarios S1-S8 use FCD as predictor; S9-S16 are the same 8 WITHOUT FCD (tests the
    circularity of FCD, R1-3 / R2-9). FCD still defines the forest domain of the maps (07).
  * Scenario selection: LOWEST RMSE on 'test_interno' (real AGB) among the 16. Everything
    downstream (07, 07b, 07c) reads the selected scenario from rf_selected_model.json.
  * Final model: selected scenario + its hyperparameters refitted on ALL rows
    of the pool (train + test_interno + all weak-labels at the scenario value).
  * Temporal validation (used once): paired USFQ sites, visits 2017 and 2021
    (stocks) and change 2021 - 2017, with skill against persistence (delta = 0).

Inputs are read from DATA\interim. All results go to RESULTADOS:
  01_kfold_tuning\        rf_cv_tuning.csv (64 combos x 16 scenarios), ntrees_convergence.csv
  02_test_interno\        rf_scenarios_internal_test.csv
  03_modelo_final\        rf_selected_model.joblib / .json, rf_permutation_importance.csv
  04_validacion_temporal\ rf_temporal_validation.csv, rf_temporal_metrics.csv
  SI_tablas\              Table_S_hyperparameter_grid.csv, Table_S_hyperparameters_by_scenario.csv,
                          Table_S_ntrees_convergence.csv, Supplementary_tables_RF.xlsx
Figures (FIGURAS): fig_rf_scenarios.tif, fig_rf_internal_test.tif, fig_rf_importance.tif,
  fig_rf_temporal_validation.tif, fig_si_ntrees_convergence.tif
Usage: python 06_rf_scenarios.py [interim_dir fig_dir results_dir]
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 11

# ── Paths ────────────────────────────────────────────────────────────────────
INTERIM = r"D:\PAPER-ECOLIN\DATA\interim"
FIG_DIR = r"D:\PAPER-ECOLIN\FIGURAS"
RES_DIR = r"D:\PAPER-ECOLIN\RESULTADOS"
if len(sys.argv) == 4:
    INTERIM, FIG_DIR, RES_DIR = sys.argv[1:4]
D_KFOLD = os.path.join(RES_DIR, '01_kfold_tuning')
D_TEST  = os.path.join(RES_DIR, '02_test_interno')
D_MODEL = os.path.join(RES_DIR, '03_modelo_final')
D_TEMP  = os.path.join(RES_DIR, '04_validacion_temporal')
D_SI    = os.path.join(RES_DIR, 'SI_tablas')
for _d in [FIG_DIR, D_KFOLD, D_TEST, D_MODEL, D_TEMP, D_SI]:
    os.makedirs(_d, exist_ok=True)
NTREES_CHECK = [25, 50, 100, 250, 500, 1000, 1500]

# ── Settings ─────────────────────────────────────────────────────────────────
SEED      = 42
N_FOLDS   = 5
TIE_TOL   = 0.0          # Mg/ha; 0 = pure lowest RMSE (parsimony only for exact ties)
N_PERM    = 50           # permutation repeats
STUDY_BBOX = (-79.2, -78.2, -0.5, 0.4)
PARAM_GRID = {
    'n_estimators':     [500],
    'max_depth':        [None, 6, 9, 12],
    'min_samples_leaf': [1, 3, 5, 10],
    'max_features':     ['sqrt', 0.33, 0.5, 1.0],
}
PCA_INFO = joblib.load(os.path.join(INTERIM, 'pca_landsat.joblib'))
PCS = PCA_INFO['pcs']                                  # e.g. ['PC1']
INDICES = ['NDVI_mean', 'GNDVI_mean', 'EVI_mean']
CORE = PCS + INDICES + ['elevation_srtm', 'slope_srtm']
FCD = ['fcd_average_period']
PALSAR = ['HH_mean', 'HV_mean']
GROUPS = {'Landsat': PCS + INDICES, 'SRTM': ['elevation_srtm', 'slope_srtm'],
          'FCD': ['fcd_average_period'], 'PALSAR': PALSAR}
LABELS = {'NDVI_mean': 'NDVI', 'GNDVI_mean': 'GNDVI', 'EVI_mean': 'EVI',
          'HH_mean': 'HH', 'HV_mean': 'HV', 'elevation_srtm': 'Elevation',
          'slope_srtm': 'Slope', 'fcd_average_period': 'FCD', **{p: f'Landsat {p}' for p in PCS}}
SCENARIOS = [  # id, weak-label AGB (None = no weak-labels), PALSAR, FCD
    ('S1', None, True, True), ('S2', 15, True, True), ('S3', 25, True, True), ('S4', 40, True, True),
    ('S5', None, False, True), ('S6', 15, False, True), ('S7', 25, False, True), ('S8', 40, False, True),
    ('S9', None, True, False), ('S10', 15, True, False), ('S11', 25, True, False), ('S12', 40, True, False),
    ('S13', None, False, False), ('S14', 15, False, False), ('S15', 25, False, False), ('S16', 40, False, False)]


def scenario_features(use_p, use_f):
    """Same order as before: Landsat, SRTM, [FCD], [PALSAR]."""
    return CORE + (FCD if use_f else []) + (PALSAR if use_p else [])

# ── Load ─────────────────────────────────────────────────────────────────────
pool = pd.read_csv(os.path.join(INTERIM, 'split_train_internal.csv'))
temporal = pd.read_csv(os.path.join(INTERIM, 'split_temporal_validation.csv'))
train_all = pool[pool.set == 'train'].reset_index(drop=True)
test_int = pool[pool.set == 'test_interno'].reset_index(drop=True)
in_area_test = (test_int.x.between(*STUDY_BBOX[:2]) & test_int.y.between(*STUDY_BBOX[2:])).values
print(f"train {len(train_all)} (WL {train_all.is_weak_label.sum()}) | "
      f"test_interno {len(test_int)} (in study area {in_area_test.sum()}) | temporal {len(temporal)}")


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    rmse = np.sqrt(mean_squared_error(y, p))
    return dict(n=len(y), mean_obs=y.mean(), mean_pred=p.mean(), RMSE=rmse, rRMSE_pct=100 * rmse / y.mean(),
                bias=np.mean(p - y), MAE=mean_absolute_error(y, p),
                R2=r2_score(y, p) if len(y) > 2 else np.nan)


def scenario_data(df, wl):
    d = df.copy()
    if wl is None:
        d = d[~d.is_weak_label]
    else:
        d.loc[d.is_weak_label, 'AGB'] = float(wl)
    return d.reset_index(drop=True)


def spatial_folds(d):
    """(train_idx, val_idx): train drops held-out blocks entirely; val = real AGB only."""
    out = []
    for k in range(N_FOLDS):
        held = d.fold == k
        out.append((np.where(~held)[0], np.where(held & ~d.is_weak_label)[0]))
    return out


# ── 1. Tuning (spatial K-fold) + internal test, per scenario ─────────────────
cv_rows, res_rows, fitted = [], [], {}
for sid, wl, use_p, use_f in SCENARIOS:
    t0 = time.time()
    feats = scenario_features(use_p, use_f)
    d = scenario_data(train_all, wl)
    gs = GridSearchCV(RandomForestRegressor(random_state=SEED, n_jobs=1), PARAM_GRID,
                      cv=spatial_folds(d), scoring='neg_root_mean_squared_error',
                      n_jobs=-1, refit=True)
    gs.fit(d[feats], d['AGB'])
    cvr = pd.DataFrame(gs.cv_results_)
    cvr.insert(0, 'scenario', sid)
    cv_rows.append(cvr[['scenario', 'params', 'mean_test_score', 'std_test_score', 'rank_test_score']])

    p_int = gs.best_estimator_.predict(test_int[feats])
    m_all = metrics(test_int.AGB, p_int)
    m_area = metrics(test_int.AGB[in_area_test], p_int[in_area_test])
    res_rows.append(dict(scenario=sid, weak_label='none' if wl is None else wl,
                         palsar='yes' if use_p else 'no', fcd='yes' if use_f else 'no', n_train=len(d),
                         best_params=json.dumps(gs.best_params_),
                         CV_RMSE=-gs.best_score_,
                         CV_RMSE_sd=cvr.loc[gs.best_index_, 'std_test_score'],
                         **{f'test_{k}': v for k, v in m_all.items()},
                         **{f'test_area_{k}': v for k, v in m_area.items()}))
    fitted[sid] = (gs.best_estimator_, feats, wl, use_p, use_f, gs.best_params_, p_int)
    print(f"{sid} WL={str(wl):>4} PALSAR={'yes' if use_p else 'no '} FCD={'yes' if use_f else 'no '}  "
          f"tuned {len(cvr)} combos x {N_FOLDS} spatial folds = {len(cvr) * N_FOLDS} fits  "
          f"(CV RMSE range {-cvr.mean_test_score.max():.1f}-{-cvr.mean_test_score.min():.1f})")
    print(f"     best {gs.best_params_}  CV RMSE={-gs.best_score_:6.1f}  "
          f"test RMSE={m_all['RMSE']:6.1f}  R2={m_all['R2']:.2f}  [{time.time() - t0:.0f}s]")

cvt = pd.concat(cv_rows, ignore_index=True)
for k in ['n_estimators', 'max_depth', 'min_samples_leaf', 'max_features']:
    cvt[k] = cvt['params'].apply(lambda p: p[k])
cvt['CV_RMSE'] = -cvt['mean_test_score']
cvt['CV_RMSE_sd'] = cvt['std_test_score']
cvt = cvt[['scenario', 'n_estimators', 'max_depth', 'min_samples_leaf', 'max_features',
           'CV_RMSE', 'CV_RMSE_sd', 'rank_test_score']]
cvt.to_csv(os.path.join(D_KFOLD, 'rf_cv_tuning.csv'), index=False)
res = pd.DataFrame(res_rows)

# ── 2. Scenario selection: lowest internal-test RMSE ─────────────────────────
res['n_assumptions'] = (res.weak_label != 'none').astype(int) + (res.palsar == 'yes').astype(int)
best_rmse = res.test_RMSE.min()
tied = res[res.test_RMSE <= best_rmse + TIE_TOL].sort_values(['n_assumptions', 'test_RMSE'])
SEL = tied.iloc[0].scenario
res['selected'] = res.scenario == SEL
res.to_csv(os.path.join(D_TEST, 'rf_scenarios_internal_test.csv'), index=False)
print(f"\nBest internal-test RMSE {best_rmse:.1f}; tied within {TIE_TOL}: {list(tied.scenario)}"
      f" -> SELECTED {SEL}")

est80, feats, wl, use_p, use_f, best_params, p_int_sel = fitted[SEL]

# ── 2b. Number of trees: convergence check (selected scenario, train only) ───
d_sel = scenario_data(train_all, wl)
nt_rows = []
for n in NTREES_CHECK:
    pr = {**best_params, 'n_estimators': n}
    r = []
    for tr_idx, va_idx in spatial_folds(d_sel):
        m = RandomForestRegressor(random_state=SEED, n_jobs=-1, **pr).fit(
            d_sel.loc[tr_idx, feats], d_sel.loc[tr_idx, 'AGB'])
        r.append(np.sqrt(mean_squared_error(d_sel.loc[va_idx, 'AGB'],
                                            m.predict(d_sel.loc[va_idx, feats]))))
    nt_rows.append(dict(scenario=SEL, n_estimators=n, CV_RMSE=np.mean(r), CV_RMSE_sd=np.std(r, ddof=1)))
    print(f"  n_estimators={n:5d}  CV RMSE={np.mean(r):.2f}")
ntc = pd.DataFrame(nt_rows)
ntc.to_csv(os.path.join(D_KFOLD, 'ntrees_convergence.csv'), index=False)

# ── 2c. Supplementary tables ─────────────────────────────────────────────────
grid_tbl = pd.DataFrame([
    dict(Hyperparameter='n_estimators', Values_tested='500 (fixed; convergence checked, Table S3)',
         Description='Number of trees'),
    dict(Hyperparameter='max_depth', Values_tested='None, 6, 9, 12',
         Description='Maximum tree depth (None = unlimited)'),
    dict(Hyperparameter='min_samples_leaf', Values_tested='1, 3, 5, 10',
         Description='Minimum number of samples in a terminal node'),
    dict(Hyperparameter='max_features', Values_tested='sqrt, 0.33, 0.5, 1.0',
         Description='Fraction of predictors considered at each split'),
])
grid_tbl.loc[len(grid_tbl)] = dict(
    Hyperparameter='Search', Values_tested=f'Full grid, {len(cvt) // len(SCENARIOS)} combinations per scenario',
    Description=f'{N_FOLDS}-fold spatial block CV on the training set; selection = lowest mean RMSE')

bp = res.best_params.apply(json.loads)
hp_tbl = pd.DataFrame({
    'Scenario': res.scenario,
    'Weak-label AGB (Mg ha-1)': res.weak_label.replace('none', '—'),
    'PALSAR': res.palsar.map({'yes': 'Yes', 'no': 'No'}),
    'FCD': res.fcd.map({'yes': 'Yes', 'no': 'No'}),
    'n_train': res.n_train,
    'n_estimators': bp.apply(lambda p: p['n_estimators']),
    'max_depth': bp.apply(lambda p: 'None' if p['max_depth'] is None else p['max_depth']),
    'min_samples_leaf': bp.apply(lambda p: p['min_samples_leaf']),
    'max_features': bp.apply(lambda p: p['max_features']),
    'CV RMSE (Mg ha-1)': res.apply(lambda r: f"{r.CV_RMSE:.1f} ± {r.CV_RMSE_sd:.1f}", axis=1),
    'Internal test RMSE (Mg ha-1)': res.test_RMSE.round(1),
    'Internal test R2': res.test_R2.round(2),
    'Internal test bias (Mg ha-1)': res.test_bias.round(1),
    'Selected': res.selected.map({True: 'Yes', False: ''}),
})
nt_tbl = ntc.assign(**{'CV RMSE (Mg ha-1)': ntc.apply(
    lambda r: f"{r.CV_RMSE:.2f} ± {r.CV_RMSE_sd:.2f}", axis=1)})[['scenario', 'n_estimators', 'CV RMSE (Mg ha-1)']]

grid_tbl.to_csv(os.path.join(D_SI, 'Table_S_hyperparameter_grid.csv'), index=False, encoding='utf-8-sig')
hp_tbl.to_csv(os.path.join(D_SI, 'Table_S_hyperparameters_by_scenario.csv'), index=False, encoding='utf-8-sig')
nt_tbl.to_csv(os.path.join(D_SI, 'Table_S_ntrees_convergence.csv'), index=False, encoding='utf-8-sig')
try:
    with pd.ExcelWriter(os.path.join(D_SI, 'Supplementary_tables_RF.xlsx')) as xw:
        grid_tbl.to_excel(xw, sheet_name='S1_grid', index=False)
        hp_tbl.to_excel(xw, sheet_name='S2_by_scenario', index=False)
        nt_tbl.to_excel(xw, sheet_name='S3_ntrees', index=False)
except Exception as e:
    print(f"xlsx not written ({e}); CSV tables are available.")

# ── 3. Permutation importance (model fit on train, scored on internal test) ──
rng = np.random.default_rng(SEED)
X_te, y_te = test_int[feats].copy(), test_int.AGB.values
base_rmse = np.sqrt(mean_squared_error(y_te, est80.predict(X_te)))

def perm_importance(cols):
    inc = []
    for _ in range(N_PERM):
        Xp = X_te.copy()
        perm = rng.permutation(len(Xp))
        Xp[cols] = X_te[cols].values[perm]          # same permutation for all cols of a group
        inc.append(np.sqrt(mean_squared_error(y_te, est80.predict(Xp))) - base_rmse)
    return np.mean(inc), np.std(inc)

imp = [dict(level='variable', name=LABELS[c], mean=m, sd=s)
       for c in feats for m, s in [perm_importance([c])]]
imp += [dict(level='group', name=g, mean=m, sd=s)
        for g, cols in GROUPS.items() if set(cols) <= set(feats)
        for m, s in [perm_importance(cols)]]
imp = pd.DataFrame(imp)
imp.to_csv(os.path.join(D_MODEL, 'rf_permutation_importance.csv'), index=False)

# ── 4. Final model on the whole pool ─────────────────────────────────────────
full = scenario_data(pool, wl)                       # train + test_interno + all WL
final = RandomForestRegressor(random_state=SEED, n_jobs=-1, **best_params).fit(full[feats], full.AGB)
joblib.dump(final, os.path.join(D_MODEL, 'rf_selected_model.joblib'))
with open(os.path.join(D_MODEL, 'rf_selected_model.json'), 'w') as f:
    json.dump(dict(scenario=SEL, weak_label=wl, palsar=use_p, fcd=use_f, features=feats,
                   pca_bands=PCA_INFO['bands'], pca_components=PCS,
                   pca_file=os.path.join(INTERIM, 'pca_landsat.joblib'),
                   hyperparameters=best_params, n_train=len(full),
                   selection='lowest internal-test RMSE',
                   internal_test_RMSE=float(res.loc[res.scenario == SEL, 'test_RMSE'].iloc[0])),
              f, indent=2)

# ── 5. Temporal validation (used once): paired visits v0 / v1 ────────────────
temporal['Year(s)_Avitabile'] = temporal['Year(s)_Avitabile'].astype(str)
VISITS = sorted(temporal['Year(s)_Avitabile'].unique())        # e.g. ['2017', '2021']
v0, v1 = VISITS[0], VISITS[-1]
temporal['AGB_pred'] = final.predict(temporal[feats])
pred_v0 = temporal[temporal['Year(s)_Avitabile'] == v0].set_index('site_id')['AGB_pred']
temporal[f'pred_{v0}'] = temporal.site_id.map(pred_v0)
temporal['delta_AGB_pred'] = np.where(temporal['Year(s)_Avitabile'] == v1,
                                      temporal.AGB_pred - temporal[f'pred_{v0}'], np.nan)
temporal.to_csv(os.path.join(D_TEMP, 'rf_temporal_validation.csv'), index=False)

tm = []
tm.append(dict(test='stock (all visits)', **metrics(temporal.AGB, temporal.AGB_pred)))
for v in VISITS:
    g = temporal[temporal['Year(s)_Avitabile'] == v]
    tm.append(dict(test=f'stock {v} (map {int(g.map_year.iloc[0])})', **metrics(g.AGB, g.AGB_pred)))
pairs = temporal.dropna(subset=['delta_AGB_obs', 'delta_AGB_pred'])
md = metrics(pairs.delta_AGB_obs, pairs.delta_AGB_pred)
rmse_persist = np.sqrt(np.mean(pairs.delta_AGB_obs ** 2))
md.update(rRMSE_pct=np.nan, RMSE_persistence=rmse_persist,
          skill_vs_persistence=1 - md['RMSE'] / rmse_persist,
          direction_agreement_pct=100 * np.mean(np.sign(pairs.delta_AGB_obs) ==
                                                np.sign(pairs.delta_AGB_pred)))
tm.append(dict(test=f'change {v1}-{v0}', **md))
tm = pd.DataFrame(tm)
tm.to_csv(os.path.join(D_TEMP, 'rf_temporal_metrics.csv'), index=False)
print("\nTemporal validation:\n" + tm.round(2).to_string(index=False))

# ── 6. Figures ───────────────────────────────────────────────────────────────
def style(ax, grid=True):
    ax.tick_params(direction='in', length=4)
    if grid:
        ax.grid(True, color='#e0e0e0', linewidth=0.5, zorder=0)
    for s in ax.spines.values():
        s.set_linewidth(0.8); s.set_color('black')

def stats_box(ax, m, extra=''):
    ax.text(0.04, 0.97, f"R$^2$ = {m['R2']:.2f}\nRMSE = {m['RMSE']:.1f}\n"
                        f"Bias = {m['bias']:.1f}\nMean observed = {m['mean_obs']:.1f}\n"
                        f"Mean predicted = {m['mean_pred']:.1f}\nn = {int(m['n'])}{extra}",
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=0.8))

def one_to_one(ax, a, b, pad=0.05):
    lo, hi = min(np.min(a), np.min(b)), max(np.max(a), np.max(b))
    r = hi - lo; lim = [lo - pad * r, hi + pad * r]
    ax.plot(lim, lim, 'k--', linewidth=1.0, zorder=2)
    ax.set_xlim(lim); ax.set_ylim(lim)

# Scenarios: CV RMSE and internal-test RMSE
fig, axes = plt.subplots(1, 2, figsize=(16, 4.6), sharey=False)
xpos = np.arange(len(res))
xt = [f"{s}\n{w}" for s, w in zip(res.scenario, res.weak_label)]
mk = ['o' if f == 'yes' else 's' for f in res.fcd]
for ax, col, sd, ttl in [(axes[0], 'CV_RMSE', 'CV_RMSE_sd', 'A. Spatial K-fold (train)'),
                         (axes[1], 'test_RMSE', None, 'B. Internal test')]:
    face = ['#606060' if p == 'yes' else 'white' for p in res.palsar]
    if sd:
        ax.errorbar(xpos, res[col], yerr=res[sd], fmt='none', ecolor='#404040',
                    elinewidth=0.8, capsize=3, zorder=2)
    for m_ in ('o', 's'):
        sel_ = np.array(mk) == m_
        ax.scatter(xpos[sel_], res[col][sel_], c=np.array(face)[sel_], marker=m_, edgecolors='black',
                   s=55, zorder=3)
    selm = res.selected.values
    ax.scatter(xpos[selm], res[col][selm], facecolors='none', edgecolors='black',
               s=200, linewidths=1.2, zorder=4)
    for xv in (3.5, 11.5):
        ax.axvline(xv, color='#808080', linestyle=':', linewidth=0.8)
    ax.axvline(7.5, color='black', linestyle='--', linewidth=0.8)
    trf = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for xv, lab_ in ((3.5, 'With FCD'), (11.5, 'Without FCD')):
        ax.text(xv, 0.97, lab_, transform=trf, ha='center', va='top', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='none', pad=1))
    ax.set_xticks(xpos); ax.set_xticklabels(xt, fontsize=9)
    ax.set_xlabel('Scenario / weak-label AGB (Mg ha$^{-1}$)')
    ax.set_ylabel('RMSE (Mg ha$^{-1}$)')
    ax.set_title(ttl, loc='left', fontweight='bold')
    style(ax)
h = [plt.Line2D([], [], marker='o', ls='', mfc='#606060', mec='black', label='With PALSAR'),
     plt.Line2D([], [], marker='o', ls='', mfc='white', mec='black', label='Without PALSAR'),
     plt.Line2D([], [], marker='o', ls='', mfc='none', mec='black', label='With FCD (circle)'),
     plt.Line2D([], [], marker='s', ls='', mfc='none', mec='black', label='Without FCD (square)'),
     plt.Line2D([], [], marker='o', ls='', mfc='none', mec='black', ms=13, label='Selected')]
fig.legend(handles=h, loc='lower center', ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_rf_scenarios.tif'), dpi=300, bbox_inches='tight',
            facecolor='white', format='tiff')
plt.close(fig)

# Internal test: observed vs predicted (selected scenario)
fig, ax = plt.subplots(figsize=(5, 4.8))
ax.scatter(test_int.AGB[~in_area_test], p_int_sel[~in_area_test], marker='o', color='#b0b0b0',
           edgecolors='#505050', s=30, linewidths=0.5, zorder=3, label='Outside study area')
ax.scatter(test_int.AGB[in_area_test], p_int_sel[in_area_test], marker='^', color='#404040',
           edgecolors='black', s=34, linewidths=0.5, zorder=4, label='Study area')
one_to_one(ax, test_int.AGB, p_int_sel)
stats_box(ax, metrics(test_int.AGB, p_int_sel))
ax.set_xlabel('Observed AGB (Mg ha$^{-1}$)'); ax.set_ylabel('Predicted AGB (Mg ha$^{-1}$)')
ax.set_title(f'A. Internal test ({SEL})', loc='left', fontweight='bold')
ax.legend(frameon=True, edgecolor='black', fontsize=9, loc='lower right')
style(ax)
fig.savefig(os.path.join(FIG_DIR, 'fig_rf_internal_test.tif'), dpi=300, bbox_inches='tight',
            facecolor='white', format='tiff')
plt.close(fig)

# Permutation importance: variables and groups
fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), gridspec_kw={'width_ratios': [1.6, 1]})
for ax, lvl, ttl in [(axes[0], 'variable', 'A. Variables'), (axes[1], 'group', 'B. Data sources')]:
    d = imp[imp.level == lvl].sort_values('mean')
    ax.barh(d.name, d['mean'], xerr=d['sd'], color='#808080', edgecolor='#404040',
            linewidth=0.6, error_kw=dict(ecolor='black', elinewidth=0.8, capsize=2), zorder=3)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Increase in RMSE when permuted (Mg ha$^{-1}$)')
    ax.set_title(ttl, loc='left', fontweight='bold')
    style(ax)
    ax.grid(axis='y', visible=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_rf_importance.tif'), dpi=300, bbox_inches='tight',
            facecolor='white', format='tiff')
plt.close(fig)

# Temporal validation: stocks and change
fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
ax = axes[0]
for yr, mk in zip(VISITS, ['o', 's', '^', 'D']):
    m = temporal['Year(s)_Avitabile'] == yr
    ax.scatter(temporal.AGB[m], temporal.AGB_pred[m], marker=mk, color='#808080',
               edgecolors='#404040', s=45, linewidths=0.6, zorder=3, label=f'Visit {yr}')
for _, g in temporal.groupby('site_id'):          # link the two visits of each site
    g = g.sort_values('Year(s)_Avitabile')
    ax.plot(g.AGB, g.AGB_pred, color='#b0b0b0', linewidth=0.6, zorder=2)
one_to_one(ax, temporal.AGB, temporal.AGB_pred)
stats_box(ax, tm.iloc[0])
ax.set_xlabel('Observed AGB (Mg ha$^{-1}$)'); ax.set_ylabel('Predicted AGB (Mg ha$^{-1}$)')
ax.set_title('A. Stocks at re-measured plots', loc='left', fontweight='bold')
ax.legend(frameon=True, edgecolor='black', fontsize=9, loc='lower right')
style(ax)
ax = axes[1]
ax.scatter(pairs.delta_AGB_obs, pairs.delta_AGB_pred, marker='o', color='#808080',
           edgecolors='#404040', s=45, linewidths=0.6, zorder=3, label=f'{v1} − {v0}')
one_to_one(ax, pairs.delta_AGB_obs, pairs.delta_AGB_pred)
ax.axhline(0, color='#808080', linewidth=0.6); ax.axvline(0, color='#808080', linewidth=0.6)
mc = tm.iloc[-1]
ax.text(0.04, 0.97, f"RMSE = {mc['RMSE']:.1f}\nBias = {mc['bias']:.1f}\n"
                    f"Mean observed = {mc['mean_obs']:.1f}\nMean predicted = {mc['mean_pred']:.1f}\n"
                    f"Skill vs persistence = {mc['skill_vs_persistence']:.2f}\n"
                    f"Direction agreement = {mc['direction_agreement_pct']:.0f}%\nn = {int(mc['n'])}",
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=0.8))
ax.set_xlabel('Observed ΔAGB (Mg ha$^{-1}$)'); ax.set_ylabel('Predicted ΔAGB (Mg ha$^{-1}$)')
ax.set_title(f'B. Change {v0}–{v1}', loc='left', fontweight='bold')
ax.legend(frameon=True, edgecolor='black', fontsize=9, loc='lower right')
style(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_rf_temporal_validation.tif'), dpi=300,
            bbox_inches='tight', facecolor='white', format='tiff')
plt.close(fig)

# SI: number of trees convergence
fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.errorbar(ntc.n_estimators, ntc.CV_RMSE, yerr=ntc.CV_RMSE_sd / np.sqrt(N_FOLDS), fmt='-o',
            color='#606060', mfc='#808080', mec='#404040', ecolor='#404040', elinewidth=0.8,
            capsize=3, zorder=3)
ax.axvline(500, color='black', linestyle='--', linewidth=0.8)
ax.set_xscale('log')
ax.set_xticks(ntc.n_estimators); ax.set_xticklabels(ntc.n_estimators)
ax.set_xlabel('Number of trees'); ax.set_ylabel('Spatial CV RMSE (Mg ha$^{-1}$)')
ax.set_title(f'A. Convergence of RF error ({SEL})', loc='left', fontweight='bold')
style(ax)
fig.savefig(os.path.join(FIG_DIR, 'fig_si_ntrees_convergence.tif'), dpi=300,
            bbox_inches='tight', facecolor='white', format='tiff')
plt.close(fig)

print("\nSummary of scenarios:")
print(res[['scenario', 'weak_label', 'palsar', 'fcd', 'CV_RMSE', 'test_RMSE', 'test_R2', 'test_bias',
           'test_area_RMSE', 'selected']].round(2).to_string(index=False))
print(f"\nResults in: {RES_DIR}")
for dd in [D_KFOLD, D_TEST, D_MODEL, D_TEMP, D_SI]:
    print(f"  {os.path.basename(dd)}: {sorted(os.listdir(dd))}")
print("\nDone.")
