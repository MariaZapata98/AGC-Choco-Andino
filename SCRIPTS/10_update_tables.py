r"""
10_update_tables.py
Updates the manuscript and Supplementary Excel tables with the outputs of 07b and 09
(run AFTER 09_rbca_recortes.py). Only the cells with values are rewritten; titles,
format and the remaining sheets (model, validation) are kept.

All RBCA values use the SAME pixels (inside the RBCA and valid at the five dates), so
Table 4 (2000 and 2022), Table 5 (sum of bands) and Table 6 (total) give the same AGC.

Inputs
  RESULTADOS\recortes\rbca_agc_by_year.csv, rbca_agc_by_elevation.csv, rbca_ledger_2000_2022.csv,
                      rbca_fcd_classes_by_year.csv, rbca_fcd_transitions_ha.csv,
                      rbca_cohort_trends.csv, rbca_cohort_trajectory.csv, rbca_heatmap_fcd_elevation.csv (09)
  RESULTADOS\07_incertidumbre\totals_by_year.csv, change.csv, ledger_uncertainty.csv,
                      ledger_threshold_sensitivity.csv (07b)
  RESULTADOS\tablas\ECOLIND-60399_tablas_manuscrito.xlsx, ECOLIND-60399_tablas_supplementary.xlsx
Outputs
  RESULTADOS\tablas\ECOLIND-60399_tablas_manuscrito_v2.xlsx, ECOLIND-60399_tablas_supplementary_v2.xlsx
Usage   python 10_update_tables.py
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT    = sys.argv[1] if len(sys.argv) > 1 else r"D:\PAPER-ECOLIN"
RES     = os.path.join(ROOT, "RESULTADOS")
REC     = os.path.join(RES, "recortes")
UNC     = os.path.join(RES, "07_incertidumbre")
TAB     = os.path.join(RES, "tablas")
F_MAN   = os.path.join(TAB, "ECOLIND-60399_tablas_manuscrito.xlsx")
F_SUP   = os.path.join(TAB, "ECOLIND-60399_tablas_supplementary.xlsx")
NY      = 22
TODAY   = datetime.date.today().strftime("%d-%b-%Y")

rd = lambda d, f: pd.read_csv(os.path.join(d, f))
by_year = rd(REC, "rbca_agc_by_year.csv").set_index("year")
elev    = rd(REC, "rbca_agc_by_elevation.csv")
led     = rd(REC, "rbca_ledger_2000_2022.csv").set_index("code")
fcls    = rd(REC, "rbca_fcd_classes_by_year.csv").set_index("year")
trans   = pd.read_csv(os.path.join(REC, "rbca_fcd_transitions_ha.csv"), index_col=0)
trends  = rd(REC, "rbca_cohort_trends.csv").set_index("code")
traj    = rd(REC, "rbca_cohort_trajectory.csv")
heat    = rd(REC, "rbca_heatmap_fcd_elevation.csv")
tot     = rd(UNC, "totals_by_year.csv").query("scenario == 'S2' and mask == 'forest_own'").set_index("year")
chg     = rd(UNC, "change.csv").query("scenario == 'S2' and mask == 'forest_own'").set_index("period")
lunc    = rd(UNC, "ledger_uncertainty.csv").set_index("code")
YEARS   = list(by_year.index)
area_rbca, area_u = by_year.area_rbca_ha.iloc[0], by_year.area_analysed_ha.iloc[0]

# ── checks: everything must add up ───────────────────────────────────────────
e22 = elev[elev.year == 2022].AGC_TgC.sum()
print(f"AGC 2000: by year {by_year.AGC_forest_TgC[2000]:.3f} | ledger {led.AGC_2000_TgC[0]:.3f} | "
      f"07b {tot.AGC_TgC[2000]:.3f}")
print(f"AGC 2022: by year {by_year.AGC_forest_TgC[2022]:.3f} | ledger {led.AGC_2022_TgC[0]:.3f} | "
      f"elevation {e22:.3f} | 07b {tot.AGC_TgC[2022]:.3f}")
assert abs(by_year.AGC_forest_TgC[2022] - led.AGC_2022_TgC[0]) < 0.01, "09 was not re-run with the common universe"


def put(ws, row, values, col0=2):
    for k, v in enumerate(values):
        if v is not None:
            ws.cell(row=row, column=col0 + k, value=v)


def note(ws, startswith, text):
    for row in ws.iter_rows(min_col=1, max_col=1):
        c = row[0]
        if isinstance(c.value, str) and c.value.startswith(startswith):
            c.value = text
            return
    print(f"  (note not found in {ws.title}: '{startswith}...')")


ci = lambda lo, hi, f="{:.2f}", sep="–": f"{f.format(lo)}{sep}{f.format(hi)}"

# ══ Manuscript tables ════════════════════════════════════════════════════════
wb = load_workbook(F_MAN)
wb["Índice"]["A1"] = f"ECOLIND-60399 — Tablas del manuscrito (actualizado {TODAY}, escenario seleccionado S2)"

# Table 4
ws = wb["Table 4 AGC by year RBCA"]
ws["A1"] = (f"Table 4. Forest area, total above-ground carbon (AGC), mean AGC density and mean forest canopy "
            f"density (FCD) in the Chocó Andino Biosphere Reserve (RBCA) for the five map dates, and net change "
            f"2000–2022. Values over the {area_u:,.0f} ha with data at all five dates. Forest = FCD ≥ 30 %.")
ws["C3"] = "Forest (% of analysed area)"
for r, y in enumerate(YEARS, start=4):
    b = by_year.loc[y]
    put(ws, r, [round(b.area_forest_ha), round(b.pct_forest, 2), round(b.AGC_forest_TgC, 2),
                ci(tot.CI95_low[y], tot.CI95_high[y]),
                f"{b.AGC_mean_forest_MgC_ha:.1f} ± {b.AGC_SD_spatial_MgC_ha:.1f}", round(b.FCD_mean, 2)])
c = chg.loc["2000-2022"]
put(ws, 9, [round(led.change_TgC[0], 2), ci(c.CI95_low, c.CI95_high),
            f"ΔAGC = {led.rate_mean_MgC_ha_yr[0]:.2f} Mg C ha⁻¹ yr⁻¹"], col0=4)
note(ws, "El total de AGC", f"Todos los valores usan los mismos píxeles (dentro del RBCA y con datos en las 5 fechas: "
     f"{area_u:,.0f} de {area_rbca:,.0f} ha), igual que el ledger (Tabla 6) y la Tabla 5, por eso los totales coinciden.")
note(ws, "RESULTADOS\\recortes\\rbca_agc_by_year.csv", "RESULTADOS\\recortes\\rbca_agc_by_year.csv (09): area_forest_ha, "
     "pct_forest, AGC_forest_TgC, AGC_mean_forest_MgC_ha, AGC_SD_spatial_MgC_ha (SD espacial entre píxeles de bosque), FCD_mean.")
note(ws, "Total AGC y 95 % CI", "95 % CI: RESULTADOS\\07_incertidumbre\\totals_by_year.csv (07b, mismos píxeles), "
     "filas mask = forest_own: CI95_low, CI95_high.")
note(ws, "Mean FCD", "Mean FCD = promedio sobre el área analizada (bosque y no bosque).")

# Table 5
ws = wb["Table 5 Elevation RBCA"]
e0, e1 = elev[elev.year == 2000].set_index("elevation_band"), elev[elev.year == 2022].set_index("elevation_band")
for r, band in enumerate(e1.index, start=4):
    a, b = e0.loc[band], e1.loc[band]
    put(ws, r, [round(b.area_ha), round(a.pct_forest, 2), round(b.pct_forest, 2), round(b.AGC_TgC, 2),
                round(100 * b.AGC_TgC / e22, 2), round(a.AGC_mean_forest_MgC_ha, 2),
                round(b.AGC_mean_forest_MgC_ha, 2), round((b.AGC_TgC - a.AGC_TgC) * 1e6 / b.area_ha / NY, 2)])

# Table 6
ws = wb["Table 6 Ledger RBCA"]
for r, k in enumerate([1, 2, 3, 4, 0], start=4):
    L = led.loc[k]
    if k in (2, 3, 4):
        u = lunc.loc[k]
        aci = ci(u.area_CI95_low, u.area_CI95_high, "{:,.0f}")
        cci = ci(u.change_CI95_low, u.change_CI95_high, sep=" to ")
    elif k == 0:
        aci, cci = None, ci(c.CI95_low, c.CI95_high, sep=" to ")
    else:
        aci, cci = "—", "—"
    put(ws, r, [round(L.area_ha), aci, round(L.pct_area, 2), round(L.AGC_2000_TgC, 2), round(L.AGC_2022_TgC, 2),
                round(L.change_TgC, 2), cci, round(L.rate_mean_MgC_ha_yr, 2), round(L.share_of_net_change_pct, 2)])
note(ws, "Píxeles con NoData", f"Píxeles con NoData en alguna fecha excluidos (área analizada {area_u:,.0f} ha de "
     f"{area_rbca:,.0f} ha); mismos píxeles que las Tablas 4 y 5.")
wb.save(F_MAN.replace(".xlsx", "_v2.xlsx"))
print("saved:", F_MAN.replace(".xlsx", "_v2.xlsx"))

# ══ Supplementary tables ═════════════════════════════════════════════════════
wb = load_workbook(F_SUP)
wb["Índice"]["A1"] = f"ECOLIND-60399 — Supplementary tables (actualizado {TODAY}, escenario seleccionado S2)"

ws = wb["Table S12 FCD classes RBCA"]                     # FCD classes by year
for r, y in enumerate(fcls.index, start=4):
    put(ws, r, [round(v, 2) for v in fcls.loc[y].values])

ws = wb["Table S13 FCD transitions"]                       # transitions 2000 -> 2022
for r, (_, row) in enumerate(trans.iterrows(), start=4):
    put(ws, r, [round(v, 2) for v in row.values] + [round(row.sum(), 2)])

ws = wb["Table S14 Canopy cohorts"]                        # cohorts of 2000
for r, k in enumerate([1, 2, 3, 4], start=4):
    t = trends.loc[k]
    a = traj[(traj.code == k) & (traj.year == 2000)].iloc[0]
    b = traj[(traj.code == k) & (traj.year == 2022)].iloc[0]
    put(ws, r, [round(t.area_ha), round(a.FCD_mean, 2), round(b.FCD_mean, 2), round(a.AGC_mean_MgC_ha, 2),
                round(b.AGC_mean_MgC_ha, 2), round(b.pct_forest, 2), round(t.FCD_slope, 2), round(t.pct_FCD_up, 2),
                round(t.AGC_slope, 2), round(t.pct_AGC_up, 2)])

ws = wb["Table S15 FCD x elevation"]                       # dAGC by FCD class x elevation
for r, (_, h) in enumerate(heat.iterrows(), start=4):
    put(ws, r, [h.fcd_2000, h.elevation_band, round(h.area_ha), round(h["mean"], 2), round(h.sd, 2)], col0=1)

wb.save(F_SUP.replace(".xlsx", "_v2.xlsx"))
print("saved:", F_SUP.replace(".xlsx", "_v2.xlsx"))
print("\nSheets updated: Tables 4, 5, 6 and S12–S15. The other sheets (model, validation, uncertainty) "
      "do not depend on 09 and are unchanged.")
