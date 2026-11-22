"""
03_extract_predictors_gee.py
============================
Extrae predictores de satélite para cada punto del CSV de entrenamiento
usando Google Earth Engine (al vuelo — sin descargar imágenes).
Parcelas USFQ de 60 x 60 m (code_des = 'USFQ'): media ponderada por área de los píxeles de 30 m
que intersectan la parcela. Resto (BP 10 x 10 m, CAM2, SAM, RBA, GEDI): píxel que contiene el punto.

Columnas nuevas añadidas al CSV:
  Landsat  : B_mean, G_mean, R_mean, NIR_mean, SWIR1_mean, SWIR2_mean
             NDVI_mean, GNDVI_mean, EVI_mean
  PALSAR-2 : HH_mean, HV_mean  (DN crudo; media 2015–año para años ≤ 2020, proxy 2020 para años > 2020)
  SRTM     : elevation_srtm, slope_srtm

Uso:
  cd D:/PAPER-ECOLIN
  python scripts/03_extract_predictors_gee.py

Requisitos:
  pip install earthengine-api pandas tqdm
  earthengine authenticate   # solo la primera vez
"""

import ee
import pandas as pd
import numpy as np
import math
import json
import time
from pathlib import Path
from tqdm import tqdm

# ── CONFIGURACIÓN ────────────────────────────────────────────────────────────
PROJECT_ID   = 'ee-modelsgee4'
ASSET_CSV    = f'projects/{PROJECT_ID}/assets/training_oct2025'
CSV_INPUT    = Path(r'D:\PAPER-ECOLIN\DATA\raw\training_oct2025.csv')
CSV_OUTPUT   = Path(r'D:\PAPER-ECOLIN\DATA\interim\training_oct2025_predictors.csv')

# Tamaño de lote para llamadas a GEE (no exceder ~100 por lote)
BATCH_SIZE   = 50
SCALE_M      = 30     # resolución común de salida (Landsat, SRTM, PALSAR reproyectado)
PALSAR_ALOS2_MIN = 2015   # primer año ALOS-2 disponible en GEE
PALSAR_ALOS2_MAX = 2020   # último año ALOS-2 disponible en GEE
# Lógica PALSAR por año del punto:
#   año < 2015  → solo 2015 (proxy, primer año ALOS-2)
#   2015–2020   → media de PALSAR desde 2015 hasta ese año
#   año > 2020  → solo 2020 (último disponible)

# CRS y grilla de referencia — todo se alinea a esta grilla Landsat
# EPSG:32717 = UTM zona 17S (cubre el Chocó Andino y gran parte del área de estudio)
# Para puntos fuera de UTM 17S GEE reproyecta correctamente igual.
CRS_REF      = 'EPSG:32717'

# Plot footprint (01-nov): the 23 USFQ plots are 60 x 60 m -> predictors = area-weighted mean of the
# 30-m pixels intersecting a 60 x 60 m square centred on the plot coordinate (assumed plot centre).
# BP plots (10 x 10 m) and all other sources (CAM2, SAM, RBA, GEDI) -> pixel containing the point.
# Column 'code_des' of the raw CSV distinguishes BP and USFQ.
PLOT_HALF_SIDE_M = {'USFQ': 30}      # half side (m) by code_des; codes not listed = point

# ── INICIALIZAR GEE ──────────────────────────────────────────────────────────
print('Inicializando Earth Engine...')
ee.Initialize(project=PROJECT_ID)
print('✓ GEE listo\n')


# ════════════════════════════════════════════════════════════════════════════
# FUNCIONES LANDSAT
# ════════════════════════════════════════════════════════════════════════════

def scale_l57(img):
    """Aplicar factor de escala SR a Landsat 5/7."""
    optical = img.select('SR_B.').multiply(0.0000275).add(-0.2)
    return img.addBands(optical, overwrite=True)

def scale_l89(img):
    """Aplicar factor de escala SR a Landsat 8/9."""
    optical = img.select('SR_B.').multiply(0.0000275).add(-0.2)
    return img.addBands(optical, overwrite=True)

def mask_range(img):
    """Enmascarar píxeles donde cualquier banda óptica RENOMBRADA queda fuera
    de [-0.2, 1.0]. Se aplica DESPUÉS del rename para evitar incluir bandas
    térmicas (SR_B10, SR_B11 en L8/L9) que tienen rangos completamente distintos.
    Elimina artefactos de saturación (DN=65535 → 1.6022) sin descartar
    negativos pequeños de calibración válidos (Blue ~ -0.05 en bosque denso)."""
    optical_bands = ['Blue','Green','Red','NIR','SWIR1','SWIR2']
    optical = img.select(optical_bands)
    valid   = optical.gte(-0.2).And(optical.lte(1.0)).reduce(ee.Reducer.min())
    # Bruma / nube delgada que la QA no detecta: índice HOT (Zhang et al. 2002),
    # HOT = Blue - 0.5*Red - 0.08 > 0. El suelo desnudo y lo urbano tienen Red >= Blue
    # → HOT negativo, se conservan (no es un filtro por NDVI).
    hot = img.select('Blue').subtract(img.select('Red').multiply(0.5)).subtract(0.08)
    valid = valid.And(hot.lte(0))
    return img.updateMask(valid)

def mask_qa(img):
    """Máscara QA completa de Landsat Collection 2 (v4, 28-oct).
    QA_PIXEL: bit 0 fill · bit 1 dilated cloud · bit 2 cirrus (L8/L9) · bit 3 cloud ·
              bit 4 cloud shadow · bits 8-9 confianza de nube MEDIA o ALTA (>= 2)
    Nube y sombra se amplían 2 píxeles (60 m) para quitar los bordes difusos (bruma).
    QA_RADSAT: cualquier banda saturada → fuera.
    Es una máscara de calidad del producto, NO un filtro por NDVI."""
    qa = img.select('QA_PIXEL')
    cloudy = (qa.bitwiseAnd(1 << 1).neq(0)                   # dilated cloud
              .Or(qa.bitwiseAnd(1 << 2).neq(0))              # cirrus
              .Or(qa.bitwiseAnd(1 << 3).neq(0))              # cloud
              .Or(qa.bitwiseAnd(1 << 4).neq(0))              # cloud shadow
              .Or(qa.rightShift(8).bitwiseAnd(3).gte(2)))    # confianza de nube media/alta
    cloudy = cloudy.focalMax(radius=2, kernelType='square', units='pixels')   # buffer 60 m
    fill = qa.bitwiseAnd(1 << 0).neq(0)
    radsat = img.select('QA_RADSAT').neq(0)
    return img.updateMask(cloudy.Or(fill).Or(radsat).Not())

def rename_l57(img):
    """Renombrar bandas L5/L7 al esquema común."""
    return img.select(
        ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7'],
        ['Blue',  'Green','Red',  'NIR',  'SWIR1','SWIR2']
    )

def rename_l89(img):
    """Renombrar bandas L8/L9 al esquema común."""
    return img.select(
        ['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'],
        ['Blue',  'Green','Red',  'NIR',  'SWIR1','SWIR2']
    )

def add_indices(img):
    """Añadir NDVI, GNDVI y EVI calculados sobre la imagen GEE.
    EVI se clampea a [-1, 1]: la fórmula puede explotar cuando Blue es alto
    y el denominador (NIR + 6R - 7.5B + 1) se acerca a cero.
    NDVI y GNDVI usan normalizedDifference que ya garantiza [-1, 1].
    """
    ndvi  = img.normalizedDifference(['NIR','Red']).rename('NDVI')
    gndvi = img.normalizedDifference(['NIR','Green']).rename('GNDVI')
    evi   = img.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {'NIR': img.select('NIR'),
         'RED': img.select('Red'),
         'BLUE': img.select('Blue')}
    ).rename('EVI').clamp(-1, 1)   # rango físico válido; evita explosión de denominador
    return img.addBands([ndvi, gndvi, evi])

def medoid(collection, bands):
    """
    Medoid de Flood (2013): devuelve el píxel real más cercano
    a la mediana espectral. Más robusto que la mediana en zonas
    con cobertura nubosa persistente como el Chocó.
    """
    median = collection.select(bands).median()
    def dist_to_median(img):
        diff = img.select(bands).subtract(median)
        # qualityMosaic se queda con el valor MÁS ALTO de la banda de calidad → se usa la
        # distancia NEGATIVA para elegir el píxel MÁS CERCANO a la mediana (medoid de Flood 2013).
        # (v2 usaba la distancia positiva → elegía el píxel más lejano: nubes, bruma, sombras.)
        neg_dist = diff.pow(2).reduce(ee.Reducer.sum()).multiply(-1).rename('neg_dist')
        return img.addBands(neg_dist)
    with_dist = collection.map(dist_to_median)
    return with_dist.qualityMosaic('neg_dist').select(bands + ['NDVI','GNDVI','EVI'])

def get_landsat_collection(start, end, year):
    """
    Construye colección Landsat filtrada entre start y end.
    Selector de sensor basado en el año central del período.
    L5+L7 ≤2012, L8 2013-2021, L9 2022+
    """
    if year <= 2012:
        col = (ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
               .filterDate(start, end)
               .map(mask_qa).map(scale_l57).map(rename_l57).map(mask_range).map(add_indices))
        col7 = (ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
                .filterDate(start, end)
                .map(mask_qa).map(scale_l57).map(rename_l57).map(mask_range).map(add_indices))
        return col.merge(col7)
    elif year <= 2021:
        return (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                .filterDate(start, end)
                .map(mask_qa).map(scale_l89).map(rename_l89).map(mask_range).map(add_indices))
    else:
        return (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
                .filterDate(start, end)
                .map(mask_qa).map(scale_l89).map(rename_l89).map(mask_range).map(add_indices))

def get_landsat_year(year, extra_semester=False):
    """
    Construye medoid para un año dado.
    Si extra_semester=True, añade julio–diciembre del año anterior
    para ampliar el pool de píxeles (usado cuando el período es año único).
    L5 2000-2012, L8 2013-2021, L9 2022+
    Devuelve imagen con bandas: Blue Green Red NIR SWIR1 SWIR2 NDVI GNDVI EVI
    """
    start = ee.Date.fromYMD(year, 1, 1)
    end   = ee.Date.fromYMD(year, 12, 31)
    col   = get_landsat_collection(start, end, year)

    if extra_semester:
        # Temporada seca del año anterior: julio–diciembre (menos nubes en Chocó)
        prev_year  = year - 1
        start_prev = ee.Date.fromYMD(prev_year, 7, 1)
        end_prev   = ee.Date.fromYMD(prev_year, 12, 31)
        col_prev   = get_landsat_collection(start_prev, end_prev, prev_year)
        col = col.merge(col_prev)

    bands = ['Blue','Green','Red','NIR','SWIR1','SWIR2']
    return medoid(col, bands)

def get_landsat_period(year_ini, year_end, with_fallback=False):
    """
    Media de medoids anuales entre year_ini y year_end (inclusive).

    with_fallback=True: para períodos de año único, añade jul–dic del año
    anterior al pool del medoid. Solo se usa si la primera extracción sin
    fallback devolvió nulos para ese punto.
    """
    year_ini, year_end = int(year_ini), int(year_end)
    single_year = (year_ini == year_end)

    years = list(range(year_ini, year_end + 1))
    n_total = len(years)

    images = []
    for i, y in enumerate(years):
        use_extra = with_fallback and single_year and (i == 0)
        images.append(get_landsat_year(y, extra_semester=use_extra))
    col = ee.ImageCollection(images)

    # Máscara válida por año: 1 si tiene dato en NIR, 0 si NoData
    valid_masks = [img.select('NIR').mask().toFloat() for img in images]
    n_valid     = ee.ImageCollection(valid_masks).sum().rename('n_years_valid')
    n_total_img = ee.Image.constant(n_total).rename('n_years_total').toFloat()

    # Promediar solo las bandas base — los índices se recalculan sobre las
    # bandas promediadas para garantizar consistencia (NDVI de las bandas
    # promediadas ≠ promedio de NDVIs anuales, especialmente en períodos multi-año)
    bands_base = ['Blue','Green','Red','NIR','SWIR1','SWIR2']
    img_mean   = col.select(bands_base).mean()
    img_mean   = add_indices(img_mean)   # recalcular NDVI, GNDVI, EVI sobre las medias

    return img_mean.addBands([n_valid, n_total_img])


# ════════════════════════════════════════════════════════════════════════════
# FUNCIONES PALSAR-2
# ════════════════════════════════════════════════════════════════════════════

def lee_filter_band(img, band_name, window_size=7):
    """Aplica filtro de Lee a una sola banda (strings Python, no ee.List)."""
    band = img.select([band_name])
    half = window_size // 2
    kernel = ee.Kernel.square(half)

    mean      = band.reduceNeighborhood(ee.Reducer.mean(),     kernel)
    var_local = band.reduceNeighborhood(ee.Reducer.variance(), kernel)

    # Varianza del ruido: estimada como mean²/ ENL (ENL≈4 para PALSAR-2)
    var_noise = mean.pow(2).divide(4.0)

    weight = (var_local.subtract(var_noise)
                       .divide(var_local.add(1e-10))
                       .max(0).min(1))

    filtered = mean.add(weight.multiply(band.subtract(mean)))
    return filtered.rename([band_name])

def lee_filter(img, window_size=7):
    """
    Filtro de Lee para reducción de speckle SAR.
    Aplica banda a banda usando strings Python para evitar
    problemas con ee.List.map() en la API Python de GEE.
    """
    # Obtener nombres de banda como lista Python (getInfo es necesario aquí
    # porque iteramos en Python, no en el servidor GEE)
    band_names = img.bandNames().getInfo()   # ['HH', 'HV']
    filtered = [lee_filter_band(img, b, window_size) for b in band_names]
    # Combinar todas las bandas en una sola imagen
    return ee.ImageCollection(filtered).toBands().rename(band_names)

def dn_to_db(img):
    """
    Convertir DN de JAXA PALSAR a backscatter en dB.
    Formula JAXA: dB = 10 * log10(DN^2 * 0.0000125)
    NOTA: No se usa — mantenemos DN crudo para consistencia con el CSV original.
    Se deja aquí por referencia.
    """
    db = img.pow(2).multiply(0.0000125).log10().multiply(10)
    return db

def get_palsar_single(year):
    """
    Mosaico PALSAR-2 para un año específico (debe existir en GEE: 2015-2020).
    Aplica filtro Lee 7×7 para reducir speckle.
    """
    start = ee.Date.fromYMD(year, 1, 1)
    end   = ee.Date.fromYMD(year, 12, 31)
    img = (ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR')
           .filterDate(start, end)
           .select(['HH', 'HV'])
           .mosaic())
    # Lee filter en resolución nativa (25 m) antes de reproyectar
    return lee_filter(img, window_size=7)

def get_palsar_period(year_ini, year_end):
    """
    Imagen PALSAR-2 para un período del CSV, con lógica por año:

      año < 2015  → solo 2015 (proxy, primer año ALOS-2 disponible en GEE)
      2015–2020   → media de los años ALOS-2 desde 2015 hasta min(año, 2020)
      año > 2020  → solo 2020 (último año disponible en GEE)

    Para períodos multi-año (e.g. '2015-2018') se usa year_end como referencia.
    """
    year_end = int(year_end)

    # Calcular rango PALSAR efectivo
    if year_end < PALSAR_ALOS2_MIN:
        # Antes de 2015: usar solo 2015 como proxy
        return get_palsar_single(PALSAR_ALOS2_MIN)

    palsar_end   = min(year_end, PALSAR_ALOS2_MAX)    # cap en 2020
    palsar_start = PALSAR_ALOS2_MIN                    # siempre empieza en 2015

    palsar_years = list(range(palsar_start, palsar_end + 1))

    if len(palsar_years) == 1:
        return get_palsar_single(palsar_years[0])

    images = [get_palsar_single(y) for y in palsar_years]
    return ee.ImageCollection(images).mean()


# ════════════════════════════════════════════════════════════════════════════
# FUNCIÓN SRTM
# ════════════════════════════════════════════════════════════════════════════

def get_srtm():
    """
    Elevación y pendiente desde SRTM (estático).
    Reproyectado a la misma grilla de referencia (CRS_REF, 30 m)
    para alinearse con Landsat y PALSAR.
    slope se calcula ANTES de reproyectar para usar la resolución
    nativa del SRTM (30 m ya coincide, pero forzamos la grilla exacta).
    """
    dem   = ee.Image('USGS/SRTMGL1_003').select('elevation')
    slope = ee.Terrain.slope(dem)
    return dem.addBands(slope)  # sin reproject — sampleRegions maneja la alineación


# ════════════════════════════════════════════════════════════════════════════
# EXTRACCIÓN POR LOTES
# ════════════════════════════════════════════════════════════════════════════

WL_AGB = 15.0   # weak-labels (AGB == 15): their period is the FCD time step (Year_ini_even), not the
                # measurement period of the source record (Year(s)_Avitabile) -> Landsat / PALSAR of that year


def record_periods(d):
    """(year_ini, year_end) per row: Year(s)_Avitabile for real records, Year_*_even for weak-labels."""
    parsed = d['Year(s)_Avitabile'].apply(parse_year_range)
    ini = parsed.apply(lambda x: x[0]).astype(int)
    end = parsed.apply(lambda x: x[1]).astype(int)
    wl = np.isclose(d['AGB'].astype(float), WL_AGB) & d['Year_ini_even'].notna()
    ini[wl] = d.loc[wl, 'Year_ini_even'].astype(int)
    end[wl] = d.loc[wl, 'Year_end_even'].astype(int)
    return ini, end


def parse_year_range(year_str):
    """
    Parsea Year(s)_Avitabile → (year_ini, year_end).
    '2020'      → (2020, 2020)
    '2008-2012' → (2008, 2012)
    """
    s = str(year_str).strip()
    if '-' in s:
        parts = s.split('-')
        return int(parts[0]), int(parts[1])
    return int(s), int(s)


def extract_batch(batch_df):
    """
    Extrae predictores para un lote de puntos.
    Usa Year(s)_Avitabile como fuente de verdad para el período,
    sin depender de Year_ini_even / Year_end_even que colapsan rangos.
    Agrupa por período para reusar imágenes.

    Para períodos de año único (year_ini == year_end):
      - Pase 1: extrae SIN fallback.
      - Identifica UIDs que quedaron sin datos de Landsat (NIR ausente).
      - Pase 2 (solo para esos UIDs): extrae CON fallback jul–dic año anterior.
      - Marca used_fallback_landsat=1 SOLO para los que realmente usaron el fallback.

    Para períodos multi-año (year_ini != year_end):
      - Un solo pase, sin fallback.

    Devuelve dict {unique_id_combined: {columna: valor, ...}}
    """
    results = {}

    # Derivar year_ini / year_end desde Year(s)_Avitabile
    batch_df = batch_df.copy()
    batch_df['_year_ini'], batch_df['_year_end'] = record_periods(batch_df)

    # Agrupar por período para calcular cada stack una sola vez
    grupos = batch_df.groupby(['_year_ini', '_year_end'])

    def make_fc(grupo_df):
        """Crea FeatureCollection de EE con los puntos de un grupo."""
        features = []
        for _, row in grupo_df.iterrows():
            geom = ee.Geometry.Point([row['x'], row['y']])
            half = PLOT_HALF_SIDE_M.get(str(row.get('code_des', '')), 0)
            if half:   # square plot footprint (axis-aligned; near the equator 1 deg lat ~ 1 deg lon)
                geom = geom.buffer(half, 0.1).bounds(0.1)
            features.append(ee.Feature(geom, {'uid': str(row['unique_id_combined'])}))
        return ee.FeatureCollection(features)

    def sample_img(img, fc, label, y_ini, y_end):
        """Extrae valores de una imagen; devuelve dict uid→props o {} si falla."""
        try:
            # reduceRegions + mean: a point gets the value of the pixel that contains it; a plot
            # polygon gets the mean of the intersecting pixels weighted by their fraction inside it
            sampled = img.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.mean(),
                scale=SCALE_M,
                tileScale=4
            )
            data = sampled.getInfo()
            out = {}
            for feat in data['features']:
                props = feat['properties']
                uid = props.pop('uid')
                out[uid] = {k: v for k, v in props.items() if v is not None}
            return out
        except Exception as e:
            print(f'\n  ⚠ Error {label} período {y_ini}-{y_end}: {e}')
            return {}

    def merge_into_results(res_landsat, res_srtm, res_palsar, fallback_uids=None):
        """
        Combina resultados de las tres fuentes y los acumula en `results`.
        fallback_uids: conjunto de UIDs que usaron el fallback Landsat.
        """
        if fallback_uids is None:
            fallback_uids = set()
        all_uids = set(res_landsat) | set(res_srtm) | set(res_palsar)
        for uid in all_uids:
            merged = {}
            merged.update(res_landsat.get(uid, {}))
            merged.update(res_srtm.get(uid, {}))
            merged.update(res_palsar.get(uid, {}))
            merged['used_fallback_landsat'] = 1 if uid in fallback_uids else 0
            if uid in results:
                results[uid].update(merged)
            else:
                results[uid] = merged

    LANDSAT_BANDS = ['Blue','Green','Red','NIR','SWIR1','SWIR2',
                     'NDVI','GNDVI','EVI','n_years_valid','n_years_total']

    for (y_ini, y_end), grupo in grupos:
        y_ini, y_end = int(y_ini), int(y_end)
        single_year = (y_ini == y_end)

        # ── Imágenes PALSAR y SRTM (siempre un solo pase) ─────────────────
        palsar_img = get_palsar_period(y_ini, y_end)
        srtm_img   = get_srtm()
        fc_all     = make_fc(grupo)

        res_srtm   = sample_img(srtm_img.select(['elevation','slope']),
                                fc_all, 'SRTM', y_ini, y_end)
        res_palsar = sample_img(palsar_img.select(['HH','HV']),
                                fc_all, 'PALSAR', y_ini, y_end)

        if not single_year:
            # ── Período multi-año: un solo pase Landsat, sin fallback ──────
            landsat_img = get_landsat_period(y_ini, y_end, with_fallback=False)
            res_landsat = sample_img(landsat_img.select(LANDSAT_BANDS),
                                     fc_all, 'Landsat', y_ini, y_end)
            merge_into_results(res_landsat, res_srtm, res_palsar)

        else:
            # ── Año único: dos pases ───────────────────────────────────────
            # Pase 1: sin fallback
            print(f'\n  → Pase 1 Landsat (sin fallback) para {y_ini}')
            landsat_img_p1 = get_landsat_period(y_ini, y_end, with_fallback=False)
            res_landsat_p1 = sample_img(landsat_img_p1.select(LANDSAT_BANDS),
                                        fc_all, 'Landsat P1', y_ini, y_end)

            # Identificar UIDs sin dato Landsat (NIR ausente en resultado)
            all_uids_grupo = set(grupo['unique_id_combined'].astype(str))
            uids_con_dato  = {uid for uid, props in res_landsat_p1.items()
                              if props.get('NIR') is not None}
            uids_sin_dato  = all_uids_grupo - uids_con_dato

            res_landsat_final = dict(res_landsat_p1)  # copia del pase 1
            fallback_uids_used = set()

            if uids_sin_dato:
                print(f'  → {len(uids_sin_dato)} UIDs sin Landsat; Pase 2 con fallback jul–dic {y_ini-1}')
                # Filtrar el grupo a solo los UIDs sin dato
                grupo_sin = grupo[grupo['unique_id_combined'].astype(str).isin(uids_sin_dato)]
                fc_sin    = make_fc(grupo_sin)

                # Pase 2: con fallback (agrega jul–dic año anterior)
                landsat_img_p2 = get_landsat_period(y_ini, y_end, with_fallback=True)
                res_landsat_p2 = sample_img(landsat_img_p2.select(LANDSAT_BANDS),
                                            fc_sin, 'Landsat P2', y_ini, y_end)

                # Integrar resultados del pase 2
                for uid, props in res_landsat_p2.items():
                    if props.get('NIR') is not None:
                        res_landsat_final[uid] = props
                        fallback_uids_used.add(uid)
                        print(f'    ✓ uid {uid}: recuperado con fallback')
                    else:
                        print(f'    ✗ uid {uid}: sigue sin dato tras fallback')

            merge_into_results(res_landsat_final, res_srtm, res_palsar,
                               fallback_uids=fallback_uids_used)

    return results


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print('Leyendo CSV de entrada...')
    df = pd.read_csv(CSV_INPUT)
    if 'code_des' not in df.columns:                   # BP = 10 x 10 m plots, USFQ = 60 x 60 m plots
        df['code_des'] = df['code']
    df['code'] = df['code'].replace({'BP': 'USFQ'})   # both networks are reported as USFQ (28-oct)
    for k, h in PLOT_HALF_SIDE_M.items():
        print(f'  {k}: {(df.code_des == k).sum()} records -> mean of pixels in a {2*h} x {2*h} m square')
    print(f"  BP: {(df.code_des == 'BP').sum()} records -> pixel containing the plot centre")
    print(f'  {len(df):,} puntos × {df.shape[1]} columnas')
    # Derivar períodos reales desde Year(s)_Avitabile
    df['_year_ini'], df['_year_end'] = record_periods(df)
    n_wl = int(np.isclose(df['AGB'], WL_AGB).sum())
    print(f'  Weak-labels (AGB = {WL_AGB:g}): {n_wl} -> predictors of their FCD year (Year_ini_even)')
    print(f'  Períodos únicos (desde Year(s)_Avitabile): {df.groupby(["_year_ini","_year_end"]).ngroups}')
    print()

    # Columnas de salida que vamos a añadir
    new_cols = ['B_mean','G_mean','R_mean','NIR_mean','SWIR1_mean','SWIR2_mean',
                'NDVI_mean','GNDVI_mean','EVI_mean',
                'HH_mean','HV_mean',
                'elevation_srtm','slope_srtm',
                'n_years_valid_landsat','n_years_total_landsat',
                'used_fallback_landsat']

    # Inicializar columnas como NaN / 0
    for col in new_cols:
        df[col] = np.nan
    # used_fallback_landsat se asigna por extract_batch (solo los que realmente usaron fallback)
    df['used_fallback_landsat'] = 0
    print(f'  Puntos con año único (candidatos a fallback): {(df["_year_ini"] == df["_year_end"]).sum()}')
    print()

    # Mapeo de nombres internos GEE → nombres de columna del CSV
    band_map = {
        'Blue'     : 'B_mean',
        'Green'    : 'G_mean',
        'Red'      : 'R_mean',
        'NIR'      : 'NIR_mean',
        'SWIR1'    : 'SWIR1_mean',
        'SWIR2'    : 'SWIR2_mean',
        'NDVI'     : 'NDVI_mean',
        'GNDVI'    : 'GNDVI_mean',
        'EVI'      : 'EVI_mean',
        'HH'       : 'HH_mean',
        'HV'       : 'HV_mean',
        'elevation'     : 'elevation_srtm',
        'slope'         : 'slope_srtm',
        'n_years_valid' : 'n_years_valid_landsat',
        'n_years_total' : 'n_years_total_landsat',
        # SRTM
        'elevation'     : 'elevation_srtm',
        'slope'         : 'slope_srtm',
    }

    # ── Procesar en lotes ──────────────────────────────────────────────────
    n_batches = math.ceil(len(df) / BATCH_SIZE)
    print(f'Procesando {len(df)} puntos en {n_batches} lotes de {BATCH_SIZE}...\n')

    all_results = {}
    failed_batches = []

    for i in tqdm(range(n_batches), desc='Lotes GEE'):
        batch = df.iloc[i * BATCH_SIZE : (i + 1) * BATCH_SIZE].copy()
        try:
            batch_results = extract_batch(batch)
            all_results.update(batch_results)
        except Exception as e:
            print(f'\n  ✗ Lote {i+1} falló: {e}')
            failed_batches.append(i)
        # Pausa breve para no saturar la API
        time.sleep(1)

    # ── Mapear resultados al DataFrame ─────────────────────────────────────
    print('\nMapeando resultados al DataFrame...')
    matched = 0
    for uid_str, props in all_results.items():
        mask = df['unique_id_combined'].astype(str) == uid_str
        if mask.any() and props:
            for gee_band, csv_col in band_map.items():
                if gee_band in props:
                    df.loc[mask, csv_col] = props[gee_band]
            # used_fallback_landsat viene directo del dict (0 o 1)
            if 'used_fallback_landsat' in props:
                df.loc[mask, 'used_fallback_landsat'] = props['used_fallback_landsat']
            matched += 1

    print(f'  Puntos con datos: {matched:,} / {len(df):,}')

    # ── Diagnóstico de nulos ───────────────────────────────────────────────
    print('\nNulos por columna nueva:')
    for col in new_cols:
        n_null = df[col].isna().sum()
        if n_null > 0:
            print(f'  {col}: {n_null} nulos')

    if failed_batches:
        print(f'\n⚠ Lotes con error: {failed_batches}')
        print('  Puedes reintentar solo esos puntos ajustando el índice de inicio.')

    # ── Resumen de calidad temporal ────────────────────────────────────────
    if 'n_years_valid_landsat' in df.columns:
        print('\nCalidad temporal Landsat (n_years_valid / n_years_total):')
        df['pct_valid'] = (df['n_years_valid_landsat'] /
                           df['n_years_total_landsat'] * 100).round(1)
        bins = [0, 25, 50, 75, 100]
        labels = ['0-25%','25-50%','50-75%','75-100%']
        df['valid_cat'] = pd.cut(df['pct_valid'], bins=bins,
                                  labels=labels, include_lowest=True)
        print(df['valid_cat'].value_counts().sort_index().to_string())
        print(f"\n  Puntos con 0 años válidos (NaN): {df['n_years_valid_landsat'].eq(0).sum()}")
        print(f"  Puntos con todos los años válidos: {(df['pct_valid']==100).sum()}")
        df.drop(columns=['pct_valid','valid_cat'], inplace=True)

    # ── Guardar CSV ────────────────────────────────────────────────────────
    if CSV_OUTPUT.exists():   # no se pierde la versión anterior (QA v2)
        bak = CSV_OUTPUT.with_name(CSV_OUTPUT.stem + '_single_pixel.csv')
        if not bak.exists():
            CSV_OUTPUT.rename(bak)
            print(f'  versión anterior guardada como {bak.name}')
    df.to_csv(CSV_OUTPUT, index=False)
    print(f'\n✓ CSV guardado: {CSV_OUTPUT}')
    print(f'  Filas: {len(df):,}  |  Columnas: {df.shape[1]}')
    print(f'  Columnas nuevas: {new_cols}')


if __name__ == '__main__':
    main()
