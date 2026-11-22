"""
02_download_images_gee.py
=========================
Descarga imágenes de predicción para el área del Chocó Andino (AOI)
usando Google Earth Engine → exporta a Google Drive → descarga local.

Períodos de predicción (5 mapas — años del FCD):
  2000, 2006, 2012, 2016, 2022

Capas descargadas por período:
  Landsat  : Blue, Green, Red, NIR, SWIR1, SWIR2, NDVI, GNDVI, EVI  (30 m)
  PALSAR-2 : HH, HV  (proxy 2015, 30 m)
  SRTM     : elevation, slope  (30 m, descarga única)

Lógica Landsat por período (año del FCD), relleno de vacíos en 2 pasos (v3):
  1. medoid del año completo (ene–dic)                         coverage = 1
  2. donde queda vacío: medoid S2 del año ANTERIOR (jul–dic)    coverage = 2
  3. donde sigue vacío: medoid S1 del año SIGUIENTE (ene–jun)   coverage = 3
  4. donde sigue vacío: NoData (enmascarado)                    coverage = 0
     → en el 07 esos píxeles son NoData (-9999, código 2 "no Landsat").

  mapa   año principal   S2 año anterior     S1 año siguiente
  2000   L5+L7 2000      L5+L7 jul–dic 1999  L5+L7 ene–jun 2001
  2006   L5+L7 2006      L5+L7 jul–dic 2005  L5+L7 ene–jun 2007
  2012   L5+L7 2012      L5+L7 jul–dic 2011  L7+L8 ene–jun 2013 (L8 opera desde abr-2013)
  2016   L8    2016      L8    jul–dic 2015  L8    ene–jun 2017
  2022   L8+L9 2022      L8    jul–dic 2021  L8+L9 ene–jun 2023

  Máscara QA completa (fill, dilated cloud, cirrus, cloud, shadow, confianza alta, saturación)
  + rango físico, IGUAL que en el 03. No hay filtro por NDVI (suelo desnudo y urbano se quedan).
  La banda 'coverage_main' dice de dónde viene cada píxel (1/2/3; 0 = sin dato).

AOI (v2): extensión del raster FCD (rectángulo lon -79.0719 a -78.3130, lat -0.3970 a 0.3622).
  La v1 usaba el asset 'study_area_chocho_andino', que en realidad era el bbox de los
  puntos del CSV dentro de un rectángulo aproximado escrito a mano (lat máx 0.15) en
  01_data_exploration.ipynb → los rasters quedaban cortados. El recorte al área de
  estudio real se hace en el script 07 (predicción), no aquí.

PALSAR (v2): se exporta cada año 2015–2020 por separado (palsar_<año>.tif). Los promedios
  que usa cada mapa (igual que en el entrenamiento, script 03) se calculan en el 07:
    2000/2006/2012 → 2015 · 2016 → media 2015–2016 · 2022 → media 2015–2020

Salida local:
  D:/PAPER-ECOLIN/DATA/interim/landsat/  → landsat_<año>.tif
  D:/PAPER-ECOLIN/DATA/interim/srtm-dem/ → srtm_dem.tif
  D:/PAPER-ECOLIN/DATA/interim/palsar-radar/ → palsar_2015.tif ... palsar_2020.tif

Uso:
  python scripts/02_download_images_gee.py

Requisitos:
  pip install earthengine-api google-api-python-client google-auth
  earthengine authenticate   # solo la primera vez (credenciales reutilizadas para Drive)
"""

import ee
import os
import time
from pathlib import Path
from tqdm import tqdm

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
PROJECT_ID   = 'ee-modelsgee4'
AOI_ASSET    = f'projects/{PROJECT_ID}/assets/study_area_chocho_andino'   # v1, NO se usa (recortado)

DRIVE_FOLDER = 'choco_andino_exports'   # carpeta en tu Google Drive

OUT_LANDSAT  = Path(r'D:\PAPER-ECOLIN\DATA\interim\landsat')
OUT_SRTM     = Path(r'D:\PAPER-ECOLIN\DATA\interim\srtm-dem')
OUT_PALSAR   = Path(r'D:\PAPER-ECOLIN\DATA\interim\palsar-radar')

SCALE_M      = 30
CRS_OUT      = 'EPSG:32717'    # UTM zona 17S — grilla de salida para todos los rasters
PALSAR_YEARS = [2015, 2016, 2017, 2018, 2019, 2020]   # ALOS-2 anual disponible

# Extensión del FCD (landtrendr_fitted_prediction_2000.tif, EPSG:4326)
FCD_BOUNDS   = [-79.07186726488217, -0.3969655240524166, -78.312970512858, 0.3622007225569911]

# Qué exportar (pon False para saltar lo que ya tengas bien descargado)
DO_LANDSAT   = True
DO_PALSAR    = False   # ya descargados (palsar_2015..2020.tif); True solo si faltan
DO_SRTM      = False   # ya descargado (srtm_dem.tif)

# Períodos de predicción FCD: (año_label, year_ini, year_end)
# Todos son año único → medoid del año; vacíos: S2 año anterior, luego S1 año siguiente
PERIODOS = [
    (2000, 2000, 2000),
    (2006, 2006, 2006),
    (2012, 2012, 2012),
    (2016, 2016, 2016),
    (2022, 2022, 2022),
]

# ── INICIALIZAR GEE ───────────────────────────────────────────────────────────
print('Inicializando Earth Engine...')
ee.Initialize(project=PROJECT_ID)
# v2: región = extensión completa del FCD (no el asset recortado)
aoi = ee.Geometry.Rectangle(FCD_BOUNDS, proj='EPSG:4326', geodesic=False)
print(f'✓ GEE listo — región de exportación = extensión FCD {FCD_BOUNDS}\n')


# ════════════════════════════════════════════════════════════════════════════
# FUNCIONES LANDSAT  (idéntica lógica al script 03_extract_predictors_gee.py)
# ════════════════════════════════════════════════════════════════════════════

def scale_l57(img):
    """Factor de escala SR a Landsat 5/7."""
    optical = img.select('SR_B.').multiply(0.0000275).add(-0.2)
    return img.addBands(optical, overwrite=True)

def scale_l89(img):
    """Factor de escala SR a Landsat 8/9."""
    optical = img.select('SR_B.').multiply(0.0000275).add(-0.2)
    return img.addBands(optical, overwrite=True)

def mask_range(img):
    """Filtro de rango físico [-0.2, 1.0] sobre bandas ópticas RENOMBRADAS.
    Se aplica después del rename para no incluir bandas térmicas (SR_B10/B11).
    Elimina saturación (DN=65535 → 1.6022) sin descartar negativos de calibración."""
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
    return img.select(
        ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7'],
        ['Blue',  'Green','Red',  'NIR',  'SWIR1','SWIR2']
    )

def rename_l89(img):
    return img.select(
        ['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'],
        ['Blue',  'Green','Red',  'NIR',  'SWIR1','SWIR2']
    )

def add_indices(img):
    ndvi  = img.normalizedDifference(['NIR','Red']).rename('NDVI')
    gndvi = img.normalizedDifference(['NIR','Green']).rename('GNDVI')
    evi   = img.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {'NIR': img.select('NIR'),
         'RED': img.select('Red'),
         'BLUE': img.select('Blue')}
    ).rename('EVI').clamp(-1, 1)
    return img.addBands([ndvi, gndvi, evi])

def medoid(collection, bands):
    """Medoid de Flood (2013): píxel real más cercano a la mediana espectral."""
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
    if year <= 2012:
        # L5 + L7 (L5 descomisionado en 2013, L7 tiene scan-line corrector fallo 2003
        # pero aún aporta píxeles útiles para completar cobertura en Chocó)
        col5 = (ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
                .filterDate(start, end)
                .map(mask_qa).map(scale_l57).map(rename_l57).map(mask_range).map(add_indices))
        col7 = (ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
                .filterDate(start, end)
                .map(mask_qa).map(scale_l57).map(rename_l57).map(mask_range).map(add_indices))
        return col5.merge(col7)
    elif year == 2013:
        # L8 opera desde abril 2013 → se suma L7 para cubrir ene–mar (relleno S1 2013 del mapa 2012)
        col7 = (ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
                .filterDate(start, end)
                .map(mask_qa).map(scale_l57).map(rename_l57).map(mask_range).map(add_indices))
        col8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                .filterDate(start, end)
                .map(mask_qa).map(scale_l89).map(rename_l89).map(mask_range).map(add_indices))
        return col7.merge(col8)
    elif year <= 2021:
        # Solo L8 (L9 lanzó sep 2021, cobertura estable desde 2022)
        return (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                .filterDate(start, end)
                .map(mask_qa).map(scale_l89).map(rename_l89).map(mask_range).map(add_indices))
    else:
        # L8 + L9 (ambos operativos; maximiza cobertura en Chocó nuboso)
        col8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                .filterDate(start, end)
                .map(mask_qa).map(scale_l89).map(rename_l89).map(mask_range).map(add_indices))
        col9 = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
                .filterDate(start, end)
                .map(mask_qa).map(scale_l89).map(rename_l89).map(mask_range).map(add_indices))
        return col8.merge(col9)

def get_landsat_year(year):
    """Medoid de un año completo (ene–dic)."""
    start = ee.Date.fromYMD(year, 1, 1)
    end   = ee.Date.fromYMD(year, 12, 31)
    col   = get_landsat_collection(start, end, year)
    bands = ['Blue','Green','Red','NIR','SWIR1','SWIR2']
    return medoid(col, bands)

def get_landsat_season(year, m_ini, m_end):
    """Medoid de los meses m_ini..m_end de un año."""
    start = ee.Date.fromYMD(year, m_ini, 1)
    end   = ee.Date.fromYMD(year, m_end, 1).advance(1, 'month')
    col   = get_landsat_collection(start, end, year)
    bands = ['Blue','Green','Red','NIR','SWIR1','SWIR2']
    return medoid(col, bands)

def get_landsat_fallback(year):
    """Relleno 1: S2 (jul–dic) del año anterior."""
    return get_landsat_season(year - 1, 7, 12)

def get_landsat_fallback_next(year):
    """Relleno 2: S1 (ene–jun) del año siguiente."""
    return get_landsat_season(year + 1, 1, 6)

def build_landsat_period(year_label, year_ini, year_end):
    """
    Construye imagen Landsat para un período de predicción.

    Año único (year_ini == year_end):
      - imagen principal: medoid del año
      - imagen fallback:  medoid jul–dic año anterior
      - resultado: where(principal tiene dato, principal, fallback)
      → sin vacíos dentro del AOI si la temporada anterior tiene cobertura

    Multi-año (year_ini != year_end):
      - media de medoids anuales (igual que en extracción de puntos)
    """
    single_year = (year_ini == year_end)
    bands_all   = ['Blue','Green','Red','NIR','SWIR1','SWIR2','NDVI','GNDVI','EVI']

    if single_year:
        img_main = get_landsat_year(year_ini).select(bands_all).toFloat()
        img_prev = get_landsat_fallback(year_ini).select(bands_all).toFloat()       # S2 año anterior
        img_next = get_landsat_fallback_next(year_ini).select(bands_all).toFloat()  # S1 año siguiente

        def has(img):
            return img.select('NIR').mask().gt(0).unmask(0)

        # mosaic(): la última imagen queda arriba → prioridad principal > S2 anterior > S1 siguiente.
        # Un píxel sin dato en las tres queda enmascarado (NoData).
        img_filled = ee.ImageCollection([img_next, img_prev, img_main]).mosaic()

        coverage = (ee.Image(0)
                    .where(has(img_next), 3)
                    .where(has(img_prev), 2)
                    .where(has(img_main), 1)
                    .rename('coverage_main').toFloat())
        # n_clear: nº de escenas despejadas (tras la máscara QA) en el año principal → diagnóstico
        n_clear = (get_landsat_collection(ee.Date.fromYMD(year_ini, 1, 1),
                                          ee.Date.fromYMD(year_ini, 12, 31), year_ini)
                   .select('NIR').count().unmask(0).rename('n_clear').toFloat())
        return img_filled.addBands([coverage, n_clear]).select(bands_all + ['coverage_main', 'n_clear'])

    else:
        # Multi-año: media de medoids sobre bandas base, índices recalculados
        years       = list(range(year_ini, year_end + 1))
        images      = [get_landsat_year(y) for y in years]
        col         = ee.ImageCollection(images)
        bands_base  = ['Blue','Green','Red','NIR','SWIR1','SWIR2']
        # Cobertura: fracción de años con dato
        valid_masks = [img.select('NIR').mask().toFloat() for img in images]
        n_valid     = ee.ImageCollection(valid_masks).sum()
        n_total     = ee.Image.constant(len(years)).toFloat()
        coverage    = n_valid.divide(n_total).rename('coverage_main')
        # Recalcular índices sobre las bandas promediadas (no promediar los índices)
        img_mean    = col.select(bands_base).mean()
        img_mean    = add_indices(img_mean)
        return img_mean.addBands(coverage)


# ════════════════════════════════════════════════════════════════════════════
# FUNCIONES PALSAR-2
# ════════════════════════════════════════════════════════════════════════════

def lee_filter_band(img, band_name, window_size=7):
    band      = img.select([band_name])
    half      = window_size // 2
    kernel    = ee.Kernel.square(half)
    mean      = band.reduceNeighborhood(ee.Reducer.mean(),     kernel)
    var_local = band.reduceNeighborhood(ee.Reducer.variance(), kernel)
    var_noise = mean.pow(2).divide(4.0)
    weight    = var_local.subtract(var_noise).divide(var_local.add(1e-10))
    weight    = weight.max(0).min(1)
    return mean.add(weight.multiply(band.subtract(mean))).rename(band_name)

def lee_filter(img, window_size=7):
    hh = lee_filter_band(img, 'HH', window_size)
    hv = lee_filter_band(img, 'HV', window_size)
    return img.addBands([hh, hv], overwrite=True)

def build_palsar(year):
    """PALSAR-2 de un año, filtro Lee 7×7, DN crudo (sin conversión a dB) — igual que script 03."""
    start = ee.Date.fromYMD(year, 1, 1)
    end   = ee.Date.fromYMD(year, 12, 31)
    img   = (ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR')
             .filterDate(start, end)
             .select(['HH','HV'])
             .mosaic())
    return lee_filter(img, window_size=7).select(['HH','HV'])


# ════════════════════════════════════════════════════════════════════════════
# FUNCIONES SRTM
# ════════════════════════════════════════════════════════════════════════════

def build_srtm():
    dem   = ee.Image('USGS/SRTMGL1_003').select('elevation')
    slope = ee.Terrain.slope(dem)
    return dem.addBands(slope)


# ════════════════════════════════════════════════════════════════════════════
# EXPORTAR A GOOGLE DRIVE
# ════════════════════════════════════════════════════════════════════════════

def export_to_drive(image, description, filename, bands=None):
    """
    Lanza tarea de exportación GEE → Google Drive.
    Castea todas las bandas a Float32 para evitar conflictos de tipo.
    Devuelve el objeto Task.
    """
    if bands:
        image = image.select(bands)

    # Castear a Float32 para que todas las bandas sean compatibles al exportar.
    # SRTM elevation viene como Int16, n_years_* como Float64 → unificar en Float32.
    image = image.toFloat()

    task = ee.batch.Export.image.toDrive(
        image          = image.clip(aoi),
        description    = description,
        folder         = DRIVE_FOLDER,
        fileNamePrefix = filename,
        region         = aoi,
        scale          = SCALE_M,
        crs            = CRS_OUT,
        maxPixels      = 1e13,
        fileFormat     = 'GeoTIFF'
    )
    task.start()
    return task

def wait_for_tasks(tasks, poll_interval=30):
    """Espera a que terminen todas las tareas, mostrando progreso."""
    print(f'\nEsperando {len(tasks)} tarea(s) GEE...')
    pending = list(tasks)
    while pending:
        still_pending = []
        for name, task in pending:
            status = task.status()['state']
            if status in ('COMPLETED',):
                print(f'  ✓ {name}')
            elif status in ('FAILED', 'CANCELLED'):
                print(f'  ✗ {name}: {task.status().get("error_message","?")}')
            else:
                still_pending.append((name, task))
        pending = still_pending
        if pending:
            print(f'  … {len(pending)} pendientes — reintentando en {poll_interval}s')
            time.sleep(poll_interval)
    print('Todas las tareas finalizaron.\n')


# ════════════════════════════════════════════════════════════════════════════
# DESCARGAR DESDE DRIVE (requiere gcloud / google-api-python-client)
# ════════════════════════════════════════════════════════════════════════════

def get_drive_credentials():
    """Reutiliza el refresh_token de `earthengine authenticate` para acceder a Drive
    (mismo método que 02c_palsar_only.py; google.auth.default() fallaba en Windows)."""
    import json
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    cred_paths = [
        Path.home() / '.config' / 'earthengine' / 'credentials',
        Path(os.environ.get('APPDATA', '')) / 'earthengine' / 'credentials',
    ]
    cred_file = next((p for p in cred_paths if p.exists()), None)
    if cred_file is None:
        raise FileNotFoundError('No se encontró la credencial de Earth Engine. Corre: earthengine authenticate')
    with open(cred_file) as f:
        data = json.load(f)
    creds = Credentials(token=None, refresh_token=data['refresh_token'],
                        token_uri='https://oauth2.googleapis.com/token',
                        client_id=data['client_id'], client_secret=data['client_secret'],
                        scopes=['https://www.googleapis.com/auth/drive.readonly'])
    creds.refresh(Request())
    return creds


def download_from_drive():
    """
    Descarga todos los GeoTIFF de DRIVE_FOLDER a las carpetas locales.
    Usa la Google Drive API (googleapiclient) que ya viene con earthengine-api.
    No requiere gcloud CLI.
    """
    print('Descargando desde Google Drive...')
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        import google.auth
        import io

        creds   = get_drive_credentials()
        service = build('drive', 'v3', credentials=creds)

        # Buscar la carpeta DRIVE_FOLDER
        folder_res = service.files().list(
            q=f"name='{DRIVE_FOLDER}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields='files(id,name)'
        ).execute()
        folders = folder_res.get('files', [])
        if not folders:
            print(f'  ⚠ Carpeta "{DRIVE_FOLDER}" no encontrada en Drive.')
            _print_manual_instructions()
            return
        folder_id = folders[0]['id']

        # Listar GeoTIFF dentro de la carpeta
        files_res = service.files().list(
            q=f"'{folder_id}' in parents and name contains '.tif' and trashed=false",
            fields='files(id,name,modifiedTime)',
            orderBy='modifiedTime desc',
            pageSize=200
        ).execute()
        drive_files = files_res.get('files', [])
        if not drive_files:
            print('  ⚠ No se encontraron archivos .tif en la carpeta Drive.')
            _print_manual_instructions()
            return

        seen = set()   # Drive admite nombres repetidos: se baja solo la exportación MÁS RECIENTE
        for f in drive_files:
            fname   = f['name']
            if fname in seen:
                continue
            seen.add(fname)
            if DO_LANDSAT is False and fname.startswith('landsat_'):
                continue
            if DO_PALSAR is False and fname.startswith('palsar_'):
                continue
            if DO_SRTM is False and fname.startswith('srtm_'):
                continue
            file_id = f['id']
            # Determinar carpeta destino según prefijo
            if fname.startswith('landsat_'):
                dest = OUT_LANDSAT / fname
            elif fname.startswith('srtm_'):
                dest = OUT_SRTM / fname
            elif fname.startswith('palsar_'):
                dest = OUT_PALSAR / fname
            else:
                continue
            print(f'  ↓ {fname}  →  {dest}')
            request  = service.files().get_media(fileId=file_id)
            with open(dest, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request, chunksize=8 * 1024 * 1024)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        print(f'       {int(status.progress() * 100)}%', end='\r')
            print(f'     ✓ {fname}')

        print('✓ Descarga completa.\n')

    except Exception as e:
        print(f'  ⚠ Error en descarga automática: {e}')
        _print_manual_instructions()


def _print_manual_instructions():
    print('\n  → Descarga manual desde drive.google.com:')
    print(f'     carpeta: {DRIVE_FOLDER}')
    print(f'     landsat_*.tif  →  {OUT_LANDSAT}')
    print(f'     srtm_dem.tif   →  {OUT_SRTM}')
    print(f'     palsar_2015..2020.tif →  {OUT_PALSAR}')
    print()


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    # Crear carpetas de salida
    for d in [OUT_LANDSAT, OUT_SRTM, OUT_PALSAR]:
        d.mkdir(parents=True, exist_ok=True)


    tasks = []

    # ── 1. LANDSAT por período ────────────────────────────────────────────
    if DO_LANDSAT:
        print('Preparando exportaciones Landsat...')
        for (year_label, year_ini, year_end) in PERIODOS:
            print(f'  Período {year_label} ({year_ini}–{year_end})')
            img  = build_landsat_period(year_label, year_ini, year_end)
            desc = f'landsat_{year_label}'
            task = export_to_drive(img, desc, desc)
            tasks.append((desc, task))
            print(f'    → tarea lanzada: {desc}')

    # ── 2. PALSAR-2: un archivo por año 2015–2020 ───────────────────────
    if DO_PALSAR:
        print('\nPreparando exportaciones PALSAR-2...')
        for yr in PALSAR_YEARS:
            palsar_desc = f'palsar_{yr}'
            task_palsar = export_to_drive(build_palsar(yr), palsar_desc, palsar_desc)
            tasks.append((palsar_desc, task_palsar))
            print(f'  → tarea lanzada: {palsar_desc}')

    # ── 3. SRTM (único) ──────────────────────────────────────────────────
    if DO_SRTM:
        print('\nPreparando exportación SRTM...')
        srtm_desc = 'srtm_dem'
        task_srtm = export_to_drive(build_srtm(), srtm_desc, srtm_desc)
        tasks.append((srtm_desc, task_srtm))
        print(f'  → tarea lanzada: {srtm_desc}')

    print(f'\nTotal tareas lanzadas: {len(tasks)}')
    print(f'Puedes monitorear en: https://code.earthengine.google.com/tasks\n')

    # ── 4. Esperar y descargar ────────────────────────────────────────────
    wait_for_tasks(tasks, poll_interval=60)
    download_from_drive()

    print('=' * 60)
    print('RESUMEN DE ARCHIVOS GENERADOS:')
    for d, label in [(OUT_LANDSAT,'Landsat'), (OUT_SRTM,'SRTM'), (OUT_PALSAR,'PALSAR')]:
        files = list(d.glob('*.tif'))
        print(f'  {label} ({len(files)} archivos): {d}')
        for f in sorted(files):
            print(f'    {f.name}')


if __name__ == '__main__':
    main()
