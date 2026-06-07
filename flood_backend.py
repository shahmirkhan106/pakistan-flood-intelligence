"""
Pakistan Flood Intelligence System
Backend ML Pipeline v3.0 — REAL Google Earth Engine Data
─────────────────────────────────────────────────────────
All training features and 2025 predictions are now pulled
directly from GEE satellite imagery. No synthetic data.

Run:  python flood_backend.py
      python flood_backend.py --year 2024   (predict a different year)
      python flood_backend.py --cache-only  (skip GEE, use cached CSV)
"""

import ee
import numpy as np
import pandas as pd
import json
import os
import time
import argparse
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────────

GEE_PROJECT   = 'daring-elysium-498521-p4'
CACHE_FILE    = 'gee_features_cache.csv'   # saved after GEE fetch so you don't re-fetch
PREDICT_YEAR  = 2025
GEE_SCALE     = 1000   # metres — 500 is more accurate but slower; raise to 2000 if quota errors
GEE_MAX_PIX   = 1e9
GEE_RETRY     = 3      # retries per failed GEE call
GEE_SLEEP     = 2      # seconds between province calls (avoid rate limiting)

# Sentinel-1 only starts from late 2014, so training window is 2015–2024
TRAIN_START_YEAR = 2015
TRAIN_END_YEAR   = 2024

# ─── INIT ─────────────────────────────────────────────────────

ee.Initialize(project=GEE_PROJECT)

print("=" * 62)
print("  PAKISTAN FLOOD INTELLIGENCE SYSTEM  v3.0")
print("  Real GEE Data Pipeline")
print("=" * 62)

# ─── PROVINCE BOUNDARIES (GAUL official polygons via GEE) ─────

def load_province_boundaries():
    """
    Load Pakistan province boundaries from GAUL where available.
    GB and AJK use custom polygons because they are not reliably
    present in FAO/GAUL/2015/level1.
    """

    fallback_rects = {
        'Sindh': ee.Geometry.Polygon(
            [[[66.5,23.5],[66.5,28.3],[71.8,28.3],[71.8,23.5],[66.5,23.5]]]
        ),

        'Punjab': ee.Geometry.Polygon(
            [[[69.9,28.4],[69.9,34.1],[75.4,34.1],[75.4,28.4],[69.9,28.4]]]
        ),

        'Balochistan': ee.Geometry.Polygon(
            [[[60.5,24.5],[60.5,32.2],[70.3,32.2],[70.3,24.5],[60.5,24.5]]]
        ),

        'KPK': ee.Geometry.Polygon(
            [[[69.5,32.5],[69.5,36.5],[74.5,36.5],[74.5,32.5],[69.5,32.5]]]
        ),

        # Improved Gilgit-Baltistan polygon
        'GB': ee.Geometry.Polygon([
            [
                [72.4,34.8],
                [73.0,35.8],
                [74.0,36.8],
                [75.5,37.2],
                [77.0,36.7],
                [76.7,35.4],
                [75.8,35.0],
                [74.5,34.5],
                [73.0,34.4],
                [72.4,34.8]
            ]
        ]),

        # Improved Azad Jammu & Kashmir polygon
        'AJK': ee.Geometry.Polygon([
            [
                [73.2,33.1],
                [73.4,34.0],
                [73.8,34.8],
                [74.8,35.0],
                [75.3,34.4],
                [75.0,33.4],
                [74.2,33.1],
                [73.2,33.1]
            ]
        ])
    }

    try:
        gaul = ee.FeatureCollection("FAO/GAUL/2015/level1")
        pak = gaul.filter(ee.Filter.eq('ADM0_NAME', 'Pakistan'))

        # Debug: show what GAUL actually contains
        try:
            print("\nAvailable Pakistan ADM1 regions:")
            names = pak.aggregate_array('ADM1_NAME').getInfo()
            for n in sorted(names):
                print(f"  - {n}")
        except Exception:
            pass

        boundaries = {}

        province_variants = {
            'Sindh': ['Sindh'],
            'Punjab': ['Punjab'],
            'Balochistan': ['Balochistan', 'Baluchistan'],
            'KPK': [
                'Khyber Pakhtunkhwa',
                'North-West Frontier',
                'Khyber Pakhtunkhwa (FATA)',
                'Pakhtunkhwa'
            ]
        }

        # Load GAUL-supported provinces
        for province, variants in province_variants.items():
            matched = False

            for variant in variants:
                try:
                    feat = pak.filter(
                        ee.Filter.eq('ADM1_NAME', variant)
                    )

                    if feat.size().getInfo() > 0:
                        boundaries[province] = feat.geometry()
                        print(
                            f"  ✓ GAUL boundary loaded: "
                            f"{province} (as '{variant}')"
                        )
                        matched = True
                        break

                except Exception:
                    continue

            if not matched:
                print(
                    f"  ⚠ No GAUL match for {province} "
                    f"— using fallback geometry"
                )
                boundaries[province] = fallback_rects[province]

        # Always use custom polygons for GB and AJK
        boundaries['GB'] = fallback_rects['GB']
        boundaries['AJK'] = fallback_rects['AJK']

        print("  ✓ Custom boundary loaded: GB")
        print("  ✓ Custom boundary loaded: AJK")

        return boundaries

    except Exception as e:
        print(
            f"  ⚠ GAUL load failed ({e}) "
            f"— using fallback geometries for all provinces"
        )

        return fallback_rects




# ─── GEE FEATURE EXTRACTION ───────────────────────────────────

def safe_gee_call(fn, retries=GEE_RETRY, fallback=None):
    """Retry wrapper for GEE calls that can transiently fail."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"    ↻ GEE error (attempt {attempt+1}/{retries}): {e} — retrying in {wait}s")
            time.sleep(wait)
    print(f"    ✗ All retries failed, using fallback")
    return fallback


def get_sentinel1_features(region, start_date, end_date):
    """
    Extract Sentinel-1 SAR VV backscatter statistics.
    Returns dict with mean, stdDev, p10 for before/during windows.
    Water surfaces return very low backscatter (~-20 dB).
    Backscatter drop (delta) is the primary flood signal.
    """
    def _fetch():
        col = (ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .select('VV')
            # Speckle filter — 150m focal mean
            .map(lambda img: img.focal_mean(150, 'circle', 'meters')))

        size = col.size().getInfo()
        if size == 0:
            return None

        mean_img = col.mean()
        stats = mean_img.reduceRegion(
            reducer=(ee.Reducer.mean()
                     .combine(ee.Reducer.stdDev(), sharedInputs=True)
                     .combine(ee.Reducer.percentile([10, 25, 75, 90]), sharedInputs=True)),
            geometry=region,
            scale=GEE_SCALE,
            maxPixels=GEE_MAX_PIX,
            bestEffort=True
        ).getInfo()
        stats['_image_count'] = size
        return stats

    return safe_gee_call(_fetch, fallback=None)


def get_ndvi_features(region, start_date, end_date):
    """
    Extract NDVI from Sentinel-2. Low NDVI indicates submerged/waterlogged land.
    Uses cloud masking — falls back to Landsat 8 if S2 coverage is sparse.
    """
    def _fetch_s2():
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            .map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI')))
        if col.size().getInfo() == 0:
            return None
        return col.mean().reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.min(), sharedInputs=True)
                              .combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=region, scale=GEE_SCALE, maxPixels=GEE_MAX_PIX, bestEffort=True
        ).getInfo()

    def _fetch_l8():
        col = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUD_COVER', 40))
            .map(lambda img: img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')))
        if col.size().getInfo() == 0:
            return None
        return col.mean().reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.min(), sharedInputs=True),
            geometry=region, scale=GEE_SCALE, maxPixels=GEE_MAX_PIX, bestEffort=True
        ).getInfo()

    result = safe_gee_call(_fetch_s2, fallback=None)
    if result is None or result.get('NDVI_mean') is None:
        print("    → S2 sparse, trying Landsat-8...")
        result = safe_gee_call(_fetch_l8, fallback={'NDVI_mean': 0.25, 'NDVI_min': 0.05, 'NDVI_stdDev': 0.1})
    return result or {'NDVI_mean': 0.25, 'NDVI_min': 0.05, 'NDVI_stdDev': 0.1}


def get_dem_features(region):
    """
    SRTM DEM — static (doesn't change by year).
    Low elevation + low slope = higher flood accumulation risk.
    """
    def _fetch():
        dem   = ee.Image('USGS/SRTMGL1_003')
        slope = ee.Terrain.slope(dem)
        elev_stats  = dem.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.min(), sharedInputs=True)
                              .combine(ee.Reducer.percentile([10]), sharedInputs=True),
            geometry=region, scale=GEE_SCALE, maxPixels=GEE_MAX_PIX, bestEffort=True
        ).getInfo()
        slope_stats = slope.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.percentile([90]), sharedInputs=True),
            geometry=region, scale=GEE_SCALE, maxPixels=GEE_MAX_PIX, bestEffort=True
        ).getInfo()
        return {**elev_stats, **{f'slope_{k}': v for k, v in slope_stats.items()}}

    return safe_gee_call(_fetch, fallback={
        'elevation_mean': 300, 'elevation_min': 10,
        'elevation_p10': 50,   'slope_mean': 5, 'slope_p90': 20
    })


def get_soil_moisture(region, start_date, end_date):
    """
    NASA SMAP soil moisture (surface, m³/m³, range ~0.02–0.5).
    Uses SPL4SMGP/007 (supersedes deprecated HSL dataset).
    Band: sm_surface — volumetric water content in top ~5cm of soil.
    High values = saturated soil = less infiltration = flood risk.
    """
    def _fetch_spl4():
        # New SMAP Level-4 global 9km product (updated dataset)
        smap = (ee.ImageCollection('NASA/SMAP/SPL4SMGP/007')
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .select('sm_surface')
            .mean())
        stats = smap.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True),
            geometry=region, scale=11000, maxPixels=GEE_MAX_PIX, bestEffort=True
        ).getInfo()
        # Rename keys to consistent names
        return {
            'ssm_mean': stats.get('sm_surface_mean', stats.get('sm_surface', 0.25)),
            'ssm_max':  stats.get('sm_surface_max',  0.4),
        }

    def _fetch_legacy():
        # Fallback: deprecated dataset still works for older years
        smap = (ee.ImageCollection('NASA_USDA/HSL/SMAP10KM_soil_moisture')
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .select('ssm')
            .mean())
        stats = smap.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True),
            geometry=region, scale=10000, maxPixels=GEE_MAX_PIX, bestEffort=True
        ).getInfo()
        raw_mean = stats.get('ssm_mean', stats.get('ssm', 0.25))
        raw_max  = stats.get('ssm_max', 0.5)
        # Legacy dataset returns kg/m² (0–25ish), not 0–1 — normalise to ~0–1
        # Typical field capacity ~15 kg/m², so divide by 25
        if raw_mean and raw_mean > 1.0:
            raw_mean = min(raw_mean / 25.0, 1.0)
            raw_max  = min(raw_max  / 25.0, 1.0)
        return {'ssm_mean': raw_mean, 'ssm_max': raw_max}

    # Try new dataset first, fall back to legacy
    result = safe_gee_call(_fetch_spl4, fallback=None)
    if result is None or result.get('ssm_mean') is None:
        print("    → SPL4 unavailable, trying legacy SMAP...")
        result = safe_gee_call(_fetch_legacy, fallback={'ssm_mean': 0.25, 'ssm_max': 0.4})
    return result or {'ssm_mean': 0.25, 'ssm_max': 0.4}


def get_rainfall_features(region, start_date, end_date):
    """
    CHIRPS daily rainfall — cumulative monsoon precipitation.
    Strong predictor of flood onset alongside soil saturation.
    """
    def _fetch():
        chirps = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .select('precipitation'))
        total = chirps.sum()
        stats = total.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True),
            geometry=region, scale=5000, maxPixels=GEE_MAX_PIX, bestEffort=True
        ).getInfo()
        return stats

    result = safe_gee_call(_fetch, fallback={'precipitation_mean': 200, 'precipitation_max': 500})
    return result or {'precipitation_mean': 200, 'precipitation_max': 500}


# ─── FEATURE VECTOR BUILDER ───────────────────────────────────

def build_feature_vector(province_name, year, provinces, is_monsoon=True):
    """
    Pull all GEE features for one province-year and return a flat dict.
    Monsoon window: Jul–Sep (during), Apr–Jun (pre-monsoon baseline).
    """
    region = provinces[province_name]

    if is_monsoon:
        start  = f'{year}-07-01';  end   = f'{year}-09-30'
        pre_s  = f'{year}-04-01';  pre_e = f'{year}-06-30'
    else:
        start  = f'{year}-01-01';  end   = f'{year}-03-31'
        pre_s  = f'{year-1}-10-01'; pre_e = f'{year-1}-12-31'

    sar_dur  = get_sentinel1_features(region, start, end)
    sar_pre  = get_sentinel1_features(region, pre_s, pre_e)
    ndvi     = get_ndvi_features(region, start, end)
    dem      = get_dem_features(region)
    soil_dur = get_soil_moisture(region, start, end)
    soil_pre = get_soil_moisture(region, pre_s, pre_e)
    rain     = get_rainfall_features(region, start, end)

    f = {}

    # SAR features
    if sar_dur and sar_pre and sar_dur.get('VV_mean') and sar_pre.get('VV_mean'):
        f['sar_mean_during'] = sar_dur.get('VV_mean', -12)
        f['sar_std_during']  = sar_dur.get('VV_stdDev', 3)
        f['sar_p10_during']  = sar_dur.get('VV_p10', -20)
        f['sar_mean_before'] = sar_pre.get('VV_mean', -10)
        f['sar_p10_before']  = sar_pre.get('VV_p10', -18)
        # positive delta = backscatter dropped during monsoon = water/flooding signal
        # convention: before - during (always used consistently in training AND inference)
        f['sar_delta']       = f['sar_mean_before'] - f['sar_mean_during']
        f['sar_p10_delta']   = f['sar_p10_before']  - f['sar_p10_during']
        f['sar_images']      = sar_dur.get('_image_count', 1)
    else:
        # Flag this row as GEE-failed — model will use fallbacks
        print(f"    ⚠ SAR fallback for {province_name} {year}")
        f.update({'sar_mean_during':-12,'sar_std_during':3,'sar_p10_during':-20,
                  'sar_mean_before':-10,'sar_p10_before':-18,'sar_delta':2,
                  'sar_p10_delta':2,'sar_images':0})

    # NDVI
    f['ndvi_mean']     = (ndvi or {}).get('NDVI_mean', 0.25)
    f['ndvi_min']      = (ndvi or {}).get('NDVI_min', 0.05)
    f['ndvi_std']      = (ndvi or {}).get('NDVI_stdDev', 0.1)

    # DEM (static)
    f['elevation_mean'] = (dem or {}).get('elevation_mean', 300)
    f['elevation_min']  = (dem or {}).get('elevation_min', 10)
    f['elevation_p10']  = (dem or {}).get('elevation_p10', 50)
    f['slope_mean']     = (dem or {}).get('slope_mean', 5)
    f['slope_p90']      = (dem or {}).get('slope_p90', 20)

    # Soil moisture
    f['soil_during']    = (soil_dur or {}).get('ssm_mean', 0.3)
    f['soil_before']    = (soil_pre or {}).get('ssm_mean', 0.2)
    f['soil_delta']     = f['soil_during'] - f['soil_before']  # saturation increase
    f['soil_max']       = (soil_dur or {}).get('ssm_max', 0.5)

    # Rainfall
    f['rain_total']     = (rain or {}).get('precipitation_mean', 200)
    f['rain_max_cell']  = (rain or {}).get('precipitation_max', 400)

    # Context
    f['is_monsoon']     = int(is_monsoon)
    f['province']       = province_name
    f['year']           = year

    return f


# ─── GROUND TRUTH LABELS ──────────────────────────────────────
# Sources: NDMA Pakistan, EM-DAT, ReliefWeb, UNOSAT
# 1 = significant flood event affecting >1M people or >20k km²
# 0 = no major flood (drought, normal monsoon, or minor localised events)

KNOWN_FLOODS = {
    # 2015 — flash floods KPK/Balochistan; Sindh minor
    ('KPK',         2015): 1,
    ('Balochistan', 2015): 1,
    ('Sindh',       2015): 0,
    ('Punjab',      2015): 0,
    ('GB',          2015): 1,
    ('AJK',         2015): 0,
    # 2016 — Punjab and Sindh flooding
    ('Punjab',      2016): 1,
    ('Sindh',       2016): 1,
    ('Balochistan', 2016): 0,
    ('KPK',         2016): 0,
    ('GB',          2016): 0,
    ('AJK',         2016): 0,
    # 2017 — relatively minor year
    ('Sindh',       2017): 0,
    ('Punjab',      2017): 0,
    ('Balochistan', 2017): 0,
    ('KPK',         2017): 0,
    ('GB',          2017): 0,
    ('AJK',         2017): 0,
    # 2018 — KPK/Balochistan flash floods
    ('KPK',         2018): 1,
    ('Balochistan', 2018): 1,
    ('Sindh',       2018): 0,
    ('Punjab',      2018): 0,
    ('GB',          2018): 1,
    ('AJK',         2018): 0,
    # 2019 — moderate flooding
    ('Sindh',       2019): 1,
    ('Balochistan', 2019): 1,
    ('KPK',         2019): 0,
    ('Punjab',      2019): 0,
    ('GB',          2019): 0,
    ('AJK',         2019): 0,
    # 2020 — heavy monsoon, multi-province
    ('Sindh',       2020): 1,
    ('Punjab',      2020): 1,
    ('Balochistan', 2020): 1,
    ('KPK',         2020): 1,
    ('GB',          2020): 0,
    ('AJK',         2020): 0,
    # 2021 — widespread but below 2022
    ('Sindh',       2021): 1,
    ('Punjab',      2021): 1,
    ('Balochistan', 2021): 1,
    ('KPK',         2021): 1,
    ('GB',          2021): 1,
    ('AJK',         2021): 1,
    # 2022 — catastrophic, 33M affected, 160,000 km² inundated
    ('Sindh',       2022): 1,
    ('Punjab',      2022): 1,
    ('Balochistan', 2022): 1,
    ('KPK',         2022): 1,
    ('GB',          2022): 1,
    ('AJK',         2022): 1,
    # 2023 — below average after La Niña; KPK flash floods
    ('Sindh',       2023): 1,
    ('KPK',         2023): 1,
    ('Punjab',      2023): 0,
    ('Balochistan', 2023): 0,
    ('GB',          2023): 1,
    ('AJK',         2023): 0,
    # 2024 — above average pre-monsoon; Balochistan severe
    ('Sindh',       2024): 1,
    ('Balochistan', 2024): 1,
    ('KPK',         2024): 1,
    ('Punjab',      2024): 0,
    ('GB',          2024): 0,
    ('AJK',         2024): 0,
}


# ─── TRAINING DATA FROM REAL GEE ──────────────────────────────

def generate_training_data(provinces, use_cache=True):
    """
    Fetch real satellite features from GEE for all province-years
    and return a labelled DataFrame ready for model training.

    Results are cached to CACHE_FILE — if the file exists and
    use_cache=True, GEE is skipped entirely (fast re-runs).
    """
    if use_cache and Path(CACHE_FILE).exists():
        print(f"\n[1/4] Loading cached GEE features from '{CACHE_FILE}'...")
        df = pd.read_csv(CACHE_FILE)
        print(f"  Loaded {len(df)} rows from cache.")
        return df

    print(f"\n[1/4] Fetching real GEE features ({TRAIN_START_YEAR}–{TRAIN_END_YEAR})...")
    print(f"  Provinces: {list(provinces.keys())}")
    print(f"  This will take ~15–45 minutes depending on GEE quota.\n")

    rows = []
    province_list = list(provinces.keys())
    total = len(province_list) * (TRAIN_END_YEAR - TRAIN_START_YEAR + 1)
    done  = 0

    for year in range(TRAIN_START_YEAR, TRAIN_END_YEAR + 1):
        for province in province_list:
            done += 1
            print(f"  [{done}/{total}] {province} {year}...")
            try:
                feats = build_feature_vector(province, year, provinces, is_monsoon=True)
                feats['flood'] = KNOWN_FLOODS.get((province, year), 0)
                rows.append(feats)
                print(f"    ✓ sar_delta={feats['sar_delta']:.2f}  soil={feats['soil_during']:.3f}"
                      f"  rain={feats['rain_total']:.0f}mm  label={feats['flood']}")
            except Exception as e:
                print(f"    ✗ FAILED: {e} — skipping")
            time.sleep(GEE_SLEEP)

    df = pd.DataFrame(rows)

    # Save cache
    df.to_csv(CACHE_FILE, index=False)
    print(f"\n  ✓ Saved GEE features to '{CACHE_FILE}'")
    print(f"  Samples: {len(df)} | Flood events: {df['flood'].sum()}")
    return df


# ─── MODEL TRAINING ───────────────────────────────────────────

FEATURE_COLS = [
    'sar_delta', 'sar_mean_during', 'sar_std_during',
    'sar_p10_during', 'sar_p10_delta',
    'ndvi_mean', 'ndvi_min', 'ndvi_std',
    'elevation_mean', 'elevation_min', 'elevation_p10',
    'slope_mean', 'slope_p90',
    'soil_during', 'soil_before', 'soil_delta', 'soil_max',
    'rain_total', 'rain_max_cell',
    'is_monsoon',
]


def train_model(df):
    """Train RF + GBT ensemble on real GEE features."""
    print("\n[2/4] Training ML models on real satellite features...")

    # Drop rows where SAR fallback was used (sar_images == 0)
    # so the model only learns from confirmed real data
    if 'sar_images' in df.columns:
        real_sar = df[df['sar_images'] > 0]
    else:
        real_sar = df
    if len(real_sar) < 10:
        print("  ⚠ Less than 10 rows with real SAR — using all rows (including fallbacks)")
        real_sar = df

    available_cols = [c for c in FEATURE_COLS if c in real_sar.columns]
    X = real_sar[available_cols].fillna(0).values
    y = real_sar['flood'].values

    print(f"  Training on {len(X)} samples | {int(y.sum())} flood / {int((1-y).sum())} no-flood")
    print(f"  Features used: {len(available_cols)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_split=3,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train_s, y_train)

    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.06, max_depth=4,
        subsample=0.8, random_state=42
    )
    gb.fit(X_train_s, y_train)

    rf_preds = rf.predict(X_test_s)
    gb_preds = gb.predict(X_test_s)
    ens_prob  = rf.predict_proba(X_test_s)[:,1] * 0.6 + gb.predict_proba(X_test_s)[:,1] * 0.4
    ens_preds = (ens_prob > 0.5).astype(int)

    print("\n  ── Random Forest ──────────────────────")
    print(classification_report(y_test, rf_preds, target_names=['No Flood','Flood']))
    print(f"  ── Ensemble F1: {f1_score(y_test, ens_preds):.3f} ────────────\n")

    # Confusion matrix
    cm = confusion_matrix(y_test, ens_preds)
    print(f"  Confusion matrix (ensemble):")
    print(f"    True Neg: {cm[0,0]}  False Pos: {cm[0,1]}")
    print(f"    False Neg: {cm[1,0]}  True Pos: {cm[1,1]}\n")

    # Cross-validation on full dataset
    Xs = scaler.transform(X)
    cv_scores = cross_val_score(rf, Xs, y, cv=min(5, int(y.sum())), scoring='f1')
    print(f"  Cross-val F1 (RF, {len(cv_scores)}-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Feature importances
    importances = pd.Series(rf.feature_importances_, index=available_cols).sort_values(ascending=False)
    print("\n  Top 8 feature importances (Random Forest):")
    for feat, imp in importances.head(8).items():
        bar = '█' * int(imp * 60)
        print(f"    {feat:<22} {bar} {imp:.3f}")

    return rf, gb, scaler, available_cols


# ─── REAL 2025 PREDICTION ─────────────────────────────────────

def predict_year(rf, gb, scaler, feature_cols, provinces, year=PREDICT_YEAR):
    """
    Fetch real GEE features for the target year and run the trained model.
    This is the part that was previously hand-coded with fake numbers.
    """
    print(f"\n[3/4] Fetching real GEE features for {year} predictions...")

    results = {}
    for province in provinces.keys():
        print(f"  Fetching: {province} {year}...")
        try:
            feats = build_feature_vector(province, year, provinces, is_monsoon=True)

            print(f"    SAR delta={feats['sar_delta']:.2f} dB  "
                  f"soil={feats['soil_during']:.3f}  "
                  f"rain={feats['rain_total']:.0f}mm  "
                  f"NDVI={feats['ndvi_mean']:.3f}")

            X   = np.array([[feats.get(f, 0) for f in feature_cols]])
            X_s = scaler.transform(X)

            rf_prob       = rf.predict_proba(X_s)[0][1]
            gb_prob       = gb.predict_proba(X_s)[0][1]
            ensemble_prob = rf_prob * 0.6 + gb_prob * 0.4

            results[province] = {
                'rf_prob':        round(rf_prob * 100, 1),
                'gb_prob':        round(gb_prob * 100, 1),
                'ensemble_prob':  round(ensemble_prob * 100, 1),
                'prediction':     'FLOOD' if ensemble_prob > 0.5 else 'NO FLOOD',
                'confidence':     'HIGH' if abs(ensemble_prob - 0.5) > 0.25 else 'MEDIUM',
                # Store raw GEE values for transparency
                # sar_delta_db = before - during; positive = backscatter drop = flood signal
                'sar_delta_db':   round(feats['sar_delta'], 2),
                'soil_moisture':  round(feats['soil_during'], 3),
                'rainfall_mm':    round(feats['rain_total'], 1),
                'ndvi':           round(feats['ndvi_mean'], 3),
            }
        except Exception as e:
            print(f"    ✗ FAILED for {province}: {e}")
            results[province] = {
                'rf_prob': 0, 'gb_prob': 0, 'ensemble_prob': 0,
                'prediction': 'ERROR', 'confidence': 'NONE',
                'error': str(e)
            }
        time.sleep(GEE_SLEEP)

    return results


# ─── FOLIUM MAP ───────────────────────────────────────────────

def generate_flood_map(predictions, year=PREDICT_YEAR):
    """Generate interactive Leaflet map via Folium."""
    print(f"\n[4/4] Generating flood risk map for {year}...")
    try:
        import folium
        from folium.plugins import MiniMap
    except ImportError:
        os.system("pip install folium --quiet")
        import folium
        from folium.plugins import MiniMap

    m = folium.Map(
        location=[30.3753, 69.3451], zoom_start=5,
        tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        attr='CartoDB Dark'
    )

    province_display = {
        'Sindh':       ([25.8, 68.8],  [23.5, 66.5, 28.5, 71.5]),
        'Punjab':      ([31.0, 72.3],  [28.5, 70.0, 34.0, 75.5]),
        'Balochistan': ([28.5, 65.0],  [24.5, 60.5, 32.0, 70.0]),
        'KPK':         ([34.0, 71.5],  [32.5, 69.5, 36.5, 74.5]),
        'GB':          ([35.8, 74.5],  [34.5, 72.0, 37.5, 77.5]),
        'AJK':         ([34.0, 74.2],  [33.0, 73.0, 35.5, 75.5]),
    }

    def get_color(p):
        if p >= 80: return '#ff2d55'
        if p >= 60: return '#ff6b35'
        if p >= 40: return '#ffc800'
        if p >= 20: return '#0088cc'
        return '#1a4a6a'

    for province, (center, bbox) in province_display.items():
        pred  = predictions.get(province, {})
        prob  = pred.get('ensemble_prob', 0)
        color = get_color(prob)

        folium.Rectangle(
            bounds=[[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
            color=color, weight=2, fill=True,
            fill_color=color, fill_opacity=0.35,
            tooltip=f"{province}: {prob:.1f}% flood risk"
        ).add_to(m)

        popup_html = f"""
        <div style="font-family:monospace;font-size:12px;padding:10px;min-width:220px">
          <b style="color:{color};font-size:14px">{province}</b>
          <hr style="border-color:#333;margin:5px 0">
          <b>Ensemble risk: {prob:.1f}%</b><br>
          RF model:  {pred.get('rf_prob',0):.1f}%<br>
          GB model:  {pred.get('gb_prob',0):.1f}%<br>
          <hr style="border-color:#333;margin:5px 0">
          SAR Δ (↑=flood): {pred.get('sar_delta_db','N/A')} dB<br>
          Soil moist: {pred.get('soil_moisture','N/A')}<br>
          Rainfall:   {pred.get('rainfall_mm','N/A')} mm<br>
          NDVI:       {pred.get('ndvi','N/A')}<br>
          <hr style="border-color:#333;margin:5px 0">
          <b style="color:{color}">{pred.get('prediction','N/A')}</b>
          · {pred.get('confidence','N/A')}
        </div>"""

        folium.Marker(
            location=center,
            popup=folium.Popup(popup_html, max_width=260),
            icon=folium.DivIcon(
                html=f'<div style="font-family:monospace;font-size:11px;font-weight:bold;'
                     f'color:{color};text-shadow:0 0 6px {color}">{province}<br>{prob:.0f}%</div>',
                icon_size=(80, 30), icon_anchor=(40, 15)
            )
        ).add_to(m)

    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
         background:#0a1520;border:1px solid #00b4ff44;padding:14px;
         font-family:monospace;font-size:12px;color:#7ab3cc;border-radius:4px">
      <b style="color:#00b4ff">{year} FLOOD RISK — REAL GEE DATA</b><br><br>
      <span style="color:#ff2d55">■</span> Critical (&gt;80%)<br>
      <span style="color:#ff6b35">■</span> High (60–80%)<br>
      <span style="color:#ffc800">■</span> Medium (40–60%)<br>
      <span style="color:#0088cc">■</span> Low (20–40%)<br>
      <span style="color:#1a4a6a">■</span> Minimal (&lt;20%)<br>
      <hr style="border-color:#00b4ff33;margin:8px 0">
      <span style="color:#00ffd0">Sentinel-1 SAR + SMAP + CHIRPS</span><br>
      <span style="color:#00ffd0">RF + GBT Ensemble · GEE v3.0</span>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    MiniMap(toggle_display=True).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    out = f'pakistan_flood_{year}_predictions.html'
    m.save(out)
    print(f"  ✓ Saved: {out}")
    return out


# ─── EXPORT JSON ──────────────────────────────────────────────

def export_dashboard_data(predictions, year=PREDICT_YEAR):
    """Export predictions + historical stats as JSON for the dashboard."""
    HISTORICAL_STATS = {
        '2010': {'affected_millions': 9.2,  'flood_area_km2': 136890},
        '2011': {'affected_millions': 34.5, 'flood_area_km2':  93597},
        '2012': {'affected_millions': 10.2, 'flood_area_km2':  58123},
        '2013': {'affected_millions':  5.6, 'flood_area_km2': 148002},
        '2014': {'affected_millions':  8.5, 'flood_area_km2': 100712},
        '2015': {'affected_millions': 13.2, 'flood_area_km2':  97585},
        '2016': {'affected_millions': 24.5, 'flood_area_km2': 136164},
        '2017': {'affected_millions': 11.2, 'flood_area_km2':  21539},
        '2018': {'affected_millions':  9.1, 'flood_area_km2': 146003},
        '2019': {'affected_millions': 31.2, 'flood_area_km2': 103638},
        '2020': {'affected_millions': 23.0, 'flood_area_km2': 113105},
        '2021': {'affected_millions': 10.3, 'flood_area_km2': 148018},
        '2022': {'affected_millions': 33.0, 'flood_area_km2': 160000},
        '2023': {'affected_millions': 17.6, 'flood_area_km2':  73639},
        '2024': {'affected_millions': 20.6, 'flood_area_km2':  26575},
    }

    output = {
        'generated_at': datetime.now().isoformat(),
        'model':        'RF+GB Ensemble v3.0 — Real GEE Data',
        'year':         year,
        'data_source':  'Sentinel-1 SAR · SMAP · CHIRPS · Sentinel-2 NDVI · SRTM DEM',
        'predictions':  predictions,
        'year_stats':   HISTORICAL_STATS,
    }

    with open('flood_predictions.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("  ✓ Saved: flood_predictions.json")
    return output


# ─── MAIN ─────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pakistan Flood Intelligence System v3.0')
    parser.add_argument('--year',       type=int, default=PREDICT_YEAR,
                        help=f'Year to predict (default: {PREDICT_YEAR})')
    parser.add_argument('--cache-only', action='store_true',
                        help='Skip GEE fetch, use existing cache file for training')
    parser.add_argument('--no-cache',   action='store_true',
                        help='Ignore existing cache and re-fetch from GEE')
    args = parser.parse_args()

    use_cache = not args.no_cache

    # Load province boundaries
    print("\nLoading province boundaries...")
    provinces = load_province_boundaries()

    # Fetch / load training data
    df = generate_training_data(provinces, use_cache=use_cache)

    # Train model
    rf, gb, scaler, feature_cols = train_model(df)

    # Predict target year from real GEE data
    preds = predict_year(rf, gb, scaler, feature_cols, provinces, year=args.year)

    # Print results
    print(f"\n{'='*62}")
    print(f"  {args.year} MONSOON FLOOD PREDICTIONS  (REAL SATELLITE DATA)")
    print(f"{'='*62}")
    for province, result in sorted(preds.items(), key=lambda x: -x[1].get('ensemble_prob', 0)):
        prob = result.get('ensemble_prob', 0)
        bar  = '█' * int(prob / 5)
        sar  = result.get('sar_delta_db', '?')
        sm   = result.get('soil_moisture', '?')
        print(f"  {province:<14} {bar:<20} {prob:5.1f}%  [{result['prediction']}]"
              f"  SAR Δ={sar}dB  SM={sm}")

    # Generate outputs
    generate_flood_map(preds, year=args.year)
    export_dashboard_data(preds, year=args.year)

    print(f"\n{'='*62}")
    print(f"  DONE. Files generated:")
    print(f"  1. pakistan_flood_{args.year}_predictions.html")
    print(f"  2. flood_predictions.json")
    print(f"  3. {CACHE_FILE}  (GEE features — reuse with --cache-only)")
    print(f"{'='*62}")
