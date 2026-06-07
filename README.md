# 🌊 Pakistan Flood Intelligence System

> **Real satellite data · RF+GB Ensemble v3.0 · Google Earth Engine**

A machine-learning pipeline that predicts monsoon flood risk across Pakistan's six provinces using live satellite imagery from Google Earth Engine (GEE). The system pulls multi-source remote sensing data, trains a Random Forest + Gradient Boosting ensemble on 10 years of labelled flood events (2015–2024), and produces province-level risk scores, an interactive Leaflet map, and a Streamlit dashboard.

---

## 📸 Output Preview

| Interactive Map | Streamlit Dashboard |
|---|---|
| Province risk overlays with SAR/SMAP/CHIRPS popups | Bar charts for risk %, rainfall, soil moisture, SAR delta |

---

## 🛰️ Data Sources

| Source | Variable | Resolution |
|---|---|---|
| **Sentinel-1 SAR** (Copernicus) | VV backscatter — water surface detection | 10 m (sampled at 1 km) |
| **NASA SMAP SPL4SMGP/007** | Surface soil moisture (m³/m³) | 9 km |
| **CHIRPS Daily** | Cumulative monsoon rainfall (mm) | 5 km |
| **Sentinel-2 SR / Landsat-8** | NDVI — vegetation/waterlogging index | 10–30 m |
| **USGS SRTM DEM** | Elevation & slope — flood accumulation risk | 30 m |

All data is fetched via the **Google Earth Engine Python API** for the monsoon window (July–September) and compared against a pre-monsoon baseline (April–June).

---

## 🤖 ML Pipeline

```
GEE Satellite Data
       │
       ▼
Feature Engineering (20 features per province-year)
  ├── SAR delta, p10 delta, image count
  ├── NDVI mean/min/std
  ├── Elevation mean/min/p10 + slope mean/p90
  ├── Soil moisture during/before/delta/max
  └── Rainfall total + max cell
       │
       ▼
RF + GBT Ensemble (0.6 RF · 0.4 GBT)
       │
       ▼
Province Risk Score (%) + Confidence Label
```

- **Training data:** 60 labelled province-year samples (2015–2024), ground-truth from NDMA, EM-DAT, ReliefWeb, UNOSAT
- **Train/test split:** 80/20, stratified; 5-fold cross-validation
- **Ensemble weights:** RF 60% + GBT 40%

---

## 🗂️ Project Structure

```
├── flood_backend.py              # GEE feature extraction + model training + prediction
├── dashboard.py                  # Streamlit dashboard
├── flood_predictions.json        # Latest predictions (auto-generated)
├── gee_features_cache.csv        # Cached GEE features (skip re-fetching)
└── pakistan_flood_YYYY_predictions.html  # Interactive Leaflet map (auto-generated)
```

---

## ⚡ Quick Start

### 1. Prerequisites

```bash
pip install earthengine-api scikit-learn pandas numpy folium streamlit plotly
```

Authenticate with Google Earth Engine:

```bash
earthengine authenticate
```

You'll need a GEE-enabled Google Cloud project. Set your project ID in `flood_backend.py`:

```python
GEE_PROJECT = 'your-gcp-project-id'
```

### 2. Run the ML Pipeline

```bash
# Full run — fetches from GEE, trains model, generates outputs (~15–45 min)
python flood_backend.py

# Predict a different year
python flood_backend.py --year 2024

# Skip GEE re-fetch, use cached CSV (fast)
python flood_backend.py --cache-only

# Force re-fetch, ignore cache
python flood_backend.py --no-cache
```

**Outputs generated:**
- `pakistan_flood_YYYY_predictions.html` — interactive map
- `flood_predictions.json` — risk scores for all provinces
- `gee_features_cache.csv` — cached satellite features

### 3. Launch the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard reads `flood_predictions.json` and renders province risk scores, bar charts for each satellite variable, and the raw JSON.

---

## 🏛️ Provinces Covered

| Province | Boundary Source |
|---|---|
| Sindh | FAO/GAUL 2015 Level-1 |
| Punjab | FAO/GAUL 2015 Level-1 |
| Balochistan | FAO/GAUL 2015 Level-1 |
| Khyber Pakhtunkhwa (KPK) | FAO/GAUL 2015 Level-1 |
| Gilgit-Baltistan (GB) | Custom polygon (GAUL incomplete) |
| Azad Jammu & Kashmir (AJK) | Custom polygon (GAUL incomplete) |

---

## ⚠️ Known Limitations

1. **`ee.Initialize()` called at module level** — runs before `argparse`, so the script crashes before `--help` can print if GEE auth is missing. Workaround: ensure GEE is authenticated before running.

2. **`os.system("pip install folium")` in `generate_flood_map()`** — installs `folium` at runtime if absent. Install it upfront via the prerequisites step instead.

3. **No `requirements.txt`** — all dependencies must be installed manually from the Quick Start command.

4. **GEE scale inconsistency** — SMAP is fetched at 11 km scale while SAR/CHIRPS use the global `GEE_SCALE=1000` (1 km). Fetching CHIRPS at 1 km is slower and consumes more GEE quota than necessary.

---

## 📊 Sample 2025 Predictions

| Province | Ensemble Risk | Prediction | Confidence |
|---|---|---|---|
| Sindh | 92.2% | FLOOD | HIGH |
| Balochistan | 87.8% | FLOOD | HIGH |
| KPK | 87.5% | FLOOD | HIGH |
| Punjab | 75.7% | FLOOD | HIGH |
| GB | 67.7% | FLOOD | MEDIUM |
| AJK | 64.2% | FLOOD | MEDIUM |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Satellite data | Google Earth Engine Python API (`earthengine-api`) |
| ML models | `scikit-learn` — `RandomForestClassifier`, `GradientBoostingClassifier` |
| Data wrangling | `pandas`, `numpy` |
| Map visualisation | `folium` + Leaflet.js |
| Dashboard | `streamlit` + `plotly` |
| Remote sensing | Sentinel-1 (ESA Copernicus), SMAP (NASA), CHIRPS (UCSB), Sentinel-2, SRTM (USGS) |

---

## 📖 Ground Truth Sources

Flood labels (2015–2024) were compiled from:
- **NDMA Pakistan** — National Disaster Management Authority situation reports
- **EM-DAT** — International Disaster Database (CRED)
- **ReliefWeb** — UN OCHA humanitarian reports
- **UNOSAT** — UN Satellite Centre flood extent maps

---

## 📄 License

MIT — see `LICENSE` for details.
