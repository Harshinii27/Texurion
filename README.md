# Texture Analysis Studio

Web + desktop application for texture analysis using **Local Binary Patterns (LBP)**
with optional **Gabor** and **opponent-colour LBP** fusion.

- **Web app (Streamlit):** `streamlit run app.py`
- **CLI:** `python cli.py images/brick.jpeg`
- **Desktop GUI (Tkinter, optional):** `python gui.py`

## Features

| Area | What it does |
|------|--------------|
| Analyzer | Upload an image, pick LBP radius/method, choose a **region of interest** (live overlay), enable multiscale LBP + Gabor + opponent-colour blocks, see the fused feature vector, metrics and rule-based descriptor |
| Models | Train a **real SVM classifier** (scikit-learn) on a labelled ZIP (`class_folder/image.jpg`), see measured train/test accuracy, save/load models |
| Batch | Process many images/ZIPs concurrently, per-file status + timing + error, CSV export |
| Performance | **Measured** latency/memory for every feature block (skimage vs custom NumPy vs custom Numba), with an equivalence check and PDF report |

## Project layout

```
core/
  io.py        image decode/validate, grayscale, resizing
  lbp.py       skimage LBP + circular custom engines (vectorized/numba/python)
  histogram.py normalised histograms, statistics, chi-square/cosine distances
  gabor.py     Gabor filter bank features and magnitude view
  pipeline.py  FeatureConfig, fused extraction, ROI, similarity, metrics
  batch.py     safe ZIP handling (path-traversal protection), concurrency, CSV
  benchmark.py real latency/RSS measurement and PDF report
app.py           Streamlit dashboard
cli.py           command-line single-image analysis
classifier.py    trained SVM (save/load/predict) + rule-based texture descriptor
analysis.py      desktop analysis orchestrator
report.py        PNG/PDF/CSV/JSON export helpers
gui.py           optional CustomTkinter desktop application
tests/           61 pytest tests
```

## Setup (Python 3.14 venv)

```powershell
py -3.14 -m venv venv314
.\venv314\Scripts\python -m pip install --upgrade pip
.\venv314\Scripts\python -m pip install -r requirements.txt
# optional: tests
.\venv314\Scripts\python -m pip install -r requirements-dev.txt
```

## Run

```powershell
# web app
.\venv314\Scripts\python -m streamlit run app.py

# CLI
.\venv314\Scripts\python cli.py images/brick.jpeg
.\venv314\Scripts\python cli.py images/brick.jpeg --radius 2 --method nri_uniform --json out.json

# desktop GUI (requires customtkinter)
.\venv314\Scripts\python gui.py
```

## Tests

```powershell
.\venv314\Scripts\python -m pytest -q
```

## Honesty & limitations (reading this matters)

- **No fabricated results.** Every metric, chart, accuracy, benchmark figure and
  probability in the UI is computed live from real inputs. Nothing is invented.
- **Rule-based descriptor is not a classifier.** The Analyzer's Smooth/Fine/
  Rough/Coarse label is a deterministic heuristic, always labelled "not a
  trained classifier".
- **Classification requires your data.** A real SVM only exists after you train
  on a labelled ZIP in the Models tab. With no model loaded, classification is
  disabled with an explicit notice; nothing pretends a model exists.
- **Consistent features.** Training and prediction use the same
  `FeatureConfig`. If your settings change the vector length, the app refuses
  to run an incompatible model instead of crashing (or silently mismatching).
- **Benchmark numbers are environment-specific.** They report measured medians
  on this machine and are labelled with the environment, image size and
  repetition count. Numba's speed advantage, if any, is only claimed where the
  equivalence check passes (bit-identical output vs the NumPy engine).
- **No MobileNet / pretrained-weight comparison** is present: no such model is
  bundled, and inventing one would violate the honesty requirement.
- **ZIP uploads are safe by construction:** member names are validated against
  path traversal, archives are decoded in memory and never extracted to disk.

## Docker

Build only after running locally; the image serves the Streamlit app.

```powershell
docker build -t texture-analysis .
docker run --rm -p 8501:8501 texture-analysis
```