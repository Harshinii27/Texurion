# Texture Analysis Using Local Binary Pattern - Manuscript Package

This folder contains an IEEE-style research paper generated from the **actual
implementation** in this repository. No metric, figure, or citation in the
paper was invented: everything is either read from the project's own source
code and stored model, computed by the project's own code, or cited from a
verified publication.

## Contents

| File | What it is |
|------|------------|
| `paper.tex` | Full IEEEtran (conference) manuscript. |
| `paper.pdf` | Compiled output of `paper.tex` (10 pages). |
| `make_figures.py` | Regenerates every figure and the measured JSON from project code. |
| `figures/` | PNG figures referenced by `paper.tex` (incl. `cv_methods.png`, `classifier_comparison.png`, `ablation.png`). |
| `experiments/benchmark_measured.json` | Raw measured latencies, environment, memory and engine-equivalence data. |
| `experiments/run_extended_eval.py` | Runs repeated cross-validation, GLCM baseline, classifier comparison, and feature-block ablation. |
| `experiments/extended_eval.json` | Raw output of the extended evaluation (source of the new result tables). |

## How to regenerate the figures and measurements

Use the project virtual environment (Windows PowerShell):

```powershell
& "venv314\Scripts\python.exe" "paper\make_figures.py"
```

This rewrites `figures/*.png` and `experiments/benchmark_measured.json`.
It uses only functions from `core/` and `classifier.py`, i.e. the same code
the web app calls.

To regenerate the extended evaluation (cross-validation, GLCM baseline,
classifier comparison, and ablation) and its three figures:

```powershell
& "venv314\Scripts\python.exe" "paper\experiments\run_extended_eval.py"
```

This rewrites `experiments/extended_eval.json`, `figures/cv_methods.png`,
`figures/classifier_comparison.png`, and `figures/ablation.png`.

> **Note on benchmark variance:** latencies are measured with
> `time.perf_counter` after a warm-up. On this shared machine the absolute
> values varied by up to roughly 3x between runs (background load), so only
> within-run comparisons are meaningful. The paper states this explicitly.
> If you regenerate, the new numbers will differ from the table in the
> current `paper.tex`; either re-run the numbers in the table or keep the
> committed JSON as the record. The paper text and the JSON must be kept
> consistent with each other.

## How to compile the paper

The manuscript targets the standard `IEEEtran` class (conference option).

1. Install a LaTeX distribution that includes `IEEEtran.cls`
   (TeX Live: `texlive-publishers`; MiKTeX installs it on demand).
2. From this folder run, twice:

    ```powershell
    pdflatex paper.tex
    pdflatex paper.tex
    ```

The compiled PDF (`paper.pdf`, 10 pages) was produced with **MiKTeX 25.12**
(installed on demand; `[MPM]AutoInstall=1` so `IEEEtran.cls` and other
packages are fetched automatically). The build is clean: no errors, no
undefined references, no overfull boxes. `hyperref` is loaded last, so the
DOIs, URLs and cross-references in the PDF are clickable while keeping IEEE
styling (`hidelinks`). To rebuild after editing:

```powershell
& "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode paper.tex
& "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode paper.tex
```

## Pre-submission verification (performed)

- **Experimental reproducibility:** re-running the pipeline via
  `experiments/run_extended_eval.py` reproduces the stored fold-level scores
  and the single-split confusion matrix exactly (17/17 checks). Full
  opponent-colour matrices were not re-extracted (each takes >12 min on this
  machine); their dimensions (`216`, `240`) were confirmed on a sample image
  and their stored statistics recompute exactly from the stored fold scores.
- **Statistical correctness:** all 70 stored means, standard deviations,
  95% confidence intervals and paired `t`/Wilcoxon results recompute exactly
  from the stored per-fold scores (`scipy.stats.ttest_rel` / `wilcoxon`; CI is
  a t-interval over the 10 independent repeat means, df = 9).
- **PDF formatting:** 10 pages; 8 figures embedded; 13 references and 25
  numbered labels; 11 distinct DOI URLs now clickable; no overfull boxes, no
  undefined references, no missing-character warnings.


## Verified vs. pending (what is trustworthy in the paper)

### Verified - measured or read directly from the project
- **Dataset:** `textures_upload1/` (and `textures.zip`) contains exactly 360
  JPEGs, 120 each in `banded/`, `dotted/`, `woven/`. The **public provenance
  of these images is not recorded anywhere in the repository**, and the
  paper says so.
- **Model (`texture_model.pkl`):** classes `banded, dotted, woven`;
  `feature_length = 54`; `n_samples = 360`; trained `2026-10-11T22:08:56`;
  train accuracy `0.8185`; test accuracy `0.7778` on `90` held-out samples
  (`test_size = 0.25`); model type `sklearn-svm-rbf` (RBF SVM wrapped in
  `CalibratedClassifierCV`, sigmoid). Confusion matrix
  `[[15,14,1],[0,29,1],[0,4,26]]`. Per-class report:
  banded `1.00/0.50/0.67`, dotted `0.62/0.97/0.75`, woven `0.93/0.87/0.90`;
  macro avg `0.85/0.78/0.77`.
- **Feature configuration of the model:** `method=uniform`, `engine=skimage`,
  scales `(1,8),(2,16),(3,24)`, Gabor and opponent-colour disabled,
  `max_side=512`, `descriptor_length=54`.
- **LBP numerical example:** the `3x3` patch and the resulting uniform code
  in the paper were computed with the project's own circular engine
  (`figures/lbp_example.png`).
- **Benchmark:** the table values are the medians from the recorded run in
  `experiments/benchmark_measured.json` (reps `3`, image `256x171`,
  Windows 11, Python 3.14.7, NumPy 2.5.4, OpenCV 5.0.0, scikit-image 0.26.0,
  SciPy 1.18.1, Numba 0.68.0). The NumPy vs. Numba engine equivalence check
  reported `equal = true`, `diff_count = 0`.
- **Library versions:** match `requirements.txt` (Streamlit 1.65.0, NumPy
  2.5.4, OpenCV 5.0.0.x, scikit-image 0.26.0, SciPy 1.18.1, Matplotlib
  3.11.2, Pillow 12.3.0, pandas 3.0.6, reportlab 5.0.1, scikit-learn 1.9.1,
  joblib 1.6.0, Numba 0.68.0, customtkinter 6.0.0; pytest 9.1.1 in
  `requirements-dev.txt`).
- **Software behaviour:** four-tab web app, ZIP path-traversal safeguards,
  61 passing tests, rule-based descriptor clearly labelled as *not* a
  trained classifier - all verified against the code.
- **Extended evaluation** (`experiments/extended_eval.json`, produced by
  `experiments/run_extended_eval.py` on the same 360 images):
  - Reproduction of the stored single split matches exactly: train
    `0.8185`, test `0.7778`, identical confusion matrix.
  - Repeated stratified 5-fold cross-validation (5x10 = 50 folds):
    LBP+SVM(RBF) `0.7650 +/- 0.0495` (95% CI `0.7563-0.7737`);
    GLCM+SVM(RBF) `0.6850 +/- 0.0468` (95% CI `0.6770-0.6930`); paired
    `t=9.477, p=1.16e-12`, Wilcoxon `p=1.38e-8`, mean difference `0.08`.
  - Classifier comparison on the same LBP features/folds: SVM-RBF
    `0.7650 +/- 0.0495`; SVM-linear `0.8103 +/- 0.0451`; RandomForest
    `0.7800 +/- 0.0519`; kNN(k=5) `0.7022 +/- 0.0529`. Paired tests vs.
    SVM-RBF: linear `p=1.15e-8`, RF `p=1.79e-2`, kNN `p=3.66e-9`.
  - Feature-block ablation (SVM-RBF, same folds): LBP only `0.7650`; +Gabor
    `0.8331`; +opponent `0.8678`; +Gabor+opponent `0.8794 +/- 0.0355`.
  - Descriptor dimensions: LBP `54`, GLCM `18`, LBP+Gabor `78`,
    LBP+opponent `216`, LBP+Gabor+opponent `240`.

### Pending experimental validation (intentionally NOT presented as results)
- Evaluation on standard public texture datasets (Brodatz, Outex, KTH-TIPS)
  and any cross-dataset generalisation.
- Comparison against deep texture representations (e.g., CNN features).
- Hyper-parameter optimisation of the SVM and the Gabor bank.
- Rotation / scale / illumination robustness sweeps.
- Benchmark latency on additional hardware/images.

### Reference list
All references cited in `paper.tex` were checked for correct title, venue,
volume/issue, pages and year; DOIs are given for the journal articles.
`paper.tex` uses only references that are actually cited and lists every
cited work. If you do not want to rely on any single source, remove both the
`\cite` and its `\bibitem`.

## Known implementation notes (fixed in this working tree)
- `core/benchmark.py` line ~88 had a bug: `(_measurers() - started)` called a
  function minus a float. It is now `(_measurers()() - started)`. The full
  test suite (61 tests) passes after the fix.
- `app.py` ROI reset uses an `on_click` callback
  (`_reset_roi`, setting `roi_x0/roi_y0/roi_x1/roi_y1`) instead of mutating
  widget keys after creation.
