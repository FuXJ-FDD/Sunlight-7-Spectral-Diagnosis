
# Sunlight-7 SDR2026 reproducibility package

This lightweight package reproduces the main calculations, clustering, sensitivity analyses, tables, and figures used in **“Spectral diagnosis of national SDG performance beyond linear aggregation.”**

## Tested software

- Python 3.13.5
- NumPy 2.3.5
- pandas 2.2.3
- SciPy 1.17.0
- scikit-learn 1.8.0
- Matplotlib 3.10.8
- openpyxl 3.1.5
- GeoPandas 1.1.2

## One-command reproduction

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

For a faster check of the pipeline, using fewer GMM initializations only for the K=2–10 diagnostic sweep:

```bash
python run_all.py --quick
```

To reproduce tables only:

```bash
python run_all.py --skip-figures
```

## Outputs

`outputs/Sunlight7_reproduction_results.xlsx` contains:

- country-level φ1–φ7 and Ωwhite values;
- rankings and reranking relative to the SDSN Index;
- four-archetype assignments, posterior probabilities, and uncertainty;
- GMM diagnostics;
- sensitivity-analysis results;
- Pearson correlation matrices;
- Figure S3 source data.

`outputs/figures/` contains Figures 1–7 and Figures S1–S4 in PNG, PDF, and SVG formats.

## Exact computational choices

The package limits BLAS/OpenMP thread pools to one thread for deterministic and lightweight execution.


1. Goal scores are divided by 100 to obtain normalized values in [0,1].
2. Missing goal-level values are not additionally imputed; each formula uses available constituent goals.
3. Population variance/standard deviation (`ddof=0`) is used inside φ5 and φ6.
4. GMM input is the seven-dimensional spectrum standardized across countries with `StandardScaler`.
5. Baseline GMM: four spherical components, `random_state=42`, `n_init=50`.
6. Input order is preserved from the SDR2026 workbook before GMM fitting.
7. Archetype names are assigned after fitting from component-level mean spectral profiles.
8. Sensitivity scenarios are classified with the baseline scaler and baseline four-component GMM, testing whether countries retain their baseline archetype.

## Reproducibility note

The country-level spectra reproduce the supplied reference workbook to machine precision. The pinned environment reproduces the four archetype counts (31, 11, 68, and 57). EM-based posterior probabilities can differ in the last decimals across BLAS libraries or future package versions; use the pinned versions above for closest reproduction.

## Data and map boundary note

The SDR2026 country-level workbook supplied for this analysis is included in `data/input/`. The low-resolution Natural Earth boundary files in `data/geodata/` are public-domain map data and are used only for Figures 2 and S1. Small island states not represented by the low-resolution polygons remain absent from the maps but are retained in all calculations.

## Validation

```bash
pytest -q
```

The tests verify the seven-dimensional scores against the reference workbook and verify the four archetype sample sizes.
