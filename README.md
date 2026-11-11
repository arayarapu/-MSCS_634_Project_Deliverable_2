# MSCS 634 — IPL Cricket Analytics, Deliverable 2

## Objective and dataset
Predict first-innings IPL runs from information available at the start of the innings. This builds on [Deliverable 1](https://github.com/arayarapu/MSCS_634_Project_Deliverable_1) with the same public IPL source dataset. The data were accessed on **2026-11-11 21:20 UTC**.

- [Match CSV](https://raw.githubusercontent.com/elavarasan-ai/IPL-Data-Analysis/main/matches.csv) — **1,095 rows, 20 attributes**.
- [Delivery CSV](https://raw.githubusercontent.com/elavarasan-ai/IPL-Data-Analysis/main/deliveries.csv) — **260,920 rows, 17 attributes**.
- Modeling subset: **1,061** ordinary, completed matches across **2008–2024**; **991** train and **70** held-out **2024** records.
- Data provenance: [elavarasan-ai/IPL-Data-Analysis](https://github.com/elavarasan-ai/IPL-Data-Analysis); original match and ball-by-ball source as described by that repository. No source CSVs are committed.

## Cleaning and feature engineering
1. Load both remote CSVs; preserve original copies and validate schema and row counts.
2. Normalize whitespace and historic franchise name changes; distinguish renamed franchises from distinct teams. Validate match IDs, dates, scores, and duplicate rows.
3. Aggregate `inning == 1` delivery runs to one target per match. Exclude matches affected by declared shortened targets, weather-adjustment methods, no result, missing date/score or contradictory team records. Fully bowled-out innings are retained.
4. Build pre-innings features: `season_year`, `month`, `batting_first`, `bowling_first`, `venue`, `city`, `toss_decision`, and `match_type`.
5. Impute, scale numeric features, and one-hot encode categories **within pipelines fit only to the training data**. Final match winner, scores, wickets, target runs, and target overs are **never used as predictors**.

### Cleaning observations
- Raw duplicate match IDs: **0**; raw exact duplicate delivery rows: **0**.
- Invalid match IDs: **0**; invalid delivery IDs: **0**; invalid dates: **0**.
- Unknown delivery match references: **0**; negative delivery runs: **0**; match records without first innings or excluded by regular-match criteria: **34** after first-innings join.

## Models and evaluation
All models use an **expanding-window 5-fold TimeSeriesSplit** on earlier matches. The newest calendar season (**2024**) is a final untouched test set. Models: mean baseline, Linear Regression, Ridge Regression (L2, alpha=50), and Random Forest Regression (200 trees; min leaf=3). The model selected for discussion is the one with the lowest CV RMSE; no hyperparameters are tuned on the test season.

### Regression performance (runs unless otherwise indicated)
| Model             |   CV_RMSE_mean |   CV_RMSE_std |   CV_MAE_mean |   CV_R2_mean |   Test_R2 |   Test_MSE |   Test_RMSE |   Test_MAE |
|:------------------|---------------:|--------------:|--------------:|-------------:|----------:|-----------:|------------:|-----------:|
| Ridge Regression  |         30.228 |         0.855 |        23.767 |       -0.039 |    -0.087 |    1597.62 |      39.97  |     30.227 |
| Mean baseline     |         30.551 |         1.548 |        23.955 |       -0.059 |    -0.437 |    2112.19 |      45.959 |     35.239 |
| Random Forest     |         31.104 |         1.006 |        24.474 |       -0.099 |    -0.047 |    1538.57 |      39.225 |     30.355 |
| Linear Regression |         32.465 |         1.399 |        25.532 |       -0.203 |    -0.126 |    1655.95 |      40.693 |     31.692 |


### Results and discussion — measured from the dataset

- **Dataset/model population:** 1,095 original match records and 260,920 delivery records; **1,061** matches meet the first-innings and regular-match inclusion criteria. The year range is 2008–2024. Model development contains 991 earlier matches and the untouched **2024** season contains 70 matches.
- **Feature engineering:** Models use **8 pre-innings fields** (2 numeric, 6 categorical). One-hot encoding, missing-value handling, and scaling occur inside every CV fold's training pipeline.
- **Cross-validation:** **Ridge Regression** had the lowest mean expanding-window CV RMSE at **30.23 ± 0.85 runs** across five validation folds. Its average CV R² was **-0.039**.
- **Unseen-season evaluation:** The CV-selected model's held-out **R² = -0.087**, **MSE = 1597.61**, **RMSE = 39.97 runs**, and **MAE = 30.23 runs**. The baseline RMSE was **45.96 runs**, a **+13.0% reduction** in RMSE for the selected model (negative means worse than baseline).
- **Model comparison:** **Random Forest** had the lowest test RMSE of **39.22 runs**, but the choice of model was determined using earlier-season CV rather than revising the decision after looking at the test season.
- **Scoring shift:** Average first-innings totals changed from **164.72 runs** in training years to **190.06 runs** in the held-out year. The selected model's mean residual was **+13.48 runs** (actual minus predicted), which shows the direction of average forecast bias.
- **Interpretation and limitations:** Lower RMSE means more accurate run forecasts; R² shows how much unseen-season variance the model explains relative to the test mean. First-innings runs depend on unrecorded pitch quality, weather, player availability, tactical changes, and squad composition, so pre-innings attributes alone may have limited forecasting power. Time-based testing is harder and more realistic than randomly splitting future and past IPL matches. Do **not** claim causation from team/venue coefficients.


## Charts and result files
- `screenshots/01_cv_rmse.png`: mean ± SD across chronological CV folds.
- `screenshots/02_holdout_errors.png`: test RMSE and MAE by model.
- `screenshots/03_actual_vs_predicted.png`: actual vs predicted runs for CV-selected model.
- `screenshots/04_residuals.png`: residual bias and error pattern.
- `screenshots/05_absolute_error_spread.png`: test absolute error distributions.
- `results/cross_validation_metrics.csv`, `results/model_comparison.csv`, `results/holdout_predictions.csv`: measured results for reproducibility.

## Challenges, decisions, and limitations
- **Season labels** in source may be strings such as `2007/08`; derive calendar year from the match date, ensuring consistent chronological splitting.
- **Franchise names** evolve; standardize clear name changes, but do not merge historically distinct franchises.
- **Rain and abandoned matches:** exclude declared shortened/adjusted games for the intended standard-match target; retain plausible extremely high/low totals rather than deleting them by a generic outlier cutoff.
- **Leakage:** one row per match; only pre-innings predictors; a chronological holdout and in-fold transformers prevent information from unseen seasons affecting training.
- **Unobserved factors:** player availability, weather, ground conditions and match strategy may prevent accurate point predictions. Cross-validation and holdout performance are reported without claiming causal interpretation.

## Reproduce on Mac (internet connection needed)
```bash
python3 -m pip install -r requirements.txt
python3 -m jupyter lab
```
Open `IPL_Deliverable_2_Regression.ipynb`, select **Run → Run All Cells**, and verify the notebook writes this README plus the chart/metrics outputs. If macOS cannot validate certificates, install system Python certificates or `certifi`; the notebook includes a verified-SSL retry and never disables validation.

## Submission: 
executed notebook 
README
requirements.txt
screenshots/ and
results/ 
