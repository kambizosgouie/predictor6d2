# General Property Predictor 6d2 — User and Technical Manual

This manual describes **6d2.py**, the extended version of the General Property Predictor. It covers how to run the app, how to use every tab, and the technical details of every method, model and metric, followed by test datasets you can download.

The quick-start commands are in section 3.

---

## Contents

1. [What the app does](#1-what-the-app-does)
2. [Files and requirements](#2-files-and-requirements)
3. [Running the app](#3-running-the-app)
4. [Data requirements](#4-data-requirements)
5. [Using the app](#5-using-the-app)
6. [Technical reference: training pipeline](#6-technical-reference-training-pipeline)
7. [Technical reference: models](#7-technical-reference-models)
8. [Technical reference: metrics](#8-technical-reference-metrics)
9. [Technical reference: correlation and association](#9-technical-reference-correlation-and-association)
10. [Technical reference: feature importance](#10-technical-reference-feature-importance)
11. [Technical reference: SHAP](#11-technical-reference-shap)
12. [Technical reference: partial dependence](#12-technical-reference-partial-dependence)
13. [Reading the results: practical guidance](#13-reading-the-results-practical-guidance)
14. [Test datasets](#14-test-datasets)
15. [Reference results](#15-reference-results)
16. [Limitations](#16-limitations)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. What the app does

You upload a table of numbers (CSV or Excel). You choose which columns are **inputs** (also called features) and which are **targets** (the quantities to predict). The app then:

- trains **14 regression models** separately for each target;
- reports how accurate each model is on data it did not train on;
- explains which inputs matter and how they influence the target, using correlation methods, permutation importance, SHAP values and partial dependence;
- lets you enter a new set of input values and see every model's prediction side by side.

It works on any numeric table.

---

## 2. Files and requirements

| File | Purpose |
|------|---------|
| `6d2.py` | The Streamlit app (run this one) |
| `analysis_utils.py` | Correlation, VIF, permutation importance, standardised coefficients and partial dependence helpers. **Required by 6d2.** |
| `data_utils.py` | File reading and numeric cleaning |
| `metrics_utils.py` | Accuracy metrics |
| `requirements.txt` | Python package list |
| `tests/` | Unit tests for the helper modules |

**Python 3.10 or newer.** The app uses Streamlit, pandas, NumPy, scikit-learn, matplotlib, XGBoost, CatBoost, SHAP and openpyxl (see `requirements.txt`).

---

## 3. Running the app

From the project folder (the one containing `6d2.py`):

**Windows**

```
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run 6d2.py
```

**Mac / Linux**

```
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m streamlit run 6d2.py
```

The app opens at `http://localhost:8501`. Press **Ctrl + C** in the terminal to stop it.

### Deploying on Streamlit Community Cloud

1. Put `6d2.py`, `analysis_utils.py`, `data_utils.py`, `metrics_utils.py` and `requirements.txt` in a GitHub repository.
2. In Streamlit Cloud, choose the repository, branch and **main file `6d2.py`**.
3. Under **Advanced settings**, choose **Python 3.12** if the default Python fails to install packages.

First training can take a minute or more on the free tier because several models (CatBoost, the neural network, SHAP) are compute-heavy.

---

## 4. Data requirements

- **Formats:** `.csv`, `.xlsx`, `.xlsm`. For Excel files, choose the worksheet in the sidebar.
- **Layout:** one observation per row; the first row holds the column names.
- **Numeric columns only.** Every column is converted with `pandas.to_numeric(errors="coerce")`. Text columns become empty and are dropped from the selectable list. Infinite values are treated as missing.
- **Duplicate column names** are renamed automatically (`name`, `name (2)`, …). Blank names become `Column N`.
- **Missing values:** rows with a missing value in any *selected* input or target are removed. Columns you did not select do not matter.
- **Minimum size:** at least **10 complete rows**. The **Train Models** button stays disabled below that. More is better; several dozen rows or more gives meaningful test scores.
- **A column cannot be both an input and a target.**
- **Default selection:** the last numeric column is the default target and the rest are inputs.

---

## 5. Using the app

### 5.1 Sidebar

1. Upload a file.
2. Check the preview (first 10 rows).
3. Choose **Input variables** and **Target variables to predict**. The sidebar shows how many rows are usable.
4. Click **Train Models**. Changing the file or the selection requires training again.

Trained models are cached, so switching tabs or dropdowns does not retrain them.

### 5.2 Tabs

| Tab | Contents |
|-----|----------|
| **Feature Analysis** | Correlation, importance, SHAP and partial dependence for each target |
| **Model Comparison** | Ranked table and bar chart of test scores for all 14 models, per target |
| **Predict** | Enter new input values and get every model's prediction |
| **One tab per model** (14) | Metrics, fitted formulas (linear-type models), coefficients (Lasso/Elastic Net), parity plots |

### 5.3 Feature Analysis tab (per target)

1. **Correlation and association** — five methods plus an input-to-input heatmap and VIF table.
2. **Feature importances** — consensus ranking, permutation importance per model, built-in tree importances, standardised coefficients.
3. **SHAP values** — for a model you choose; bar, beeswarm, dependence and waterfall plots.
4. **Partial dependence** — one curve per input.

### 5.4 Figures: thumbnails and full screen

Each figure appears as a medium-sized thumbnail. Click **⛶ Full screen — <figure name>** to open a larger version in a pop-up. Close it with the × or by pressing Esc. Hovering over an image also shows Streamlit's own fullscreen icon.

### 5.5 Predict tab

Each input is pre-filled with the **median** of that column. Enter your values and click **Predict**. The table lists one row per model and one column per target.

Predictions are only trustworthy **inside the range of your training data**. See [Limitations](#16-limitations).

### 5.6 Individual model tabs

- **Metrics table** for each target: test and full-data versions of R², MAE, RMSE, MAPE, AAD and MAE as a percentage of the mean (see [Metrics](#8-technical-reference-metrics)).
- **Fitted formulas** for Linear, Polynomial (degree 2) and Ridge regression, written in terms of your column names.
- **Standardised coefficients** for Lasso and Elastic Net.
- **Parity plots:** predicted versus true values, for the training+test data combined. Points on the red dashed diagonal are perfect predictions.

---

## 6. Technical reference: training pipeline

1. The file is read and converted to numeric columns.
2. Selected inputs and targets are kept and rows with any missing value are removed.
3. The rows are split **80% train / 20% test** with `train_test_split(test_size=0.2, random_state=42)`. The same split is used for every model and every target, so results are reproducible and directly comparable.
4. For **each target separately**, all 14 models are fitted on the training rows only.
5. "Test" metrics use the held-out 20%. "All" metrics use train and test rows together and therefore look better, because they include data the model has seen.

There is no cross-validation and no hyperparameter search. Every model uses fixed, sensible defaults (next section). On small datasets, the single 80/20 split makes test scores noisy: a different random split could change rankings between close models.

---

## 7. Technical reference: models

All random seeds are fixed at 42.

| # | Model | Configuration | Notes |
|---|-------|---------------|-------|
| 1 | Linear Regression | ordinary least squares | Interpretable; cannot capture curvature |
| 2 | Polynomial Regression deg2 | degree-2 polynomial features (no bias column) → linear regression | Adds squares and pairwise products of inputs; can extrapolate wildly |
| 3 | Random Forest | 200 trees | Averages many deep trees; cannot extrapolate outside the training range |
| 4 | Gradient Boosting | 300 trees, learning rate 0.05, depth 3 | scikit-learn implementation |
| 5 | XGBoost | 300 trees, learning rate 0.05, depth 4, subsample 0.9, column sample 0.9 | |
| 6 | Ridge Regression | L2 penalty, alpha 1.0 | On the raw (unscaled) inputs |
| 7 | KNN | 5 neighbours, distance-weighted | Inputs are **not** scaled, so large-valued columns dominate distances |
| 8 | Neural Network | inputs standardised; MLP with hidden layers (64, 32), ReLU, L2 alpha 0.05, learning rate 0.001, up to 2000 iterations; target also standardised and transformed back | |
| 9 | CatBoost | 300 iterations, learning rate 0.05, depth 6 | |
| 10 | Lasso Regression | inputs and target standardised; L1 penalty alpha 0.01 | Drives unhelpful coefficients to exactly zero |
| 11 | Elastic Net | inputs and target standardised; alpha 0.01, L1 ratio 0.5 | Mix of L1 and L2 penalties |
| 12 | SVR (RBF) | inputs and target standardised; RBF kernel, C = 10, epsilon = 0.05 | Smooth non-linear fit |
| 13 | Extra Trees | 300 trees | Like Random Forest but with randomised split thresholds |
| 14 | HistGradientBoosting | up to 300 iterations, learning rate 0.05, minimum 3 samples per leaf | Fast gradient boosting on binned inputs |

### Model families

- **Linear-type** (Linear, Ridge, Lasso, Elastic Net, Polynomial): produce an explicit formula; stable on smooth, roughly linear trends; unreliable outside the data range.
- **Tree-based** (Random Forest, Gradient Boosting, XGBoost, CatBoost, Extra Trees, HistGradientBoosting): capture non-linearities and interactions automatically; their predictions are bounded by values seen in training, so they never extrapolate.
- **Others** (KNN, Neural Network, SVR): flexible, distance-based or smooth; sensitive to input scaling (handled for the neural network, Lasso, Elastic Net and SVR, but not KNN).

### Why scaling matters for some models

Lasso, Elastic Net and SVR use penalties or distances that depend on the *size* of each input. Without scaling, an input measured in the thousands would dominate one measured in fractions. The app therefore standardises (subtracts the mean, divides by the standard deviation) inside these models. Predictions are converted back to your original units automatically.

---

## 8. Technical reference: metrics

With true values `y`, predictions `ŷ` and `n` samples:

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| **R²** | `1 − Σ(y − ŷ)² / Σ(y − ȳ)²` | 1 is perfect; 0 equals always predicting the mean; **negative** means worse than the mean |
| **MAE** | `mean(abs(y − ŷ))` | Average error in the target's own units |
| **RMSE** | `sqrt(mean((y − ŷ)²))` | Like MAE but penalises large errors more; never smaller than MAE |
| **MAPE (%)** | `mean(abs((y − ŷ) / y)) × 100`, over rows where `y ≠ 0` | Average relative error |
| **AAD (%)** | Same formula as MAPE in this app | Reported under both names for users who use either convention |
| **MAE% of Mean** | `MAE / mean(y) × 100` | MAE relative to the average target value; `NaN` if the mean is zero |

Practical notes:

- Rows where the true value is exactly 0 are skipped in MAPE/AAD. If many targets are zero or near zero, MAPE becomes huge or misleading; use MAE and RMSE instead.
- Compare **test** metrics across models. "All" metrics are optimistic, especially for flexible models like KNN, which can reproduce training rows exactly.
- A large gap between "R² (all)" and "R² (test)" indicates overfitting.

---

## 9. Technical reference: correlation and association

Shown in **Feature Analysis → 1. Correlation and association**, computed on all usable rows (train + test). The signed methods go in the heatmap; the unsigned ones go in the bar chart. All values are also listed in a table.

| Method | Range | Detects | Notes |
|--------|-------|---------|-------|
| **Pearson** | −1 to 1 | Linear relationships | Sensitive to outliers |
| **Spearman** | −1 to 1 | Monotonic relationships (always rising or always falling) | Pearson on ranks; robust to outliers and curved-but-monotonic trends |
| **Kendall** (τ) | −1 to 1 | Monotonic relationships | Based on concordant/discordant pairs; more robust on small samples; typically smaller in magnitude than Spearman for the same data |
| **Distance correlation** | 0 to 1 | **Any** dependence, including non-monotonic (e.g. U-shapes) | 0 only if the variables are independent. Unsigned. Computed from double-centred pairwise distance matrices (Székely et al.), so memory use grows with the square of the number of rows |
| **Mutual information** | 0 to ∞ (nats) | **Any** dependence | Estimated with scikit-learn's k-nearest-neighbour estimator (`mutual_info_regression`, seed 42). Not normalised, so compare features **within** one target only; estimates are noisy on small samples |

**How to read the combination:**

- Pearson high, Spearman high → clean linear/monotonic relationship.
- Spearman much higher than Pearson → monotonic but curved or outlier-affected (for example `age` vs concrete strength).
- Pearson near 0 but distance correlation or mutual information clearly above 0 → a non-linear, non-monotonic relationship that Pearson cannot see.

### Input-to-input correlation and VIF

A second heatmap shows Pearson correlation **between inputs**. Strongly correlated inputs make importance measures share credit between them.

The **variance inflation factor** quantifies this. For each input, it is regressed on all the other inputs (ordinary least squares with an intercept) and:

```
VIF = 1 / (1 − R²)
```

where R² comes from that regression. Rules of thumb: **VIF < 5** is fine; **5–10** is moderate; **> 10** signals serious multicollinearity (an input that is almost a linear combination of the others). VIF is shown only when there are at least two inputs, and is reported as infinite for perfectly collinear inputs.

---

## 10. Technical reference: feature importance

### 10.1 Permutation importance (per model)

Computed **on the held-out test set** for every one of the 14 models:

1. Measure the model's test R².
2. Randomly shuffle one input column, breaking its link to the target.
3. Measure R² again; the **drop** is that input's importance.
4. Repeat 10 times with different shuffles and average (seed 42), then repeat for every input.

Properties:

- Works for **any** model, which makes importances comparable across model types.
- Based on *predictive* contribution to unseen data, not on how the model happens to use a feature internally.
- Can be slightly negative: shuffling made the model marginally *better*, meaning the input contributes only noise.
- With correlated inputs, shuffling one still leaves its twin intact, so importance can appear **split or understated**.
- The test set is only 20% of your rows; small test sets give noisy values. A heatmap shows models as rows and inputs as columns.

### 10.2 Consensus importance

For each model, negative values are clipped to 0 and the scores are divided by that model's total so they sum to 1. The consensus is the **mean across all models**. It highlights features that matter regardless of the algorithm, and reduces the influence of any single poorly-fitted model. A model whose importances are all zero is ignored in the average.

Models that fit poorly produce unreliable importances. If a few models are clearly bad (test R² near or below 0), treat the consensus with caution, or compare it with the permutation heatmap restricted to your best models.

### 10.3 Built-in tree importances

For Random Forest, Gradient Boosting, XGBoost, CatBoost and Extra Trees the app plots each model's own `feature_importances_`, which measure how much each input reduces prediction error across the splits of the trees. They are fast and familiar but **biased toward inputs with many distinct values** and share credit among correlated inputs. Prefer permutation importance or SHAP when they disagree.

### 10.4 Standardised coefficients (linear-type models)

For Linear, Ridge, Lasso and Elastic Net, the chart shows the change in the target, **in target standard deviations**, for a **one-standard-deviation increase** in an input, holding the others fixed.

- Linear and Ridge: `coefficient × std(input) / std(target)` (computed from the raw coefficients).
- Lasso and Elastic Net: the coefficients of the internally standardised model.

This puts inputs measured in different units on one comparable scale. The sign gives direction. A value of exactly 0 for Lasso or Elastic Net means the penalty removed that input.

---

## 11. Technical reference: SHAP

SHAP (SHapley Additive exPlanations) assigns each input a contribution to each individual prediction, based on Shapley values from cooperative game theory. For one row:

```
prediction = base value + sum of SHAP values of all inputs
```

The base value is the average model output. A positive SHAP value pushed that prediction **above** average; a negative one pushed it **below**. SHAP values are in the **units of the target**.

### 11.1 Choosing the explained model

Below the importance section, a dropdown selects which of the 14 models to explain. It defaults to the model with the highest test R². Each of the targets has its own dropdown.

### 11.2 Explainer used

| Model type | Explainer | Rows explained | Exact? |
|-----------|-----------|----------------|--------|
| Random Forest, Gradient Boosting, XGBoost, CatBoost, Extra Trees | `TreeExplainer` | up to 300 random rows of train+test | Yes |
| Linear Regression, Ridge | `LinearExplainer` | all rows of train+test | Yes |
| Everything else (Polynomial, KNN, Neural Network, Lasso, Elastic Net, SVR, HistGradientBoosting) | Model-agnostic permutation explainer using the model's `predict` | up to 60 random rows of train+test | Approximate, sampling-based |

The model-agnostic explainer uses a background sample of up to 50 training rows to represent "feature absent". The caption under the SHAP heading states which explainer ran and how many rows were explained. Row sampling uses seed 42.

Because SHAP is computed over train and test rows together, it describes **how the model behaves on your data**, not how well it generalises. Use test R² for generalisation.

### 11.3 Plots

- **Bar plot:** mean absolute SHAP value per input — overall importance in target units.
- **Beeswarm:** every row is a dot per input. Horizontal position is the SHAP value; colour is the input's value (red high, blue low). A cloud of red dots on the right means high values push predictions up.
- **Dependence plot:** SHAP value of one chosen input against its value; colours show the strongest interacting input. Vertical spread at a given x-value indicates interactions with other inputs.
- **Waterfall:** step-by-step breakdown of one chosen row from the base value to its prediction. Choose the row number with the numeric field (row numbers refer to the explained sample, not the original file).

---

## 12. Technical reference: partial dependence

Shown in **Feature Analysis → 4. Partial dependence**, for the model selected in the SHAP dropdown.

For each input:

1. Take 25 values spanning the input's observed range (evenly spaced quantiles of all rows; duplicate values are merged).
2. For each value, set that input to it **for every row** in the data, leaving the other inputs as observed.
3. Predict, then average the predictions.
4. Plot the averages against the value.

The curve shows the model's average response to one input in isolation: rising, falling, flat, saturating, or U-shaped. Cautions:

- Averaging hides interactions; a flat curve can mean no effect **or** opposing effects in different parts of the data.
- With strongly correlated inputs, the sweep creates input combinations that never occur in reality, where the model's behaviour is unconstrained. Check VIF first.
- Tree models give step-like curves; smooth models give smooth ones. This is a property of the model, not of reality.

---

## 13. Reading the results: practical guidance

A sensible order for an analysis session:

1. **Model Comparison tab.** Which models reach a high test R²? If even the best is poor (below roughly 0.5), your inputs may not contain enough information, or there are too few rows.
2. **Check for overfitting** in each model tab: a big gap between "R² (all)" and "R² (test)".
3. **Parity plots.** Look for systematic curvature or a funnel shape (errors growing with value), which RMSE alone hides.
4. **Correlation tables.** Compare Pearson and Spearman; look for non-linear signals in distance correlation and mutual information.
5. **VIF.** If several inputs exceed 5–10, interpret individual importances cautiously.
6. **Consensus and permutation importance**, then **SHAP** on the best model, to see *how* the top inputs act.
7. **Partial dependence** to see the shape of each effect.
8. **Predict tab** for new conditions, staying within the training range.

Typical patterns:

| Observation | Likely meaning |
|-------------|----------------|
| Tree models ≫ linear models | Strong non-linearity or interactions |
| Polynomial deg 2 ≫ Linear | Mostly smooth curvature |
| Linear ≈ trees | Relationship is close to linear |
| KNN test R² far below "all" R² | KNN is memorising; also check unscaled inputs |
| Lasso zeroes an input | Little extra information beyond the other inputs |
| Negative R² | Model is worse than predicting the mean; poor fit, tiny test set, or extrapolation |

---

## 14. Test datasets

These public datasets let you try the app. Download each file (see below), then upload it in the sidebar.

### 14.1 Public datasets (verified download links)

| Dataset | Direct link | Suggested inputs | Suggested target | Rows | Notes |
|---------|-------------|------------------|------------------|------|-------|
| **Concrete compressive strength** (recommended) | [concrete.csv](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv) | `cement`, `slag`, `ash`, `water`, `superplastic`, `coarseagg`, `fineagg`, `age` | `strength` | 1,030 | All numeric; non-linear; ideal for the full feature analysis |
| **Auto MPG** | [mpg.csv](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv) | `cylinders`, `displacement`, `horsepower`, `weight`, `acceleration`, `model_year` | `mpg` | 398 | A few missing `horsepower` values test the missing-row handling (expect 392 usable rows); text columns `origin` and `name` are ignored |
| **Boston housing** | [BostonHousing.csv](https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv) | `crim`, `nox`, `rm`, `age`, `dis`, `tax`, `lstat` | `medv` | 506 | Strongly correlated inputs: good for VIF |
| **Medical insurance charges** | [insurance.csv](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv) | `age`, `bmi`, `children` | `charges` | 1,338 | Text columns (`sex`, `smoker`, `region`) are dropped automatically. The most important driver, `smoker`, is text, so scores will be modest. To include it, convert yes/no to 1/0 in a spreadsheet first |
| **Cars93 (with missing values)** | [Cars93_miss.csv](https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv) | `Horsepower`, `Weight`, `EngineSize`, `Length`, `Wheelbase` | `Price` | 93 | Small and has gaps; tests behaviour on limited data |

The Boston housing dataset is widely used for teaching but contains a variable (`b`) that is now considered ethically problematic. It is suggested here only as a test of the software; `b` is excluded from the suggested inputs.

### 14.2 How to download them

**PowerShell (Windows):**

```powershell
mkdir test-data; cd test-data
curl.exe -L -o concrete.csv https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv
curl.exe -L -o mpg.csv https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv
curl.exe -L -o BostonHousing.csv https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv
curl.exe -L -o insurance.csv https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv
curl.exe -L -o Cars93_miss.csv https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv
```

Use `curl.exe`, not `curl`, because plain `curl` is an alias for another command in PowerShell.

**Mac / Linux:**

```bash
mkdir test-data && cd test-data
curl -L -O https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv
curl -L -O https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv
```

**Browser:** open the link, press **Ctrl + S**, and save as a `.csv` file (set "Save as type" to *All files* so the extension is not changed).

### 14.3 Quick end-to-end check with the concrete dataset

1. Upload `concrete.csv`.
2. Select the eight columns other than `strength` as inputs and `strength` as the target.
3. Click **Train Models**.
4. In **Predict**, enter the first row of the file in the order of the columns shown: `540, 0, 0, 162, 2.5, 1040, 676, 28`. The true strength is 79.99; tree models should predict roughly 70–75 and linear models about 52.

Take care to enter the values into the matching fields. Putting values in the wrong fields (for example, water = 676) creates a physically impossible mixture, and the linear and polynomial models may then give negative strengths.

---

## 15. Reference results

Measured with 6d2 on `concrete.csv` (1,030 rows; 8 inputs; target `strength`; 80/20 split, seed 42). Your numbers should match these to the displayed precision when using the same file and settings.

**Test-set accuracy**

| Model | R² (test) | RMSE | MAE |
|-------|-----------|------|-----|
| HistGradientBoosting | 0.919 | 4.57 | 2.99 |
| CatBoost | 0.915 | 4.67 | 3.29 |
| XGBoost | 0.914 | 4.70 | 3.31 |
| Gradient Boosting | 0.902 | 5.03 | 3.70 |
| Extra Trees | 0.893 | 5.26 | 3.38 |
| Random Forest | 0.881 | 5.53 | 3.78 |
| Neural Network | 0.874 | 5.71 | 4.04 |
| SVR (RBF) | 0.874 | 5.71 | 3.97 |
| KNN | 0.805 | 7.09 | 5.15 |
| Polynomial Regression deg2 | 0.784 | 7.46 | 5.97 |
| Ridge Regression | 0.628 | 9.80 | 7.75 |
| Linear Regression | 0.628 | 9.80 | 7.75 |
| Elastic Net | 0.626 | 9.82 | 7.84 |
| Lasso Regression | 0.624 | 9.85 | 7.87 |

**Association with `strength`**

| Input | Pearson | Spearman | Kendall | Distance corr. | Mutual info |
|-------|---------|----------|---------|----------------|-------------|
| cement | 0.498 | 0.478 | 0.327 | 0.458 | 0.311 |
| slag | 0.135 | 0.164 | 0.119 | 0.167 | 0.178 |
| ash | −0.106 | −0.078 | −0.060 | 0.119 | 0.122 |
| water | −0.290 | −0.308 | −0.206 | 0.354 | 0.360 |
| superplastic | 0.366 | 0.348 | 0.250 | 0.333 | 0.212 |
| coarseagg | −0.165 | −0.184 | −0.124 | 0.191 | 0.255 |
| fineagg | −0.167 | −0.180 | −0.122 | 0.186 | 0.214 |
| age | 0.329 | 0.596 | 0.449 | 0.458 | 0.359 |

**Multicollinearity (VIF)**

| cement | slag | ash | water | superplastic | coarseagg | fineagg | age |
|--------|------|-----|-------|--------------|-----------|---------|-----|
| 7.49 | 7.28 | 6.17 | 7.00 | 2.96 | 5.08 | 7.01 | 1.12 |

What this illustrates:

- Tree and boosting models clearly beat the linear ones (0.88–0.92 vs about 0.63): the relationship between mix and strength is strongly non-linear.
- `age` has a much higher Spearman (0.596) than Pearson (0.329): the effect is monotonic but curved, which the signed correlation methods expose.
- `water` shows a larger distance correlation and mutual information than its Pearson value suggests.
- Several inputs have VIF above 5, so credit among cement, slag, water and fineagg is shared.

---

## 16. Limitations

- **Small data.** With a few dozen rows, the 20% test set contains only a handful of points. R², permutation importance and rankings can change a lot between random splits.
- **Single split, fixed hyperparameters.** There is no cross-validation or tuning; a model's score reflects these defaults, not its best possible performance.
- **Extrapolation.** Tree-based models cannot predict outside the range of the training targets. Linear, polynomial and neural-network models can, but often unreliably. Always keep new inputs within the ranges of your data. The Predict tab does not warn you.
- **Correlation is not causation.** Importance, SHAP and partial dependence describe the fitted model's behaviour on your data, not physical cause and effect.
- **Correlated inputs.** Importance can be split or understated; partial dependence can evaluate impossible combinations.
- **SHAP sampling.** For non-tree, non-linear-explainer models the values are approximate and based on up to 60 rows.
- **KNN is unscaled,** so large-valued inputs dominate it.
- **Numeric data only.** Text/categorical columns are ignored; encode them as numbers first if they matter.
- **Runtime.** CatBoost, the neural network, SVR and SHAP take time on large files; the app caches results so re-visiting tabs is fast. Very large files (many tens of thousands of rows) may be slow, especially the distance-correlation step, whose memory grows with the square of the row count.

---

## 17. Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: analysis_utils` | Make sure `analysis_utils.py` is in the same folder as `6d2.py` (and in the repository when deploying) |
| `ModuleNotFoundError` for another package on Streamlit Cloud | Check `requirements.txt` is committed; set Python to 3.12 in Advanced settings and reboot the app |
| **Train Models** is greyed out | Select at least one input and one target, make sure no column is in both lists, and ensure there are at least 10 complete rows |
| "The selected sheet needs at least two numeric columns" | The file has fewer than two numeric columns; check that numbers are not stored as text (for example with thousands separators or units) |
| Negative or absurd predictions in the Predict tab | Values are probably outside the training range or typed in the wrong fields; compare against rows of your data |
| Very low or negative R² on a model tab | Too few rows, noisy data, or inputs unrelated to the target; check the Model Comparison tab to see if all models struggle |
| Analysis is slow | Large files take longer; wait for the spinners, and avoid retraining by keeping the selection unchanged |
| Port 8501 already in use | Add `--server.port 8502` to the run command |
| Plots look unchanged after editing the code | Reload the browser tab or use **Rerun** from the Streamlit menu |

---

## Credits and licences

The test datasets are third-party data hosted in public GitHub repositories; each remains under its original licence and terms. They are linked for convenience and are not redistributed by this project.
