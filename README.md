# Walmart Weekly Sales Predictor

[![GitHub](https://img.shields.io/badge/GitHub-Walmart__Sales-181717?style=flat&logo=github)](https://github.com/Gargamelch/Walmart_Sales)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3130/)

<img src="https://full-stack-assets.s3.eu-west-3.amazonaws.com/Walmart_logo_(2008).svg.png" alt="Walmart Logo" width="300"/>

## Project

Walmart's marketing service wants a machine learning model that can estimate weekly sales for its stores with the best possible precision. Such a model helps them understand how sales are influenced by economic indicators (temperature, fuel price, CPI, unemployment) and holidays, and can support future marketing campaigns.

## Goals

1. Perform an exploratory data analysis (EDA) and the necessary preprocessing to prepare the data for machine learning.
2. Train a baseline linear regression model.
3. Train regularized regression models (Lasso and Ridge) to reduce overfitting and compare performance.

## Dataset

The dataset (`Walmart_Store_sales.csv`) contains weekly sales figures for different Walmart stores along with economic indicators:

| Column | Description |
|---|---|
| `Store` | Store identifier (categorical) |
| `Date` | Week date |
| `Weekly_Sales` | Target variable — weekly sales for the store |
| `Holiday_Flag` | Whether the week includes a holiday (categorical) |
| `Temperature` | Average temperature for the week |
| `Fuel_Price` | Fuel price in the region |
| `CPI` | Consumer Price Index |
| `Unemployment` | Unemployment rate |

The raw file contains **150 rows and 8 columns** — a fairly small dataset for a regression task.

## Approach

### 1. Setup

- Standard imports for data handling/visualization, modeling, and regression metrics.
- Custom helper functions:
  - `read_file_safe`: tries multiple encodings (`utf-16`, `utf-8`, `latin1`) to safely load the CSV.
  - `normalize_cols`: standardizes column names (lowercase, snake_case).
  - `ml_metrics`: builds a tidy dataframe of R², MAE, and RMSE for train/test sets given a model's predictions.

### 2. EDA and Preprocessing

- Loaded and normalized the dataset, parsed `date` as a datetime column.
- Checked for duplicates and missing values, and reviewed dtypes.
- **Dropped rows with missing `weekly_sales`** (the target) rather than imputing them, to avoid introducing bias into the labels — 14 rows dropped. Rows with a missing `date` were also dropped, since no date-based features could be built for them — 18 further rows dropped. **118 rows remained** after this step.
- **Created date-based features**: `year`, `month`, `day`, and `day_of_week`.
  - Found that `day_of_week` was constant (always Friday, day = 4) across all observations, so it carried no predictive information and was **dropped**.
- **Removed outliers/invalid values** on `temperature`, `fuel_price`, `cpi`, and `unemployment` using a ±3 standard deviation rule around the mean (missing values were kept, not treated as outliers, so they could still be handled by imputation later). **113 rows remained** for modeling after this step.
- Explored the data visually:
  - Average weekly sales over time.
  - Seasonality: average weekly sales by year, by month, and year-month trends by year.
  - Distribution of weekly sales (histogram with mean/median markers).
  - Weekly sales vs. holidays (boxplot).
  - Weekly sales vs. each economic indicator (regression scatter plots with correlation coefficients).
  - Store-to-store sales gap (top vs. bottom stores by average weekly sales).
  - Correlation matrix / heatmap across the continuous numeric features and `holiday_flag`.

### 3. Feature/Target Split

- **Target (Y):** `weekly_sales`
- **Categorical features:** `store`, `holiday_flag`, `year`, `month` — cast to strings and one-hot encoded, since none of the four should be treated as an ordered numeric quantity (a store ID, a holiday flag, or a calendar year/month has no linear relationship with sales).
- **Numerical features:** `temperature`, `fuel_price`, `cpi`, `unemployment`.
- The raw `day` column was **dropped** before modeling: it only encoded "which Friday of the month," carried no useful signal on its own, and one-hot encoding it would have added ~31 near-empty dummy columns to a training set of only ~90 rows — a clear overfitting risk. `date` was dropped after all its components had been extracted.

### 4. Baseline Model — Linear Regression

- Split the data into train/test sets (80/20, 90 train rows / 23 test rows).
- Built a `ColumnTransformer` pipeline:
  - Numeric features: mean imputation + `StandardScaler`.
  - Categorical features: most-frequent imputation + `OneHotEncoder` (dropping the first category, ignoring unknown categories).
- Trained a `LinearRegression` model and evaluated R², MAE, and RMSE on both train and test sets.
- Plotted feature importance using the model's coefficients. Because `store` was one-hot encoded, the largest coefficients are almost always specific store dummies — different stores have structurally different baseline sales levels (size, location, traffic, etc.) that dwarf the effect of week-to-week economic variation. Among the scaled numeric features, `cpi` has the largest magnitude, making it the economic indicator the model relies on most.

### 5. Regularized Models — Lasso and Ridge

Both models were trained directly on `weekly_sales` (no target transformation) and tuned with `GridSearchCV` (5-fold CV) over `alpha` and `fit_intercept: [True, False]`.

**Lasso**
- `alpha` grid: `[0.001, 0.1, 0.5, 1.0, 10.0, 100.0, 1000.0, 1100.0, 1200.0, 1280.0, 1289.0, 1290.0, 1291.0, 1300.0, 1500.0, 2000.0]`
- Hit a `ConvergenceWarning` at the default 1,000 iterations, resolved by raising `max_iter` to 10,000 — no target transformation was needed.
- Best hyperparameters: `alpha = 1290.0`, `fit_intercept = True` (mean 5-fold CV R² ≈ 0.912).
- Feature importance was plotted from the Lasso coefficients, and the number of features zeroed out by the model was reported (Lasso performs implicit feature selection) — **6 of 36 features** were zeroed: `month_10`, `month_5`, `month_3`, `cpi`, `year_2011`, `month_8`.

**Ridge**
- `alpha` grid: `[0.001, 0.01, 0.07, 0.077, 0.078, 0.079, 0.08, 0.1, 0.2, 0.5, 1.0, 10.0, 100.0, 1000.0]`
- Best hyperparameters: `alpha = 0.079`, `fit_intercept = True` (mean 5-fold CV R² ≈ 0.898).
- Feature importance was plotted the same way as for Lasso/Linear Regression.

### 6. Results

| Model | Set | R² | MAE | RMSE |
|---|---|---|---|---|
| Linear Regression | train | 0.985 | ~$60k | ~$80k |
| Linear Regression | test | 0.969 | ~$107k | ~$122k |
| Ridge (α = 0.079) | train | 0.984 | ~$63k | ~$85k |
| Ridge (α = 0.079) | test | 0.977 | ~$89k | ~$104k |
| Lasso (α = 1290) | train | 0.981 | ~$67k | ~$92k |
| Lasso (α = 1290) | test | 0.982 | ~$76k | ~$93k |

**Key takeaways:**
- **Linear Regression overfits**: train R² (0.985) drops to test R² (0.969) — a gap of 0.017 — and test MAE (~$107k) is ~78% higher than train MAE (~$60k).
- **Ridge improves generalization**: test R² rises to 0.977 (vs. 0.969 for LR) and test MAE drops to ~$89k, a ~17% reduction. The train-test gap also shrinks (0.006 vs. 0.017).
- **Lasso generalizes best**: test R² (0.982) is essentially identical to train R² (0.981) — no meaningful overfitting. Test MAE (~$76k) is a ~29% reduction vs. LR, and test RMSE drops from ~$122k (LR) to ~$93k. It also zeroes out 6 of 36 features, producing a simpler, more interpretable model.
- Note that the R² used to select each model's `alpha` was the 5-fold cross-validated score on the training set (~0.91 for Lasso, ~0.90 for Ridge) — noticeably lower than the single train/test split numbers above, a useful reminder that those splits are optimistic given how little data this is.
- **Lasso is the recommended model**, offering the best test-set performance across R², MAE, and RMSE, no train-test gap, and a simpler feature set.
- These comparisons rest on a very small test set (23 rows out of 113 usable rows total), so the ranking should be read as directionally reliable rather than statistically conclusive.

## Tech Stack

- Python
- pandas, numpy
- matplotlib, seaborn, plotly
- scikit-learn (Pipeline, ColumnTransformer, LinearRegression, Ridge, Lasso, GridSearchCV)

## Project Structure

```
.
├── Walmart_Store_sales.csv   # Dataset
├── Walmart_Sales.ipynb       # Full analysis: EDA, preprocessing, modeling, evaluation
└── README.md
```

## Data source

- [Dataset (CSV)](https://julie-resources.s3.eu-west-3.amazonaws.com/analysis-fullstack-fulltime-v2/machine-learning-projects-dafs/walmart-predict-weekly-sales-dafs/Walmart_Store_sales.csv)

## Author's Note

This project was completed as part of a machine learning exercise using a custom Walmart sales dataset (adapted from a Kaggle competition).
