# Walmart Weekly Sales Predictor

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

- Standard imports for data handling/visualization for modeling and for regression metrics .
- Custom helper functions:
  - `read_file_safe`: tries multiple encodings (`utf-16`, `utf-8`, `latin1`) to safely load the CSV.
  - `normalize_cols`: standardizes column names (lowercase, snake_case).
  - `ml_metrics`: builds a tidy dataframe of R², MAE, and RMSE for train/test sets given a model's predictions.

### 2. EDA and Preprocessing

- Loaded and normalized the dataset, parsed `date` as a datetime column.
- Checked for duplicates and missing values, and reviewed dtypes.
- **Dropped rows with missing `weekly_sales`** (the target) rather than imputing them, to avoid introducing bias into the labels.
- **Created date-based features**: `year`, `month`, `day`, and `day_of_week`.
  - Found that `day_of_week` was constant (always Friday, day = 4) across all observations, so it carried no predictive information and was **dropped**.
- **Removed outliers/invalid values** on `temperature`, `fuel_price`, `cpi`, and `unemployment` using a ±3 standard deviation rule around the mean (missing values were kept for later imputation).
- Explored the data visually:
  - Average weekly sales over time.
  - Seasonality: average weekly sales by month, and year-month trends by year.
  - Distribution of weekly sales (histogram with mean/median markers).
  - Weekly sales vs. holidays (boxplot).
  - Weekly sales vs. each economic indicator (regression scatter plots).
  - Correlation matrix / heatmap across numeric features.

### 3. Feature/Target Split

- **Target (Y):** `weekly_sales`
- **Categorical features:** `store`, `holiday_flag`
- **Numerical features:** `temperature`, `fuel_price`, `cpi`, `unemployment`, `year`, `month`, `day`
- `store` was cast to string so it wouldn't be treated as a numeric feature; `date` was dropped after extracting its components.

### 4. Baseline Model — Linear Regression

- Split the data into train/test sets (80/20).
- Built a `ColumnTransformer` pipeline:
  - Numeric features: mean imputation + `StandardScaler`.
  - Categorical features: most-frequent imputation + `OneHotEncoder` (dropping the first category, ignoring unknown categories).
- Trained a `LinearRegression` model and evaluated R², MAE, and RMSE on both train and test sets.
- Plotted feature importance using the model's coefficients. Because `store` was one-hot encoded, the largest coefficients are almost always specific store dummies, different stores have structurally different baseline sales levels (size, location, traffic, etc.) that dwarf the effect of week-to-week economic variation. Among the scaled numeric features, `cpi` has the largest magnitude, making it the economic indicator the model relies on most.

### 5. Regularized Models — Lasso and Ridge

For both models, hyperparameters were tuned with `GridSearchCV` (5-fold CV) over:
- `alpha`: `[0.001, 0.1, 0.5, 1.0, 10.0, 100.0, 1000.0]`
- `fit_intercept`: `[True, False]`

**Lasso**
- Fitting on the raw target produced a convergence warning, so the target was transformed with `log10(Y)` before training, and predictions were back-transformed (`10 ** prediction`) to the original dollar scale before computing metrics, making all R²/MAE/RMSE values directly comparable to the other models.
- Feature importance was plotted from the Lasso coefficients, and the number of features zeroed out by the model was reported (Lasso performs implicit feature selection).

**Ridge**
- Fitted directly on `weekly_sales` (no target transformation needed).
- Feature importance was plotted the same way as for Lasso/Linear Regression.

### 6. Results

| Model | Set | R² | MAE | Notes |
|---|---|---|---|---|
| Linear Regression | train | 0.973 | ~$84k | |
| Linear Regression | test | 0.940 | ~$125k | Noticeable overfitting |
| Ridge (α = 0.001) | train / test | ≈ same as LR | ≈ same as LR | Penalty negligible at optimal α |
| Lasso (log10 target) | train | 0.9623 | ~$84k | |
| Lasso (log10 target) | test | 0.9619 | ~$84k | Best generalization, ~33% lower test MAE than LR, drops 3 features |

**Key takeaways:**
- Linear Regression and Ridge show clear signs of overfitting: training R² (0.973) drops noticeably to test R² (0.940), with test MAE (~$125k) far above train MAE (~$84k).
- Ridge performs almost identically to plain Linear Regression, since its optimal penalty (α = 0.001) is negligible.
- **Lasso generalizes best**: virtually no train-test gap (R² 0.9623 → 0.9619), a ~33% reduction in test MAE compared to Linear Regression, and it simplifies the model by zeroing out 3 uninformative features.
- On this small dataset (~90 training rows), regularization — especially Lasso — effectively controls overfitting without sacrificing training fit.
- **Lasso is the recommended model**, offering the best generalization, lowest prediction error, and greatest stability across train/test splits.

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
