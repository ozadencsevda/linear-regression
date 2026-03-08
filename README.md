#  Linear Regression Analysis — Multi-Dataset Comparison

A supervised machine learning project applying **Linear Regression** across three different datasets to explore how data characteristics affect model performance. Each dataset goes through the same full ML pipeline and is benchmarked against a DummyRegressor baseline.

---

##  Project Overview

| Dataset | Target Variable | R² Score |
|---|---|---|
|  Ames Housing | `SalePrice` (house price) | 0.05 |
|  Wine Quality | `class` (quality score) | 0.38 |
|  CPU Activity | `usr` (CPU usage %) | 0.88 |

---

##  Pipeline (Applied to Each Dataset)

```
Raw Data → EDA → Preprocessing → Train/Test Split → Feature Scaling → Model Training → Evaluation
```

**1. Exploratory Data Analysis (EDA)**
- Dataset shape and type inspection
- Missing value detection
- Target variable distribution
- Correlation heatmap across numerical features

**2. Preprocessing**
- Median imputation for numerical columns, mode for categorical
- One-Hot Encoding for categorical variables (Ames Housing)
- IQR-based outlier removal on the target variable

**3. Model Training**
- 80/20 train-test split (`random_state=42`)
- `StandardScaler` applied to features
- Baseline: `DummyRegressor(strategy='mean')`
- Main model: `LinearRegression()`

**4. Evaluation**
- MAE, RMSE, R² metrics
- Actual vs. Predicted scatter plot
- Residual plot
- Cross-dataset comparison charts

---

## 📈 Key Findings

- **CPU Activity** achieved the strongest results (R² = 0.88), reflecting a clear linear relationship between features and CPU usage.
- **Wine Quality** showed moderate performance (R² = 0.38); the discrete integer nature of the target variable limits linear regression's fit.
- **Ames Housing** performed poorly (R² = 0.05) due to high dimensionality after one-hot encoding — 79 features expanding into a sparse feature space.

In all three cases, Linear Regression meaningfully outperformed the baseline model.

---

##  Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

##  Usage

```bash
git clone https://github.com/<your-username>/lineer-regresyon-analizi.git
cd lineer-regresyon-analizi
jupyter notebook lineerRegresyonAnalizi.ipynb
```

---

##  Repository Structure

```
├── lineerRegresyonAnalizi.ipynb   # Main notebook
└── README.md
```

---

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-green?logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
