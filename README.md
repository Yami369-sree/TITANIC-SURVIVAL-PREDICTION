# TITANIC-SURVIVAL-PREDICTION
# Titanic Survival Prediction 🚢🔮

Machine‑learning pipeline that predicts whether a Titanic passenger survived, using the classic Kaggle Titanic data set.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Quickstart](#quickstart)
4. [Repository Structure](#repository-structure)
5. [Modeling Approach](#modeling-approach)
6. [Results](#results)
7. [CLI Usage](#cli-usage)
8. [Contributing](#contributing)
9. [License](#license)

---

## Project Overview
This repo contains:
- **Exploratory analysis** and rich visualisations (📝 `notebooks/01_exploration.ipynb`)
- **Clean feature pipeline** (`src/features.py`) handling missing values, encoding, scaling
- **Model zoo** (`src/model.py`) with Logistic Regression, Random Forest, Gradient Boosting, XGBoost, & LightGBM
- **CLI scripts** to train and predict without notebooks
- **Unit tests** ensuring data and code quality

Targets:
- **Binary classification** (`Survived` = 0 | 1)
- **Primary metric**: F1‑score, with accuracy & ROC‑AUC as auxiliaries

---

## Dataset
- **Source**: [Kaggle – Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- **Fields Used**

| Column      | Description                              |
|-------------|------------------------------------------|
| `Survived`  | 0 = No, 1 = Yes (target)                |
| `Pclass`    | Ticket class (1 = Upper, 2 = Middle, 3 = Lower) |
| `Name`      | Passenger name                           |
| `Sex`       | Male / Female                            |
| `Age`       | Age in years                             |
| `SibSp`     | # siblings / spouses aboard              |
| `Parch`     | # parents / children aboard              |
| `Ticket`    | Ticket number                            |
| `Fare`      | Passenger fare (£)                       |
| `Cabin`     | Cabin number                             |
| `Embarked`  | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## Quickstart

```bash
# 1. Clone repo & install deps
git clone https://github.com/<you>/titanic-survival-prediction.git
cd titanic-survival-prediction
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate titanic-torch

# 2. Download Kaggle CSVs to data/raw/
kaggle competitions download -c titanic -p data/raw --unzip

# 3. Run Jupyter notebooks
jupyter lab

# 4. Or train directly via CLI
python -m src.train --model random_forest
