# K-Nearest Neighbours Classifier

Binary classification on a social-network ads dataset using the K-NN algorithm with decision-boundary visualisation.

## What It Does

Predicts whether a user will purchase a product based on **Age** and **Estimated Salary**. The model trains a K-NN classifier (K=5, Minkowski distance) on 75 % of the data, evaluates on the remaining 25 %, and renders colour-coded decision boundaries for both splits.

Implementations are provided in **Python** and **R**.

## Dataset

`Social_Network_Ads.csv` — 400 records with columns `User ID`, `Gender`, `Age`, `EstimatedSalary`, `Purchased` (0/1). Only `Age` and `EstimatedSalary` are used as features.

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3 | Main implementation |
| 📊 scikit-learn | KNeighborsClassifier, train/test split, scaling |
| 📈 matplotlib | Decision-boundary plots |
| 🐼 pandas / NumPy | Data loading and array ops |
| 📉 R | Alternative implementation (`knn.R`) |

## Getting Started

```bash
# Install dependencies
pip install numpy pandas matplotlib scikit-learn

# Run
python knn.py
```

The script prints the confusion matrix and displays two decision-boundary plots (training set, test set).

### R version

```r
# Requires: caTools, class, ElemStatLearn
Rscript knn.R
```

## ⚠️ Known Issues

- `knn.R` depends on the **ElemStatLearn** package, which was archived from CRAN. Install from archive or use an alternative dataset-visualisation approach.
- Decision-boundary plotting is slow on large feature ranges due to the fine mesh grid step (`0.01`).
