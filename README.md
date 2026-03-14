# K-Nearest Neighbours (KNN) Classifier

A classification project using the K-Nearest Neighbours algorithm to predict whether a user will purchase a product based on age and estimated salary. Implementations provided in both Python and R.

## Methodology

1. Load the `Social_Network_Ads.csv` dataset containing user demographics and purchase history.
2. Extract features (Age, Estimated Salary) and the target variable (Purchased).
3. Split data into 75% training and 25% test sets.
4. Apply feature scaling (standardization) so distance-based KNN isn't biased by feature magnitude.
5. Train a KNN classifier with K=5 using the Minkowski distance metric (p=2, i.e. Euclidean).
6. Evaluate performance via a confusion matrix.
7. Visualize decision boundaries for both training and test sets.

## Tech Stack

| Component | Technology |
|-----------|------------|
| 🐍 Language | Python 3, R |
| 📊 ML Library | scikit-learn, class (R) |
| 🔢 Numerical | NumPy |
| 🗂️ Data Handling | pandas, caTools (R) |
| 📈 Visualization | matplotlib, ElemStatLearn (R) |

## Dependencies

### Python

```
numpy
pandas
matplotlib
scikit-learn
```

Install with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

### R

```r
install.packages(c("caTools", "class", "ElemStatLearn"))
```

## How to Run

### Python

```bash
cd K-Nearest-Neighbours
python knn.py
```

### R

```bash
cd K-Nearest-Neighbours
Rscript knn.R
```

Or open `knn.R` in RStudio and run interactively.

## Dataset

`Social_Network_Ads.csv` — 400 records with columns:

| Column | Description |
|--------|-------------|
| User ID | Unique identifier |
| Gender | Male / Female |
| Age | User age |
| EstimatedSalary | Annual estimated salary |
| Purchased | 0 = No, 1 = Yes |

## Known Issues

- The R script depends on `ElemStatLearn`, which has been archived from CRAN. Install from archive or use an alternative visualization approach.
- Decision boundary visualization can be slow on large datasets due to fine-grained meshgrid (step=0.01).
- The dataset is small (400 samples); results may not generalize to production-scale data.

## License

See [LICENSE](LICENSE) for details.
