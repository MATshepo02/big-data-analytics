# ITBDA4-14 Project 1 — Big Data Analytics

**Module:** ITBDA4-14 | NQF Level 8  
**Author:** Matshepo Tshabangu  
**Institution:** Eduvos  

---

## Overview

This repository contains R implementations for all four questions of the ITBDA4-14 Big Data Analytics Project 1. The project covers core data science and machine learning techniques including dimensionality reduction, classification, regression, and clustering.

---

## Project Structure

```
├── Question1_PCA.R               # Principal Component Analysis (15 marks)
├── Question2_HeartDisease.R      # Logistic Regression — Heart Disease (20 marks)
├── Question3_RegressionModels.R  # MLR vs Decision Tree Regression (34 marks)
├── Question4_Clustering.R        # KMeans Clustering + PCA (25 marks)
├── README.md
└── data/
    ├── heart_disease.csv         # Place dataset here
    ├── wine-clustering.csv       # Place dataset here
    └── your_dataset.csv          # Q3 dataset (rename accordingly)
```

---

## Questions Summary

### Question 1 — Principal Component Analysis (15 Marks)
**File:** `Question1_PCA.R`

| Sub-question | Description | Marks |
|---|---|---|
| 1.1 | Explanation of PCA | 6 |
| 1.2 | Eigenvalue decomposition description & implementation | 6 |
| 1.3 | Template R code for n-dimensional PCA transformation | 3 |

**Key concepts:** Covariance matrix, Eigendecomposition, Scree plot, Biplot, Dimensionality reduction

---

### Question 2 — Heart Disease Diagnosis (20 Marks)
**File:** `Question2_HeartDisease.R`  
**Dataset:** `heart_disease.csv`

| Sub-question | Description | Marks |
|---|---|---|
| 2.1 | Load data, ensure numeric types, preview records | 2 |
| 2.2 | Normalise features + 80/20 train/test split | 4 |
| 2.3 | Logistic Regression — precision, recall, accuracy | 6 |
| 2.4 | Bar graph of feature contributions (sorted by abs. coefficient) | 8 |

**Key concepts:** Min-max normalisation, Logistic regression, Confusion matrix, Feature importance

---

### Question 3 — Regression Models (34 Marks)
**File:** `Question3_RegressionModels.R`

| Sub-question | Description | Marks |
|---|---|---|
| Q3 setup | Load and explore dataset | — |
| 3.3a | Multiple Linear Regression | 4 |
| 3.3b | Decision Tree Regression | 4 |
| 3.4 | R² and RMSE on train + test sets | 8 |
| 3.5 | Goodness of fit scatter plots + residual plots | 8 |
| 3.6 | Model performance commentary + recommendations | 5 |

**Key concepts:** MLR, Decision Trees, R², RMSE, Overfitting, Model comparison

---

### Question 4 — Advanced Clustering Analysis (25 Marks)
**File:** `Question4_Clustering.R`  
**Dataset:** `wine-clustering.csv`

| Sub-question | Description | Marks |
|---|---|---|
| 4.1 | Load data, summarise attributes and record count | 3 |
| 4.2 | PCA transformation + 2D scatter plot | 8 |
| 4.3 | KMeans with Elbow method + WCSS line graph | 7 |
| 4.4 | 2D scatter plot with cluster labels on PCA axes | 7 |

**Key concepts:** PCA, KMeans, Elbow method, WCSS, Silhouette analysis, Cluster profiling

---

## Setup & Requirements

### R Packages Required

```r
install.packages(c(
  "caret",        # ML framework (train/test split, confusion matrix)
  "ggplot2",      # Data visualisation
  "dplyr",        # Data manipulation
  "e1071",        # SVM / caret dependency
  "rpart",        # Decision Tree
  "rpart.plot",   # Decision Tree visualisation
  "factoextra",   # PCA and clustering visualisation
  "cluster",      # Silhouette analysis
  "Metrics"       # RMSE calculation
))
```

### Running the Code

1. Clone the repository
2. Place the required CSV datasets in the `data/` folder (or update file paths in each script)
3. Open each `.R` file in RStudio
4. Run the full script (`Ctrl+Shift+Enter`) or section by section

```r
# Example: run Question 1
source("Question1_PCA.R")

# Example: run Question 4 (ensure wine-clustering.csv is available)
source("Question4_Clustering.R")
```

---

## Key Results

| Question | Model/Technique | Key Metric |
|---|---|---|
| Q2 | Logistic Regression | Accuracy, Precision, Recall on heart disease test set |
| Q3 | MLR vs Decision Tree | R² and RMSE comparison on train + test sets |
| Q4 | KMeans (k=3) | WCSS, Between/Total SS ratio on wine clusters |

---

## Technologies Used

- **Language:** R
- **IDE:** RStudio
- **Libraries:** caret, ggplot2, factoextra, rpart, cluster, dplyr
- **Techniques:** PCA, Logistic Regression, Multiple Linear Regression, Decision Trees, KMeans Clustering

---

## Academic Integrity

This project was completed individually in accordance with academic integrity policies. All code was written by the author. No AI-generated code was submitted as original work without understanding.

---

*Eduvos | BSc IT (Hons) Data Science | 2026*
