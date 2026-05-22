# =============================================================================
# ITBDA4-14 Project 1 | Question 1 (15 Marks)
# Topic: Principal Component Analysis (PCA)
# Author: Matshepo Tshabangu
# =============================================================================

# -----------------------------------------------------------------------------
# Q1.1 - What is Principal Component Analysis? (6 Marks)
# -----------------------------------------------------------------------------

# Principal Component Analysis (PCA) is an unsupervised dimensionality reduction
# technique that transforms a dataset with many correlated features into a smaller
# set of uncorrelated variables called "principal components".
#
# Key concepts:
#   - Each principal component is a linear combination of the original features
#   - PC1 captures the most variance in the data
#   - PC2 captures the second most variance, orthogonal (perpendicular) to PC1
#   - And so on for PC3, PC4, ...
#
# Why use PCA?
#   1. Reduces computational cost when training ML models on high-dimensional data
#   2. Removes multicollinearity between features
#   3. Enables 2D/3D visualisation of n-dimensional data
#   4. Helps reduce overfitting by removing noise dimensions
#
# How it works (step-by-step):
#   Step 1: Standardise the data (zero mean, unit variance)
#   Step 2: Compute the covariance matrix of the standardised data
#   Step 3: Compute Eigenvalues and Eigenvectors of the covariance matrix
#   Step 4: Sort Eigenvectors by their Eigenvalues in descending order
#   Step 5: Select the top k Eigenvectors (principal components)
#   Step 6: Project the original data onto the new k-dimensional subspace


# -----------------------------------------------------------------------------
# Q1.2 - Eigenvalue Implementation of PCA (6 Marks)
# -----------------------------------------------------------------------------

# The Eigenvalue decomposition is at the mathematical heart of PCA.
#
# COVARIANCE MATRIX:
#   Given a standardised data matrix X (n x p), the covariance matrix is:
#   C = (1 / (n-1)) * t(X) %*% X
#   This p x p matrix captures how each pair of features varies together.
#
# EIGENDECOMPOSITION:
#   We solve: C * v = λ * v
#   Where:
#     λ (lambda) = Eigenvalue  → represents the AMOUNT of variance explained
#     v           = Eigenvector → represents the DIRECTION of that variance
#
# SELECTION:
#   - Sort Eigenvalues in descending order: λ1 ≥ λ2 ≥ ... ≥ λp
#   - The proportion of variance explained by component i = λi / sum(λ)
#   - Choose k components that explain a sufficient amount of variance (e.g. ≥ 95%)
#
# PROJECTION:
#   New data matrix Z (n x k) = X %*% W
#   Where W is the p x k matrix of selected Eigenvectors (loadings)

# Manual Eigenvalue-based PCA implementation:
manual_pca <- function(X, n_components = 2) {
  # Step 1: Standardise (centre and scale)
  X_scaled <- scale(X)
  
  # Step 2: Compute covariance matrix
  cov_matrix <- cov(X_scaled)
  cat("Covariance Matrix (first 3x3):\n")
  print(round(cov_matrix[1:min(3, nrow(cov_matrix)), 1:min(3, ncol(cov_matrix))], 4))
  
  # Step 3: Eigendecomposition
  eigen_result <- eigen(cov_matrix)
  eigenvalues  <- eigen_result$values
  eigenvectors <- eigen_result$vectors
  
  # Step 4: Variance explained per component
  var_explained <- eigenvalues / sum(eigenvalues) * 100
  cum_var       <- cumsum(var_explained)
  
  cat("\nEigenvalues:\n")
  print(round(eigenvalues, 4))
  
  cat("\nVariance Explained (%) per Component:\n")
  print(round(var_explained, 2))
  
  cat("\nCumulative Variance Explained (%):\n")
  print(round(cum_var, 2))
  
  # Step 5: Select top k Eigenvectors
  W <- eigenvectors[, 1:n_components]
  
  # Step 6: Project data
  Z <- X_scaled %*% W
  colnames(Z) <- paste0("PC", 1:n_components)
  
  return(list(
    scores       = as.data.frame(Z),
    eigenvalues  = eigenvalues,
    eigenvectors = eigenvectors,
    var_explained = var_explained,
    cum_var      = cum_var
  ))
}


# -----------------------------------------------------------------------------
# Q1.3 - Template Code: n-dimensional PCA transformation in R (3 Marks)
# -----------------------------------------------------------------------------

cat("\n=== Q1.3: PCA Template Code ===\n")

# --- Using built-in prcomp() (recommended approach) ---

# Load or simulate data (replace with your actual dataset)
set.seed(42)
n <- 150
p <- 8
X_demo <- as.data.frame(matrix(rnorm(n * p), nrow = n,
                                 dimnames = list(NULL, paste0("Feature", 1:p))))

cat("Dataset dimensions:", nrow(X_demo), "rows x", ncol(X_demo), "cols\n")

# Step 1: Run PCA with scaling
pca_result <- prcomp(X_demo, center = TRUE, scale. = TRUE)

# Step 2: Summary — variance explained
cat("\nPCA Summary:\n")
summary(pca_result)

# Step 3: Scree plot — how many components to keep
explained_var <- (pca_result$sdev^2) / sum(pca_result$sdev^2) * 100

plot(explained_var,
     type = "b",
     pch  = 19,
     col  = "steelblue",
     xlab = "Principal Component",
     ylab = "Variance Explained (%)",
     main = "Scree Plot — Variance Explained by Each PC")
abline(h = 5, col = "red", lty = 2)
legend("topright", legend = "5% threshold", col = "red", lty = 2, cex = 0.8)

# Step 4: Project data onto first 2 PCs for 2D visualisation
pca_scores <- as.data.frame(pca_result$x[, 1:2])

plot(pca_scores$PC1, pca_scores$PC2,
     pch  = 19,
     col  = "steelblue",
     xlab = paste0("PC1 (", round(explained_var[1], 1), "% variance)"),
     ylab = paste0("PC2 (", round(explained_var[2], 1), "% variance)"),
     main = "2D Projection of n-Dimensional Data via PCA")
abline(h = 0, v = 0, col = "grey70", lty = 2)

# Step 5: Biplot — shows both scores and loadings
biplot(pca_result,
       main  = "PCA Biplot — Scores and Loadings",
       col   = c("steelblue", "tomato"),
       cex   = 0.7)

# Step 6: Manual Eigenvalue method (for verification / deeper understanding)
cat("\n--- Manual Eigenvalue PCA ---\n")
manual_result <- manual_pca(X_demo, n_components = 2)

# Step 7: Loadings — which original features contribute most to each PC?
cat("\nPC Loadings (top contributors):\n")
loadings_df <- as.data.frame(pca_result$rotation[, 1:2])
print(round(loadings_df, 4))

cat("\nDone. PCA reduced", p, "dimensions to 2 principal components.\n")
