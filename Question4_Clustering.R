# =============================================================================
# ITBDA4-14 Project 1 | Question 4 (25 Marks)
# Topic: Advanced Clustering Analysis — Wine Dataset
# Dataset: wine-clustering.csv
# Author: Matshepo Tshabangu
# =============================================================================

# Install packages if needed:
# install.packages(c("ggplot2", "factoextra", "cluster", "dplyr"))

library(ggplot2)
library(factoextra)   # fviz_eig, fviz_pca_ind, fviz_cluster, fviz_nbclust
library(cluster)      # silhouette
library(dplyr)


# =============================================================================
# Q4.1 — Load dataset, summarise attributes and record count (3 Marks)
# =============================================================================

cat("=== Q4.1: Load and Explore Wine Dataset ===\n")

# Load the wine clustering dataset
wine <- read.csv("wine-clustering.csv", stringsAsFactors = FALSE)

cat("Number of records (rows):", nrow(wine), "\n")
cat("Number of attributes (columns):", ncol(wine), "\n")

cat("\nAttribute names:\n")
print(names(wine))

cat("\nAttribute data types:\n")
print(sapply(wine, class))

cat("\nFirst 6 rows (data preview):\n")
print(head(wine))

cat("\nSummary statistics per attribute:\n")
print(summary(wine))

cat("\nMissing values per attribute:\n")
print(colSums(is.na(wine)))

# Drop any NA rows
wine <- na.omit(wine)
cat("\nFinal clean dataset:", nrow(wine), "rows x", ncol(wine), "cols ✓\n")

# Check for non-numeric columns
non_numeric <- names(wine)[!sapply(wine, is.numeric)]
if (length(non_numeric) > 0) {
  cat("\nConverting non-numeric columns:", paste(non_numeric, collapse = ", "), "\n")
  wine <- as.data.frame(lapply(wine, function(col) as.numeric(as.character(col))))
  wine <- na.omit(wine)
}

cat("\nInterpretation:\n")
cat("  The wine dataset contains chemical composition measurements of wines\n")
cat("  sourced from Spanish farmers. Each row represents one wine sample and\n")
cat("  each column is a chemical constituent (e.g. Alcohol, Malic Acid, Ash).\n")


# =============================================================================
# Q4.2 — PCA transformation + 2D scatter plot (8 Marks)
# =============================================================================

cat("\n=== Q4.2: PCA Transformation and 2D Visualisation ===\n")

# Step 1: Scale the data (mandatory before PCA)
wine_scaled <- scale(wine)
cat("Data scaled (zero mean, unit variance) ✓\n")

cat("\nScaling verification (mean and sd per column should be ~0 and ~1):\n")
scaling_check <- data.frame(
  Mean = round(colMeans(wine_scaled), 6),
  SD   = round(apply(wine_scaled, 2, sd), 6)
)
print(scaling_check)

# Step 2: Apply PCA
pca_result <- prcomp(wine_scaled, center = FALSE, scale. = FALSE)
# Note: already scaled manually above, so center=FALSE, scale.=FALSE

cat("\nPCA completed ✓\n")

# Step 3: Variance explained
explained_var  <- (pca_result$sdev^2) / sum(pca_result$sdev^2) * 100
cum_var        <- cumsum(explained_var)

var_df <- data.frame(
  PC            = paste0("PC", 1:length(explained_var)),
  Variance_Pct  = round(explained_var, 2),
  Cumulative_Pct = round(cum_var, 2)
)
cat("\nVariance Explained per Principal Component:\n")
print(var_df)

cat(sprintf("\nPC1 + PC2 together explain %.2f%% of total variance.\n",
            explained_var[1] + explained_var[2]))

# Step 4: Scree plot
fviz_eig(pca_result,
         addlabels = TRUE,
         ylim      = c(0, max(explained_var) + 10),
         main      = "Scree Plot — Variance Explained by Each Principal Component",
         xlab      = "Principal Components",
         ylab      = "Percentage of Variance Explained (%)",
         barfill   = "steelblue",
         barcolor  = "steelblue",
         linecolor = "tomato")

# Step 5: 2D scatter plot using PC1 and PC2
pca_scores <- as.data.frame(pca_result$x[, 1:2])
colnames(pca_scores) <- c("PC1", "PC2")

ggplot(pca_scores, aes(x = PC1, y = PC2)) +
  geom_point(colour = "steelblue", alpha = 0.7, size = 2.5) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey60") +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey60") +
  labs(
    title    = "2D PCA Scatter Plot — Wine Dataset",
    subtitle = paste0("PC1 (", round(explained_var[1], 1), "%) vs PC2 (",
                       round(explained_var[2], 1), "%)  |  Total: ",
                       round(explained_var[1] + explained_var[2], 1), "% variance"),
    x        = paste0("Principal Component 1 (", round(explained_var[1], 1), "%)"),
    y        = paste0("Principal Component 2 (", round(explained_var[2], 1), "%)")
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

# Biplot — shows loadings (feature arrows) overlaid on scores
fviz_pca_ind(pca_result,
             geom.ind     = "point",
             col.ind      = "steelblue",
             pointsize    = 2,
             addEllipses  = FALSE,
             title        = "PCA Individual Plot — Wine Samples",
             repel        = TRUE)

cat("\nInterpretation:\n")
cat("  - Each point in the scatter plot is one wine sample projected to 2D\n")
cat("  - PC1 captures the most variance; PC2 the second most\n")
cat("  - Clusters in this plot suggest groupings of similar wines\n")


# =============================================================================
# Q4.3 — KMeans Clustering with Elbow Method + WCSS plot (7 Marks)
# =============================================================================

cat("\n=== Q4.3: KMeans Clustering — Elbow Method ===\n")

# Data must be scaled before clustering
# wine_scaled is already computed above

# Elbow method: compute WCSS (Within Cluster Sum of Squares) for k = 1 to 12
set.seed(42)
k_max  <- 12
wcss   <- numeric(k_max)

for (k in 1:k_max) {
  km        <- kmeans(wine_scaled, centers = k, nstart = 25, iter.max = 300)
  wcss[k]   <- km$tot.withinss
}

# WCSS table
wcss_df <- data.frame(K = 1:k_max, WCSS = round(wcss, 2))
cat("WCSS values per k:\n")
print(wcss_df)

# Elbow line graph
ggplot(wcss_df, aes(x = K, y = WCSS)) +
  geom_line(colour = "steelblue", linewidth = 1.2) +
  geom_point(colour = "tomato", size = 3) +
  scale_x_continuous(breaks = 1:k_max) +
  labs(
    title    = "Elbow Method — Within Cluster Sum of Squares (WCSS)",
    subtitle = "Choose k at the 'elbow' where WCSS reduction flattens",
    x        = "Number of Clusters (k)",
    y        = "Total Within-Cluster Sum of Squares (WCSS)"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

# Automated elbow detection via factoextra
fviz_nbclust(wine_scaled, kmeans,
             method  = "wss",
             k.max   = k_max,
             nstart  = 25) +
  labs(title    = "Optimal k — Elbow Method (factoextra)",
       subtitle = "Red dashed line marks the suggested elbow point") +
  theme_minimal()

# Also check silhouette method for confirmation
fviz_nbclust(wine_scaled, kmeans,
             method  = "silhouette",
             k.max   = k_max,
             nstart  = 25) +
  labs(title = "Optimal k — Average Silhouette Method") +
  theme_minimal()

# Set optimal k based on elbow observation (update if your elbow differs)
optimal_k <- 3
cat(sprintf("\nOptimal k selected from elbow: k = %d\n", optimal_k))
cat("Rationale: WCSS reduction slows significantly after k =", optimal_k, "\n")


# =============================================================================
# Q4.4 — 2D Scatter Plot with Cluster Labels on PCA Components (7 Marks)
# =============================================================================

cat("\n=== Q4.4: Cluster Visualisation on PCA Space ===\n")

# Run KMeans with the optimal k
set.seed(42)
km_final <- kmeans(wine_scaled,
                   centers  = optimal_k,
                   nstart   = 25,
                   iter.max = 300)

cat("KMeans clustering complete ✓\n")
cat("Cluster sizes:\n")
print(table(km_final$cluster))
cat("Total WCSS:", round(km_final$tot.withinss, 2), "\n")
cat("Between-cluster SS:", round(km_final$betweenss, 2), "\n")
cat("Between / Total SS ratio:", round(km_final$betweenss / km_final$totss * 100, 2), "% (higher is better)\n")

# Combine PCA scores with cluster labels
cluster_df <- data.frame(
  PC1     = pca_scores$PC1,
  PC2     = pca_scores$PC2,
  Cluster = as.factor(km_final$cluster)
)

# Cluster centroids in PCA space
centroids <- cluster_df %>%
  group_by(Cluster) %>%
  summarise(PC1 = mean(PC1), PC2 = mean(PC2))

# 2D scatter plot with cluster colours
ggplot(cluster_df, aes(x = PC1, y = PC2, colour = Cluster)) +
  geom_point(alpha = 0.75, size = 2.5) +
  geom_point(data = centroids, aes(x = PC1, y = PC2),
             shape = 8, size = 6, stroke = 1.5, colour = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey70") +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey70") +
  scale_colour_manual(
    values = c("1" = "#e74c3c", "2" = "#2980b9", "3" = "#27ae60",
               "4" = "#f39c12", "5" = "#8e44ad"),
    name   = "Cluster"
  ) +
  labs(
    title    = paste0("KMeans Clustering (k=", optimal_k, ") — Wine Dataset"),
    subtitle = paste0("2D projection via PCA (PC1 + PC2 = ",
                       round(explained_var[1] + explained_var[2], 1), "% variance)  |  ★ = centroids"),
    x        = paste0("PC1 (", round(explained_var[1], 1), "% variance)"),
    y        = paste0("PC2 (", round(explained_var[2], 1), "% variance)")
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title      = element_text(face = "bold"),
    legend.position = "right"
  )

# Alternative using factoextra for polished cluster plot
fviz_cluster(km_final,
             data         = wine_scaled,
             geom         = "point",
             ellipse.type = "convex",
             palette      = c("#e74c3c", "#2980b9", "#27ae60"),
             ggtheme      = theme_minimal(),
             main         = paste0("KMeans Cluster Plot (k=", optimal_k, ") — Wine Dataset"),
             sub          = "Convex hulls enclose each cluster; axes are first 2 PCA components")

# Cluster profile: mean of each original feature per cluster
cat("\nCluster Profiles (mean of each feature per cluster):\n")
wine_with_clusters <- as.data.frame(wine_scaled)
wine_with_clusters$Cluster <- km_final$cluster
cluster_profiles <- wine_with_clusters %>%
  group_by(Cluster) %>%
  summarise(across(everything(), ~ round(mean(.), 3)))
print(as.data.frame(cluster_profiles))

cat("\nInterpretation:\n")
cat("  - Each cluster represents a distinct chemical profile of wine\n")
cat("  - Clusters with high separation in PCA space have very different compositions\n")
cat("  - The cluster profiles table shows which chemical constituents characterise each group\n")
cat("  - WNX can use these clusters to classify new wine stock from Spanish farmers\n")
cat("    into the corresponding wine type without manual analysis\n")
