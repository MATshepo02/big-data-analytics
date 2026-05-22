# =============================================================================
# ITBDA4-14 Project 1 | Question 2 (20 Marks)
# Topic: Heart Disease Diagnosis — Logistic Regression
# Dataset: heart_disease.csv
# Author: Matshepo Tshabangu
# =============================================================================

# Install packages if needed (run once):
# install.packages(c("caret", "ggplot2", "dplyr", "e1071"))

library(caret)
library(ggplot2)
library(dplyr)

# -----------------------------------------------------------------------------
# Q2.1 — Load dataset, ensure numeric types, preview data (2 Marks)
# -----------------------------------------------------------------------------

cat("=== Q2.1: Load and Inspect Data ===\n")

# Load dataset (update path if needed)
heart <- read.csv("heart_disease.csv", stringsAsFactors = FALSE)

cat("Dimensions:", nrow(heart), "rows x", ncol(heart), "columns\n\n")

# Preview first few rows
cat("First 6 rows:\n")
print(head(heart))

# Check data types
cat("\nColumn data types:\n")
print(sapply(heart, class))

# Convert all columns to numeric (handles any character columns)
heart <- as.data.frame(lapply(heart, function(col) as.numeric(as.character(col))))

# Check for missing values after conversion
cat("\nMissing values per column:\n")
print(colSums(is.na(heart)))

# Remove rows with NA (if any introduced by coercion)
heart <- na.omit(heart)
cat("\nClean dataset dimensions:", nrow(heart), "rows x", ncol(heart), "cols\n")

# Summary statistics
cat("\nDataset Summary:\n")
print(summary(heart))

# Distribution of target variable
cat("\nTarget variable distribution:\n")
print(table(heart$target))


# -----------------------------------------------------------------------------
# Q2.2 — Normalise features + 80/20 train/test split (4 Marks)
# -----------------------------------------------------------------------------

cat("\n=== Q2.2: Normalise and Split ===\n")

# Separate features (X) and target (y)
# Assumes the target column is named 'target' — update if different
target_col <- "target"
X <- heart[, !names(heart) %in% target_col]
y <- heart[[target_col]]

# Min-Max normalisation function
min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# Normalise all feature columns
X_norm <- as.data.frame(lapply(X, min_max_norm))
cat("Features normalised (min-max scaling) ✓\n")

# Combine normalised features with target
heart_norm <- cbind(X_norm, target = as.factor(y))

# Set seed for reproducibility
set.seed(123)

# 80/20 stratified split
train_index <- createDataPartition(heart_norm$target, p = 0.8, list = FALSE)
train_data  <- heart_norm[ train_index, ]
test_data   <- heart_norm[-train_index, ]

cat("Training set:", nrow(train_data), "rows\n")
cat("Test set:    ", nrow(test_data),  "rows\n")
cat("Train/Test split ratio: 80/20 ✓\n")

# Verify class balance in split
cat("\nClass distribution — Training set:\n")
print(table(train_data$target))
cat("\nClass distribution — Test set:\n")
print(table(test_data$target))


# -----------------------------------------------------------------------------
# Q2.3 — Logistic Regression: precision, recall, accuracy (6 Marks)
# -----------------------------------------------------------------------------

cat("\n=== Q2.3: Logistic Regression Model ===\n")

# Train logistic regression
log_model <- glm(target ~ .,
                 data   = train_data,
                 family = binomial(link = "logit"))

cat("Model trained successfully ✓\n\n")
cat("Model Summary:\n")
print(summary(log_model))

# Predict on test set (probabilities → class labels at 0.5 threshold)
pred_prob  <- predict(log_model, newdata = test_data, type = "response")
pred_class <- ifelse(pred_prob >= 0.5, 1, 0)
pred_class <- as.factor(pred_class)

# Confusion matrix
cm <- confusionMatrix(pred_class, test_data$target, positive = "1")
cat("\nConfusion Matrix:\n")
print(cm$table)

# Extract metrics
accuracy  <- cm$overall["Accuracy"]
precision <- cm$byClass["Precision"]
recall    <- cm$byClass["Recall"]
f1        <- cm$byClass["F1"]

cat("\n--- Performance Metrics ---\n")
cat(sprintf("Accuracy  : %.4f (%.2f%%)\n", accuracy,  accuracy  * 100))
cat(sprintf("Precision : %.4f\n", precision))
cat(sprintf("Recall    : %.4f\n", recall))
cat(sprintf("F1 Score  : %.4f\n", f1))

cat("\nInterpretation:\n")
cat("  - Accuracy  = overall correct predictions / total predictions\n")
cat("  - Precision = of all predicted positives, how many were actually positive\n")
cat("  - Recall    = of all actual positives, how many did the model find\n")


# -----------------------------------------------------------------------------
# Q2.4 — Bar graph: feature contribution to logistic regression (8 Marks)
# -----------------------------------------------------------------------------

cat("\n=== Q2.4: Feature Importance Bar Graph ===\n")

# Extract coefficients (excluding intercept)
coeff_df <- data.frame(
  Feature     = names(coef(log_model))[-1],
  Coefficient = coef(log_model)[-1]
)

# Take absolute value and sort descending
coeff_df$AbsCoefficient <- abs(coeff_df$Coefficient)
coeff_df <- coeff_df[order(coeff_df$AbsCoefficient, decreasing = TRUE), ]
coeff_df$Feature <- factor(coeff_df$Feature, levels = coeff_df$Feature)

cat("Logistic Regression Coefficients (sorted by absolute value):\n")
print(coeff_df)

# Bar plot
ggplot(coeff_df, aes(x = Feature, y = AbsCoefficient, fill = AbsCoefficient)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "#a8d8ea", high = "#1a5276",
                      name = "Absolute\nCoefficient") +
  coord_flip() +
  labs(
    title    = "Feature Contribution to Heart Disease Diagnosis",
    subtitle = "Logistic Regression | Absolute Coefficient Values (Descending)",
    x        = "Feature (Predictor Variable)",
    y        = "Absolute Logistic Regression Coefficient",
    caption  = "Higher value = stronger discriminating power between sick and healthy patients"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(colour = "grey40"),
    axis.text.y   = element_text(size = 10),
    legend.position = "right"
  )

cat("\nBar graph plotted. Top discriminating features:\n")
cat("  1st:", as.character(coeff_df$Feature[1]), "—", round(coeff_df$AbsCoefficient[1], 4), "\n")
cat("  2nd:", as.character(coeff_df$Feature[2]), "—", round(coeff_df$AbsCoefficient[2], 4), "\n")
cat("  3rd:", as.character(coeff_df$Feature[3]), "—", round(coeff_df$AbsCoefficient[3], 4), "\n")

cat("\nFinding: Features with the highest absolute coefficients are most influential\n")
cat("in discriminating between healthy and sick patients in the logistic model.\n")
