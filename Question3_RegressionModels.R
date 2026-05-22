# =============================================================================
# ITBDA4-14 Project 1 | Question 3 (34 Marks)
# Topic: Multiple Linear Regression vs Decision Tree Regression
# Author: Matshepo Tshabangu
# =============================================================================

# Install packages if needed:
# install.packages(c("caret", "rpart", "rpart.plot", "ggplot2", "dplyr", "Metrics"))

library(caret)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(dplyr)
library(Metrics)   # for rmse()

# =============================================================================
# STEP 1: Load and Explore the Dataset
# =============================================================================

cat("=== Step 1: Load Dataset ===\n")

# Load dataset — update filename/path as needed
df <- read.csv("your_dataset.csv", stringsAsFactors = FALSE)

cat("Dimensions:", nrow(df), "rows x", ncol(df), "columns\n")
cat("\nColumn names:\n")
print(names(df))

cat("\nFirst 6 rows:\n")
print(head(df))

cat("\nData types:\n")
print(sapply(df, class))

cat("\nMissing values:\n")
print(colSums(is.na(df)))

cat("\nSummary statistics:\n")
print(summary(df))

# Convert any non-numeric columns
df <- as.data.frame(lapply(df, function(col) as.numeric(as.character(col))))
df <- na.omit(df)
cat("\nClean dimensions:", nrow(df), "rows x", ncol(df), "cols ✓\n")


# =============================================================================
# STEP 2: Feature Selection & Target Variable
# =============================================================================

cat("\n=== Step 2: Feature / Target Setup ===\n")

# UPDATE this to your actual target column name
target_col <- names(df)[ncol(df)]   # defaults to last column
cat("Target variable:", target_col, "\n")

X <- df[, !names(df) %in% target_col]
y <- df[[target_col]]

cat("Number of predictor features:", ncol(X), "\n")


# =============================================================================
# STEP 3: Train / Test Split (80/20)
# =============================================================================

cat("\n=== Step 3: Train/Test Split (80/20) ===\n")

set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
train_data  <- df[ train_index, ]
test_data   <- df[-train_index, ]

cat("Training rows:", nrow(train_data), "\n")
cat("Test rows:    ", nrow(test_data),  "\n")


# =============================================================================
# Q3.3a — Multiple Linear Regression (4 Marks)
# =============================================================================

cat("\n=== Q3.3a: Multiple Linear Regression ===\n")

# Train model
mlr_model <- lm(as.formula(paste(target_col, "~ .")), data = train_data)

cat("MLR Model Summary:\n")
print(summary(mlr_model))

# Predictions
mlr_train_pred <- predict(mlr_model, newdata = train_data)
mlr_test_pred  <- predict(mlr_model, newdata = test_data)

cat("\nSample predictions vs actuals (test set, first 10):\n")
comparison_mlr <- data.frame(
  Actual    = test_data[[target_col]][1:10],
  Predicted = round(mlr_test_pred[1:10], 3)
)
print(comparison_mlr)


# =============================================================================
# Q3.3b — Decision Tree Regression (4 Marks)
# =============================================================================

cat("\n=== Q3.3b: Decision Tree Regression ===\n")

# Train model
dt_model <- rpart(
  formula = as.formula(paste(target_col, "~ .")),
  data    = train_data,
  method  = "anova",    # "anova" for regression
  control = rpart.control(minsplit = 10, cp = 0.01)
)

cat("Decision Tree Summary:\n")
print(summary(dt_model))

# Visualise the tree
rpart.plot(dt_model,
           main   = "Decision Tree Regression",
           type   = 2,
           extra  = 101,
           fallen.leaves = TRUE,
           cex    = 0.7)

# Predictions
dt_train_pred <- predict(dt_model, newdata = train_data)
dt_test_pred  <- predict(dt_model, newdata = test_data)

cat("\nSample predictions vs actuals (test set, first 10):\n")
comparison_dt <- data.frame(
  Actual    = test_data[[target_col]][1:10],
  Predicted = round(dt_test_pred[1:10], 3)
)
print(comparison_dt)


# =============================================================================
# Q3.4 — R² and RMSE on Training AND Test Sets (8 Marks)
# =============================================================================

cat("\n=== Q3.4: Coefficient of Determination (R²) and RMSE ===\n")

# Helper: R-squared
r_squared <- function(actual, predicted) {
  ss_res <- sum((actual - predicted)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  1 - (ss_res / ss_tot)
}

# --- Multiple Linear Regression ---
mlr_r2_train   <- r_squared(train_data[[target_col]], mlr_train_pred)
mlr_r2_test    <- r_squared(test_data[[target_col]],  mlr_test_pred)
mlr_rmse_train <- rmse(train_data[[target_col]], mlr_train_pred)
mlr_rmse_test  <- rmse(test_data[[target_col]],  mlr_test_pred)

# --- Decision Tree Regression ---
dt_r2_train   <- r_squared(train_data[[target_col]], dt_train_pred)
dt_r2_test    <- r_squared(test_data[[target_col]],  dt_test_pred)
dt_rmse_train <- rmse(train_data[[target_col]], dt_train_pred)
dt_rmse_test  <- rmse(test_data[[target_col]],  dt_test_pred)

# Summary table
results_table <- data.frame(
  Model    = c("Multiple Linear Regression", "Decision Tree Regression"),
  R2_Train = round(c(mlr_r2_train,   dt_r2_train),   4),
  R2_Test  = round(c(mlr_r2_test,    dt_r2_test),    4),
  RMSE_Train = round(c(mlr_rmse_train, dt_rmse_train), 4),
  RMSE_Test  = round(c(mlr_rmse_test,  dt_rmse_test),  4)
)

cat("\nModel Performance Summary:\n")
print(results_table)

cat("\nInterpretation:\n")
cat("  R² (Coefficient of Determination):\n")
cat("    - Ranges from 0 to 1; closer to 1 = better fit\n")
cat("    - R² = 1 means the model explains 100% of variance in the target\n")
cat("    - Large gap between train R² and test R² suggests overfitting\n\n")
cat("  RMSE (Root Mean Squared Error):\n")
cat("    - Lower is better; measured in the same units as the target\n")
cat("    - Penalises large errors more heavily than small ones\n")
cat("    - Train RMSE << Test RMSE suggests overfitting\n")


# =============================================================================
# Q3.5 — Goodness of Fit Scatter Plots (8 Marks)
# =============================================================================

cat("\n=== Q3.5: Goodness of Fit Scatter Plots ===\n")

# Prepare data frames for plotting
plot_mlr <- data.frame(
  Actual    = test_data[[target_col]],
  Predicted = mlr_test_pred,
  Model     = "Multiple Linear Regression"
)
plot_dt <- data.frame(
  Actual    = test_data[[target_col]],
  Predicted = dt_test_pred,
  Model     = "Decision Tree Regression"
)
plot_all <- rbind(plot_mlr, plot_dt)

# Combined goodness-of-fit scatter plot
ggplot(plot_all, aes(x = Actual, y = Predicted, colour = Model)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_abline(slope = 1, intercept = 0, colour = "black",
              linetype = "dashed", linewidth = 0.8) +
  facet_wrap(~ Model, ncol = 2) +
  scale_colour_manual(values = c("Multiple Linear Regression" = "#2980b9",
                                  "Decision Tree Regression"   = "#27ae60")) +
  labs(
    title    = "Goodness of Fit — Actual vs Predicted (Test Set)",
    subtitle = "Dashed line = perfect prediction (y = x)",
    x        = "Actual Values",
    y        = "Predicted Values",
    caption  = paste0(
      "MLR: R² = ", round(mlr_r2_test, 3), ", RMSE = ", round(mlr_rmse_test, 3),
      " | DT: R² = ", round(dt_r2_test, 3), ", RMSE = ", round(dt_rmse_test, 3)
    )
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold"),
    strip.text    = element_text(face = "bold"),
    legend.position = "none"
  )

# Residuals plot
plot_mlr$Residuals <- plot_mlr$Actual - plot_mlr$Predicted
plot_dt$Residuals  <- plot_dt$Actual  - plot_dt$Predicted
plot_resid <- rbind(plot_mlr, plot_dt)

ggplot(plot_resid, aes(x = Predicted, y = Residuals, colour = Model)) +
  geom_point(alpha = 0.5, size = 1.8) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "black") +
  facet_wrap(~ Model, ncol = 2) +
  scale_colour_manual(values = c("Multiple Linear Regression" = "#2980b9",
                                  "Decision Tree Regression"   = "#27ae60")) +
  labs(
    title    = "Residual Plot — Predicted vs Residuals (Test Set)",
    subtitle = "Points should be randomly scattered around 0 for a good model",
    x        = "Predicted Values",
    y        = "Residuals (Actual - Predicted)"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"), legend.position = "none")


# =============================================================================
# Q3.6 — Model Performance Commentary & Recommendations (5 Marks)
# =============================================================================

cat("\n=== Q3.6: Model Performance Commentary ===\n")

cat("
COMMENTARY ON MODEL PERFORMANCES:
-----------------------------------

Multiple Linear Regression (MLR):
  - Assumes a LINEAR relationship between predictors and the target variable
  - Interpretable: each coefficient directly shows the effect of a feature
  - May underfit if the true relationship is non-linear
  - Sensitive to outliers and multicollinearity among features

Decision Tree Regression (DTR):
  - Captures NON-LINEAR relationships without feature scaling
  - Easy to visualise and interpret via tree structure
  - Prone to overfitting (high variance), especially deep trees
  - Performance often improves with pruning or ensemble methods

Comparison (based on R² and RMSE):
  - A higher R² on test data indicates better generalisation
  - A lower RMSE indicates more accurate individual predictions
  - If Decision Tree has high train R² but low test R², it is overfitting

RECOMMENDATIONS TO IMPROVE MODEL PERFORMANCE:
----------------------------------------------
  1. Feature Engineering: create interaction terms, polynomial features, or
     log-transform skewed features to improve linear model fit
  2. Cross-Validation: use k-fold CV (e.g. k=10) for more robust evaluation
  3. Regularisation: apply Ridge or Lasso regression to reduce MLR overfitting
  4. Tree Pruning: reduce Decision Tree depth (maxdepth) or increase cp value
     to prevent overfitting
  5. Ensemble Methods: replace single DT with Random Forest or Gradient
     Boosting for significantly better performance and reduced variance
  6. Outlier Removal: identify and handle outliers that distort regression fits
  7. Feature Selection: remove low-importance or highly correlated predictors
")
