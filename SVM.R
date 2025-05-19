cat("\014") # clears console
rm(list = ls()) # clears global environment
install.packages("LiblineaR")
library(LiblineaR)
# Load required packages
library(e1071)
library(caret)
library(tidyverse)
library(Matrix)
library(dplyr)
library(ggplot2)
library(LiblineaR)
library(pROC)


# Set seed for reproducibility
set.seed(123)

# Load data
train <- read.csv("C:/Users/dangn/Downloads/ALY/ALY6040 Data Mining/Final Project/Fianl Project Dataset/train.csv")

# Step 1: Preprocessing
# Cap and impute Annual_Premium
cap <- quantile(train$Annual_Premium, 0.99)
med_prem <- median(train$Annual_Premium, na.rm = TRUE)
train$Annual_Premium <- ifelse(train$Annual_Premium > cap, NA, train$Annual_Premium)
train$Annual_Premium[is.na(train$Annual_Premium)] <- med_prem

# Convert categorical variables to factor
factor_vars <- c("Gender", "Vehicle_Age", "Vehicle_Damage",
                 "Policy_Sales_Channel", "Driving_License", "Previously_Insured")
train[factor_vars] <- lapply(train[factor_vars], as.factor)
train$Response <- as.factor(train$Response)

# Feature engineering
train <- train %>%
  mutate(Premium_Per_Day = Annual_Premium / Vintage)

# Step 2: Train/Validation Split
train_index <- createDataPartition(train$Response, p = 0.8, list = FALSE)
train_data <- train[train_index, ]
val_data <- train[-train_index, ]
# 1. Convert factor response to numeric (0 and 1)
train_label <- as.numeric(as.character(train_data$Response))  # Must be numeric for LiblineaR
val_label   <- as.numeric(as.character(val_data$Response))

# 2. Prepare matrix using sparse.model.matrix
train_matrix <- sparse.model.matrix(Response ~ . - id - 1, data = train_data)
val_matrix   <- sparse.model.matrix(Response ~ . - id - 1, data = val_data)

# Combine training data and labels for downsampling
train_df <- as.data.frame(as.matrix(train_matrix))
train_df$Response <- as.factor(train_label)

# Apply downsampling using caret
down_train <- downSample(x = train_df[, -ncol(train_df)], 
                         y = train_df$Response, 
                         yname = "Response")

# Prepare data for LiblineaR
x_down <- as.matrix(down_train[, -ncol(down_train)])
y_down <- as.numeric(as.character(down_train$Response))  # ensure it's 0/1

# Train SVM using LiblineaR
svm_model <- LiblineaR(data = x_down, 
                       target = y_down,
                       type = 6,         # L1-regularized L2-loss SVM (dual)
                       cost = 1, 
                       bias = TRUE,
                       verbose = TRUE)

# 5. Predict on validation data
val_pred <- predict(svm_model, val_matrix, proba = TRUE)
predicted_classes <- val_pred$predictions
val_probs <- val_pred$probabilities[, 2]  # probability of class "1"

# 6. Evaluate performance
conf_mat <- confusionMatrix(as.factor(predicted_classes), 
                            as.factor(val_label), 
                            positive = "1")
print(conf_mat)

roc_val <- roc(val_label, val_probs)
plot(roc_val, col = "purple", main = "SVM ROC (Validation)")
print(sprintf("Validation AUC = %.4f", auc(roc_val)))


# 1. Reload the test set
test <- read.csv("C:/Users/dangn/Downloads/ALY/ALY6040 Data Mining/Final Project/Fianl Project Dataset/test.csv")

# 2. Apply the same preprocessing
cap <- quantile(train$Annual_Premium, 0.99)
med_prem <- median(train$Annual_Premium, na.rm = TRUE)
test$Annual_Premium <- ifelse(test$Annual_Premium > cap, NA, test$Annual_Premium)
test$Annual_Premium[is.na(test$Annual_Premium)] <- med_prem

factor_vars <- c("Gender", "Vehicle_Age", "Vehicle_Damage",
                 "Policy_Sales_Channel", "Driving_License", "Previously_Insured")
test[factor_vars] <- lapply(test[factor_vars], as.factor)

# Align levels with training data
for (var in factor_vars) {
  test[[var]] <- factor(test[[var]], levels = levels(train[[var]]))
}

# Feature engineering
test <- test %>%
  mutate(Premium_Per_Day = Annual_Premium / Vintage)

# Filter complete cases to avoid matrix build errors
test <- test %>% filter(complete.cases(.))

# 3. Build sparse matrix for test set
test_matrix <- sparse.model.matrix(~ . - id - 1, data = test)

# 4. Align test matrix columns with training matrix
train_cols <- colnames(train_matrix)
common_cols <- intersect(colnames(test_matrix), train_cols)
test_matrix_aligned <- test_matrix[, common_cols, drop = FALSE]

missing_cols <- setdiff(train_cols, common_cols)
if (length(missing_cols) > 0) {
  mat_zeros <- Matrix::Matrix(0, 
                              nrow = nrow(test_matrix_aligned), 
                              ncol = length(missing_cols),
                              sparse = TRUE)
  colnames(mat_zeros) <- missing_cols
  test_matrix_aligned <- cBind(test_matrix_aligned, mat_zeros)
}

# Reorder columns to match training
test_matrix_aligned <- test_matrix_aligned[, train_cols]

# 5. Predict with the trained SVM model
test_probs_svm <- predict(svm_model, test_matrix_aligned, proba = TRUE)
test_pred_svm <- ifelse(test_probs_svm$probabilities[, 2] > 0.5, 1, 0)

# 6. Create submission file
submission_svm <- data.frame(id = test$id, Response = test_pred_svm)
write.csv(submission_svm, "svm_submission.csv", row.names = FALSE)
