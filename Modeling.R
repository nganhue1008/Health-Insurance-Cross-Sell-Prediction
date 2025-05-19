cat("\014") # clears console
rm(list = ls()) # clears global environment
install.packages("randomForest")
install.packages("xgboost")
install.packages("Matrix")
# Load libraries
library(tidyverse)
library(skimr)
library(caret) #for model splitting
library(pROC)
library(randomForest)
library(Matrix)
library(xgboost)
library(dplyr)
library(ggplot2)

# Load datasets
train <- read.csv("C:/Users/dangn/Downloads/ALY/ALY6040 Data Mining/Final Project/Fianl Project Dataset/train.csv")
test <- read.csv("C:/Users/dangn/Downloads/ALY/ALY6040 Data Mining/Final Project/Fianl Project Dataset/test.csv")

# Set a random seed for reproducibility
set.seed(123)
# Handle Outliers for Annual_Premium (based on Train data)
cap <- quantile(train$Annual_Premium, 0.99) # 99th percentile cap

train$Annual_Premium <- ifelse(train$Annual_Premium > cap, NA, train$Annual_Premium)
test$Annual_Premium <- ifelse(test$Annual_Premium > cap, NA, test$Annual_Premium)

# Step 2: Impute Missing Values (Annual_Premium) with Median
median_premium <- median(train$Annual_Premium, na.rm = TRUE)
train$Annual_Premium[is.na(train$Annual_Premium)] <- median_premium
test$Annual_Premium[is.na(test$Annual_Premium)] <- median_premium

# Step 3: Convert Variables to Factors
factor_vars <- c("Gender", "Vehicle_Age", "Vehicle_Damage",
                 "Policy_Sales_Channel", "Driving_License", "Previously_Insured")

train[factor_vars] <- lapply(train[factor_vars], as.factor)
test[factor_vars] <- lapply(test[factor_vars], as.factor)

# Response variable is only in train
train$Response <- as.factor(train$Response)

# Step 4: Feature Engineering - Premium per Day
train <- train %>%
  mutate(Premium_Per_Day = Annual_Premium / Vintage)

test <- test %>%
  mutate(Premium_Per_Day = Annual_Premium / Vintage)

# Step 5: Split Train into Training and Validation Sets (80/20 split)
train_index <- createDataPartition(train$Response, p = 0.8, list = FALSE)
train_data <- train[train_index, ]
val_data <- train[-train_index, ]

#Fix Factor Levels (Important for Categorical Variables)
train_data$Policy_Sales_Channel <- factor(train_data$Policy_Sales_Channel)
val_data$Policy_Sales_Channel <- factor(val_data$Policy_Sales_Channel, 
                                        levels = levels(train_data$Policy_Sales_Channel))




## LOGISTIC REGRESSION
# Step 1: Handle Class Imbalance with Class Weights
# Create manual weights
class_weights <- ifelse(train_data$Response == 1,
                        (nrow(train_data) / sum(train_data$Response == 1)),
                        (nrow(train_data) / sum(train_data$Response == 0)))

# Step 2: Train Logistic Regression Model
logit_model <- glm(Response ~ . - id, 
                   data = train_data, 
                   family = binomial(link = "logit"), 
                   weights = class_weights)

# Step 3: Predict on Validation Set
val_probs <- predict(logit_model, newdata = val_data, type = "response")
# threshold at 0.5 to classify
val_pred <- ifelse(val_probs > 0.5, 1, 0)
val_actual <- as.numeric(as.character(val_data$Response))

# Step 4: Evaluate Performance
confusionMatrix(as.factor(val_pred), as.factor(val_actual), positive = "1")

# ROC Curve and AUC
roc_curve <- roc(val_actual, val_probs)
plot(roc_curve, col = "blue")
auc(roc_curve)

test$Policy_Sales_Channel <- factor(test$Policy_Sales_Channel, 
                                    levels = levels(train_data$Policy_Sales_Channel))

# Predict probabilities on test set
test_probs <- predict(logit_model, newdata = test, type = "response")

# Convert probabilities to 0 or 1 (using 0.5 cutoff)
test_pred <- ifelse(test_probs > 0.5, 1, 0)

# Create a submission dataframe
submission_logit <- data.frame(id = test$id, Response = test_pred)
# Save to CSV
write.csv(submission_logit, "logistic_submission.csv", row.names = FALSE)


#RANDOM FOREST
# Step 1: Train Random Forest Model
# Convert Policy_Sales_Channel to numeric
train_data$Policy_Sales_Channel <- as.numeric(as.character(train_data$Policy_Sales_Channel))
val_data$Policy_Sales_Channel <- as.numeric(as.character(val_data$Policy_Sales_Channel))
test$Policy_Sales_Channel <- as.numeric(as.character(test$Policy_Sales_Channel))

# We handle class imbalance using the 'classwt' argument
rf_model <- randomForest(Response ~ . - id, 
                         data = train_data, 
                         ntree = 500,            # Number of trees
                         mtry = 5,               # Number of variables tried at each split
                         importance = TRUE,      # Calculate variable importance
                         classwt = c("0" = 1, "1" = nrow(train_data) / sum(train_data$Response == 1)))

# Step 2: Predict on Validation Set
val_pred_rf <- predict(rf_model, newdata = val_data, type = "response")
val_actual_rf <- val_data$Response

# Step 3: Evaluate Model Performance
confusionMatrix(val_pred_rf, val_actual_rf, positive = "1")

# ROC Curve and AUC
val_probs_rf <- predict(rf_model, newdata = val_data, type = "prob")[,2]
roc_curve_rf <- roc(as.numeric(as.character(val_actual_rf)), val_probs_rf)
plot(roc_curve_rf, col = "forestgreen")
auc(roc_curve_rf)

# Predict on the test set
test_pred_rf <- predict(rf_model, newdata = test, type = "response")

# Create a submission dataframe
submission_rf <- data.frame(id = test$id, Response = test_pred_rf)

# Save to CSV
write.csv(submission_rf, "random_forest_submission.csv", row.names = FALSE)

# XGBoost
# Build full design matrix once
full_train_matrix <- sparse.model.matrix(Response ~ . - id - 1, data = train)
full_train_label  <- as.numeric(as.character(train$Response))

# Split via the same index you used earlier
train_matrix <- full_train_matrix[train_index, ]
val_matrix   <- full_train_matrix[-train_index, ]

train_label  <- full_train_label[train_index]
val_label    <- full_train_label[-train_index]

# Create DMatrix
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dval   <- xgb.DMatrix(data = val_matrix,   label = val_label)
# Calculate scale_pos_weight from your train labels
scale_pos_weight <- sum(train_label == 0) / sum(train_label == 1)

# Re-define the params list
params <- list(
  booster          = "gbtree",
  objective        = "binary:logistic",
  eval_metric      = "auc",
  scale_pos_weight = scale_pos_weight,
  eta              = 0.1,
  max_depth        = 6,
  subsample        = 0.8,
  colsample_bytree = 0.8
)


# Then train as before:
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(val = dval),
  early_stopping_rounds = 10,
  verbose = 1
)
# 12. Evaluate on validation set
val_probs <- predict(xgb_model, newdata = val_matrix)
val_pred  <- ifelse(val_probs > 0.5, 1, 0)
print(confusionMatrix(as.factor(val_pred), as.factor(val_label), positive = "1"))
roc_val <- roc(val_label, val_probs)
plot(roc_val, col = "red", main = "XGBoost ROC (Validation)")
print(sprintf("Validation AUC = %.4f", auc(roc_val)))


# 1. Reload test.csv fresh
test <- read.csv("C:/Users/dangn/Downloads/ALY/ALY6040 Data Mining/Final Project/Fianl Project Dataset/test.csv")

# 2. Preprocessing

# 2a. Outlier cap
cap      <- quantile(train$Annual_Premium, 0.99)
med_prem <- median(train$Annual_Premium, na.rm = TRUE)

test$Annual_Premium <- ifelse(test$Annual_Premium > cap, NA, test$Annual_Premium)
test$Annual_Premium[is.na(test$Annual_Premium)] <- med_prem

# 2b. Fix Policy_Sales_Channel safely
test$Policy_Sales_Channel <- as.character(test$Policy_Sales_Channel)
train_levels <- levels(train$Policy_Sales_Channel)
test$Policy_Sales_Channel[!test$Policy_Sales_Channel %in% train_levels] <- "Other"
test$Policy_Sales_Channel <- factor(test$Policy_Sales_Channel, levels = c(train_levels, "Other"))

# 2c. Fix other categorical variables
test$Gender             <- factor(test$Gender, levels = levels(train$Gender))
test$Driving_License    <- factor(test$Driving_License, levels = levels(train$Driving_License))
test$Previously_Insured <- factor(test$Previously_Insured, levels = levels(train$Previously_Insured))
test$Vehicle_Age        <- factor(test$Vehicle_Age, levels = levels(train$Vehicle_Age))
test$Vehicle_Damage     <- factor(test$Vehicle_Damage, levels = levels(train$Vehicle_Damage))

# 2d. Feature engineer Premium_Per_Day
if (!"Premium_Per_Day" %in% names(test)) {
  test$Premium_Per_Day <- test$Annual_Premium / test$Vintage
}

# ✅ 2e. ***NOW filter complete cases before building matrix***
test <- test %>% filter(complete.cases(.))

# 3. Build matrix
test_matrix_raw <- sparse.model.matrix(~ . - id - 1, data = test)

# 4. Align columns
train_cols <- colnames(train_matrix)
common_cols <- intersect(colnames(test_matrix_raw), train_cols)
test_matrix_aligned <- test_matrix_raw[, common_cols, drop = FALSE]

missing_cols <- setdiff(train_cols, common_cols)
if (length(missing_cols) > 0) {
  mat_zeros <- Matrix::Matrix(0, 
                              nrow = nrow(test_matrix_aligned), 
                              ncol = length(missing_cols),
                              sparse = TRUE)
  colnames(mat_zeros) <- missing_cols
  test_matrix_aligned <- cBind(test_matrix_aligned, mat_zeros)
}
test_matrix_aligned <- test_matrix_aligned[, train_cols]

# 5. Predict
test_probs <- predict(xgb_model, newdata = test_matrix_aligned)
test_pred  <- ifelse(test_probs > 0.5, 1, 0)

# ✅ Now finally dimensions match
dim(test_matrix_aligned)
length(test$id)
length(test_pred)

# 6. Save submission
submission_xgb <- data.frame(id = test$id, Response = test_pred)
write.csv(submission_xgb, "xgboost_submission.csv", row.names = FALSE)

# Create a dataframe with model names and their AUC scores
model_perf <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "XGBoost"),
  AUC = c(0.8460, 0.8364, 0.8571)
)

# Plotting
ggplot(model_perf, aes(x = Model, y = AUC, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = round(AUC, 4)), vjust = -0.5, size = 4) +
  coord_cartesian(ylim = c(0.8, 0.9)) +   # Use coord_cartesian instead of ylim
  labs(title = "Model Validation AUC Comparison",
       x = "Model",
       y = "AUC Score") +
  theme_minimal() +
  theme(legend.position = "none")
