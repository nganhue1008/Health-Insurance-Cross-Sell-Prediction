cat("\014") # clears console
rm(list = ls()) # clears global environment

# Step 1: Loading Data and Initial Inspection
# Load libraries
library(tidyverse)
library(skimr)
library (lattice)
library(caret) #for model splitting

# Load the dataset
train <- read.csv("train.csv")

# Inspect dataset structure and summary
str(train)
skim(train)

# Step 2: Check for Missing Values and Duplicates
# Check for missing values
missing_values <- colSums(is.na(train))
print(missing_values)

# Check for duplicates
duplicates <- train[duplicated(train), ]
nrow(duplicates)

#Step 3: Data Cleaning and Transformation
# Convert variables to appropriate types
train <- train %>%
  mutate(
    Gender = as.factor(Gender),
    Driving_License = as.factor(Driving_License),
    Previously_Insured = as.factor(Previously_Insured),
    Vehicle_Age = as.factor(Vehicle_Age),
    Vehicle_Damage = as.factor(Vehicle_Damage),
    Policy_Sales_Channel = as.factor(Policy_Sales_Channel),
    Response = as.factor(Response)
  )

# Verify the transformations
str(train)

#Step 4: Visualizing distributions 
#categorical variables
# Gender distribution
ggplot(train, aes(Gender, fill = Gender)) +
  geom_bar() + theme_minimal()

# Response distribution (Target variable)
ggplot(train, aes(Response, fill = Response)) +
  geom_bar() + theme_minimal()
#Continuos variables 
# Age distribution
ggplot(train, aes(Age)) +
  geom_histogram(bins = 30, fill = 'blue', color='white') + theme_minimal()

# Annual Premium distribution
ggplot(train, aes(Annual_Premium)) +
  geom_histogram(bins = 30, fill = 'green', color='white') + theme_minimal() +
  xlim(quantile(train$Annual_Premium, c(0.01, 0.99))) # Removes extreme outliers visually

#Step 5: Handling outlier 
# Visual inspection for outliers in Annual_Premium
ggplot(train, aes(y = Annual_Premium)) +
  geom_boxplot(fill='orange') + theme_minimal()

# Replace extreme Annual_Premium values with NA (outside 99th percentile)
threshold <- quantile(train$Annual_Premium, 0.99)
train$Annual_Premium <- ifelse(train$Annual_Premium > threshold, NA, train$Annual_Premium)

# Check missing again after outlier removal
sum(is.na(train$Annual_Premium))

# Step 6: Covariation Between Variables

# Categorical vs Continuous

# Boxplot of Annual Premium vs Vehicle Age
ggplot(train, aes(Vehicle_Age, Annual_Premium, fill=Vehicle_Age)) +
  geom_boxplot() + theme_minimal()

# Two Categorical Variables
# Response by Gender
ggplot(train, aes(Gender, fill = Response)) +
  geom_bar(position = "fill") + theme_minimal()

# Two Continuous Variables
# Age vs Annual Premium scatterplot
ggplot(train, aes(Age, Annual_Premium)) +
  geom_point(alpha=0.3) + theme_minimal()

# Key Statistical Summaries
# Summary statistics
summary(train)

# Response rate by Vehicle Damage
train %>%
  group_by(Vehicle_Damage) %>%
  summarise(Response_rate = mean(as.numeric(as.character(Response)))) 

