# Health-Insurance-Cross-Sell-Prediction

## Business questions 
To address low cross-sell conversion rates, our client seeks to refine targeting by predicting which health insurance customers are likely to buy vehicle insurance. 
1.	Which demographic and behavioral traits are most indicative of interest in vehicle insurance?
2.	What sales channels and geographical regions convert most effectively?
3.	How can marketing resources be deployed more strategically based on predictive segmentation?

## Data Preparation 
•	Dataset: Over 380,000 customer records with demographic and behavioral attributes
•	Preprocessing: Outliers in Annual_Premium were capped at the 99th percentile; missing values were imputed using the median
•	Factor conversion: Categorical variables such as Vehicle_Damage and Policy_Sales_Channel were recast as factors
•	Feature engineering: Introduced Premium_Per_Day as a normalized cost indicator (premium divided by tenure)

## Modeling Techniques
Technique		Validation AUC		Sensitivity		Specificity
Logistic Regression	0.8460			92.9%			67.3%
Random Forest		0.8364			11.4%			97.2%
XGBoost			0.8571			92.9%			67.3%
Linear SVM		0.8459			94.4%			63.7%

•	XGBoost performed best overall, balancing precision and recall
•	SVM achieved the highest recall, valuable for minimizing missed conversions
•	Random Forest had the highest specificity fewer false positives, but poor recall for interested customers

## Key insights
Customer Behavior Driver
•	Vehicle_Damage = Yes significantly correlates with higher response likelihood (~24%) ~ 0.5% for No 
•	Customers between 30 and 50 years old with moderate Annual_Premium levels show a higher propensity to respond positively to vehicle insurance offers. (See Figure 1.)
•	Feature: Premium_Per_Day identified high conversion for cost-conscious users Conversion by Channel and Region
•	Channels 152, 26, and 124 yield the most conversions and highest customer engagement 
•	Channels 163 and 157 showed the highest conversion rates (efficiency)
Segmentation-Driven Recommendations
•	High conversion groups share combinations of vehicle damage history, mid-range premiums, and active driving status
•	Feature Premium_Per_Day helps rank customers by financial sensitivity, guiding value-based targeting 

## Interpretations 
•	Which features matter most for customer interest (vehicle damage, premium-per-day, driving license status)
•	Which top-performing channels (Channels 152, 26, and 124) and regions for resource allocation.
•	How segmentation can improve conversion (Premium_Per_Day tiers)

## Limitations 
•	Premium_Per_Day is highly skewed, which may introduce bias when used for segmentation. Normalization or binning to reduce distortion may benefit future preprocessing.
•	Regional codes are anonymized and may contain confounding factors (e.g., urban vs. rural differences) that are not currently captured. Enriching the dataset with socioeconomic or geographic context would improve segmentation.
•	Data Gaps: Key variables such as income, claims history, or digital behavior (e.g., clicks, app activity) were not available, which may affect both accuracy and business relevance.

## Recommendations 
Customer Targeting:
•	Focus on those with prior vehicle damage, mid-range premiums, and no prior insurance
Channel Development
•	Double down on Channel 152 and analyze what practices differentiate it
•	Reduce investment in underperforming channels or test new messaging strategies
Feature Expansion for Future Cycles
•	Income Level: Can help determine affordability and risk tolerance
•	Credit Score or Risk Band: Often correlates with financial reliability and future payment behavior
•	Historical Claims Data: Reveals prior insurance activity or risk exposure
•	Policy Bundling Status: Indicates multi-policy customers who are more receptive to cross-sell offers
•	Customer Tenure & Engagement Metrics: App usage frequency, login counts, or email click rates reflect digital engagement and interest
