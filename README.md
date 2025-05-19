# Health-Insurance-Cross-Sell-Prediction

## Business questions 
To address low cross-sell conversion rates, our client seeks to refine targeting by predicting which health insurance customers are likely to buy vehicle insurance. 
1.	Which demographic and behavioral traits are most indicative of interest in vehicle insurance?
2.	What sales channels and geographical regions convert most effectively?
3.	How can marketing resources be deployed more strategically based on predictive segmentation?

## Data Preparation 

![Screenshot 2025-05-19 151504](https://github.com/user-attachments/assets/8305d84a-f8e2-464e-aa53-80790de0f4d7)

![Screenshot 2025-05-19 151515](https://github.com/user-attachments/assets/9d4f1745-1f97-49ff-b7a1-677eed1b048b)


![Screenshot 2025-05-19 151527](https://github.com/user-attachments/assets/3b90280e-997a-4ef1-8fec-4febba176886)


## Modeling Techniques
Technique		Validation AUC		Sensitivity		Specificity
Logistic Regression	0.8460			92.9%			67.3%
Random Forest		0.8364			11.4%			97.2%
XGBoost			0.8571			92.9%			67.3%
Linear SVM		0.8459			94.4%			63.7%
![Validation AUC scores](https://github.com/user-attachments/assets/b673d054-77c1-452b-b836-42897be90cf0)

•	XGBoost performed best overall, balancing precision and recall
•	SVM achieved the highest recall, valuable for minimizing missed conversions
•	Random Forest had the highest specificity fewer false positives, but poor recall for interested customers
![Screenshot 2025-05-19 151536](https://github.com/user-attachments/assets/08610b1f-cede-4f26-9948-81f07f5c69e5)

## Interpretations 
•	Which features matter most for customer interest (vehicle damage, premium-per-day, driving license status)
•	Which top-performing channels (Channels 152, 26, and 124) and regions for resource allocation.
•	How segmentation can improve conversion (Premium_Per_Day tiers)

![Screenshot 2025-05-19 151545](https://github.com/user-attachments/assets/194e9215-73ae-4ae3-9a64-c0174f2b7b79)

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
