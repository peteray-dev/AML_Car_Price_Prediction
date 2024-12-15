# AML_Car_Price_Prediction

# Autotrader Car price prediction model

## DATA PROCESSING FOR MACHINE LEARNING

The car sale advert dataset used in this task was gotten from Autotrader, one of the university industry partners. The dataset contains several information about different vehicles including their prices, other features include mileage, year of registration, reg_code, standard model, standard make, vehicle condition, body type, crossover van and cars, fuel type, body type, standard colour, and public reference. The dataset has a shape of 402,005 rows and 12 columns. In this dataset there are some missing values which might have happened because of mistakes during entry or technical issue, all these were technically handles, however, reg_code and year of registration have the highest number of null values. Having a statistical understanding of the data helped to identify patterns, understand how data points are distributed and identify potential outliers, this creates a foundation for making a data driven decision.

 ![image](https://github.com/user-attachments/assets/d0e57b68-6795-43c3-89e5-e3e3de8e3591)

Figure: Overview of the dataset

These missing values were replaced using several methods. It was noticed that the new vehicles present have no year of registration and registration code. Since the data collected has a maximum year of 2020, then all the new vehicles were given a year of registration of 2021 which is the following year. According to the vehicle registration in the United Kingdom, vehicles registered between March and August have its registration code as the last two digit of that year, for instance if the year is 2018, then the code will be 18, else 50 will be added to the last two digit of the year, in this case (18+50 = 68). Therefore, for all the new vehicles registered in 2021 which is the following year, 21 and 71 were shuffled between the cars because the actual registration period cannot be predicted. Other methods such as simple imputer and KNN imputer were also used to fill in the missing values through pipeline.

Furthermore, the data contains outliers, all outliers in mileage were removed using the interquantile range technique, values below the lower limit and above the upper limit were removed. From the dataset analysis, it was identified that some vehicles that were said to be registered in period above 2001 have a registration code with alphabet which violates the vehicle registration specification, all these vehicles were removed as well as those below 2001 that has a number registration code. Since this is a regression task, all categorical features were encoded using target encoding and one- hot encoding, this transformation converts them into numerical data making it suitable to build a machine learning model. To ensure that the features with higher values do not dominate the model learning, rescaling of the data was employed to have all the features on similar scale.
 
The dataset was stratified based on the standard model and standard make so that certain classes which are the majority won’t overshadow the minority during splitting. Subsequently, the dataset was split into training, validation, and test in the ratio of 80:10:10 respectively. The validation dataset was used for hyperparameters fine tuning before final evaluation on the test set.

 ![image](https://github.com/user-attachments/assets/39066dea-0f00-45a2-93ee-f7785375b06b)

Figure: Splitting of the dataset into train, validation, and test

## 2.	FEATURE ENGINEERING
Based on the domain knowledge, some features were generated from the existing features, for instance, the vehicle age which reveals how long a car has been used before being sold. This was further analyzed with the price and mileage which shows a slightly negative and positive correlation respectively. The mileage level feature was created from mileage so that the transformation will be in a more interpretable format for the model.

 ![image](https://github.com/user-attachments/assets/ee82c4fb-c22d-43ca-a0b6-f148d5bb5634)

Figure: feature engineering based on domain knowledge

To capture the non- linear relationship between the features and the target variable, polynomial feature of the second order was used to automatically generate features. This is because some features and the target variable exhibit non-linear relationships. Also, the interaction helps with the understanding of the effect of multiple features on target variable. These techniques generated 77 independent features.
 
 ![image](https://github.com/user-attachments/assets/080c2c4f-9783-48ab-a5c0-c92859c34306)

Figure: Polynomial and Interaction Features

## 3.	FEATURE SELECTION AND DIMENSIONALITY REDUCTION

There are some features that are not contributing to the prediction of vehicle prices based on basic knowledge of the feature in the dataset. Features like ‘crossover car and van’, ‘reg_code’, ‘public reference’ have no contribution to price prediction based on intuition. Univariate and correlation analysis reveals a perfect insight into some of the features being more important to predicting price.

![image](https://github.com/user-attachments/assets/aeb3e4f3-c295-4696-87e8-2e9c9d4c7369)

Figure: Univariate Analysis

From the above figure, ‘public reference’ and ‘crossover cars and van’ may not be a better feature for price prediction because of their distribution. Although this may not necessarily be the case, it is important that more analysis is performed, most especially the understanding in relation to price (target variable). Using heatmap (bivariate analysis), some features have a very low correlation with price, these features might contribute less to prediction, it is worthy of note that some features selected earlier are also showing a very low correlation, also some features are highly correlated which may mean that they both represent the same information of which we can exclude one of them with less importance based on the domain knowledge, for instance ‘year of registration’ and ‘reg_code’.
 
![image](https://github.com/user-attachments/assets/8884e203-d4df-4709-a631-d9e323d24eee)
 

Figure: Correlation Analysis

To uncover hidden patterns that might be difficult to uncover using the above analysis, we can consider using automatic feature selection because it reduces bias. Automatic feature selection such as SelectKBest, Recursive Feature Elimination, and Sequential Feature Selection was used in the process and a technique was selected based on its performance and efficiency using linear regression as an estimator because it is computationally efficient. Each of these was carried out on both the original features and polynomial feature to understand the performance improvements. Generally, there was a performance improvement on the usage automatic feature selection on the polynomial features compared to the original features. RFE automatically selected 15 features out of 77 features generated by polynomial and interaction while the sequential only selected 3 features. Overall, RFE gave a better performance and selected reasonable features.

![image](https://github.com/user-attachments/assets/15a82b6d-114a-4e1c-a3e1-749883010a68)

![image](https://github.com/user-attachments/assets/478f277d-531f-488a-9b8d-ce1d72e7f055)

Figure: Automatic feature selection and dimensionality reduction

The dimensionality reduction technique used is Principal Component Analysis because it reduces the number of features generated by polynomials and interactions while preserving key information as much as possible. The number of components selected was based on explained variance ratio which captured about 98% of the variance which is 9 components. The selected number of components was experimented on how it contributes to the model performance.
 
## 4.	MODEL BUILDING
## 4.1.	Linear Model

By analyzing how models respond to feature selection and dimensionality reduction techniques on both the original dataset features and their polynomial transformations, it is possible to choose models with superior performance. Doing this contributed to studying the model performance. Generally, using models on the original dataset shows a sense of overfitting and a very large Mean Absolute Error (MAE).

![image](https://github.com/user-attachments/assets/a2ab47d8-2aba-4e61-b83a-def6b90b1754)

Figure: Linear regression with SelectKBest

Linear Regression on Polynomial and Interaction Transformation
	Training
Score	Validation
Score	Training
MAE	Validation
MAE
Linear Regression +
KBest	0.96	0.77	4069.28	4160.83
Linear Regression +
RFECV	0.97	0.83	3365.05	3431.81
Linear Regression +
SFS	0.96	0.82	3637.04	3728.26
Linear Regression +
PCA	0.91	0.66	5753.69	5642.35
Table: Linear Regression score and metrics

From the table above, linear regression with recursive feature elimination stands out. It exhibits a lower difference in both mean absolute error and score between the training and validation data. This combined observation suggests the model might be performing well and generalizing effectively when compared to others. Linear regression with PCA shows overfitting and a very high mean absolute error.

## 4.2.	Random Forest Regression

Random forest is simply the combination of multiple decision trees to improve performance. At initial stage, there was no hyperparameter tuning, the default parameters for random forest were used which includes: n_estimators = 100, max_depth = None, min_samples_leaf = 1, etc, according to sklearn documentation. From the table below, the model performance using different types of feature selection techniques shows a better performance. As highlighted earlier, recursive feature selection is the best technique for selection, it is also shown when used with random forest.
 
 ![image](https://github.com/user-attachments/assets/372b0878-a814-4901-81a6-3eb149d1868f)

Figure: Random Forest Regression

Random Forest Regressor on Polynomial and Interaction Transformation
	Training
Score	Validation
Score	Training
MAE	Validation
MAE
Random	Forest
Regression	+ SelectKBest	0.89	0.86	1204.82	2531.62
Random	Forest
Regression + RFECV	0.89	0.89	896.84	2203.61
Random	Forest
Regression + SFS	0.91	0.87	1091.03	2561.97
Random	Forest
Regression + PCA	0.87	0.86	1144.15	2893.00

Table: Random Forest score and metrics

Random forest shows a lower mean absolute error when compared to linear model, also the case of overfitting was also resolved using random forest.
To select the best parameter with optimal performance, performing grid search proved computationally expensive, taking a long time to execute even after reducing the parameters. Despite using basic parameters, the grid search only identified options with relatively low scores (around 50%) as the best. To resolve this, more parameters should be provided.

## 4.3.	Gradient Boosting Regression

![image](https://github.com/user-attachments/assets/c1276cd5-a92c-46d5-8f00-77e3fc10d318)

Figure: Gradient Boosting Regression
 
Gradient Boosting Regression on Polynomial and Interaction Transformation
	Training
Score	Validation
Score	Training
MAE	Validation
MAE
Gradient	Boosting Regression		+
SelectKBest	0.98	0.83	3351.32	3436.44
Gradient	Boosting
Regression + RFECV	0.98	0.86	2920.04	3032.77
Gradient	Boosting
Regression + SFS	0.98	0.85	3157.64	3272.71
Gradient	Boosting
Regression + PCA	0.97	0.82	3829.93	3957.50
Table: Gradient Boosting Regression score and metrics.


The training accuracy score is high compared to other models and at the same time it has a high mean absolute error compared to random forest results. The model with recursive feature elimination performs much better, this confirms that RFECV is a better selection technique for this task. To enhance the model, grid search will help identify the optimal hyperparameter which will prevent overfitting and potentially generalize more accurately, the problem faced at this stage was because the operation was computationally intensive which take a longer time to run.

Overall, random forest appears to be a strong model for this task because it has a very low mean absolute error and there is also a low difference between the training and validation score.

4.4.	Averager/Voter/Stacker Ensemble
Voting regression is an ensembling technique that makes predictions based on the average of multiple regression models, in this case, linear regression, random forest, and gradient boosting regression, to achieve a better performance.

![image](https://github.com/user-attachments/assets/dacde31f-4883-4db5-a461-50095cecde24)

Figure: Voting Regression


	Training
Score	Validation
Score	Training
MAE	Validation
MAE
Ensemble + RFECV	0.98	0.88	2219.93	2641.47
 
Using the feature selected by RFECV, the ensemble model shows signs of slight overfitting, this can be avoided by grid search on random forest and gradient boosting regressor to select the best hyperparameter to improve performance. Although this is computationally intensive.

## 5.	MODEL EVALUATION AND ANALYSIS

## 5.1.	Overall Performance with Cross-Validation

To further improve the performance of the models, a 5-fold cross validation using a negative mean absolute error as evaluation metrics was used. This process involved the splitting of the data into 5 folds, training and testing repeatedly on all folds to ensure a more robust estimate of performance on unseen data.

 ![image](https://github.com/user-attachments/assets/fa30ddff-fda9-4a72-98c4-2e1c4bc9fe2e)

Figure: Cross validation of the models

![image](https://github.com/user-attachments/assets/7bcb0766-bbd3-4763-9463-1146248337cb)

Figure: Result of cross validation

From the above result, the standard deviation of the gradient boosting regression and ensemble is low on both training and testing data indicating a more consistent performance across the data split. Random forest regression and linear regression shows consistency on the training data but has a high variation on the test data. It is important that the model shows a lower MAE and STD to ensure an accurate price prediction. Amongst all, the ensemble model appears to be the best because of its low mean absolute error and standard deviation on the test data.

5.2.	True vs Predicted Analysis
This analysis helps to understand how far the predicted price by each model deviates from the actual price. The first 10 car instances in the test data were used for this analysis.
 
 ![image](https://github.com/user-attachments/assets/e072e867-a087-4c31-84ab-12798bb01b3e)

Figure: Predicted price vs actual price

From the above figure, it is seen that the model predicted prices are very close to the true price. The predictions above 30,000 seem to be higher than the actual value, but random forest predicted a more closer price than other model.
Based on time analysis, it took a longer execution time for ensemble and random forest compared to linear model and gradient boosting. Looking at the scatterplot of actual vs predicted prices for ensemble model, it is seen that this model predicts almost an accurate price, with points closer to the diagonal line.

5.3.	Global and Local Explanation with SHAP
The importance of local explanation is to understand the contribution of each feature to the prediction of vehicle price of an instance. Each prediction made has different feature contributing effect. In this task, the explanation of predictions made by gradient boosting regression will be examined.

![image](https://github.com/user-attachments/assets/2e2cb859-0eab-4e60-a97d-ecfa7cab2fe1)

Figure: Explanation of predicted price of an instance

The baseline ∑[f(X)]– the average predicted probability is 16607.217. The first instance in the test data has a high predicted price of 18694.786. Features such as ‘year of registration’, ‘year of
 
registration fuel type’ and ‘mileage’ with values 0.966, 0.121 and 0.106 respectively contributed to the increment of the predicted price, on the other hand the low value of standard model (0.019) decreases the predicted price. The addition of these values to baseline price resulted in the price prediction. Based on the car information, the explanation made by Shap corresponds with what one would have expected based on domain knowledge.
The features with the large mean absolute shapley values are classified as the most important features. In this case, the ‘standard model’, ‘reg_code standard_make’, ‘mileage vehicle condition_USED’ and ‘mileage’ are the most important features. The figure below explains the feature importance with feature effect, the high value of the first two important features contributes to the increase in the predicted price while the lower value does not really show a significant reduction in the price. Only the low value of ‘mileage standard_model’ significantly reduces the predicted price. It is worthy to note that each point represents the shapley value of a feature and an instance.

![image](https://github.com/user-attachments/assets/c6151d87-9b5d-4e84-8c54-2552ac4ad259)

Figure: Shap Summary Plot (Global Explanation)

# 5.4.	Partial Dependency Plots

Partial Dependency plot helps to show the effect that one or two features have on the predicted prices, this shows a pictorial representation of relationship that exists between the target and independent features, whether they are linear or complex. The PDP analysis below interprets the ensemble model’s prediction, an increase in mileage resulted in a perfect increase in price prediction.
From the analysis, the increase in year of registration^2 (polynomial feature) slightly increases the price in opposition to year of registration which decreases the price. The PDP analysis of interaction feature like ‘standard_model body_type’ and ‘standard_model fuel_type’ shows a non- linear relationship, and slightly drops as the feature increases. Moreso, price increases as the standard_model increases while price decreases as fuel_type increases. For interacting features, such as ‘standard_model fuel_type’, the price stops increasing and flattens at 0.0035, indicating that any increase from this point has no effect on the price prediction.

Further analysis was also done on random forest and gradient boosting prediction, it was revealed that these models also made a similar plot as ensemble model.
 
 ![image](https://github.com/user-attachments/assets/5875cd25-5c56-451d-af56-5536b45a0732)

![image](https://github.com/user-attachments/assets/3f92115f-8a68-401c-9ab9-f085c2e463a5)

![image](https://github.com/user-attachments/assets/01cf879e-bd94-44c0-9952-db80058da2e5)

Figure: Partial Dependency Plot for some features

![image](https://github.com/user-attachments/assets/e2749921-5bda-41b6-ad47-36735babbd68)

Figure: 2D Partial Dependence Plot analysis

The figure above explains the price prediction for ‘year of registration’ and ‘standard model’, it is seen that when a vehicle of high model and lower year of registration is given to the model, a high price is predicted, which corresponds with the domain knowledge/analysis.


