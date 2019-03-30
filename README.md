# Diabetes

Abstract:

opportunity (Diabetes)
challenge (To predict the relationship between BMI and expected Y value)
action (Primarily use linear regression along with cross-validation)
resolution (Inconclusive due to high errors and low scores)
Diabetes is on the rise in recent years and the trend shows an increase, especially among youths (aged 10-19), in both diagnosed and undiagnosed cases, serious complications can develop without timely diagnosis and treatments. The risk of developing type 2 diabetes in those who are obese are at four times greater.  
 
The dataset provided can be obtained on sklearn here: Dataset, This Diabetes sample consists of 442 diabetes patients having 10 numeric feature measurements including age, sex, bmi, bp, S1, S2, S3, S4, S5, S6 (S* = particular type of Serum) with an expected value of Y, measured over a year period. The study is to find out the likelihood of diabetes in relation to one’s body mass. There are plenty evidence that shows Obesity dramatically increase the chances of becoming diabetic or/and being undiagnosed. The purpose of this study is to find a model that will best predict such occurrences.  

Introduction:

A study into whether we could predict the type of diabetes patients would have been a very enlightening project. Based on past studies which have shown that Type I diabetes tend to affect more youth, though new trends are showing that the rate at which youths who are afflicted by Diabetes II are catching up with Type I. Nonetheless, our dataset appears to contain very few data points for patients in that age group (10-19). The focus of our study will be on investigating the relationship between Increase in bmi (body weight) and also increases in Y(Expected value).

The diabetes dataset involves supervised learning with continuous output values. Below, we will discuss the various regressors ranging from linear regression, cross-validation (LASSO) clustering are to be employed in an attempt to help answer some interesting questions.   


Methods:

Regression analysis - The goal of this exercise is to fit a linear regression line to each feature fitted against the expected value Y (a quantitative measure of disease progression one year later). The sklearn data have been mean centered and scaled by the standard deviation times number of samples, in other words, the sum of squares of each columns amount to 1. The scaled data should produce no side effects on the shape of the resulting figures as the data are still proportional. First, we will plot the data on a scatter plot per individual feature and then fit a regression line through the data points. Mean squared error and R2 values will be studied to determine the correlations of each feature.    

Cross Validation - The details of the implementation using plotly can be referred to this page. This analysis was taken out of the plot cross-validation with linear models section. The objective is to evaluate the trustworthiness of the selection of alpha. 

LASSO - Lasso cross-validation is a cross-validation estimator that returns an alphas value. The greater the value the better fit it is for modeling. The pseudocode can be found here. Using internal alpha parameters as well as external ones to make performance comparison.  


Results:


Regression analysis

|![regr_BMI](https://user-images.githubusercontent.com/42139196/55275748-f8084200-52c0-11e9-8db7-f36d8f04a76f.png)|
![regr_S5](https://user-images.githubusercontent.com/42139196/55275752-140be380-52c1-11e9-8ea4-df40cb91de31.png)|
![regression_analysis_table](https://user-images.githubusercontent.com/42139196/55275820-41a55c80-52c2-11e9-97f6-e4f2ec7c58fc.png)

The MSE and R2 score are the key criterias in this analysis, in this particular study both the BMI (Body Mass Index) and S5 have the lowest MSE (mean square error) and both have the highest R2 scores. Even though the higher values of R2, we favored the findings on the low MSE figures in selecting the preferred prediction model, in this case, BMI and S5, further minimized the errors by computing the RMSE values. S5 as one of 6 blood serums not very clearly explained in the case summary and therefore difficult to make conjecture about that feature, however we know from common sense that obesity often contributes to a more likely outcome of a diabetic prognosis. Regardless, the R2 score at 47% and 39% for BMI and S5 respectively are still not sufficient for us to draw any substantial conclusions from this model.   


Cross Validation

![cv_plotly](https://user-images.githubusercontent.com/42139196/55275727-90ea8d80-52c0-11e9-894b-e7222f9f4f9e.png)


