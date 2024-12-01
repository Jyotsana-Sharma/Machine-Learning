from turtle import mode
import pandas as pd

import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import model_selection

from math import sqrt
import numpy as np
import warnings
warnings.filterwarnings( "ignore" )


data=pd.read_csv('https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv',index_col=0)
print(data.head())
data.columns=['TV','Radio','Newspaper','Sales']
print(data.shape)

fig,axes=plt.subplots(1,3,sharey=False)# 1 rows and 3 column means that i will be having one x-axis and 3 y-axis that means 3 graphs in a 1 row
data.plot(kind='scatter',x='TV',y='Sales',ax=axes[0],figsize=(16,8))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axes[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axes[2])
plt.show()

feature_cols=['TV']
x=data[feature_cols]
y=data.Sales
lm=LinearRegression()#Initializing the Model
lm.fit(x,y)
print("Intercept(c) : ",lm.intercept_) #printing the intercept of the equation 
print("Coefficient(m) : ",lm.coef_)#printing the coefficient of result using linear equation
#now as we have y that is our sales and x as tv only so by looking at coefficient of linear regression model
#we can say that a unit increases in tv ad spending is associated with the 0.04753664 unit increases in sales 

#using the previous value to calculate the new value lets say that there is a new sale where tv ad spending 50000 
#what would we predict the sales in that market 
#Intercept: In this equation, the value 'c' is called the intercept of the line. 
# The intercept measures the length where the line cuts the y-axis, from the origin. 
# It can also be interpreted as the point (0, c) on the y-axis, through which the line is passing.
#y=mx+c 7.032593549127695+0.04753664*50 will be new value for sales 
print(7.032593549127695+0.04753664*50 ) 

#Lets predict the new X_values 
X_new=pd.DataFrame({'TV':[50]})
print(X_new)
print(X_new.head())
print(lm.predict(X_new))
#lets make predictions for the smallest and the largest observed values of x.
#and then use the predicted values to plot the least square lines
X_new=pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
print(X_new)
print(type(X_new))
X_new=X_new.to_numpy()
#predicting the values for our new data min and max values for tv column 
preds=lm.predict(X_new)
print(preds)
print(type(preds))
#Initializing the scatter plot for Tv and sales 
data.plot(kind="scatter",x='TV',y='Sales')
#High bias of a machine learning model is a condition where the output of the machine learning model is quite far off from the actual output.
#  This is due to the simplicity of the model. 
# We saw earlier that a model with high bias has both, high error on the training set and the test set
plt.plot(X_new,preds,c='red',linewidth=2)
plt.show()
#Looking at the plot we can see that there is high bias and low variance
#https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229
#there is no relationship between tv ads and sales <- Null Hypothesis assumption
#we shall reject the null hypothesis if the 95% confidence interval does not include zero

#{y}=b_1x+b_0
#In OLS method, https://www.geeksforgeeks.org/ordinary-least-squares-ols-using-statsmodels/
#https://medium.com/analytics-vidhya/ordinary-least-square-ols-method-for-linear-regression-ef8ca10aadfc
#https://economictheoryblog.com/ordinary-least-squares-ols/#:~:text=In%20data%20analysis%2C%20we%20use,estimator%20by%20a%20simple%20formula.
# OLS for estimating the unknown parameters in a linear regression model. 
# The goal is minimizing the differences between the collected observations in some arbitrary dataset and the responses predicted by the linear approximation of the data. 
#https://www.datarobot.com/blog/ordinary-least-squares-in-python/
#https://www.ritchieng.com/machine-learning-evaluate-linear-regression-model/
lm=smf.ols(formula='Sales~TV',data=data).fit()
print("Confidence Interval ",lm.conf_int()) #returns the confidence interval
#printing the p-values 
print(lm.pvalues)
#in this case the pvalue for tv is so far less than the 0.5
#so we infer that there is relationship between the 

#How well does the model fits the data ?
#The most common way to evaluate the overall fit of a linear model is to use the rsquared method 
print(lm.rsquared)
#By just looking at the rsquared value it is difficult to say that it is a good rsquared value 
#thats why it is mostly used the rsquared value while comparing different models
#Earlier we have used tv as the feature column 
# Now just take all the independent variables in our feature column to build a multiple linear regression model
#Lets start building the multiple linear regression model by taking three variables in feature column
feature_cols=['TV','Radio','Newspaper']
X=data[feature_cols]
y=data.Sales 
#splitting to training and testing dataset
xtrain,xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=0.3,random_state=42)
#Applying linear regression
lm=LinearRegression()
lm.fit(X,y)
print("Intercept for X: ",lm.intercept_)
print("Coefficient for X: ",lm.coef_)

#Applying linear regression on the testing data 
ls=LinearRegression()
ls.fit(xtrain,ytrain)
print("Intercept for training : ",ls.intercept_)
print("Coefficient for training : ",ls.coef_)
# predictions 
predictions=lm.predict(xtest)
print(sqrt(mean_squared_error(ytest,predictions)))

#using the ols results of summary () we will get more information 
lk=smf.ols('Sales~TV+Newspaper+Radio',data=data).fit()
print(lk.conf_int())
print(lk.summary())
#the rsquared value is 0.897 which is better than the previous values that means 
# this model provides better fit for the data than the model that includes only TV as feature
#How do we check which feature need to be included in a linear model ?
#1.Try out different models and only keep the predictors if only they have small pvalue
#2.check whether the rsquared values goes up when you add up the new predictors 
#demo for rsquared value 
lp=smf.ols("Sales~TV+Radio",data=data).fit()
print(lp.rsquared)

lf=smf.ols("Sales~TV+Newspaper+Radio",data=data).fit()
print(lf.rsquared)
#Observing that the rsquared value always increases as we add up the feature to the model 
#even if they are unrrelated to the response(Target variable which is Sales )
#thus selecting the model with the highest rsquared value is not a reliable approach for choosing the best linear regression model 

#handling categorical variable with only two categories lets create a new variable as size 
np.random.seed(12345)
nums=np.random.rand(len(data))
mask_large=nums>0.5
data['Size']='small'
data.loc[mask_large,'Size']='large'
print(data.head())
#creating dummy variable that represents the categories as binary values 
data['Islarge']=data.Size.map({'small':0,'large':1})
print(data.head())
#lets redo the multiple linear regression and include the Islarge variable 
feature_cols=['TV','Radio','Newspaper','Islarge']
X=data[feature_cols]
y=data.Sales
lo=LinearRegression()
lo.fit(X,y)
list(zip(feature_cols, lo.coef_))

nums = np.random.rand(len(data))
mask_suburban = (nums > 0.33) & (nums < 0.66)
mask_urban = nums > 0.66
data['Area'] = 'rural'
# Series.loc is a purely label-location based indexer for selection by label
data.loc[mask_suburban, 'Area'] = 'suburban'
data.loc[mask_urban, 'Area'] = 'urban'
data.head()
area_dummies = pd.get_dummies(data.Area, prefix='Area').iloc[:, 1:]
area_dummies.head()
# concatenate the dummy variable columns onto the DataFrame (axis=0 means rows, axis=1 means columns)
data = pd.concat([data, area_dummies], axis=1)
data.head()
# create X and y
feature_cols = ['TV', 'Radio', 'Newspaper', 'Size_large', 'Area_suburban', 'Area_urban']
X = data[feature_cols]
y = data.Sales

# instantiate and fit
lm2 = LinearRegression()
lm2.fit(X, y)

# print the coefficients
list(zip(feature_cols, lm2.coef_))




