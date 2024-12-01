import pandas as pd
import numpy as np
# importing the models

from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression

df=pd.read_csv("/Users/jyotsanasharma/Desktop/ML/Feature Engineering/forward_feature_Selection.csv")
# print(df.index)
# print(df.columns)
#print(df.head)
# print(df.shape)
# print(df.info)
#print(df.describe)
#print(df.describe(include='all'))

#To find the null values 
#print(df.isnull().sum())
#So we found that there is no null values in our dataset 

#Let's drop some of the columns which we donot need to make simpler for applying forward_feature_selection and create a new dataset
data=df.drop(['datetime','atemp','casual','registered'],axis=1)
#print(data.columns)

#Now let's use our new data set "data" for the forward_feature_Selection
#print(data.shape)

#Let's Identify our target variable and other Independent variables
X=data.iloc[:,0:7]
#print("X")
#print(X)
Y=data.iloc[:,1]
#print("Y")
#print(Y)

#lets install mlxtend library (contains useful wrapper methods) 
#Documentation for mlxtend library : https://gdcoder.com/mlxtend-feature-selection-tutorial/ 
#As we are going to implement the feature forward selection lets import the SequentialFeatureSelector from sklearn library 
#We will import the linear regression from sklearn as this is a linear regression problem 
#How to choose which algorithm should be applied 
#https://medium.com/analytics-vidhya/which-machine-learning-algorithm-should-you-use-by-problem-type-a53967326566 
#Best : https://blogs.sas.com/content/subconsciousmusings/2020/12/09/machine-learning-algorithm-use/  
# creating the training data
X = data.drop(['count'], axis=1)
#print("X")
#print(X)
Y = data['count']
#print("Y")
#print(Y)
#print(X.shape, Y.shape)
# calling the linear regression model

lreg = LinearRegression()
sfs1 = sfs(lreg, k_features=4, forward=True, verbose=2, scoring='neg_mean_squared_error')
#About sbs 
#http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/

sfs1 = sfs1.fit(X, Y)
#print(sfs1)

feat_names = list(sfs1.k_feature_names_)
#print(feat_names)

# creating a new dataframe using the above variables and adding the target variable
new_data = data[feat_names]
new_data['count'] = data['count']

# first five rows of the new data
print(new_data.head())




