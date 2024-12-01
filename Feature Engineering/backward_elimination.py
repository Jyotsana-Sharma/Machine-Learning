#https://www.analyticsvidhya.com/blog/2021/04/backward-feature-elimination-and-its-implementation/
#Importing the necessary libraries 
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression

#Reading the dataset and printing the first five observations 
data=pd.read_csv("/Users/jyotsanasharma/Desktop/ML/Feature Engineering/forward_feature_Selection.csv")
#print(data.head())
#New Dataset axis=1 means deletion along the column name
df=data.drop(['datetime','atemp','casual','registered'],axis=1)
#print(df.head())
#print(df.shape)
#Checking for the missing values in the dataset
#print(df.isnull().sum())

#Creating the training dataset 
X=df.drop(['count'],axis=1)
Y=df['count']
#print(X.shape,Y.shape)

lreg = LinearRegression()

sfs1 = sfs(lreg, k_features=4, forward=False, verbose=1, scoring='neg_mean_squared_error')
sfs1 = sfs1.fit(X, Y)

feat_names = list(sfs1.k_feature_names_)
#print(feat_names)

#Lets put these features into a new dataframe  and compare the shapes of our dataframe
new_data=df[feat_names]
new_data['count']=df['count']
#print(new_data.head())
print("Shape of original dataset")
print(df.shape)
print("Shape of featured dataset")
print(new_data.shape)
print(df.head())




