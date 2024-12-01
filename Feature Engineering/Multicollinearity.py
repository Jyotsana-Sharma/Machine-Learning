#https://www.analyticsvidhya.com/blog/2021/03/multicollinearity-in-data-science/#:~:text=Multicollinearity%20occurs%20when%20two%20or,variable%20in%20a%20regression%20model.

#Importing Necessary Libraries
import numpy as np 
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import load_boston
boston=load_boston()

#To see the description of boston dataset 
#print(boston.DESCR)
#Independent Variables
X=boston["data"]
Y=boston["target"]

#Name for all the features
names=list(boston["feature_names"])
# print("X")
# print(X)
# print("Y")
# print(Y)
# print("Names")
# print(names)

df=pd.DataFrame(X,columns=names)
#print(df.head())

for index in range(0,len(names)):
    y=df.loc[:,df.columns==names[index]]
    x=df.loc[:,df.columns!=names[index]]
    model=sm.OLS(y,x)
    results=model.fit()
    rsq=results.rsquared
    vif=round(1/(1-rsq),2)
    print("R Square value of {} column is {} keeping all other columns as independent features".format(names[index], (round(rsq, 2)) ))
    print("Variance Inflation Factor of {} column is {} n".format(names[index], vif))
