#https://www.youtube.com/watch?v=3fzYdnuvEfk

from socket import create_server
from cv2 import split
import pandas as pd
from torch import rand 
df=pd.read_csv("https://raw.githubusercontent.com/krishnaik06/Types-Of-Cross-Validation/main/cancer_dataset.csv")
# print(df.head())

#Independent and Dependent Features 
X=df.iloc[:,2:]
Y=df.iloc[:,1]
# print("X",X)
# print("Y",Y)

X=X.dropna(axis=1)#dropna(axis=1) will drop all the columns which contains the  missing values 
# print(X.head())
#In order to check whether a dataset is balanced or imbalanced , we will be using value_counts()
# print(Y.value_counts())

#Hold Out Validation Approach ->  Train and Test Split 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=1)
model=DecisionTreeClassifier()
model.fit(X_train,Y_train)
result=model.score(X_test,Y_test)

print("Testing result of Holdout validation :",result)

#KFOLD CROSS VALIDATION 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np 
model1=DecisionTreeClassifier()
kfold=KFold(10)
result1=cross_val_score(model,X,Y,cv=kfold)
print("Kfold_Validation score :",result1)
print("Accuracy by taking mean : ",np.mean(result1))

#Stratified KFold Cross Validation 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

skfold=StratifiedKFold(n_splits=5)
model3=DecisionTreeClassifier()
scores=cross_val_score(model3,X,Y,cv=skfold)
print("Stratified k fold cross validation : ",np.mean(scores))

#Leave one out cross validation 
from sklearn.model_selection import LeaveOneOut 
model4=DecisionTreeClassifier()
leave_validation= LeaveOneOut()
result3=cross_val_score(model4,X,Y,cv=leave_validation)
print("Leave one out cross validation : ",np.mean(result3))

#Repeated Random training-test split 
from sklearn.model_selection import ShuffleSplit 
model5=DecisionTreeClassifier()
sshuffle=ShuffleSplit(n_splits=10,test_size=0.30)
result4=cross_val_score(model,X,Y,cv=sshuffle)
print("Repeated Random Training-test split : ",result4)
print("Repeated random training-test split : ",np.mean(result4))
