#https://machinelearningmastery.com/rfe-feature-selection-in-python/

#Recursive Feature Elimination 
## ##################RFE FOR CLASSIFICATION ################

#Importing Libraries 
import imp
import sklearn
from sklearn import pipeline
from sklearn.feature_selection import RFE
from torch import cross

#checking the sklearn version 
# print(sklearn.__version__)

#First, we can use the make_classification() function 
# to create a synthetic binary classification problem with 1,000 examples and 10 input features,
#  five of which are important and five of which are redundant.

from sklearn.datasets import make_classification 
X,Y=make_classification(n_samples=1000,n_features=10,n_informative=5,n_redundant=5,random_state=1)

#Summarizing the dataset 
# print(X.shape, Y.shape)

#Next, we can evaluate an RFE feature selection algorithm on this dataset. 
#We will use a DecisionTreeClassifier to choose features and set the number of features to five.
# We will then fit a new DecisionTreeClassifier model on the selected features.

#We will evaluate the model using repeated stratified k-fold cross-validation, 
# with three repeats and 10 folds. 
# We will report the mean and standard deviation of the accuracy of the model across all repeats and folds.

#evaluate RFE for classification 
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import RepeatedStratifiedKFold 
from sklearn.feature_selection import RFE 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.pipeline import Pipeline 

#creating a pipeline 
rfe=RFE(estimator=DecisionTreeClassifier(),n_features_to_select=5)
model=DecisionTreeClassifier()
pipeline=Pipeline(steps=[('s',rfe),('m',model)])

#evaluate model 
cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)
n_scores=cross_val_score(pipeline,X,Y,scoring="accuracy",cv=cv,n_jobs=-1,error_score="raise")
print("Accuracy %.3f (%.3f)" % (mean(n_scores),std(n_scores)))







