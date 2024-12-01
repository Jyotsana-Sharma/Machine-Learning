#https://www.youtube.com/watch?v=8MxDgHMJnic  



from traceback import print_tb
import pandas as pd
from sklearn import pipeline 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris 

#Loaded the iris dataset from sklearn.datasets
# data=load_iris()
#print(data)
#Converted the arrays of data to pandas dataframe 
# data=pd.DataFrame(data.data,columns=data.feature_names)
# df=data.to_csv()
#print(data.head())
#print(data.tail())
csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# using the attribute information as the column names
col_names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Class']
iris =  pd.read_csv(csv_url, names = col_names)
# print(iris.head())
X=iris.iloc[:,0:4]
print(X.head())
Y=iris.iloc[:,-1]
# print(Y.head())
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=50)

#LOGISTIC regression
pipeline_log=Pipeline([
    ('Scaler',StandardScaler()),
    ('Logistic Regression',LogisticRegression())
])
pipeline_log.fit(X_train,Y_train)
score_model=pipeline_log.score(X_test,Y_test)
# print(score_model)

#Decision Tree
pipeline_dt=Pipeline([
    ('Scaler',StandardScaler()),
    ('DecisionTreeClassifier',DecisionTreeClassifier())
])

pipeline_dt.fit(X_train,Y_train)
score_dt=pipeline_dt.score(X_test,Y_test)
# print("Score of Decision Tree Classifier : ",score_dt)

pipeline_rf=Pipeline([ 
    ('Scaler',StandardScaler()),
    ('RandomForestClassifier',RandomForestClassifier())
])

pipeline_rf.fit(X_train,Y_train)
score_rf=pipeline_rf.score(X_test,Y_test)
# print("Score of Random Forest Classifier : ",score_rf)

#Defining more steps in pipeline 
pipeline_more=Pipeline([ 
    ('Scaler',StandardScaler()),
    ('PCA',PCA(n_components=2)),
    ('LogisticRegression',LogisticRegression())
])

pipeline_more.fit(X_train,Y_train)
score_more=pipeline_more.score(X_test,Y_test)
# print("Score : ",score_more)

#Difference between pipeline and make_pipeline 
# https://www.youtube.com/watch?v=lkFwwquv_ss






