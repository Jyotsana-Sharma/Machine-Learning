#To use categorical variables in a machine learning model, 
# you first need to represent them in a quantitative way. 
# The two most common approaches are to one-hot encode the variables using or to use dummy variables. 
#one hot encoding  includes explainable features
#dummy encoding creates n-1 columns , this encoding includes necessary information without duplication 
#we can check the number of occurrences of any categorical values using value_counts() method 
#we can limit the columns as once we have the count/occurrences using value_counts() method then we can do masking 
#that is first create a mask of values that occurres less than n times 
# for eg if a columm country has India -> 4times occurring , USA -> 8 TIMES , FRANCE -> 2TIMES , JAPAN-> 1 time 
# then we can mask the france and japan value with our choice like 'others '
# syntax : counts=df['country'].value_counts()
# df['country'].isin(counts[counts<5].index)
#df['country'][mask]='other'
#one hot encoding ->pd.get_dummies(df,colums=['country'],prefix="c")
#dummy encoding -> pd.get_dummies(df,columns=['country'],drop_first=true,prefix='c')