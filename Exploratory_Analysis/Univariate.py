
import numpy as np
import pandas as p
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns
df=p.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')
#print(df.head())
#print(df.shape)

#Univariate Analysis
df_setosa=df.loc[df['species']=='setosa']
df_versicolor=df.loc[df['species']=='versicolor']
df_virginica=df.loc[df['species']=='virginica']

#np.zeros_like returns an array of zero 
plt.plot(df_setosa['sepal_length'],np.zeros_like(df_setosa['sepal_length']),'o')
plt.plot(df_versicolor['sepal_length'],np.zeros_like(df_versicolor['sepal_length']),'o')
plt.plot(df_virginica['sepal_length'],np.zeros_like(df_virginica['sepal_length']),'o')
plt.xlabel('petal length')
plt.show()

# blue line on the figure shows setosa , green color shows versicolor and yellow color shows virginica 

