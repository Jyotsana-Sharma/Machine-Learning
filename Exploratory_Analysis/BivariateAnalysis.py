import numpy as np
import pandas as p
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns
df=p.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')
df_setosa=df.loc[df['species']=='setosa']
df_versicolor=df.loc[df['species']=='versicolor']
df_virginica=df.loc[df['species']=='virginica']
#In bivariate analysis we take two features
# hue is basically features on which we are categorizing  
sns.FacetGrid(df,hue='species',height=5,).map(plt.scatter,'petal_length','sepal_width').add_legend()
#add_legend() is used to plot on the graph that which color represent which category for eg blue color represent the setosa and yellow versicolor
#plt.show()

