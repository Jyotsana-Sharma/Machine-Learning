import numpy as np
import pandas as p
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns
df=p.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')
# df_setosa=df.loc[df['species']=='setosa']
# df_versicolor=df.loc[df['species']=='versicolor']
# df_virginica=df.loc[df['species']=='virginica']
sns.pairplot(df,hue='species',height=3)
plt.show()