import seaborn as sns
from matplotlib.pyplot import plt
df=sns.load_dataset('iris')
print("Shape: ",df.shape)
print("Total rows: ",df.index)
print("Total columns : ",df.columns)
print(df.corr())
sns.pairplot(df)
plt.show()