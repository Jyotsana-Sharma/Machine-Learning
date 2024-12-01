# As we know that most of the ml models require numerical variables 
#However if our raw data is numerical still we can do a lot to improve the features 
# So if we have data from restaurant that will have two rows restaurant name and the restuarant violation this is numeric 
# and this restaurant violation having data as {0,0,,0,1,1,2,1.2,3,3,4...}
#loc is label-based, which means that you have to specify rows and columns based on their row and column labels.
#With loc, we can use the syntax A:B to select data from label A to label B (Both A and B are included)
#With iloc, we can also use the syntax n:m to select data from position n (included) to position m (excluded)
#With loc, we just need to pass the condition to the loc statement. df.loc[df.Humidity > 50, :]
#For iloc, we will get a ValueError if pass the condition straight into the statement:
#We get the error because iloc cannot accept a boolean Series. It only accepts a boolean list. We can use the list() function to convert a Series into a boolean list.
#df.iloc[list(df.Humidity > 50)]
#Binarizing numerical variable 
#While numeric values can often be used without any feature engineering, 
# there will be cases when some form of manipulation can be useful. 
# For example on some occasions, you might not care about the magnitude of a value but only care about its direction, 
#or if it exists at all. In these situations, you will want to binarize a column.
#df['binary_violation']=0 # creating new column for the binary_violation
#using .loc method to find all the rows where the number of voilations are more than 0
#df.loc[df['NumberofViolations']>0,'BinaryVoilations']=1
#Binning Numeric Variables
#This is an extension of binarizing numeric variables if we wish to group a numeric variable into more than two bins.
#this is often useful when we have ages , wages etc. where exact number are less relevant than the general magnitude of value 
# so this time we will create three groups : Group1 : with 0 violations , Group2 : 1 or 2 violations, Group3 : 3 or more violations 
#df['Binary_Group']=pd.cut(df['Number_of_Violations'],bins=[-np.inf,0,2,np.inf],labels=[1,2,3])
#so bins are created using pandas cut () we can define the intervals in cut () function using the bins argument and we can pass the labels using labels argument 
#
#DataCamp