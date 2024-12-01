#Statistical Functions using numpy
#amin()=>to find the minimum element from an array
import numpy as np
a=np.array([[1,2,3],[4,5,6],[7,8,9]])
print("Printing the smallest element in array a",np.amin(a))
print("The smallest element along each row in an array a: ",np.amin(a,axis=1))
print("The smallest element along each column in an array a : ",np.amin(a,axis=0))
#amax()=> to find the maximum element in an array
print("Printing the largest element in an array a",np.amax(a))
print("The largest element along each row in an array a: ",np.amax(a,axis=1))
print("The largest element along each column in an array a : ",np.amax(a,axis=0))
#mean()=> to find the mean of an array 
print("Printing the mean of all the elements in an array a",np.mean(a))
print("The mean along each row in an array a: ",np.mean(a,axis=1))
print("The mean along each column in an array a: ",np.mean(a,axis=0))
#median()=> to find the median of an array 
print("Printing the median of all the elements in an array a",np.median(a))
print("Median of each row in an array a:  ",np.median(a,axis=1))
print("Median of each column in an array a: ",np.median(a,axis=0))
#average => to find the average
print("Average of complete array a : ",np.average(a))
print("Average each row in an array a: ",np.average(a,axis=0))
print("Average each column in an array a: ",np.average(a,axis=1))
#Weighted average that means there will be weights which get multiplied with each row/column and then divided by the sum of weights i mean total of all weights
w=[1,2,3]
#print("weighted array ",np.average(a,weights=w)) # this way we cannot calculate we need to mention the axis. Axis must be specified when shapes of a and weights differ
print("Weighted average along row : ", np.average(a,weights=w,axis=1))
print("Weighted average along column : ",np.average(a,weights=w,axis=0))

#Variance: How can we calculate variance is that taking mean firstly and the taking difference of each element with the mean and squaring it.
#for example taking b array =[[4,9,8],[5,7,6],[1,2,3]] , mean =45/9=5 , variance=(4-5)^2+(9-5)^2+..+(3-5)^2/9 total elements in an array
b=np.array([[4,9,8],[5,7,6],[1,2,3]])
print("The variance : ",np.var(b))
print("The standard deviation is basically the square root of variance: ",np.std(b))
