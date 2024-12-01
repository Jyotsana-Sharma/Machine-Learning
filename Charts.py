#Line Chart
from signal import pthread_kill
import numpy as np
import matplotlib.pyplot as plt
from pandas import array
#linspace creates a 1d array with 1000 evenly spaced points between zero
x=np.linspace(0,20,1000) #0 is start and 20 is the step size
#print(x)
y=np.sin(x)+0.2*x # completely random equation created for y variable
#print(y)
#plt.plot(x,y)
#to label x-axis
#plt.xlabel('input')
#plt.ylabel('output')
#plt.title('My Line Graph/Chart')
#plt.show()

#ScatterPlot
#np.random.randn(100,2)
#If positive int_like arguments are provided, randn generates an array of shape (d0, d1, ..., dn),
#  filled with random floats sampled from a univariate "normal" (Gaussian) distribution of mean 0 and variance 1. 
# A single float randomly sampled from the distribution is returned if no argument is provided.
#plt.scatter(X-axis,y-axis)
scatter_x=np.random.randn(100,2)
#plt.scatter(scatter_x[:,0],scatter_x[:,1])
#plt.show()

#for different colors and classes in scatter plot
color_x=np.random.randn(200,2)
color_x[:50]+=3
color_y=np.zeros(200)
print(color_y)
color_y[:50]=1
#print(color_y[:50])
#A 2D array in which the rows are RGB or RGBA.
#A sequence of colors of length n.
#A single color format string.
# You can even set a specific color for each dot by using an array of colors as value for the c argument:
#Example:np.array(["red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"])
#Example:plt.scatter(x, y, c=colors)
print(color_y)
plt.scatter(color_x[:,0],color_x[:,1],c=color_y)
plt.show()

#Histograms
#Lets create a sample of 10000 random numbers from the standard normal
hist_x=np.random.randn(10000)
#hist_x=np.array([1,1,1,1,2,2,3,3,3,3,3]) 
#so on x-axis we have elements from array hist_x and on the y_axis we have count of it 
plt.hist(hist_x,bins=50)
plt.show()
#when we have bins : https://www.khanacademy.org/math/statistics-probability/displaying-describing-data/quantitative-data-graphs/a/histograms-review
#random function samples form from the uniform distribution
random_uniform_x=np.random.random(10000)
plt.hist(random_uniform_x,bins=50) # the distribution is between 0 and 1
plt.show()

#Plotting Images
from PIL import Image
im=Image.open('/Users/jyotsanasharma/Desktop/udemy_Python/Screenshot 2021-12-10 at 2.38.02 PM copy.png')
print(type(im))
arr=np.array(im)
#images are being represented in computers are in form of arrays below we can see that
print(arr)
print(arr.shape) #(1800, 2880, 4) 1800=height of image, 2880->width of image the 4 is for each location in the image where we need to store the color of that pixel
plt.imshow(arr)
plt.show()
plt.imshow(im)
plt.show()
gray=arr.mean(axis=2)
plt.imshow(gray)
plt.show()
plt.imshow(gray,cmap='gray')

#https://towardsdatascience.com/master-the-art-of-subplots-in-python-45f7884f3d2e

#Scatter subplot
#Let’s now create our very first two subplots with a single row and two columns. 
# Since theaxes object contains two subplots, you can access them using indices [0] and [1] because indexing starts at 0 in Python.
subplot_x= np.linspace(0., 3*np.pi, 100) # 0 to 3*Pi in 100 steps
y_1 = np.sin(subplot_x) 
y_2 = np.cos(subplot_x)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))
axes[0].plot(subplot_x, y_1, '-', c='orange', label='sin(subplot_x)')
axes[1].plot(subplot_x, y_2, '-', c='magenta', label='cos(subplot_x)')
axes[0].legend(fontsize=16, frameon=False)
axes[1].legend(fontsize=16, frameon=False)
fig.suptitle('Subplots without shared y-axis')