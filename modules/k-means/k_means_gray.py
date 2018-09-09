# K- mean Clustering Algorithm for Gray Scale Image
import numpy as np 
from pylab import *
import matplotlib.pyplot as plt

import cv2 #used only for reading and displaying the output 

# Function to find the centroids of the clusters
def NewCentroid(points,x):
    summ= 0
    for i in points[x]:
        summ = summ + I[i[0],i[1]]
        newpoints= summ/len(points[x]) 
    return newpoints

# Entrylevel main function for finding the k means
def KmeansGrayScale(I):
    # Manually defined intensities for the cluster points
    k=[0, 70, 180]
    iterations= 4
    for x in range(iterations):    
        points=[[],[],[]]
        for i in range(len(I[:,0])):
            for j in range(len(I[0,:])):
                d = [abs(I[i,j]-k[0]), abs(I[i,j]-k[1]), abs(I[i,j]-k[2])]              
                if d[0] == min(d): 
                    points[0].append([i,j])
                if d[1] == min(d):
                    points[1].append([i,j])
                if d[2] == min(d):
                    points[2].append([i,j])
        
        #Update the centroid points                         
        k[0]= NewCentroid(points,0)
        k[1]= NewCentroid(points,1)
        k[2]= NewCentroid(points,2)
    
    im0= np.ones(shape(I)) *255
    im1= np.ones(shape(I)) *255
    im2= np.ones(shape(I)) *255
    
    for i in points[0]:
        im0[i[0],i[1]]=0
    for i in points[1]:
        im1[i[0],i[1]]=0
    for i in points[2]:
        im2[i[0],i[1]]=0
    
    return im2

I = cv2.imread('input2.jpg', 0)
Im= KmeansGrayScale(I)
plt.figure()
plt.imshow(Im, cmap = cm.gray)  
plt.show()
out = cv2.imread('out2.jpg',0)
S = shape(Im)
TP = 0.0
TN = 0.0
FP = 0.0
FN = 0.0
for i in range(S[0]):
    for j in range(S[1]):
        if Im[i,j] == 0 and out[i,j] == 0:
            TP += 1
        elif Im[i,j] == 0 and out[i,j] == 255:
            TN += 1
        elif Im[i,j] == 255 and out[i,j] == 0:
            FN += 1
        elif Im[i,j] == 255 and out[i,j] == 255:
            FP += 1
            
        F = 2 * TP / ((2*TP)+FP+FN)

print "F Score: ", F
