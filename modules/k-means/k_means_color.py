# K means for color image (please change the Working Directory)
import cv2 # used for reading and displaying input image
from sympy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as num
import random
from pylab import *

# Function returns the Segmentation of the input color image
def KmeansColor(Ic,Ig):
    Shape= shape(Ic)
    Im= np.zeros(shape(Ig))
    
    #RGB Histogram for choosing centroid for the clusters
    H=np.zeros(256)
    for i in range(Shape[0]):
        for j in range(Shape[1]):
           Im[i,j] = np.sqrt(Ic[i,j][0]**2 + Ic[i,j][1]**2 + Ic[i,j][2]**2)
           H[Ic[i,j]] =  H[Ic[i,j]]+1    
  
    # Random Centroids for the cluster points
    r1=randint(0,len(Ig[:,0])/8)
    c1=randint(0,len(Ig[0,:])/8) 
    r2=randint(len(Ig[:,0])/8,len(Ig[:,0])/4)
    c2=randint(len(Ig[0,:])/8,len(Ig[0,:])/4)
    r3=randint(len(Ig[:,0])/4,len(Ig[:,0])/2)
    c3=randint(len(Ig[0,:])/4,len(Ig[0,:])/2)
    r4=randint(len(Ig[:,0])/2,len(Ig[:,0]))
    c4=randint(len(Ig[0,:])/2,len(Ig[0,:]))
   
    
    d1= np.zeros(shape(Ig))
    d2= np.zeros(shape(Ig))
    d3= np.zeros(shape(Ig))
    d4= np.zeros(shape(Ig))
   
    # Finding the intensity difference of the respective randomly selected 
    # centroids of the cluster points
     
    while True :     
        p1=[]
        p2=[]
        p3=[]
        p4=[]
        for i in range(len(Ig[:,0])):
            for j in range(len(Ig[0,:])):
              idif1 = Ic[i,j] - Ic[r1,c1]
              dist1 = np.sqrt(int(idif1[0])**2 +int(idif1[1])**2 + int(idif1[2])**2)
              d1[i,j] = dist1
        
        for i in range(len(Ig[:,0])):
            for j in range(len(Ig[0,:])):
              idif2 = Ic[i,j] - Ic[r2,c2]
              dist2 = np.sqrt(int(idif2[0])**2 +int(idif2[1])**2 + int(idif2[2])**2)
              d2[i,j] = dist2
        
        for i in range(len(Ig[:,0])):
            for j in range(len(Ig[0,:])):
              idif3 = Ic[i,j] - Ic[r3,c3]
              dist3 = np.sqrt(int(idif3[0])**2 +int(idif3[1])**2 + int(idif3[2])**2)
              d3[i,j] = dist3
        
        for i in range(len(Ig[:,0])):
            for j in range(len(Ig[0,:])):
              idif4 = Ic[i,j] - Ic[r4,c4]
              dist4 = np.sqrt(int(idif4[0])**2 +int(idif4[1])**2 + int(idif4[2])**2)
              d4[i,j] = dist4
             
        #Append the minimum intensities of the above difference to p1,p2,p3,p4
        for i in range(len(Ig[:,0])):
            for j in range(len(Ig[0,:])):
                if(d1[i,j]<d2[i,j] and d1[i,j]<d3[i,j] and d1[i,j]<d4[i,j]):
                    p1.append((i,j))
                elif(d2[i,j]<d1[i,j] and d2[i,j]<d3[i,j] and d2[i,j]<d4[i,j]):
                    p2.append((i,j))
                elif(d3[i,j]<d1[i,j] and d3[i,j]<d2[i,j] and d3[i,j]<d4[i,j]):
                    p3.append((i,j))
                elif(d4[i,j]<d1[i,j] and d4[i,j]<d2[i,j] and d4[i,j]<d3[i,j]):
                    p4.append((i,j))

        #Updating the centroid points
        u1=v1=0
        for i in range(len(p1)):
           u1 = u1+ p1[i][0]
           v1 = v1+ p1[i][1]
           k11 = floor(u1/(1+len(p1)))
           k12 = floor(v1/(1+len(p1)))
        u2=v2=0
        for i in range(len(p2)):
           u2 += p2[i][0]
           v2 += p2[i][1]
           k21 = floor(u2/(1+len(p2)))
           k22 = floor(v2/(1+len(p2)))
        u3=v3=0
        for i in range(len(p3)):
           u3 += p3[i][0]
           v3 += p3[i][1]
           
           k31 = floor(u3/(1+len(p3)))
           k32 = floor(v3/(1+len(p3)))
        u4=v4=0
        for i in range(len(p4)):
           u4 += p4[i][0]
           v4 += p4[i][1]
           k41 = floor(u4/(1+len(p4)))
           k42 = floor(v4/(1+len(p4)))
        # Exit the while loop if this condition satisfies and break and update to the new centroids    
        if abs(r1-k11)<20 and abs(c1-k12)<20 and abs(r2-k21)<20 and abs(c2-k22)<20 and abs(r3-k31)<20 and abs(c3-k32)<20 and abs(r4-k41)<20 and abs(c4-k42):
            for i in range(len(p1)):
                Ic[p1[i][0]][p1[i][1]] = 0  
            for i in range(len(p2)):
                Ic[p2[i][0]][p2[i][1]] = 63
            for i in range(len(p3)):
                Ic[p3[i][0]][p3[i][1]] = 127 
            for i in range(len(p4)):
                Ic[p4[i][0]][p4[i][1]] = 255
        
            plt.imshow(Ic,cmap=cm.gray)
            plt.show()      
        
            break
        
        else:
            r1 =k11
            c1 =k12
            r2 =k21
            c2 =k22
            r3 =k31
            c3 =k32
            r4 =k41
            c4 =k42

#Read the image in Color
Ic = cv2.imread('input2.jpg')
#Read the image in grayscale
Ig = cv2.imread('input2.jpg',0)

KmeansColor(Ic,Ig)

"""
plt.figure()
plt.imshow(Im, cmap = cm.gray)  
plt.show()

out = cv2.imread('out2.jpg')
S = =shape(Im)
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
"""
