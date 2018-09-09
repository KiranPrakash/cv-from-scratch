import cv2
from sympy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as num
import random

I = cv2.imread('input2.jpg')
Shape1 = num.shape(I) #Ic
Inew = num.zeros((321,481)) #Ig
Shape = num.shape(Inew) #Ig

###################################
#HISTOGRAM
hist = num.zeros(256)


for i in range(Shape1[0]):
    for j in range(Shape1[1]):
        Inew[i,j] = num.sqrt((I[i,j][0]**2)+(I[i,j][1]**2)+(I[i,j][2]**2))
        pix = I[i,j]
        hist[pix] = hist[pix]+1
        

###################################

## Taking a random centroids for the K-means
x1= random.randint(0,Shape[0]/8)

y1= random.randint(0,Shape[1]/8)

x2= random.randint(Shape[0]/8,Shape[0]/4)

y2= random.randint(Shape[1]/8,Shape[1]/4)

x3=random.randint(Shape[0]/4,Shape[0]/2)

y3= random.randint(Shape[1]/4,Shape[1]/2)

x4=random.randint(Shape[0]/2,Shape[0])

y4=random.randint(Shape[1]/2,Shape[1])

int_dist1 = num.zeros(Shape)
int_dist2 = num.zeros(Shape)
int_dist3 = num.zeros(Shape)
int_dist4 = num.zeros(Shape)

n = 0
while True:
    n = n+1
    print "Number of iterations=",n
    C1 = []
    C2 = []
    C3 = []
    C4 = []

    ## Forming the intensity difference matrix for all the four clusters
    for i in range(Shape[0]):
        for j in range(Shape[1]):
            a1 = I[i,j]-I[x1,y1]           
            d1 = num.sqrt(int(a1[0])**2+int(a1[1])**2+int(a1[2])**2)
            int_dist1[i,j] = d1
    
    for i in range(Shape[0]):
        for j in range(Shape[1]):
            a2 = I[i,j]-I[x2,y2]
            d2 = num.sqrt(int(a2[0])**2+int(a2[1])**2+int(a2[2])**2)
            int_dist2[i,j] = d2
    
    for i in range(Shape[0]):
        for j in range(Shape[1]):
            a3 = I[i,j]-I[x3,y3]
            d3 = num.sqrt(int(a3[0])**2+int(a3[1])**2+int(a3[2])**2)
            int_dist3[i,j] = d3
    
    for i in range(Shape[0]):
        for j in range(Shape[1]):
            a4 = I[i,j]-I[x4,y4]
            d4 = num.sqrt(int(a4[0])**2+int(a4[1])**2+int(a4[2])**2)
            int_dist4[i,j] = d4

    

    for i in range(Shape[0]):
        for j in range(Shape[1]):   ## taking the intensity differences and sending them to the different clusters
            if(int_dist1[i,j]<int_dist2[i,j] and int_dist1[i,j]<int_dist3[i,j] and int_dist1[i,j]<int_dist4[i,j]):
                C1.append((i,j))
            elif(int_dist2[i,j]<int_dist1[i,j] and int_dist2[i,j]<int_dist3[i,j] and int_dist2[i,j]<int_dist4[i,j]):
                C2.append((i,j))
            elif(int_dist3[i,j]<int_dist1[i,j] and int_dist3[i,j]<int_dist2[i,j] and int_dist3[i,j]<int_dist4[i,j]):
                C3.append((i,j))
            elif(int_dist4[i,j]<int_dist1[i,j] and int_dist4[i,j]<int_dist2[i,j] and int_dist4[i,j]<int_dist3[i,j]):
                C4.append((i,j))
     
          
            
            
  
    ## Calculating the new x and y coordinates for the four clusters
    s1 = 0
    w1 = 0
    for i in range(len(C1)):
        s1 = s1 + C1[i][0]
        w1 = w1 + C1[i][1]
    K1x = floor(s1/(1+len(C1)))
    K1y = floor(w1/(1+len(C1)))


    s2 = 0
    w2 = 0
    for i in range(len(C2)):
        s2 = s2 + C2[i][0]
        w2 = w2 + C2[i][1]
    K2x = floor(s2/(1+len(C2)))
    K2y = floor(w2/(1+len(C2)))


    s3 = 0
    w3 = 0
    for i in range(len(C3)):
        s3 = s3 + C3[i][0]
        w3 = w3 + C3[i][1]
    K3x = floor(s3/(1+len(C3)))
    K3y = floor(w3/(1+len(C3)))


    s4 = 0
    w4 = 0
    for i in range(len(C4)):
        s4 = s4 + C4[i][0]
        w4 = w4 + C4[i][1]
    K4x = floor(s4/(1+len(C4)))
    K4y = floor(w4/(1+len(C4)))


    ## Stop condition of the while loop
    if abs(x1-K1x)<20 and abs(y1-K1y)<20 and abs(x2-K2x)<20 and abs(y2-K2y)<20 and abs(x3-K3x)<20 and abs(y3-K3y)<20 and abs(x4-K4x)<20 and abs(y4-K4y)<20:

        for i in range(len(C1)):  ## Changing the intensity value of all the elements of Cluster1 to 0
            e=C1[i][0]
            t=C1[i][1]
            I[e][t]=0

        for i in range(len(C2)):  ## Changing the intensity value of all the elements of Cluster2 to 60
            e=C2[i][0]
            t=C2[i][1]
            I[e][t]=60

        for i in range(len(C3)):  ## Changing the intensity value of all the elements of Cluster3 to 180
            e=C3[i][0]
            t=C3[i][1]
            I[e][t]=180

        for i in range(len(C4)):  ## Changing the intensity value of all the elements of Cluster4 to 255
            e=C4[i][0]
            t=C4[i][1]
            I[e][t]=255

        ## Printing the binary image after the intensity changes
        plt.imshow(I,cmap=cm.gray)
        plt.show()

        break
    ## Updating the new K1 and K2 values
    else:
        x1=K1x
        y1=K1y
        x2=K2x
        y2=K2y
        x3=K3x
        y3=K3y
        x4=K4x
        y4=K4y


