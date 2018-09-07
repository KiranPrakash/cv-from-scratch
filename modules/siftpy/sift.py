# -*- coding: utf-8 -*-
#####################################################################
## "Distinctive Image Features from Scale-Invariant Key-Points"    ##
##  by David G. Lowe. Implementation of SIFT                       ##
##  Author: Kiran Prakash                                          ##
#####################################################################


import cv2
from pylab import *
import copy as cp
from scipy import ndimage
import math
#from numpy.linalg import inv




# Function declaration for finding the key Points in SIFT algorithm goes here

def SIFT(image1,n):    
    

    
    
    #SCALE-SPACE EXTREMA DETECTION
    
    
    #detect keypoints using a cascade filtering approach
    
    sigma = 1.7
    k = 2**0.5
    scales = 5
    octaves = 4
    base_image = np.zeros((shape(image1)))
    base_image[:]= image1
    
    #Using Guassian Kernel to smooth the image for each octaves with 
    #respective scalings for creating the pyramid structure 
    image_octaveList=[]
    image_baseList = []
    for i in range(octaves):
        image_scaleList=[]
        for j in range(scales):
            
            if i==0 and j==0:
                temp1=cp.deepcopy(base_image)
                image_scaleList.append(temp1)
            elif i>0 and j==0:
                temp2=ndimage.zoom(image_baseList[i-1][0],0.5, order =1)
                temp3=cp.deepcopy(temp2)
                image_scaleList.append(temp3)
        
        image_baseList.append(image_scaleList)
     
      
    #Guassian Blurring based on the histogram 
    #which can be computed from the difference of two nearby scales separated by a
    #constant multiplicative factor k
    for i in range(octaves):
        image_scaleList=[]
        for j in range(scales):
            
            if j==0:
                temp1 =np.zeros(np.shape(image_baseList[i][0]))
                temp1[:]=image_baseList[i][0]
            sigma=math.pow(k,j)*1.7
            histogram_size= int(math.ceil(7*sigma))
            histogram_size= 2*histogram_size+1
            
            temp2=temp3=np.zeros(np.shape(temp1))
            temp2=cv2.GaussianBlur(temp1,(histogram_size,histogram_size),sigma,sigma)
            image_scaleList.append(temp2)
       
        image_octaveList.append(image_scaleList)  
        #print shape(image_octaveList)     
    
    
    #Difference of Guassian computation at each of the Ocatves
    # List initiation by taking difference between consecutive Gaussian blurred images
    
    DoG_List=[]
    for i in range(octaves):
        image_scaleList=[]
        for j in range(1,scales):
            
            difference = np.zeros(np.shape(image_octaveList[i][0]))
            difference[:]= np.subtract(image_octaveList[i][j],image_octaveList[i][j-1])
            image_scaleList.append(difference)
            
        DoG_List.append(image_scaleList)
        #print shape(DoG_List)
        
            
                
                    
                        
                        
                                
    
    #LOCAL EXTREMA DETECTION


    #each sample point is compared to its eight neighbors in the current image and nine 
    #neighbors in the scale above and below
    
    #Finding the EXTREMUM, That is minima/maxima
    c1=0 #count
       
    #Remove the boundary pixels 
    #Binarize Image of Extremum Points
    image_extremumList=[]
    for i in range(octaves):
        image_scaleList=[]
        for j in range(1, scales-2):
            image_extremum=np.zeros(DoG_List[i][j].shape,dtype=np.float64)
            for l in range(1, DoG_List[i][j].shape[0]):
                for m in range(1, DoG_List[i][j].shape[1]):
                    #It is selected only if it is larger than all of these neighbors or smaller than all
                    #of them
                    ext_points= DoG_List[i][j][l][m]
                    if ext_points == max(DoG_List[i][j][l-1:l+2, m-1:m+2].max(), DoG_List[i][j-1][l-1:l, m-1:m+2].max(), DoG_List[i][j+1][l-1:l+2, m-1:m+2].max()):
                        image_extremum[l][m]= ext_points
                        c1+=1
                    elif ext_points== min(DoG_List[i][j][l-1:l+2, m-1:m+2].min(), DoG_List[i][j-1][l-1:l+2, m-1:m+2].min(), DoG_List[i][j+1][l-1:l+2, m-1:m+2].min()):
                        image_extremum[l][m]= ext_points
                        c1+=1
            image_scaleList.append(image_extremum)
        image_extremumList.append(image_scaleList)
    print("Number of Scaled Space Extremum Points:",c1)
    
    
    #Finding Candidate KeyPoints in each of the middle two layers of DoG
    #keypoint detection is to identify locations and scales that can be
    #repeatably assigned under differing views of the same object 
    
    key_points=0
    sigma_nonzero=[]
    extremum_nonzero=[]
    for i in range(octaves):
        image_sigmaList=[]
        image_scaleList=[]
        for j in range(scales-3):
            temp4=[]
            temp4[:] = np.transpose(image_extremumList[i][j].nonzero())
            key_points+=len(temp4)
            image_scaleList.append(temp4)
            image_sigmaList.append(math.pow(k,j)*1.6)
        extremum_nonzero.append(image_scaleList)
        sigma_nonzero.append(image_sigmaList)

    #plot all the non-zero extremum points
    plt.gray()
    plt.figure(n+1)
    plt.imshow(image1)
    for i in range(octaves):
        for j in range(0,2):
            for l in range(len(extremum_nonzero[i][j])):
                x=math.pow(2,i)*extremum_nonzero[i][j][l][0]
                y=math.pow(2,i)*extremum_nonzero[i][j][l][1]
                x1= [x]
                y1 =[y]
                plt.plot(y1,x1, 'b*')
    plt.title('Non-Zero Extremum Points')
   

   
      
      
      
            
        
    #ACCURATE KEYPOINT LOCALIZATION
    
    
    #finding a detailed fit to the nearby data for location, scale, 
    #and ratio of principal curvatures
    c2=1 # counter for the finding the key points counter
    c3=0 # counter 
    extremum_points= []
    for i in range(octaves):
        image_scaleList=[]
        for j in range(2):
            c2=1
            keyPointsPerScale =[]
            
            for l in range(len(extremum_nonzero[i][j])):
                matrix_A = np.zeros((3,3))
                matrix_B = np.zeros((3,1))
                x_coord= extremum_nonzero[i][j][l][0]
                y_coord= extremum_nonzero[i][j][l][1]
                sigma_current = sigma_nonzero[i][j] 
                
                #simply locate keypoints at the location and scale of the central sample point
                ##Finding the derivativess and filling the 3x3 Linear Systems
                if(x_coord+1 < DoG_List[i][0].shape[0] and y_coord+1 < DoG_List[i][0].shape[1] and x_coord-1 >-1 and y_coord-1 >-1):
                    x_newcoord=x_coord
                    y_newcoord=y_coord
                    xnew=np.zeros((3,1))
                    sigma_new = sigma_current
                    
                    matrix_A[0][0] = DoG_List[i][j][x_coord][y_coord] - 2*DoG_List[i][j+1][x_coord][y_coord] + DoG_List[i][j+2][x_coord][y_coord]
                    matrix_A[0][1] = DoG_List[i][j+2][x_coord+1][y_coord] -DoG_List[i][j+2][x_coord-1][y_coord] - DoG_List[i][j][x_coord+1][y_coord] + DoG_List[i][j][x_coord-1][y_coord]
                    matrix_A[0][2] = DoG_List[i][j+2][x_coord][y_coord+1] -DoG_List[i][j+2][x_coord][y_coord-1] - DoG_List[i][j][x_coord][y_coord+1] + DoG_List[i][j-2][x_coord][y_coord-1]
                   
                    matrix_A[1][0] = matrix_A[0][2]
                    matrix_A[1][1] = DoG_List[i][j+1][x_coord+1][y_coord] - 2*DoG_List[i][j+1][x_coord][y_coord] + DoG_List[i][j+1][x_coord-1][y_coord]
                    matrix_A[1][2] = DoG_List[i][j+1][x_coord-1][y_coord-1] - DoG_List[i][j+1][x_coord+1][y_coord-1]  - DoG_List[i][j+1][x_coord-1][y_coord+1] + DoG_List[i][j+1][x_coord+1][y_coord+1]
                   
                    matrix_A[2][0] = matrix_A[0][2]
                    matrix_A[2][1] = matrix_A[1][2]
                    matrix_A[2][2] = DoG_List[i][j+1][x_coord][y_coord+1] - 2*DoG_List[i][j+1][x_coord][y_coord] + DoG_List[i][j+1][x_coord][y_coord-1]

                    matrix_B[0][0] =  DoG_List[i][j+2][x_coord][y_coord] - DoG_List[i][j][x_coord][y_coord]
                    matrix_B[1][0] =  DoG_List[i][j+1][x_coord+1][y_coord]- DoG_List[i][j+1][x_coord-1][y_coord]
                    matrix_B[2][0] =  DoG_List[i][j+1][x_coord][y_coord+1]- DoG_List[i][j+1][x_coord][y_coord-1]
                    
                    xdash=np.dot(np.linalg.pinv(matrix_A),matrix_B)
                    xnew[:] = xdash
                    

                    #If the offset ˆx is larger than 0.5 in any dimension, then it means that the extremum
                    #lies closer to a different sample point
                    # Change points having offset greater than 0.5 in any dimensions
                    skipPoint=0
                    if abs(xdash[0][0])>0.5 or abs(xdash[1][0])>0.5 or abs(xdash[2][0])>0.5:
                        skipPoint=1      
                        if abs(xdash[1][0])>0.5 :
                            x_newcoord = x_coord + round(xdash[1][0])
                            xnew[1][0] = xdash[1][0]- round(xdash[1][0])
                            if (x_newcoord > image_octaveList[i][0].shape[0]-1) or x_newcoord <0:
                                skipPoint =1
                                
                        if abs(xdash[2][0])>0.5:
                            y_newcoord= y_coord + round(xdash[2][0])
                            xnew[2][0] = xdash[2][0] - round(xdash[2][0])
                            if (y_newcoord > image_octaveList[i][0].shape[1]-1) or y_newcoord<0:
                                skipPoint =1
                        
                        if abs(xdash[0][0])>0.5:
                            if xdash[0][0]> 0 :
                                sigma_new = math.pow(k, (j+1))*1.6
                                xnew[0][0] = (sigma_new - math.pow(k,j)*1.6) - xdash[0][0]
                            else:
                                sigma_new = math.pow(k,(j-1))*1.6
                                xnew[0][0] = (math.pow(k,j)*1.6 - sigma_new) + xdash[0][0]
    
  
                    # Eliminating Low Contrast KeyPoints and checking for poor edge localizations 
                    if (skipPoint==0):
                        contrast_keypoint = DoG_List[i][j+1][x_newcoord][y_newcoord] + 0.6 * matrix_B[1][0] *xnew[2][0] + matrix_B[2][0]*xnew[2][0] + matrix_B[0][0] * xnew[0][0]
                    
                    #all extrema with a value of |D(ˆx)| less than 0.03 were
                    #discarded (as before, we assume image pixel values in the range [0,1]).
                   
                    #Hessian Part                      
                        if abs(contrast_keypoint)>0.03:
                            diff_xx = DoG_List[i][j+1][x_coord+1][y_coord] - 2*DoG_List[i][j+1][x_coord][y_coord] + DoG_List[i][j+1][x_coord-1][y_coord]
                            diff_xy = DoG_List[i][j+1][x_coord-1][y_coord-1] - DoG_List[i][j+1][x_coord+1][y_coord-1] + DoG_List[i][j+1][x_coord-1][y_coord+1] +DoG_List[i][j+1][x_coord+1][y_coord+1]
                            diff_yy = DoG_List[i][j+1][x_coord][y_coord+1] - 2*DoG_List[i][j+1][x_coord][y_coord] + DoG_List[i][j+1][x_coord][y_coord-1]
                           
                            trace_H = diff_xx + diff_yy
                            determinant_H = diff_xx * diff_yy - diff_xy**2
                            curvature_ratio = (trace_H*trace_H)/determinant_H
                            #Eliminating edge responses
                            #the curvatures have different signs so the
                            #point is discarded as not being an extremum
                            
                            
                            #experiments in the paper use a value of r = 10,
                            #which eliminates keypoints that have a ratio between the principal curvatures greater than 10
                            if abs(curvature_ratio)<10.0:
                                key_attributePoints = []
                               
                                key_attributePoints.append(c2)
                                key_attributePoints.append(x_newcoord)
                                key_attributePoints.append(y_newcoord)
                                
                                key_attributePoints.append(sigma_new)
                                
                                key_attributePoints.append(xnew[0][0])
                                key_attributePoints.append(xnew[1][0])
                                key_attributePoints.append(xnew[2][0])
                                
                                key_attributePoints.append(x_coord)
                                key_attributePoints.append(y_coord)
                                
                                key_attributePoints.append(sigma_current)
                                key_attributePoints.append(j+1)
                               
                                c2= c2+1
                                keyPointsPerScale.append(key_attributePoints)
                                c3 +=1
            
            image_scaleList.append(keyPointsPerScale)
        extremum_points.append(image_scaleList)
    print("The initial key points locations at maxima and minima of the difference-of-Gaussian function:", c3)
    
    #keypoint selection on a natural image
    # Get All the KEYPOINTS
    plt.gray()
    plt.figure(n+2)
    plt.imshow(image1)
    
    for i in range(octaves):
        for j in range(2):
            for l in range(len(extremum_points[i][j])):
                x=math.pow(2,i)*extremum_points[i][j][l][1]
                y=math.pow(2,i)*extremum_points[i][j][l][2]
                x1= [x]
                y1 =[y]
                plt.plot(y1,x1, 'b*') 
    plt.title('Key Points')            
    
    
    
    
    
    
    
    # ORIENTATION ASSIGNMENT
    
    
    #In order to avoid too much clutter, a low-resolution pixel image is used and keypoints are shown as
    # vectors giving the location, scale, and orientation of each keypoint
    c4 = []
    c5 = 0
    for i in  range (octaves):
        image_scaleList = []
        for j in range(scales-3):
            c2 = 1

            keyPointsPerScale = []
            for p in  range(len(extremum_points[i][j])):
                x_coord = extremum_points[i][j][p][1]
                y_coord = extremum_points[i][j][p][2]
                
                sig = extremum_points[i][j][p][3]
                IOr = np.zeros(image_octaveList[i][j].shape)
                IOr = image_octaveList[i][j]
                
                histogram_size = int(math.ceil(7*sig))
              
                Iblur = np.zeros(IOr.shape)
               
                H = cv2.getGaussianKernel(histogram_size,int(sig));
                Iblur[:                                                                                                                                                                                                                                                                                                                                                                                                                                                     ] = cv2.filter2D(IOr,-1,H);

                bins = np.zeros((1,36));
                
                #keypoint descriptor can be represented relative to this orientation and therefore achieve invariance
                #to image rotation
                
                #The highest peak in the histogram is detected, and then any other local peak that is within
                #80% of the highest peak is used to also create a keypoint with that orientation
                for s in range(-histogram_size,histogram_size+1):
                    for t in range(-histogram_size,histogram_size+1):
                        if (((x_coord + s)>0) and ((x_coord + s)<(Iblur.shape[0]-1)) and ((y_coord+ t)>0) and ((y_coord + t)<(Iblur.shape[1]-1))):
                            xmag1 = Iblur[x_coord+s+1][y_coord+t]
                            xmag2 = Iblur[x_coord+s-1][y_coord+t]
                            
                            ymag1 = Iblur[x_coord+s][y_coord+t+1]
                            ymag2 = Iblur[x_coord+s][y_coord+t-1]
                            m = math.sqrt(math.pow((xmag1-xmag2),2) + math.pow((ymag1-ymag2),2))
                            den = xmag2-xmag1
                            
                            #An orientation histogram is formed from the gradient orientations of sample points within
                            #a region around the keypoint                            
                            if den==0:
                               den = 5
                            theta = math.degrees(math.atan((ymag2-ymag1)/(den)))
                            #The orientation histogram has 36 bins covering the 360 degree
                            #range of orientations. 
                            
                            if(theta<0):
                                theta = 360 + theta                           
                            binary = (int)((theta/360)*36)%36
                            
                           
                            if binary ==36:
                                binary = 35
                            bins[0][binary] = bins[0][binary] + m

                maxBinNo = np.argmax(bins)
                maxtheta = maxBinNo*10
                maxmag = bins[0][maxBinNo]
                
                extremum_points[i][j][p].append(maxtheta)
                extremum_points[i][j][p].append(maxmag)


                nbins = 36
                threshold = 0.8
                o = 0
                for y in range(0,36):
                    orientation = 0
                    y_prev = (y-1+nbins)%nbins
                    y_next = (y+1)%nbins
                    
                    if bins[0][y] > threshold*maxtheta and bins[0][y] > bins[0][y_prev] and  bins[0][y]> bins[0][y_next]:
                        offset = (bins[0][y_prev] - bins[0][y_next])/(2*(bins[0][y_prev]+bins[0][y_next]-2*bins[0][y]))
                        exact_bin = y + offset
                        orientation = exact_bin*360/float(36)
                        
                        #Each sample added to the histogram is weighted by its gradient magnitude
                        #and by a Gaussian-weighted circular window with a σ that is 1.5 times that of the scale
                        #of the keypoint.
                        if orientation>360:
                            orientation-=360
                       
                        o+=1
                        extPtskey_attributePoints = []
                        extPtskey_attributePoints[:] = extremum_points[i][j][p]
                        extPtskey_attributePoints[11] = orientation
                        keyPointsPerScale.append(extPtskey_attributePoints)
            c5 +=len(keyPointsPerScale)
            image_scaleList.append(keyPointsPerScale)
        c4.append(image_scaleList)
    print("Principal Oreintation points after Thresholding",c5)
 
    
    
    
    
    
    
    
    #THE LOCAL IMAGE DESCRIPTOR
    

    #compute gradient for scale space
    dx_list  = []
    dy_list  = []
    for i in range(len(image_octaveList)):
        image_scaleList1 = []
        image_scaleList2 = []
        for j in range(scales):
            dx,dy = np.gradient(image_octaveList[i][j])
            image_scaleList1.append(dx)
            image_scaleList2.append(dy)
        dx_list.append(image_scaleList1)
        dy_list.append(image_scaleList2)

    const = 3
    plt.gray()
    plt.figure(n+3)
    plt.imshow(image1)
            
    for i in range(octaves):
        for j in range(2):
            for l in range(len(extremum_points[i][j])):
                x=math.pow(2,i)*extremum_points[i][j][l][1]
                y=math.pow(2,i)*extremum_points[i][j][l][2]
                dx =  const*extremum_points[i][j][l][3] * math.degrees(math.cos(extremum_points[i][j][l][10]))
                dy =  const*extremum_points[i][j][l][3] * math.degrees(math.sin(extremum_points[i][j][l][10]))
                x1= [x]
                y1 =[y]
                plt.plot(y1,x1, 'b*')
    plt.title('Image Descriptor Key Points')
    plt.figure(n+4)
    plt.imshow(image1)
    plt.title('Original Image')
    plt.show()
    
    
  
  
  
  
  
  
  
 
    
       
             
###################################################################
# REading the input image
input_image1= cv2.imread('SIFT-input1.png',0)
input_image2 = cv2.imread('SIFT-input2.png',0)

# Function returns the KeyPoints on the input images and corresponding SIFT plots
SIFT(input_image1,0)
SIFT(input_image2,3)