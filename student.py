# Projection Matrix Stencil Code
# Written by Eleanor Tursman, based on previous work by Henry Hu,
# Grady Williams, and James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech

import numpy as np
from sympy import Point3D
import cv2
import numpy as np
import os
import glob
from helpers import plot3dview, plot3dview2


def calculate_projection_matrix(Points_2D, Points_3D):
   
    print('Randomly setting matrix entries as a placeholder')
    M = np.array([[0.1768, 0.7018, 0.7948, 0.4613],
                  [0.6750, 0.3152, 0.1136, 0.0480],
                  [0.1020, 0.1725, 0.7244, 0.9932]])
    ##################
    #Solving using least square  AU=B 
    #dimensons :
    # A->  (N*2,12)  |||| B->(N*2,1)   ||| U -> (12,1) to be rehsaped 
    print(Points_2D)
    N=Points_3D.shape[0]          ## N
    A=np.zeros((N*2,11))
    B=np.zeros((N*2,1))
    i=0                     ## counter to iterate over marix
    #constructing A
    for j in range (N):
        ##buiding x row
        A[i,0:3]=Points_3D[j,:]   #first 3 elemnts in each row of A
        A[i,3]=1                  #fourth elemet in A
        last_elemnts=Points_2D[j,0]*Points_3D[j,:]  ##last 3 elements of each x row 
        A[i,8:11]=-1*last_elemnts
        #A[i,11]=-1*Points_2D[j,0]

        ##Building y row 
        A[i+1,4:7]=Points_3D[j,:]   #first 3 elemnts in each row of A
        A[i+1,7]=1                  #fourth elemet in A
        last_elemnts=Points_2D[j,1]*Points_3D[j,:]  ##last 3 elements of each x row 
        A[i+1,8:11]=-1*last_elemnts
        #A[i+1,11]=-Points_2D[j,1]
        i+=2
    #print(A)
    B=np.reshape(Points_2D,(-1,1))
    AT_A=np.matmul(A.T,A )  # A^T *A
    inverse =np.linalg.inv(AT_A)
    AT_B=np.matmul(A.T,B)
    U=np.matmul(inverse,AT_B)
    U=np.append(U,1)
    M=np.reshape(U,(3,4))
    return M


# Returns the camera center matrix for a given projection matrix
# 'M' is the 3x4 projection matrix
# 'Center' is the 1x3 matrix of camera center location in world coordinates
def compute_camera_center(M):
   
    Center = np.array([1, 1, 1])
    Q=M[:,0:3]
    m4=M[:,3]
    Center=np.matmul(-1*np.linalg.inv(Q),m4)


    return Center



   ## Calibrate Camera function 
def CameraCalibrate(o,ffname):

    ##inputs is :
    #1- o : number of image of wanted projection matrix 
    #2- ffname : name of the folder containing the images 
    print("Calibrating Camera ....")
    #Defining the dimensions of checkerboard
    CHECKERBOARD = (6,8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 


    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    path='./'+ffname+'/*.jpg'
    images = glob.glob(path)

    #p#rint(images)
    p=0
    for fname in images:
        #print(fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,cv2.CALIB_CB_ADAPTIVE_THRESH+
            cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
       # print(ret)
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            
            imgpoints.append(corners2)
            if p==o:
                points_3d= np.array(objp).reshape(-1,3)
                points_2d=np.array(corners2).reshape(-1,2)
                
            
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)
            p+=1

    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    print("Reprojection Error: \n",ret)
    Rotaion_matrix = (cv2.Rodrigues(rvecs[o]))[0]

    translation_vector = tvecs[o]
    ##conctaneting rotaion atrix and translaton vector 
    Rotaion_translation = np.concatenate([Rotaion_matrix,translation_vector], axis=1) # [R|t]
    ##Calculating projection Matrix 
    projection_matrix= np.dot(mtx,Rotaion_translation) # A[R|t]
    print("Done Calibrating.....")
    return projection_matrix,points_3d,points_2d,objpoints,imgpoints,mtx,dist,rvecs, tvecs,gray





def part_3 (folderName,E,f1,f2,M2,M3) :

    ##inputs 
    #foldername containing the images 
    #essiential matrix 
    #f1: camera matrix 1
    #f2 : camera matrix 2
    # m2,m3 :projecton matrix of the two cameras
    print("part3 start....  ")
    path='./'+folderName+'/*.jpg'
   # path='./hh/*.jpg'
    images = glob.glob(path)
    i=0
    
    for fname in images:
        img = cv2.imread(fname)
        # convert to greyscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        #print(np.array(keypoints).shape)
       
        
       
        
        #print(int(str(keypoints[0][1]),16))
        if i==0:
            k1=keypoints
            d1=descriptors
            img1=img
            i+=1
        else:
             k2=keypoints
             d2=descriptors

        sift_image = cv2.drawKeypoints(gray, keypoints, img)
        # show the image
        cv2.imshow('image', sift_image)
        # save the image
       # cv2.imwrite("table-sift.jpg", sift_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            #sift detect
        
  #sift detect
    bf = cv2.BFMatcher()
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(d1,d2,k=2)
    #matches = bf.knnMatch(d1,d2,k=2)
    #print((np.array(matches)).shape)
    #matchesarray=np.array(matches)
    #epipolar_matches=np.zeros((matchesarray.shape))
    #bf=cv2.BFMatcher()
    #matches=bf.match(d1,d2)
    #matches=sorted(matches, key= lambda x:x.distance)

    # Initialize lists
    #list_kp1 = []
    #list_kp2 = []

    # For each match...
    #for mat in matches:

        # Get the matching keypoints for each of the images
      #  img1_idx = mat.queryIdx
       # img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
       # (x1, y1) = k1[img1_idx].pt
        #(x2, y2) = k2[img2_idx].pt

        # Append to each list
        #print(x1,y1)
       # list_kp1.append((x1, y1,f1))
        #list_kp2.append((x2, y2,f2))
        
    #c1_co=np.array(list_kp1).reshape(-1,3)
    #c2_co=np.array(list_kp2).reshape(-1,3)

    #checking the epolar constrain
  #  for j in range (c1_co.shape[0]):
   #     con=np.matmul(c2_co[0,:].T,E)
    #    consrain=np.matmul(con,c1_co[0,:])
     #   if consrain==0:
      #      print("Match!!!")



    
    #print(f1,f2)
    #E=E/np.linalg.norm(E)
    m1=np.linalg.inv(f1)
    m2=np.linalg.inv(f2)
    img1_points=[]
    img2_points=[]
    good = []
    for m ,n in matches:
        pts1=np.zeros((1,3))
        pts2=np.zeros((1,3))
        (x1, y1) = k2[m.trainIdx].pt
        (x2, y2) = k1[m.queryIdx].pt
        img1_points.append([x1,y1])
        img2_points.append([x2,y2])
       # print(k2[m.trainIdx].pt)
        pts2=np.array([x2,y2,1])
       
        #pts2.append((f2))
        pts1=np.array([x1,y1,1])
        ##Normalizing the two points 
        v2=np.matmul(m2,pts1)
        v1=np.matmul(m1,pts2)
        
        ##Epipolar cnstrains
        con=np.matmul(v2.T,E)
        consrain=np.matmul(con,v1)
        ##threshold to detecet if epipolar constraint is met
        if (consrain>-.2 and consrain<.2):
            print("Match!!!")
            good.append([m])
    
    img1_points=np.array(img1_points).reshape(-1,2)
    img2_points=np.array(img2_points).reshape(-1,2)
    img3 = cv2.drawMatchesKnn(img1,k1,img,k2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('image', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ##Reconstruction and Visualization
 
    points=cv2.triangulatePoints(M3,M2,img1_points.T,img2_points.T).T
    points = points[:, :3] / points[:, 3:]
    plot3dview2(points)
    

    





