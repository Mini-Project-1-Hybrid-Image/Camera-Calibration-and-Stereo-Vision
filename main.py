# Camera Calibration Stencil Code
# Transferred to python by Eleanor Tursman, based on previous work by Henry Hu,
# Grady Williams, and James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
#
# This script
# (1) Loads 2D and 3D data points and images
# (2) Calculates the projection matrix from those points    (you code this)
# (3) Computes the camera center from the projection matrix (you code this)
# (4) Estimates the fundamental matrix                      (you code this)
# (5) Adds noise to the points if asked                     (you code this)
# (6) Estimates the fundamental matrix using RANSAC         (you code this)
#     and filters away spurious matches                                    
# (7) Visualizes the F Matrix with homography rectification
#
# The relationship between coordinates in the world and coordinates in the
# image defines the camera calibration. See Szeliski 6.2, 6.3 for reference.
#
# 2 pairs of corresponding points files are provided
# Ground truth is provided for pts2d-norm-pic_a and pts3d-norm pair
# You need to report the values calculated from pts2d-pic_b and pts3d
import warnings
import numpy as np
import os
import argparse
import cv2 

#from regex import F
from student import (calculate_projection_matrix, compute_camera_center,CameraCalibrate,part_3)
from helpers import (evaluate_points, visualize_points, plot3dview)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main(args):
    #data_dir = os.path.dirname(__file__) + '../data/'
    #data_dir = 'E:\Year4_2\Vision\Mini-Project 4/data'

   # print("here is dict :" ,data_dir)

    ########## Part (1)
    Points_2D = np.loadtxt('../data/pts2d-norm-pic_a.txt')
    Points_3D = np.loadtxt('../data/pts3d-norm.txt')

    # (Optional) Run this once you have your code working
    # with the easier, normalized points above.
    if args.hard_points:
        Points_2D = np.loadtxt('../data/pts2d-pic_b.txt')
        Points_3D = np.loadtxt('../data/pts3d.txt')

    # Calculate the projection matrix given corresponding 2D and 3D points
    # !!! You will need to implement calculate_projection_matrix. !!!
    print("shapes : ",Points_2D.shape,Points_3D.shape)
    M = calculate_projection_matrix(Points_2D, Points_3D)
    print('The projection matrix is:\n {0}\n'.format(M))

    Projected_2D_Pts, Residual = evaluate_points(M, Points_2D, Points_3D)
    print('The total residual is:\n {0}\n'.format(Residual))

    if not args.no_vis:
        visualize_points(Points_2D, Projected_2D_Pts)

    # Calculate the camera center using the M found from previous step
    # !!! You will need to implement compute_camera_center. !!!
    Center = compute_camera_center(M)
    print('The estimated location of the camera is:\n {0}\n'.format(Center))

    if not args.no_vis:
        plot3dview(Points_3D, Center)
    R=[]
    for u in range(1):
        ##you change here in num of iterations of for loop
        M2,world,image,objpoints,imgpoints1,mtx1,dist1,rvecs1, tvecs1,gray1=CameraCalibrate(u,'pattern2')
        #print('The projection matrix is:\n {0}\n'.format(M2))
        
        Projected_2D_Pts, Residual = evaluate_points(M2,image,world)
        visualize_points(image, Projected_2D_Pts)
        R.append(Residual)
        print('The total residual is:\n {0}\n'.format(Residual))

    for u in range(1):
        ##you change here in num of iterations of for loop
        M3,world,image,objpoints2,imgpoints2,mtx2,dist2,rvecs2, tvecs2,gray2=CameraCalibrate(u,'pattern1')
        #print('The projection matrix is:\n {0}\n'.format(M2))
        
        Projected_2D_Pts, Residual = evaluate_points(M3,image,world)
        visualize_points(image, Projected_2D_Pts)
        R.append(Residual)
        print('The total residual is:\n {0}\n'.format(Residual))

    ##stereoCalibrate 
    stereocalib_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
   # retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)
    retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix,fundamentalMatrix=cv2.stereoCalibrate(objpoints,
    imgpoints1,imgpoints2,mtx1,dist1,mtx2,dist2,gray1.shape[::-1],stereocalib_criteria,flags)

    print("essentialMatrix: \n",essentialMatrix)



   #part3 
    part_3('Test',essentialMatrix,mtx1,mtx2,M2,M3)







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hard_points",
        help="Use the harder, unnormalized points, to be used after you have it \
                working with normalized points (for parts 1 and 2)",
        action="store_true")
    parser.add_argument("--no-vis",
                        help="Disable visualization for part 1",
                        action="store_true")
    arguments = parser.parse_args()
    main(arguments)
