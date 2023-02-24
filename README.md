# Introduction
	In this project ,we have gone through three main parts :
	● Part1: in which we calculate the projection matrix from the 2d image points and their
	corresponding world 3d points.
	● Part2 :calibrating two cameras using a chessboard pattern forming stereovision.
	● parts 3: performing sift detection and matching between two of our captured images and
	applying epipolar constraint to increase the accuracy of the matches
# Part1 
	We calculated the projection matrix through the given 2d and 3d correspondences by solving
	System of equations in the form AU=B using least square where U is the projection matrix then
	we calculated the camera center through the following steps :
		● calculate_projection_matrix
		1. constructing the matrix A from the given 2d and 3d points
		2. Constructing matrix B by reshaping the given 2d points
		3. Solving using the linear technique where U=(A.T*A)^-1 *A.T*B
		● compute_camera_center(M)
		1. M=[Q|m4]
		2. c=Q^-1*m4
# Results of part1
![p4](https://user-images.githubusercontent.com/49596777/221073306-0cfbfcb0-b950-46c3-8ac6-5379344975fd.PNG)
![p42](https://user-images.githubusercontent.com/49596777/221073310-368f6788-93a1-4396-80f4-b06011f5f72d.PNG)


# Part2
	In this part we calibrated our own cameras using chessboard pattern and then we
	estimated the essential matrix using stereocalibrate function :
	● CamerCalibrate() [function implementation using chessboard pattern]
	1. Starting by defining the chessboard dimension which is (6,8)
	2. Then defining the world coordinate for the 3d points of the chessboard
	3. For each image ,find the corners in the chessboard using
	findchessborad function
	4. If the corners found ,then append the image points and and object
	points
	5. Calibrate the camera using Camera calibrate function
	6. Calculate rotation matrix using opencv Rodrigues
	7. Concatenating the rotation matrix and translation vector (RT)
	8. Getting the projection matrix using by multiplying the camera matrix
	with RT
	
# Part 2 results
	● First camera :
		1. Reprojection Error: 0.5689131935306923
		2. The total residual is: 133.69416547523434
		
![p2](https://user-images.githubusercontent.com/49596777/221073915-8a157dcb-f6ee-4296-a838-b9419a50929a.PNG)

	● Second camera :
		1. Reprojection Error: 0.664144274461620
		2. The total residual is: 98.63853631015635
	
![p22](https://user-images.githubusercontent.com/49596777/221073926-501a51e4-5937-41a2-8137-0b0236fef795.PNG)

	
        ● Comments 
                Although the total residual is little high for both cameras but the reprojection error is good
		(in subpixels accuracy) ,also the visualization is not bad
       
     
 # Part3:
	In this part we calibrated our own cameras as done before. Then we captured images
	of a selected object.
	● Then we used sift_create.detect and compute to calculate keypoints and
	descriptors.
	● We used flannbasedmatcher then we applied knnmatch to find the matches
	● After that we applied the ratio test to detect the uncertainties
	● We found x and y coordinates, then we improve the matches by calculating the
	epipolar constraint=0 but we find a threshold between 0.1 and -0.1
	● After that we used the image points after applying the epipolar constraints in
	triangulatepoints, and we plot the point in a 3d view.
	● We compared the length after the reconstruction with the length between the
	main corners in the object, and we found that they are near each in some
	lengths.
	At last, we recommend using a better camera than ours to help better with the
	matching and the calibration.
![p3](https://user-images.githubusercontent.com/49596777/221074279-e1f1b202-0c4d-4a7c-8f43-e6efdac68c69.PNG)
![p32](https://user-images.githubusercontent.com/49596777/221074282-2fdc78c6-814c-4d2b-855a-f1e0aba65fa1.PNG)

	
	
	
