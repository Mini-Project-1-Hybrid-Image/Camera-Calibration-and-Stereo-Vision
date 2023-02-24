# Introduction
	In this project ,we have gone through three main parts :
	● Part1: in which we calculate the projection matrix from the 2d image points and their
	corresponding world 3d points.
	● Part2 :calibrating two cameras using a chessboard pattern forming stereovision.
	● parts 3: performing sift detection and matching between two of our captured images and
	applying epipolar constraint to increase the accuracy of the matches
	
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
		
	![p2](https://user-images.githubusercontent.com/49596777/221073533-d8d11870-b5fd-42b0-9c16-8478a09aaeeb.PNG)
	
	● Second camera :
		1. Reprojection Error: 0.664144274461620
		2. The total residual is: 98.63853631015635
	
	![p22](https://user-images.githubusercontent.com/49596777/221073653-6c3972a9-0025-4dfb-b8ee-1d1aaeb7493f.PNG)
	
        ●Comments 
                Although the total residual is little high for both cameras but the reprojection error is good
		(in subpixels accuracy) ,also the visualization is not bad
       
	
	
	
