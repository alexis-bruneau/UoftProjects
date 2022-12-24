# AER1515 Assignment 3
====================

The purpose of this assignment is to perform depth estimation, object detection of cars using YOLOv3, and perform image segmentation on the object detected (cars). The dataset used is from the KITTI dataset. 

The folder path should be the follwing:

Main File
* Calibration_Assignment_2
* kitti_dataHandler
* part1_estimate_depth
* part2_yolo
* part3_segmentation
* data (folder)
	* test (folder)
		* Bounding_box_Images (folder)
		* calib (folder)
		* disparity (folder)
		* est_segmentation (folder)
		* estimated_box (folder)
		* left (folder)
	* train (folder)
		* calib (folder)
		* disparity (folder)
		* est_segmentation (folder)
		* estimated_box (folder)
		* estimated_depth (folder)
		* gt_depth (folder)
		* gt_labels (folder)
		* gt_segmentation (folder)
		* left (folder)



## Code Requirements
----

1) Create virtual environment Python 3.10.7

2) run the requirements.txt file


### Running the code
----------------

Step 3.1 (Question 1)
To run question 1, open the main file directory and open part1_estimate_depth.py (* To note, it call Calibration_Assignment2.py) from assignment 2

3.2 ( Question 2)
* On line 223, you can modifry the parameters to the desired ones and then run the code 
* To note, Line 144 and 145 has the original values that was tested to find the best confidence threshold and threshold

3.3 (Question 3)
* No modification is required if you just want to run the code using the best values found
* Uncomment line 324 to 330 if you want to test out the values that were tried to find the best cropping value and distance threshold
* Line 333 & 334 shows the best parameters found looping through all the values
* Line 337 & 340 calls function to create segmentation for train & test images. 