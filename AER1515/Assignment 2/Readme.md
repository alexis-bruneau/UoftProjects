# AER1515 Assignment 2
====================

The homework is split into 2 sections: Feature Point Detection and Correspondences, 3D Point Cloud Registration

The first part consists of matching features between stereo camera pairs, filtering out incorrect correspondences using an outlier rejetion algorithm. The rectified stereo image pairs are from [Kitti](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) dataset. The section is devided in the following three sections:

* Feature Detection
* Feature Matching
* Outlier Rejection

The second Part consists of 3D Point Cloud Registration. The goal is to take a rigid transform from one point cloud to another such that they align together. The code consists of the three following steps:

* Nearest Neighbour Search
* Pose Estimation from the Correspondence
* ICP Registration

## Code Requirements
----

1) Create virtual environment Python 3.10.7

2) run the requirements.txt file

3) There's two folder, one for each question. The folder named Feature_Matching_Correspondence have the files/code answering 
Question 1. The folder named Point_Cloud_Registraion has the file/folder answering question 2.

### Feature Matching Correspondence

4) Steps to run Feature_Matching_Correspondence code:
In this file, 4 functions were created to answer q1.1, q1.2, and q1.33. By default running the file will run all 3 questions one after each  other.
The 4 functions are located at the end of the start_code_feature file. 
	4.1) function question1_1, question1_2, and question1_3 will illustrates the keypoints and matches. Their parameter is whether to show the training (training = True) or testing results (training = False)
	4.2) The eval_result function evaluates the accuracy of question 2 and 3. Its parameter shows which RMSE to calculate (2 for 1.2 or 3 for 1.3)
	4.3) File output explanation
		4.3.1) P2_result_test is the output of question 2 for test images
		4.3.2) P2_result_training is the output of question 2 for trainning images
		4.3.3) P3_result is the output of question 3 for testing images (File asked in assignment)
		4.3.4) P3_result_training is the output of question 3 for training images

### Point_Cloud_Registration

5) Steps to run Point_Cloud_Registration
Make sure you are in the correct folder directory and the folder has access to the library mentionned.
After that, all you need to do is run the python file. It should display the point cloud registration before icp and after for the 3 dataset,it will show the graph the euclidean mean and translation graph
the ground truth, estimated pose and the error. The code might take a few minutes before displaying any outputs. 
