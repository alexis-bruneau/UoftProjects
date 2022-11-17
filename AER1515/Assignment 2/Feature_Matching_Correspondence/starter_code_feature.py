import enum
import cv2
from cv2 import illuminationChange
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import csv
import os

from pyparsing import original_text_for

class FrameCalib:
    """Frame Calibration

    Fields:
        p0-p3: (3, 4) Camera P matrices. Contains extrinsic and intrinsic parameters.
        r0_rect: (3, 3) Rectification matrix
        velo_to_cam: (3, 4) Transformation matrix from velodyne to cam coordinate
            Point_Camera = P_cam * R0_rect * Tr_velo_to_cam * Point_Velodyne
        """

    def __init__(self):
        self.p0 = []
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.r0_rect = []
        self.velo_to_cam = []


def read_frame_calib(calib_file_path):
    """Reads the calibration file for a sample

    Args:
        calib_file_path: calibration file path

    Returns:
        frame_calib: FrameCalib frame calibration
    """

    data_file = open(calib_file_path, 'r')
    data_reader = csv.reader(data_file, delimiter=' ')
    data = []

    for row in data_reader:
        data.append(row)

    data_file.close()

    p_all = []

    for i in range(4):
        p = data[i]
        p = p[1:]
        p = [float(p[i]) for i in range(len(p))]
        p = np.reshape(p, (3, 4))
        p_all.append(p)

    frame_calib = FrameCalib()
    frame_calib.p0 = p_all[0]
    frame_calib.p1 = p_all[1]
    frame_calib.p2 = p_all[2]
    frame_calib.p3 = p_all[3]

    # Read in rectification matrix
    tr_rect = data[4]
    tr_rect = tr_rect[1:]
    tr_rect = [float(tr_rect[i]) for i in range(len(tr_rect))]
    frame_calib.r0_rect = np.reshape(tr_rect, (3, 3))

    # Read in velodyne to cam matrix
    tr_v2c = data[5]
    tr_v2c = tr_v2c[1:]
    tr_v2c = [float(tr_v2c[i]) for i in range(len(tr_v2c))]
    frame_calib.velo_to_cam = np.reshape(tr_v2c, (3, 4))

    return frame_calib


class StereoCalib:
    """Stereo Calibration

    Fields:
        baseline: distance between the two camera centers
        f: focal length
        k: (3, 3) intrinsic calibration matrix
        p: (3, 4) camera projection matrix
        center_u: camera origin u coordinate
        center_v: camera origin v coordinate
        """

    def __init__(self):
        self.baseline = 0.0
        self.f = 0.0
        self.k = []
        self.center_u = 0.0
        self.center_v = 0.0


def krt_from_p(p, fsign=1):
    """Factorize the projection matrix P as P=K*[R;t]
    and enforce the sign of the focal length to be fsign.


    Keyword Arguments:
    ------------------
    p : 3x4 list
        Camera Matrix.

    fsign : int
            Sign of the focal length.


    Returns:
    --------
    k : 3x3 list
        Intrinsic calibration matrix.

    r : 3x3 list
        Extrinsic rotation matrix.

    t : 1x3 list
        Extrinsic translation.
    """
    s = p[0:3, 3]
    q = np.linalg.inv(p[0:3, 0:3])
    u, b = np.linalg.qr(q)
    sgn = np.sign(b[2, 2])
    b = b * sgn
    s = s * sgn

    # If the focal length has wrong sign, change it
    # and change rotation matrix accordingly.
    if fsign * b[0, 0] < 0:
        e = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    if fsign * b[2, 2] < 0:
        e = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    # If u is not a rotation matrix, fix it by flipping the sign.
    if np.linalg.det(u) < 0:
        u = -u
        s = -s

    r = np.matrix.transpose(u)
    t = np.matmul(b, s)
    k = np.linalg.inv(b)
    k = k / k[2, 2]

    # Sanity checks to ensure factorization is correct
    if np.linalg.det(r) < 0:
        print('Warning: R is not a rotation matrix.')

    if k[2, 2] < 0:
        print('Warning: K has a wrong sign.')

    return k, r, t


def get_stereo_calibration(left_cam_mat, right_cam_mat):
    """Extract parameters required to transform disparity image to 3D point
    cloud.

    Keyword Arguments:
    ------------------
    left_cam_mat : 3x4 list
                   Left Camera Matrix.

    right_cam_mat : 3x4 list
                   Right Camera Matrix.


    Returns:
    --------
    stereo_calibration_info : Instance of StereoCalibrationData class
                              Placeholder for stereo calibration parameters.
    """

    stereo_calib = StereoCalib()
    k_left, r_left, t_left = krt_from_p(left_cam_mat)
    _, _, t_right = krt_from_p(right_cam_mat)

    stereo_calib.baseline = abs(t_left[0] - t_right[0])
    stereo_calib.f = k_left[0, 0]
    stereo_calib.k = k_left
    stereo_calib.center_u = k_left[0, 2]
    stereo_calib.center_v = k_left[1, 2]

    return stereo_calib

def question1_1(training = False):
    if training == False:
        left_image_dir = os.path.abspath('./test/left')  
        right_image_dir = os.path.abspath('./test/right')   
        sample_list = ['000011', '000012', '000013', '000014','000015'] # Sample list testing
    else:
        left_image_dir = os.path.abspath('./training/left')  
        right_image_dir = os.path.abspath('./training/right')  
        sample_list = ['000001', '000002', '000003', '000004','000005', '000006', '000007', '000008', '000009', '000010'] # Sample list training

    ## Main
    for sample_name in sample_list:
        left_image_path = left_image_dir +'/' + sample_name + '.png'     
        img_left = cv.imread(left_image_path, 0)
        
        right_image_path = right_image_dir +'/' + sample_name + '.png'     
        img_right = cv.imread(left_image_path, 0)
        
        # TODO: Initialize a feature detector with 1000 keypoints
        orb = cv.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(img_left, None) # Detect and compute keypoints using ORB
        kp2, des2 = orb.detectAndCompute(img_right, None) # Detect and compute keypoints using ORB

        # Two methods are used to draw the keypoints
        img = cv.drawKeypoints(img_left, kp1, None, color=(0,255,0), flags=0) # draw only keypoints location,not size and orientation
        plt.imshow(img), plt.show() # Show image
        
        img2 = cv.drawKeypoints(img_left, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(img2), plt.show() # Show image
      


def question1_2(training):

    '''
    This function does the feature matching between each image pairs with epipolar constraint

    Input
    --------------
    training: type boolean, true when performing on training dataset and False for testing
    
    '''
    # File path for training vs testing
    if training == True:
        left_image_dir = os.path.abspath('./training/left') # File path left image
        right_image_dir = os.path.abspath('./training/right') # File path right image
        calib_dir = os.path.abspath('./training/calib') 
        sample_list = ['000001', '000002', '000003', '000004','000005', '000006', '000007', '000008', '000009', '000010'] # Sample list training
        output_file = open("P2_result_training.txt", "a")  
    
    else:
        left_image_dir = os.path.abspath('./test/left') # File path left image
        right_image_dir = os.path.abspath('./test/right') # File path right image
        calib_dir = os.path.abspath('./test/calib') 
        sample_list = ['000011', '000012', '000013', '000014','000015'] # Sample list testing
        output_file = open("P2_result_test.txt", "a") 
    
    
    output_file.truncate(0)

    #Loop through each image pairs
    for sample_name in sample_list:

        #Get left and right images
        left_image_path = left_image_dir +'/' + sample_name + '.png'
        right_image_path = right_image_dir +'/' + sample_name + '.png'      
        img_left = cv.imread(left_image_path, 0)
        img_right = cv.imread(right_image_path, 0)

                
        # TODO: Initialize a feature detector with 1000 keypoints using ORB
        orb = cv.ORB_create(nfeatures=1000)

        #Detect and compute 
        kp_left, des1 = orb.detectAndCompute(img_left, None)
        kp_right, des2 = orb.detectAndCompute(img_right, None)

        # TODO: Perform feature matching using Brute Force        
        # Brute Force Matching 
        bf = cv.BFMatcher(cv.NORM_HAMMING) 

        # Create mask to enforce epipolar constraint and positive disparities only
        mask = np.zeros((1000, 1000), dtype=np.uint8)
        for i in range(1000):
            for j in range(1000):
                if kp_left[i].pt[0] < kp_right[j].pt[0]: continue  # Disparity enforcement
                if abs(kp_left[i].pt[1] - kp_right[j].pt[1]) < 1:  # Epipolar enforcement
                    mask[i][j] = 1

        # Match and showcases features from left and right image
        matches = bf.match(des1, des2,mask)                
        img = cv.drawMatches(img_left, kp_left, img_right, kp_right, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img)
        plt.show()

        #Read calibration
        frame_calib = read_frame_calib(calib_dir + '/' + sample_name + '.txt')
        stereo_calib = get_stereo_calibration(frame_calib.p2, frame_calib.p3)

        # Find disparity and depth
        pixel_u_list = [] # x pixel on left image
        pixel_v_list = [] # y pixel on left image
        disparity_list = []
        depth_list = []
        
        #For each match       
        for i, match in enumerate(matches):
            # Get the matching keypoints for each of the images
            img1_idx = match.queryIdx # Return index row of the kp1 interest point 
            img2_idx = match.trainIdx # Return index row of the kp2 interest point

            # Get the coordinates of x and y for kp_left and kp_right
            (x1, y1) = kp_left[img1_idx].pt
            (x2, y2) = kp_right[img2_idx].pt
            
            x_difference = x1 - x2

            if x_difference == 0: # Would cause issue when calculating depth
                continue
            # Append to each list
            pixel_u_list.append(x1)
            pixel_v_list.append(y1)         
            disparity_list.append(x_difference) # Caluclating disparity
            depth_list.append(stereo_calib.f * stereo_calib.baseline / x_difference) # Calculating depth
        

        
        for u, v, disp, depth in zip(pixel_u_list, pixel_v_list, disparity_list, depth_list):
            line = "{} {:.2f} {:.2f} {:.2f} {:.2f}".format(sample_name, u, v, disp, depth)
            output_file.write(line + '\n') 

    ## Output    
    output_file.close()

    
def question1_3(training):
    '''
    This function does the feature matching between each image pairs using RANSAC outlier detection (discussed in class) and no epipolar constraints

    Input
    --------------
    training: type boolean, true when performing on training dataset and False for testing
    
    '''
    # File path for training vs testing
    if training == True:
        left_image_dir = os.path.abspath('./training/left') # File path left image
        right_image_dir = os.path.abspath('./training/right') # File path right image
        calib_dir = os.path.abspath('./training/calib') 
        sample_list = ['000001', '000002', '000003', '000004','000005', '000006', '000007', '000008', '000009', '000010'] # Sample list training
        output_file = open("P3_result_training.txt", "a") 
    
    else:
        left_image_dir = os.path.abspath('./test/left') # File path left image
        right_image_dir = os.path.abspath('./test/right') # File path right image
        calib_dir = os.path.abspath('./test/calib') 
        sample_list = ['000011', '000012', '000013', '000014','000015'] # Sample list testing
        output_file = open("P3_result.txt", "a")
    #Output

    output_file.truncate(0)

    
    #Loop through each image pairs
    for sample_name in sample_list:

        #Get left and right images
        left_image_path = left_image_dir +'/' + sample_name + '.png'
        right_image_path = right_image_dir +'/' + sample_name + '.png'      
        img_left = cv.imread(left_image_path, 0)
        img_right = cv.imread(right_image_path, 0)

                
        # TODO: Initialize a feature detector with 1000 keypoints using SIFT
        sift = cv.SIFT_create(nfeatures=1000)

        #Detect and compute 
        kp_left, des1 = sift.detectAndCompute(img_left, None)
        kp_right, des2 = sift.detectAndCompute(img_right, None)

        # TODO: Perform feature matching using Brute Force      
        bf = cv.BFMatcher()       
        matches = bf.knnMatch(des1, des2, k=2)               
               
        # Store all the good matches as per Lowe's ratio test.
        # https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html

        pts_left = []
        pts_right = []
        lowes_match = []

        for m,n in matches:          
            if m.distance < 0.5*n.distance:                
                lowes_match.append(m)
                pts_left.append([kp_left[m.queryIdx].pt])
                pts_right.append([kp_right[m.trainIdx].pt])
       
        #Convert pts_left and right to array
        pts_left = np.array(pts_left)
        pts_right = np.array(pts_right)
              
        #Using Ransac Method for Outliers     
        # Estimation of fundamental matrix using RANSAC algorithm with cv.findFundamentalMat
        # http://amroamroamro.github.io/mexopencv/matlab/cv.findFundamentalMat.html
        # https://python.hotexamples.com/examples/cv2/-/findFundamentalMat/python-findfundamentalmat-function-examples.html

        F, mask = cv.findFundamentalMat(pts_left, pts_right, method=cv.FM_RANSAC)
        
        new_matches = []
            
        # Loop through matches that were returned using Lowe's ratio test
       
        for i in range(len(lowes_match)):            
            if mask[i] == 1:                          
                new_matches.append(lowes_match[i])
       
        #Read calibration
        frame_calib = read_frame_calib(calib_dir + '/' + sample_name + '.txt')
        stereo_calib = get_stereo_calibration(frame_calib.p2, frame_calib.p3) 

        # Find disparity and depth
        pixel_u_list = [] # x pixel on left image
        pixel_v_list = [] # y pixel on left image
        disparity_list = []
        depth_list = []
        

        distance_match = [] # Get filtered matches under 80 meters

        for i, match in enumerate(new_matches):          

                     
            # Get the matching keypoints for each of the images         
            img1_idx = kp_left[match.queryIdx].pt # Return index row of the kp1 interest point 
            img2_idx = kp_right[match.trainIdx].pt # Return index row of the kp2 interest point

            #Calculate Depth 
            x_difference = img1_idx[0]-img2_idx[0]
            depth = stereo_calib.f * stereo_calib.baseline / x_difference            

            #Remove any matches that has a depth higher than 80 meters
            if depth < 80:
                #Add to list
                distance_match.append(new_matches[i])

                #Append x and y value from left image
                pixel_u_list.append(img1_idx[0])
                pixel_v_list.append(img1_idx[1])

                #Calculate disparity 
                x_difference = img1_idx[0]-img2_idx[0]
                disparity_list.append(img1_idx[0]-img2_idx[0])
                depth_list.append(stereo_calib.f * stereo_calib.baseline / x_difference) # Calculating depth
                ################## REMINDER APPEND TO GRAPH ####################
        
        # Calculating how many matches were dropped for each algorithm
        original_number = len(matches)
        Lowes_dropout = original_number - len(lowes_match)
        Ransac_dropout = original_number - Lowes_dropout - len(new_matches)
        Depth_dropout = original_number - Lowes_dropout - Ransac_dropout - len(pixel_u_list)
        Final_length = len(pixel_u_list)
        

        print(f'After filtering, we went from {original_number} to {Final_length} matches. Lowes removed {Lowes_dropout} matches, Ransac removed {Ransac_dropout} matches, and the maximum depth removed {Depth_dropout}')
        
        # cv.drawMatchesKnn expects list of lists as matches.
        img = cv.drawMatches(img_left,kp_left,img_right,kp_right,distance_match,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img),plt.show()

        # Output
        for u, v, disp, depth in zip(pixel_u_list, pixel_v_list, disparity_list, depth_list):
            line = "{} {:.2f} {:.2f} {:.2f} {:.2f}".format(sample_name, u, v, disp, depth)
            output_file.write(line + '\n') 


def eval_result(question_num):
    '''
    This function is used to calculate the performance of question 1.2 and 1.3.
    The result is going to be calculated based on disparity based on keypoint matches
    
    Input:
    Question number: Integer (2 or 3)
    Training: Boolean (Decide which sample list to look at)
    '''
    sample_list = ['000001', '000002', '000003', '000004','000005', '000006', '000007', '000008', '000009', '000010'] # Sample list training
    dir = os.path.abspath('./training/gt_depth_map') 
    RMSE_List = [] # Contain RMSE of each Sample
    
    for sample in sample_list:

        if question_num == 2:
            file = open("P2_result_training.txt", 'r')

        else:
            file = open("P3_result_training.txt", 'r')
        img_path = dir +'/' + sample + '.png' 
        
        gt_img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
                
        #Initialize values
        depth_diff = [] # Store depth difference between estimated and true
        zero_depth_count = 0
        total_sample_count = 0
              
        Lines = file.readlines() # Return all lines in the file, as a list where each line is an item in the list object:
        for line in Lines:
            line_strip = line.split() # Split a string into a list 
            sample_name = line_strip[0] # Get the first item from the line (sample)            
            if sample == sample_name:
                x_val, y_val, est_depth = float(line_strip[1]),float(line_strip[2]),float(line_strip[3])
                depth = gt_img[round(y_val),round(x_val)] # Get pixel value by inputing row and column value of picture 
                total_sample_count += 1
                if depth == 0:
                    zero_depth_count += 1
                difference = abs(est_depth-depth)
                depth_diff.append(difference) # calculates the difference between estimated depth and real depth and append to list
                
        #calculate RMSE
        RMSE = ((sum(depth_diff))**2/(len(depth_diff)))**0.5
        RMSE_List.append(RMSE)
    
    Average_RMSE = sum(RMSE_List)/len(RMSE_List)
    print(f"The average RMSE for question {question_num} is {Average_RMSE}")
        
# Call function default is calling question 1,2,3 for testing
question1_1(training=False) # Run question 1.1 set training to True if training set is desired
question1_2(training=False) # Run question 1.2
question1_3(training=False)
eval_result(3) # Input represent question to evaluate (1.2 or 1.3)