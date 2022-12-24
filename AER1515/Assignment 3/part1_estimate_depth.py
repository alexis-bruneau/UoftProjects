import os
import sys

import cv2
import numpy as np
import kitti_dataHandler
from Calibration_Assignment_2 import * # Use work done in Assignment 2 for claibration

def main():

    ################
    # Options
    ################
    # Input dir and output dir
    disp_dir = 'data/train/disparity'
    output_dir = 'data/train/estimated_depth'
    calib_dir = 'data/train/calib'
    sample_list = ['000001', '000002', '000003', '000004', '000005','000006','000007','000008','000009','000010']
    ################

    for sample_name in (sample_list):
        # Read disparity map
        print(sample_name)
        disparity_map = cv2.imread(f"{disp_dir}/{sample_name}.png", cv2.IMREAD_GRAYSCALE)
        
        # Read calibration info
        frame_calib = read_frame_calib(calib_dir + '/' + sample_name + '.txt')
        stereo_calib = get_stereo_calibration(frame_calib.p2, frame_calib.p3)

        # Calculate depth (z = f*B/disp)
        depth_map = np.zeros_like(disparity_map)

        #loop through each values in depth map
        for i in range(disparity_map.shape[0]):
            for j in range(disparity_map.shape[1]):
                # Assign depth of 0 if disparity map is 0
                if disparity_map[i][j] == 0:
                    depth = 0
                else:
                    # Recall depth (z = f*B/disp)
                    depth = stereo_calib.f*stereo_calib.baseline / disparity_map[i][j]

                    # Discard pixels past 80m
                    # Assign depth of 0 if pixel is greater than 80 m or less than 10 cm
                    if depth > 80 or depth < 0.1:
                        depth = 0                
                depth_map[i][j] = depth       

        # Save depth map
        cv.imwrite(f"{output_dir}/{sample_name}.png", depth_map)


if __name__ == '__main__':
    main()
