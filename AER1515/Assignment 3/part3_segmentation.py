import os
import sys

import cv2
import numpy as np
import kitti_dataHandler


def main(cropping_val,distance_threshold,Train = True):
    
    
    ################
    # Options
    ################
    # Input dir and output dir
    if Train == True:
        depth_dir = 'data/train/gt_depth'
        label_dir = 'data/train/gt_labels'
        output_dir = 'data/train/est_segmentation'
        estimated_depth_dir = 'data/train/estimated_depth'
        gt_segmentation_dir = 'data/train/gt_segmentation'
        boxes_dir = 'data/train/estimated_box'
        sample_list = ['000001', '000002', '000003', '000004', '000005','000006','000007','000008','000009','000010']

    else:
        depth_dir = 'data/test/gt_depth'
        label_dir = 'data/test/gt_labels'
        output_dir = 'data/test/est_segmentation'        
        sample_list = ['000011', '000012', '000013', '000014', '000015']
    ################

    for sample_name in sample_list:
    	# Read depth map
        # Discard depths less than 10cm from the camera ( already discarded in part 1)
        depth_map = cv2.imread(f"{estimated_depth_dir}/{sample_name}.png", cv2.IMREAD_GRAYSCALE)
       
        # Read 2d bbox
        file = open(f"{boxes_dir}/{sample_name}.txt",'r')
        Lines = file.readlines()

        # For each bbox
        segmentation_mask = np.zeros_like(depth_map)
        segmentation_mask -= 1 # Set all pixels to 255
        for box in Lines:         
            # Reminder  
            # Get the x,y,w,h values of each box          
           
            x = float(box.split()[0]) # x_left
            y = float(box.split()[1]) # y_bottom
            w = float(box.split()[2]) # Width of box
            h = float(box.split()[3]) # height of box
            
        
            # Get min / max pixel value of depth_map (our search area cannot go past that)
            y_min = 0
            x_min = 0 
            y_max = np.shape(depth_map)[0]
            x_max = np.shape(depth_map)[1]
           
            x_min_search = x
            x_max_search = int(x+w)
            y_min_search = y
            y_max_search = y + h
           

            #cropping_val = 0.05
            # Make sure that bounding box does not go out of the image
            x_min = np.clip(x, 0, depth_map.shape[1] - 1)
            x_max = np.clip(x+w, 0, depth_map.shape[1] - 1)
            y_min = np.clip(y, 0, depth_map.shape[0] - 1)
            y_max = np.clip(y+h, 0, depth_map.shape[0] - 1)

            if x_min_search < x_min:
                 x_min_search = x_min
            elif x_max_search > x_max:
                 x_max_search = x_max
            
            if y_min_search < y_min:
                y_min_search = y_min
            elif y_max_search > y_max:
                y_max_search = y_max

            # make search range tighter for imprecise values near the edges.
            x_min_search = int(x_min + cropping_val*(x_max - x_min))
            x_max_search = int(x_max - cropping_val*(x_max - x_min))
            y_min_search = int(y_min + cropping_val*(y_max - y_min))
            y_max_search = int(y_max - cropping_val*(y_max - y_min))
        
            # Estimate the average depth of the objects
            depth_map_search = depth_map[ y_min_search:y_max_search, x_min_search:x_max_search]
            average_depth = np.sum(depth_map_search)/np.count_nonzero(depth_map_search) # Don't cont zero values
            
            # Find the pixels within a certain distance from the centroid
            for i in range(x_min_search,x_max_search):
                for j in range(y_min_search,y_max_search):
                    if abs(depth_map[j][i]-average_depth) < distance_threshold:
                        # If the distance is smaller  than the computed average, replace segmentation mask value to 0
                        segmentation_mask[j][i] = 0  

        # Save the segmentation mask
        cv2.imwrite(f"{output_dir}/{sample_name}.png",segmentation_mask)
       



def map_precision_recall(cropping_val,distance_threshold):
    
    gt_seg_dir = 'data/train/gt_segmentation'
    est_segmentation_dir = 'data/train/est_segmentation'

    sample_list = ['000001','000002', '000003', '000004', '000005','000006','000007','000008','000009','000010']
    

    
    # Initialse precision and recall
    average_precision = 0
    average_recall = 0 
    print(f"Testing crop value of {cropping_val} and distance threshold of {distance_threshold}")

    
    precision_under_75 = 0
    recall_under_75 = 0 
    for sample in sample_list:
        #Get segmentation map
        est_segmentation_map = cv2.imread(f"{est_segmentation_dir}/{sample}.png",cv2.IMREAD_GRAYSCALE)
        gt_segmentation_map = cv2.imread(f"{gt_seg_dir}/{sample}.png",cv2.IMREAD_GRAYSCALE)

        # Count the following to get precision
        # TP = True Positive
        # FP = False Positive 
        # FN = False Negative
        TP = 0
        FP = 0
        FN = 0
        
        for i in range(gt_segmentation_map.shape[1]):
            for j in range(gt_segmentation_map.shape[0]):
                # Get true positive
                if gt_segmentation_map[j][i] < 255 and est_segmentation_map[j][i] < 255:
                    TP += 1
                elif gt_segmentation_map[j][i] == 255 and est_segmentation_map[j][i] < 255:
                    FP += 1
                elif gt_segmentation_map[j][i] < 255 and est_segmentation_map[j][i] == 255:
                    FN += 1
        
        # Calculate Precision & Recall
        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)
        average_precision += Precision
        average_recall += Recall

        # Count if sample has precision or recall under 75%
        if Precision < 0.75:            
            precision_under_75 +=1 
        if Recall < 0.75:
            recall_under_75 += 1 
            
    
    # Calculate average precision & recall throughout all samples
    average_precision = average_precision/len(sample_list)
    average_recall = average_recall/len(sample_list)
    print(f"The results for a cropping value of {cropping_val} and a distance threshold of {distance_threshold} is:")
    print(f"There was {precision_under_75} boxes that had under 75% precision and {recall_under_75} boxes under 75% for recall")
    print(f"The average precision is {average_precision} and the average recall is {average_recall}")


def testing(cropping_val,distance_threshold):
    '''
    Function used to output testing segmentation results
    '''
    depth_dir = 'data/test/gt_depth'
    label_dir = 'data/test/gt_labels'
    disparity_map = 'data/test/disparity'
    output_dir = 'data/test/est_segmentation'        
    sample_list = ['000011', '000012', '000013', '000014', '000015']
    boxes_dir = 'data/test/estimated_box'
    

    
    for sample_name in sample_list:
        # Read depth map
        # Discard depths less than 10cm from the camera ( already discarded in part 1)
        depth_map = cv2.imread(f"{disparity_map}/{sample_name}.png", cv2.IMREAD_GRAYSCALE)
        
        # Read 2d bbox
        file = open(f"{boxes_dir}/{sample_name}.txt",'r')
        Lines = file.readlines()

        # For each bbox
        segmentation_mask = np.zeros_like(depth_map)
        segmentation_mask -= 1 # Set all pixels to 255
        for box in Lines:         
            # Reminder  
            # Get the x,y,w,h values of each box          
            
            x = float(box.split()[0]) # x_left
            y = float(box.split()[1]) # y_bottom
            w = float(box.split()[2]) # Width of box
            h = float(box.split()[3]) # height of box
            
        
            # Get min / max pixel value of depth_map (our search area cannot go past that)
            y_min = 0
            x_min = 0 
            y_max = np.shape(depth_map)[0]
            x_max = np.shape(depth_map)[1]
            
            x_min_search = x
            x_max_search = int(x+w)
            y_min_search = y
            y_max_search = y + h
            

            #cropping_val = 0.05
            # clip because some bb are out of image bounds
            x_min = np.clip(x, 0, depth_map.shape[1] - 1)
            x_max = np.clip(x+w, 0, depth_map.shape[1] - 1)
            y_min = np.clip(y, 0, depth_map.shape[0] - 1)
            y_max = np.clip(y+h, 0, depth_map.shape[0] - 1)

            if x_min_search < x_min:
                    x_min_search = x_min
            elif x_max_search > x_max:
                    x_max_search = x_max
            
            if y_min_search < y_min:
                y_min_search = y_min
            elif y_max_search > y_max:
                y_max_search = y_max

                # Extract variables            

            # make search range tighter for imprecise values near the edges.
            x_min_search = int(x_min + cropping_val*(x_max - x_min))
            x_max_search = int(x_max - cropping_val*(x_max - x_min))
            y_min_search = int(y_min + cropping_val*(y_max - y_min))
            y_max_search = int(y_max - cropping_val*(y_max - y_min))
        
            # Estimate the average depth of the objects
            depth_map_search = depth_map[ y_min_search:y_max_search, x_min_search:x_max_search]
            print(depth_map_search)
            average_depth = np.sum(depth_map_search)/np.count_nonzero(depth_map_search) # Don't cont zero values
            
            # Find the pixels within a certain distance from the centroid
            for i in range(x_min_search,x_max_search):
                for j in range(y_min_search,y_max_search):
                    if abs(depth_map[j][i]-average_depth) < distance_threshold:
                        # If the distance is smaller  than the computed average, replace segmentation mask value to 0
                        segmentation_mask[j][i] = 0  

        # Save the segmentation mask
        cv2.imwrite(f"{output_dir}/{sample_name}.png",segmentation_mask)
            



def map_precision_recall(cropping_val,distance_threshold):
    
    gt_seg_dir = 'data/train/gt_segmentation'
    est_segmentation_dir = 'data/train/est_segmentation'

    sample_list = ['000001', '000002', '000003', '000004', '000005','000006','000007','000008','000009','000010']
    

    
    # Initialse precision and recall
    average_precision = 0
    average_recall = 0 
    print(f"Testing crop value of {cropping_val} and distance threshold of {distance_threshold}")

    
    precision_under_75 = 0
    recall_under_75 = 0 
    for sample in sample_list:
        #Get segmentation map
        est_segmentation_map = cv2.imread(f"{est_segmentation_dir}/{sample}.png",cv2.IMREAD_GRAYSCALE)
        gt_segmentation_map = cv2.imread(f"{gt_seg_dir}/{sample}.png",cv2.IMREAD_GRAYSCALE)

        #cv2.imshow(f"{sample} estimate",est_segmentation_map)
        #cv2.imshow(f'{sample} gt',gt_segmentation_map)
        #cv2.waitKey(0)
        # Count the following to get precision
        # TP = True Positive
        # FP = False Positive 
        # FN = False Negative
        TP = 0
        FP = 0
        FN = 0
        
        for i in range(gt_segmentation_map.shape[1]):
            for j in range(gt_segmentation_map.shape[0]):
                # Get true positive
                if gt_segmentation_map[j][i] < 255 and est_segmentation_map[j][i] < 255:
                    TP += 1
                elif gt_segmentation_map[j][i] == 255 and est_segmentation_map[j][i] < 255:
                    FP += 1
                elif gt_segmentation_map[j][i] < 255 and est_segmentation_map[j][i] == 255:
                    FN += 1
        
        # Calculate Precision
        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)
        average_precision += Precision
        average_recall += Recall

        if Precision < 0.75:            
            precision_under_75 +=1 
        if Recall < 0.75:
            recall_under_75 += 1 
            
    
    
    average_precision = average_precision/len(sample_list)
    average_recall = average_recall/len(sample_list)
    print(f"The results for a cropping value of {cropping_val} and a distance threshold of {distance_threshold} is:")
    print(f"There was {precision_under_75} boxes that had under 75% precision and {recall_under_75} boxes under 75% for recall")
    print(f"The average precision is {average_precision} and the average recall is {average_recall}")
 
if __name__ == '__main__':

    
    # Step 1 First step Finding the best cropping value and distance threshold (uncomment if you want to test)

    # cropping_val_list = [0.05,0.1,0.15]
    # distance_threshold_list = [1,2.5,5,7.5,10]

    # for cropping_val in cropping_val_list:
    #     for distance_threshold in distance_threshold_list:
    #         main(cropping_val,distance_threshold,Train = True)
    #         map_precision_recall(cropping_val,distance_threshold)

    # Step 2 Store best cropping val found
    cropping_val = 0.05
    distance_threshold = 10

    # Step 3 Get Training Segmentation images with best values
    main(cropping_val,distance_threshold,Train = True)

    # Step 4 Find Segmentation for Test Images
    testing(cropping_val,distance_threshold)

    
            
