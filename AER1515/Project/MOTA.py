# Python file to calculate MOTA metrics
def MOTA(distance_threshold):
    '''
    Function to calculate MOTA score 
    input -> file name string
    output Mota score   
    '''
    
    # Data we collected & groundtruth
    result_dir = 'Results/'
    ground_truth_dir = 'Videos/gt_PETS09-S2L1-raw.txt'
    gt_file = open(ground_truth_dir, 'r')
    Lines_gt = gt_file.readlines()    
    rst_file = open(result_dir+ "test" + str(distance_threshold) + ".txt", "r") 
    Lines_rst = rst_file.readlines() 
    number_lines = len(Lines_gt)
    match_gt = True
    count_gt = 0 
    frame_gt = 1
    FN_id_match = False
    
    rows_gt = len(Lines_gt) # Number of rows in ground truth
    rows_rst = len(Lines_rst) # Number of rows in results
     
    lines_rst = Lines_rst[(0)]
    id_rst = lines_rst.split(' ')[1]   
  

    def FN():
        '''
        Function used to calculate number of False Negative between results and Ground Truth        
        '''
        FN_count = 0
        # Count False Negative
        for j in range(rows_gt):

            line_gt = Lines_gt[(j)] # Groudn truth line
            id_gt = line_gt.split(',')[1] # Ground truth ID
            frame_gt = line_gt.split(',')[0] # Ground truth Frame
            
            for i in range(rows_rst):
                lines_rst = Lines_rst[(i)] # Result Line
                id_rst = lines_rst.split(' ')[1] # ID frame
                frame_rst = lines_rst.split(' ')[0] # Result frame
                
                # Verify if frames are the same and id (looks for matches FN)
                # If no match found FN will stay false and count will be incremented by 1
                if frame_rst == frame_gt and id_gt == id_rst:
                    FN_id_match = True
            if FN_id_match == False:
                FN_count += 1
        
        print(FN_count)
    

    def FP():
        '''
        Function used to calculate nubmer of False Positive between Results and groundtruth
        '''
        index_match_rst = [] # Store index not matched in results (FP)
        index_match_gt = [] # Store index already matched in ground truth
        for i in range(rows_rst):

            Index_Found_match = False # Set that no matches has been found in result
            lines_rst = Lines_rst[(i)] # Result Line
            id_rst = lines_rst.split(' ')[1] # ID frame
            frame_rst = lines_rst.split(' ')[0] # Result frame
            
            for j in range(rows_gt):                
                line_gt = Lines_gt[(j)] # Ground truth line
                id_gt = line_gt.split(',')[1] # Ground truth ID
                frame_gt = line_gt.split(',')[0] # Ground truth Frame

                # Match when frame and index is the same and it wasnt already matched in the ground truth
                if (frame_rst == frame_gt) and (id_rst == id_gt) and (j not in index_match_gt):                    
                    
                    index_match_gt.append(j)
                    Index_Found_match = True
                
            # Add index of row not found in resukts
            if Index_Found_match == False:
                index_match_rst.append(i)
            
        # Number of FP        
        Count_FP = len(index_match_rst)
        return Count_FP

    #FN()
    num_fp = FP()
    print(num_fp)


MOTA('25')