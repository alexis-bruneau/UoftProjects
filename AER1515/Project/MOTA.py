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
    FN_count = 0
    rows_gt = len(Lines_gt)
    rows_rst = len(Lines_rst)
     
    lines_rst = Lines_rst[(0)]
    id_rst = lines_rst.split(' ')[1]   
  

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
    
  

        

MOTA('25')