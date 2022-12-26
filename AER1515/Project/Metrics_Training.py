# Python file to calculate MOTA metrics
result_dir = 'Results/'
metric_result_file = open(result_dir+'MetricScore.txt',"a")

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
            FN_id_match = False
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

        print(f"The False Negative is {FN_count}")
        return FN_count
        
    

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
        print(f"The False positive count is {Count_FP}")
        return Count_FP

    def IDS(count_FP=0):
        '''
        function that calculates the number of ID switch  
        '''
        IDS_count = 0 
        IDS_Switch = False
        ID_match_list = [] # Store tuple of matches
        gt_in_list = False
        for i in range(100):           
            lines_rst = Lines_rst[(i)] # Result Line
            Frame_rst = lines_rst.split(' ')[0] # Frame Result
            Id_gt = lines_rst.split(' ')[1] # GT ID Match
            Id_rst = lines_rst.split(' ')[2] # Result original ID
            match_combination = [Id_gt,Id_rst]

            if Frame_rst == str(1):                             
                ID_match_list.append(match_combination)

            
            else:                
                for match_index,match in enumerate(ID_match_list):                    
                    
                    if Id_gt == match[0] and Id_rst != match[1]:                        
                        IDS_count += 1                             
                        ID_match_list[match_index][1] = Id_rst                 
                       
                    
                    # Add to list if gt id is not in list
                    if Id_gt == match[0]:
                        gt_in_list = True

                if gt_in_list == False:
                    ID_match_list.append(match_combination)

                gt_in_list = False  

        print(f"There is {IDS_count} IDS switch")
        return IDS_count
      


    
    
    FN_count = FN()
    Count_FP = FP()
    IDS_count = IDS()

    MOTA = round((1 - (FN_count + Count_FP + IDS_count)/number_lines),3)
       
   
    return MOTA, FN_count, Count_FP, IDS_count

 
    
        
#MOTA('50')


def MOTP(distance_threshold):
    '''
    Function to calculate MOTP score
    input -> file name string
    output MOTP score
    '''

    # Data we collected & groundtruth
    result_dir = 'Results/'
    ground_truth_dir = 'Videos/gt_PETS09-S2L1-raw.txt'
    # read txt files
    gt_file = open(ground_truth_dir, 'r')
    Lines_gt = gt_file.readlines()
    rst_file = open(result_dir + "test" + str(distance_threshold) + ".txt", "r")
    Lines_rst = rst_file.readlines()
    # the number of txt line
    rows_gt = len(Lines_gt)  # Number of rows in ground truth
    rows_rst = len(Lines_rst)  # Number of rows in results

    lines_rst = Lines_rst[(0)] # frame
    id_rst = lines_rst.split(' ')[1] # ID

    total_match = 0
    total_distance = 0
    
    # Count False Negative
    for j in range(rows_gt):
        match_found = False
        line_gt = Lines_gt[(j)]  # Groudn truth line
        id_gt = line_gt.split(',')[1]  # Ground truth ID
        frame_gt = line_gt.split(',')[0]  # Ground truth Frame
        x_bottom_gt =float(line_gt.split(',')[2])# ground truth x bottom
        y_top_gt = float(line_gt.split(',')[3]) # ground truth y top
        width_gt = float(line_gt.split(',')[4]) # ground truth width
        height_gt = float(line_gt.split(',')[5]) # ground truth height

        for i in range(rows_rst):
            lines_rst = Lines_rst[(i)]  # Result Line
            id_rst = lines_rst.split(' ')[1]  # ID frame
            frame_rst = lines_rst.split(' ')[0]  # Result frame
            x_bottom_rst = float(lines_rst.split(' ')[3]) # result x bottom
            y_top_rst = float(lines_rst.split(' ')[4]) # result y top
            width_rst = float(lines_rst.split(' ')[5]) # result width
            height_rst = float(lines_rst.split(' ')[6]) # result height

            # Verify if frames are the same and id
            if frame_rst == frame_gt and id_gt == id_rst:
                A = (x_bottom_gt, (y_top_gt - height_gt))  # (X,Y coordinates bottom left)  Bounding box --> (X, Ytop - Height)
                B = ((x_bottom_gt + width_gt), y_top_gt)  # (X,Y coordinates top right) Bounding Box --> (X_left + width, Y_top)
                C = (x_bottom_rst, (y_top_rst - height_rst))  # (X,Y coordinates bottom left)  Bounding box --> (X, Ytop - Height)
                D = ((x_bottom_rst + width_rst), y_top_rst)  # (X,Y coordinates top right) Bounding Box --> (X_left + width, Y_top)
                x, y = 0, 1
                width = min(B[0], D[0]) - max(A[0], C[0])
                height = min(B[1], D[1]) - max(A[y], C[y])

                if min(width, height) > 0:
                    AUI = width * height
                else:
                    AUI = 0

                # Calculate AOU
                area1 = (B[x] - A[x]) * (B[y] - A[y])
                area2 = (D[x] - C[x]) * (D[y] - C[y])
                intersect = AUI
                Overlap = area1 + area2 - intersect

                # Calculate IoU
                if intersect != 0:
                    IoU = intersect/Overlap
                else:
                    IoU = 0

                distance = 1 - IoU
                total_match +=1
                match_found = True

        if match_found == True:        
            total_distance += distance
        

    
        

    MOTP = total_distance/total_match
    return round(MOTP,3) 

    

MOTA, FN_count, Count_FP, IDS_count = MOTA(50)
MOTP =  MOTP(50)

# Append to txt file
metric_result_file.write(f"\nHyperparameter (250, 0.35, 0.55): Mota Score = {MOTA}, MOTP Score {MOTP}, FN = {FN_count}, FP - {Count_FP}, ISD = {IDS_count}")