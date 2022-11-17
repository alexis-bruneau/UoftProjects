from turtle import distance
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings("ignore") # Added at the end to Remove warnings for cmap

def load_point_cloud(path):
    # Load the point cloud data (do NOT change this function!)
    data = pd.read_csv(path, header=None)
    point_cloud = data.to_numpy()
    return point_cloud


def nearest_search(pcd_source, pcd_target):
    # TODO: Implement the nearest neighbour search
    # TODO: Compute the mean nearest euclidean distance between the source and target point cloud
    corr_target = []
    corr_source = []
    ec_dist_mean = 0
       
    rows = len(pcd_source) # Number of rows in source
    rows_target = len(pcd_target) # Number of rows in target
    # Brute Force approach to find matching p and q values / assuming no matches overlap

    for i in range(rows):
        min_value = float('inf') # Initialize minium value
        # Compare each number to find smallest distance
        for j in range(rows_target):

            x_diff_squared = (pcd_source[i][0]- pcd_target[j][0])**2 # Compare x values between source and target and square it
            y_diff_squared = (pcd_source[i][1]- pcd_target[j][1])**2 # Compare y values between source and target and square it
            z_diff_squared = (pcd_source[i][2]- pcd_target[j][2])**2 # Compare z values between source and target and square it

            euclid_dist = (x_diff_squared+y_diff_squared+z_diff_squared)**0.5 # Calculate euclidean distance

            #Update smallest distance
            if euclid_dist < min_value:
                min_value = euclid_dist
                target_index = j # save target index that gives smallest distance

        # append matching source and target
        corr_source.append(pcd_source[i])
        corr_target.append(pcd_target[target_index])

    # Calculating euclidean mean with matching points
    total_eucl_dist = 0 # Initialize total euclidean distance
    for i in range(rows):
        x_diff_squared = (corr_source[i][0] - corr_target[i][0])**2
        y_diff_squared = (corr_source[i][1] - corr_target[i][1])**2
        z_diff_squared = (corr_source[i][2] - corr_target[i][2])**2
        euclid_dist = (x_diff_squared+y_diff_squared+z_diff_squared)**0.5 #Calculate euclidean distance for one match
        total_eucl_dist += euclid_dist # Add euclid distanceto total
    
    ec_dist_mean = total_eucl_dist/rows # Divide total by number of matches 
    
    
    
    #Currently corr_target and corr_source are a list of array. Convert to matrix form
    corr_source = np.stack(corr_source,axis=0)
    corr_target= np.stack(corr_target, axis=0)
    
    return corr_source, corr_target, ec_dist_mean

def estimate_pose(corr_source, corr_target):
    # TODO: Compute the 6D pose (4x4 transform matrix)
    # TODO: Get the 3D translation (3x1 vector)

    pose = np.identity(4)
    translation_x = 0
    translation_y = 0
    translation_z = 0

    #Find center of masses of both point clouds
    rows = len(corr_source) # Same number of rows for source and target
    
    # Find average x y z for source and target
    centroid_source_average = np.mean(corr_source, axis = 0) # Return average for mean of x,y,z (source)
    centroid_target_average = np.mean(corr_target, axis = 0) # Return average for mean of x,y,z (target)
    centroid_source_average = np.matrix(centroid_source_average).T # Transform x y z average to matrix of size 3 x 1
    centroid_target_average = np.matrix(centroid_target_average).T # Transform x y z average to matrix of size 3 x 1

    #Initialize matrix for SVD
    H = np.zeros((3,3)) # Initialize the matrix 3 x 3 
    for i in range(rows):
        term1=np.matrix(corr_target[i]).T-centroid_target_average
        term2 = np.matrix(corr_source[i]).T-centroid_source_average
        
        
        H += np.matmul(term1, np.transpose(term2)) # This gives a 3 x 3 matrix (3x1 times 1x3)
        
    
    H = H/rows # Divide H by total number of rows
    
      
    # use SVD function to get V, D, and U_T
    V, D, U_T = np.linalg.svd(H) 
    
    #Calcualte Rotation Matrix
    R = np.matmul(V,U_T)
   
    # Get translation x y z    
    t = centroid_target_average-(np.matmul(R,centroid_source_average))

    # Add Rotation and translation to pose
    pose[:3, :3] = R #Put rotation matrix in pose
    pose[:3,3:] = t
    
    #Assign values to translation x,y,z
    translation_x = np.float64(t[0])
    translation_y = np.float64(t[1])
    translation_z = np.float64(t[2]) 
    
    return pose, translation_x, translation_y, translation_z


def icp(pcd_source, pcd_target):
    # TODO: Put all together, implement the ICP algorithm
    # TODO: Use your implemented functions "nearest_search" and "estimate_pose"
    # TODO: Run 30 iterations
    # TODO: Show the plot of mean euclidean distance (from function "nearest_search") for each iteration
    # TODO: Show the plot of pose translation (from function "estimate_pose") for each iteration

    #Initialize values / help for graph
    pose = np.identity(4)    
    euclidean_mean_list = []
    icp_itteration_list = []
    translation_x_list = []
    translation_y_list = []
    translation_z_list = []

    rows = len(pcd_source) # Total number of rows in pcd_source
    total_pose = np.identity(4)

    #We want 30 ICP itterations
    for i in range(30):

        #Call nearest_serach function, use neareast search output as input to estimate pose
        corr_source, corr_target, ec_dist_mean = nearest_search(pcd_source, pcd_target)
        pose, translation_x, translation_y, translation_z = estimate_pose(corr_source, corr_target)

        #Get rotation and translation matrix
        R = pose[:3,:3]
        t = pose[:3,3:]
        
        pcd_source = np.matmul(R,np.transpose(pcd_source)) + np.tile(t, (1, rows))
        pcd_source = np.transpose(pcd_source)

        #Plot Nearest neighbor search for each ICP itterations
        icp_itteration_list.append(i)
        euclidean_mean_list.append(ec_dist_mean)
        
        #Plot for translation x, y, and z
        translation_x_list.append(translation_x)
        translation_y_list.append(translation_y)
        translation_z_list.append(translation_z)

        total_pose = np.matmul(total_pose,pose) # calculate multiplication of all poses

    # Plotting euclidean mean vs icp itteration
    plt.plot(icp_itteration_list,euclidean_mean_list) 
    plt.title("Euclidean Mean vs ICP Iteration")
    plt.xlabel("ICP Iteration")
    plt.ylabel("Euclidean Mean")
    plt.grid()
    plt.show()
    plt.close()

    ## Plotting euclidean mean vs icp itteration
    plt.plot(icp_itteration_list,translation_x_list, label = 'Translation X') 
    plt.plot(icp_itteration_list,translation_y_list, label = 'Translation Y')
    plt.plot(icp_itteration_list,translation_z_list, label = 'Translation Z')  
    plt.title(" 3D Translation (mm) vs ICP Iteration")
    plt.xlabel("ICP Iteration")
    plt.ylabel("3D Translation (mm)")
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()

    pose = total_pose

    return pose


def main():
    # Dataset and ground truth poses
    #########################################################################################
    # Training and test data (3 pairs in total)
    train_file = ['bunny', 'dragon']
    test_file = ['armadillo']

    # Ground truth pose (from training data only, used for validating your implementation)
    GT_poses = []
    gt_pose = [0.8738,-0.1128,-0.4731,24.7571,
            0.1099,0.9934,-0.0339,4.5644,
            0.4738,-0.0224,0.8804,10.8654,
            0.0,0.0,0.0,1.0]
    gt_pose = np.array(gt_pose).reshape([4,4])
    GT_poses.append(gt_pose)
    gt_pose = [0.7095,-0.3180,0.6289,46.3636,
               0.3194,0.9406,0.1153,3.3165,
               -0.6282,0.1191,0.7689,-6.4642,
               0.0,0.0,0.0,1.0]
    gt_pose = np.array(gt_pose).reshape([4,4])
    GT_poses.append(gt_pose)
    #########################################################################################

    # Training (validate your algorithm)
    ##########################################################################################################
    for i in range(2):
        # # Load data
        path_source = './training/' + train_file[i] + '_source.csv'
        path_target = './training/' + train_file[i] + '_target.csv'
        pcd_source = load_point_cloud(path_source)
        pcd_target = load_point_cloud(path_target)
        gt_pose_i = GT_poses[i]

        # Visualize the point clouds before the registration
        ax = plt.axes(projection='3d')
        ax.scatter3D(pcd_source[:,0], pcd_source[:,1], pcd_source[:,2], cmap='Greens')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], cmap='Reds')
        plt.legend(["Source Point Cloud" , "Target Point Cloud"])
        ax.set_title('Point Clouds Before Registration')
        plt.show()

        # TODO: Use your implemented ICP algorithm to get the estimated 6D pose (from source to target point cloud)
        # Call ICP function and return calculated pose
        pose = icp(pcd_source, pcd_target)
               
        # Transform the point cloud
        # TODO: Replace the ground truth pose with your computed pose and transform the source point cloud
        pts = np.vstack([np.transpose(pcd_source), np.ones(len(pcd_source))])
        cloud_registered = np.matmul(pose, pts)  
        cloud_registered = np.transpose(cloud_registered[0:3, :])

        # # TODO: Evaluate the rotation and translation error of your estimated 6D pose with the ground truth pose
        
        #Caluclate relative error for each elements of pose
        matrix_error = np.zeros((4,4)) # Inintialize MAE
        # Loop through matrix element wise and calculate error
        for j in range(4):
            for k in range(4):
                matrix_error[j,k] = abs((pose[j,k]-gt_pose_i[j,k])/gt_pose_i[j,k])*100

        #Printing results
        np.set_printoptions(suppress=True) # Remove scientific notation in matrix display
        print(f"The following are the results for {train_file[i]}:\n")

        print("The ground truth is:\n")
        print(gt_pose_i.round(3))
        
        print("The calculated pose is:")
        print(pose.round(3))

        print(f"The pose error in % for {train_file[i]} is:")
        print(matrix_error.round(3))
        
        # Visualize the point clouds after the registration
        ax = plt.axes(projection='3d')
        ax.scatter3D(cloud_registered[:,0], cloud_registered[:,1], cloud_registered[:,2], cmap='Greens')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], cmap='Reds')
        plt.legend(["Transformed Source Point Cloud", "Target Point Cloud"])
        ax.set_title('Point Clouds After Registration')
        plt.show()
        
    ##########################################################################################################



    # Test
    ####################################################################################
    for i in range(1):
        # Load data
        path_source = './test/' + test_file[i] + '_source.csv'
        path_target = './test/' + test_file[i] + '_target.csv'
        pcd_source = load_point_cloud(path_source)
        pcd_target = load_point_cloud(path_target)

      
        # Visualize the point clouds before the registration
        ax = plt.axes(projection='3d')
        ax.scatter3D(pcd_source[:,0], pcd_source[:,1], pcd_source[:,2], cmap='Greens')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], cmap='Reds')
        plt.legend(["Source Point Cloud" , "Target Point Cloud"])
        ax.set_title('Point Clouds Before Registration')
        plt.show()

        # TODO: Use your implemented ICP algorithm to get the estimated 6D pose (from source to target point cloud)
        pose = icp(pcd_source, pcd_target)
        print(f"\nThe results for the {test_file[i]}:")
        print(f"The 6D pose output for the armadilo is:\n {pose}")
        
        
        # TODO: Show your outputs in the report
        # TODO: 1. Show your estimated 6D pose (4x4 transformation matrix)
        #print('After 30 ICP Iterations, the 6D pose for {test_file[i]} is {pose}.')
        # TODO: 2. Visualize the registered point cloud and the target point cloud

        pts = np.vstack([np.transpose(pcd_source), np.ones(len(pcd_source))])
        cloud_registered = np.matmul(pose, pts)
        cloud_registered = np.transpose(cloud_registered[0:3, :])

        #Plot point cloud after registration
        ax = plt.axes(projection='3d')
        ax.scatter3D(cloud_registered[:,0], cloud_registered[:,1], cloud_registered[:,2], cmap='Greens')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], cmap='Reds')
        plt.legend(["Transformed Source Point Cloud", "Target Point Cloud"])
        ax.set_title('Point Clouds After Registration')
        plt.show()


if __name__ == '__main__':
    main()