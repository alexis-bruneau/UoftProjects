# People Counter Using Multiple Object Tracker

## Results 

The purpose of this project is to detect and track people using YOLOv7 to detect people and implement a tracking algorithm to count people going in and out of a store. MOTA and MOTP was used to tuned hyperparameters of the algorithm. The training videos and their ground truth came from the [MOT Challenge](https://motchallenge.net/). The testing video used was from a cctv video found [here](https://www.youtube.com/watch?v=-IvBKBx0UBo&ab_channel=HDSecurityStore). 

The algorithm is easily modifiable to track different objects and change the region of interest. Our parameters were tuned to obtain the best MOTA and MOTP results. We obtain a MOTA score of 94.3% and a MOTP score of 39.1%. One limitation of our method is that occlusion of the objects reduces the accuracy. We found that the best set up was to have a camera setup and point downward. Our methodology and results can be found in AER1515_report(People Counter Using Multiple Object Tracker).

![Results-People_Counter](https://user-images.githubusercontent.com/106686997/209584702-16f70050-fbaf-42d7-89dd-c0b5d2f12df1.gif)


### Initial Setup \ Dependencies
====================

* Create a virtual environment
* Clone YOLOv7 repository https://github.com/WongKinYiu/yolov7
* Add The following to the yolov7 folder
	* Videos (Folder)
	* Results (Folder)
	* detect_and_track
	* Metrics_Training
	* requirements_gpu
	* requirements_tracking
* Replace the file plots in utils with the one provided

* Install requirements
 * pip install -r requirements.txt
 * pip install -r requirements_gpu.txt (If gpu is available)

### Training and Testing
====================

The videos we used for the training and testing are available in the Videos folder. However, if you want to test you own videos, add the in the videos folder. For testing videos, add the ground truth as a txt file in the same folder and with the same name as the video. 

To Note:
It is note necessary to perform the training step. This step is only used if tunning is desired to obtain the best MOTA and MOTP results.
When testing or training, you can uncomment line 97 and 98 to stop the script when the video reaches a certain frame. 
 
#### Training Steps
====================

* Set Training = True on line 21 of detect_and_track
* The following 3 parameters were tested in our case:
	* distance_threshold (line 31)
	* confindence level (--conf-thresh when running from terminal)
	* Iou-threshold (--iou-thresh when running from terminal)

* Modify the y ratio on line 366 (Depending on the scale of the video, the y value might need to be scalled differently)
* Run the following line:
 -- python detect_and_track.py --weights yolov7.pt --conf-thres  0.25 --iou-thres 0.45 --img-size 640 --source Videos\PETS09-S2L1-raw.mp4 --view-img --no-trace --classes 0
 * Things to consider
	* The weights can be changed to different such as YOLOv7-E6E ...
	* Change the img-size accordingly
	* Switch to the correct Video name
	* The class can be changed to track different objects (ex 0 = person, 2 = car ...). Go in yolov7 -> data -> coco.yaml to see all 80 classes
 
* Go on the Metrics_Training python file to calculate the MOTA and MOTP score.
	* Change the parameter in line 240 & 241. It represents the distance threshold that was tested
	* The results will appear in Results -> MetricScore

#### Testing Steps
====================
* Set Trainig = False on line 21 of detect_and_track
*  Change region 1 and 2 boundary points (line 174 to 186) 
*  Run the following line python detect_and_track.py --weights yolov7.pt --conf-thres  0.5 --iou-thres 0.3 --img-size 1280  --source Videos\Test_Video.mp4 --view-img --no-trace --classes 0
* Things to consider
	* The weights can be changed to different such as YOLOv7-E6E ...
	* Change the img-size accordingly
	* Switch to the correct Video name
	* Switch to the conf-thresh and iou-thresh that gave the best score. If training was skipped, conf-thres and iou-thres can be removed.
	* The class can be changed to track different objects (ex 0 = person, 2 = car ...). Go in yolov7 -> data -> coco.yaml to see all 80 classes