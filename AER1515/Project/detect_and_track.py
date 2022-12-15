import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import operator
import math
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect(save_img=False,Training=True
):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    distance_threshold = 25
    result_dir = 'Results/'
    ground_truth_dir = 'Videos/gt_PETS09-S2L1-raw.txt'
    gt_file = open(ground_truth_dir, 'r')
    gt_id_match = None
    Lines_gt = gt_file.readlines()    
    count_gt = 0
    result_dir = open(result_dir+ "test" + str(distance_threshold) + ".txt", "a") 
    result_dir.truncate(0) 
    number_lines = len(Lines_gt)
    

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    z = 0
    global_people_id = {} # Store location and id of object
    global_people_id_old = {}
    counter = 0
    global_count_gt = 0
    ID_count = 0 
    

    #comparisson_list = [{}]
   
    region = None
    for path, img, im0s, vid_cap in dataset:
        ### REMOVE AFTER TO CUT VIDEO SHORT 
        # if z == 500:
        #     break
        
        z += 1
        ### REMOVE 
       
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image            
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Draw vertical line
                #width, height = im0.shape[:2]

                # Draw 4 lines


                height = im0.shape[0]
                width = im0.shape[1]         
                        
                start_point = ((int(width/2)),0)
                end_point = ((int(width/2)),height)
                color = (0,255,0)
                thickness = 2
                ## Implement vertical line unccoment next time 
                #cv2.line(im0, start_point, end_point, color, thickness)

                # Created our desired region of interst for our vide 
                # !!! Red dot in Plots file !!!

                #  Create region 1
                p1 = (425,600)
                p2 = (225,950)
                p3 = (1400,600)
                p4 = (1500,950)
                point_list = [p1,p2,p3,p4]               

                # Create region 2
                color_2 = (255,0,0)
                p1_2 = (425,600)
                p2_2 = (1400,600)
                p3_2 = (1350,400)
                p4_2 = (525,400)
                point_list_2 = [p1_2,p2_2,p3_2,p4_2]
                
                
                if Training != True:
                    cv2.line(im0, p1,p2, color, thickness)
                    cv2.line(im0, p1,p3, color, thickness)
                    cv2.line(im0, p3, p4, color, thickness)
                    cv2.line(im0, p4,p2 , color, thickness)
                    cv2.line(im0, p1_2,p2_2, color_2, thickness)
                    cv2.line(im0, p2_2,p3_2, color_2, thickness)
                    cv2.line(im0, p3_2, p4_2, color_2, thickness)
                    cv2.line(im0, p4_2,p1_2 , color_2, thickness)
                

                # Print results    
                for c in det[:, -1].unique():                  
                    n = (det[:, -1] == c).sum()  # detections per class                    
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                people_tracker = {}
                min_distance = float('inf')
                max_distance = 0
                person_id = 0
                found_new_value = False
                for *xyxy, conf, cls in reversed(det):                    
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'                        
                        cx, cy, inbound, region, x_bot, y_top, width, height =  plot_one_box(xyxy, im0, point_list,point_list_2, label=label, color=colors[int(cls)], line_thickness=1) 

                        if Training == True:
                            bb_area = float(width) * float(height)
                            people_tracker[('person' + str(person_id))] = [(cx,cy),region,x_bot, y_top, width, height, bb_area,gt_id_match]                            
                            person_id += 1
                        elif inbound == True:
                            bb_area = float(width) * float(height)
                            people_tracker[('person' + str(person_id))] = [(cx,cy),region,x_bot, y_top, width, height, bb_area,gt_id_match]                            
                            person_id += 1
                            
                            
                
                if len(global_people_id) == 0:       
                    global_people_id = people_tracker   
                    ID_count += len(global_people_id)

                elif len(people_tracker) < len(global_people_id):
                    k2_match = [] #Track people we considered closest points
                    for k,v in people_tracker.items():                         
                        for k2,v2 in global_people_id.items():
                            if k2 in k2_match:
                                continue                       
                            # Calculate the distance between each tracked person and find closest match   
                            distance_x = abs((v[0][0] - v2[0][0])**2)
                            distance_y = abs((v[0][1]-v2[0][1])**2)
                            distance = math.sqrt(distance_x+distance_y)                  
                
                            if distance < min_distance:
                                min_distance = distance 
                                min_distance_xy = v2
                                smallest_k2 = k2
                            if distance > max_distance:
                                max_distance = distance
                                max_distance_xy = v2
                                biggest_k2 = k2                       
                
                        k2_match.append(smallest_k2)                     
                        global_people_id[smallest_k2] = min_distance_xy                    
                        min_distance = float('inf') # Reset minimum distance  

                    k_remove = []
                    for k,v in global_people_id.items():
                        if k not in k2_match:
                            k_remove.append(k)

                    for item in k_remove:
                        global_people_id.pop(item)                   
                                  
                else:


                    k2_match = [] #Track people we considered closest points
                    for k,v in global_people_id.items():                         
                        for k2,v2 in people_tracker.items():
                            if k2 in k2_match:
                                continue                       
                            # Calculate the distance between each tracked person and find closest match   
                            distance_x = abs((v[0][0] - v2[0][0])**2)
                            distance_y = abs((v[0][1]-v2[0][1])**2)
                            distance = math.sqrt(distance_x+distance_y)                  
                        
                            if distance < min_distance:
                                min_distance = distance 
                                min_distance_xy = v2
                                smallest_k2 = k2
                            if distance > max_distance:
                                max_distance = distance
                                max_distance_xy = v2
                                biggest_k2 = k2

                        
                                
                        k2_match.append(smallest_k2)                     
                        global_people_id[k] = min_distance_xy                    
                        min_distance = float('inf') # Reset minimum distance                  
                                
                    if len(people_tracker) > len(global_people_id):
                        for k,v in people_tracker.items():
                            if k not in k2_match:                                                                                
                                new_person_distance = v                            
                                found_new_value = True
                                
                    if found_new_value == True:
                        found_new_value = False
                        ID_count += 1
                        person_name = str('person'+str(ID_count))    
                        global_people_id[person_name] = new_person_distance      

            if len(global_people_id) != 0 and len(global_people_id_old) != 0 and z > 1 and len(global_people_id) == len(global_people_id_old):   
                for k,v in global_people_id.items():     
                    distance_x = abs(global_people_id[k][0][0] - global_people_id_old[k][0][0])
                    distance_y = abs(global_people_id[k][0][1] - global_people_id_old[k][0][1])
                    total_distance = distance_x + distance_y

                    if global_people_id[k][1] != global_people_id_old[k][1]:   
    
                        if global_people_id_old[k][1] == 'R2' and global_people_id[k][1] == "R1" and total_distance < distance_threshold:
                            counter += 1
                        else:
                            counter -= 1
                    
            cv2.putText(im0, str(counter), (100,200), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 255), 10)

            # Print time (inference + NMS)
            #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

        
        global_people_id_old = global_people_id.copy()

       
        # Compare information to groundtruth 
        max_IoU = 0 # Initalize minimum IoU
        add_line = global_count_gt
       

        for k,v in global_people_id.items():
            
            v[3] = float(v[3]/1.33)
            # Define Bottom Left and Top right of Bounding box 
            
            A = ((v[2]),(v[3] - v[5])) # (X,Y coordinates bottom left)  Bounding box --> (X, Ytop - Height) 
            B = ((v[2] + v[4]),(v[3])) # (X,Y coordinates top right) Bounding Box --> (X_left + width, Y_top)
            count_gt = 0

            if Training == True:
            # frame id xbottom ytop width height
                #if gt_id_match == None:
                    
            
                # Reads info for groundtruth            
                diffent_frame = True
                
                for i in range(number_lines):
                      
                    line = Lines_gt[(i)]
                    if line.split(',')[0] == str(z):
                        
                        # Define bottom left and top right corners of GT                    
                        # Ground Truth
                        global_count_gt += 1
                        C = ((float(line.split(',')[2])),float(line.split(',')[3]) - float(line.split(',')[5])) # (X,Y coordinates bottom left) --> (X, Ytop - Height) 
                        D = (float(line.split(',')[2]) + float(line.split(',')[4]), float(line.split(',')[3])) # (X,Y coordinates top right) Bounding Box --> (X_left + width, Y_top)

                        # Calculate Area of Intersection 
                        x, y = 0, 1
                        width = min(B[0], D[0]) - max(A[0], C[0])
                        height = min(B[1], D[1]) - max(A[y], C[y])
                    
                        if min(width, height) > 0:
                            AUI = width*height
                        else:
                            AUI = 0
                        
                        # Calculate AOU
                        area1 = (B[x]-A[x]) * (B[y]-A[y])
                        area2 = (D[x]-C[x]) * (D[y]-C[y])
                        intersect = AUI
                        Overlap = area1 + area2 - intersect

                        # Calculate IoU
                        if intersect != 0:
                            IoU = Overlap/intersect
                        else:
                            IoU = 0
                        # Update minimum IoU
                        if IoU > max_IoU:                                
                            max_IoU = IoU                            
                            Id_match = line.split(',')[1] # Assign ID  
                            
                     
                          
                        count_gt += 1
                        
                    else:
                        diffent_frame = False
                        Inner_loop = False
                        max_IoU = 0 
                                        
                        
            global_count_gt = global_count_gt + global_count_gt/len(global_people_id)        
            result_dir.write(f"{z} {Id_match} {k} {v[2]} {v[3]} {v[4]} {v[5]}\n")

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
