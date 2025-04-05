# ---------------------------------------------------------------------------------------------------------------------
"""
JiffyDetect:  YOLOV8 version
Cycronix
1/29/2024
"""

# ---------------------------------------------------------------------------------------------------------------------

#import argparse
import os
import sys
from pathlib import Path
# import torch
import resource
import psutil

import time
import urllib3
from datetime import datetime
import math
import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------------------------------------------------
# rtsp-camera fetch

Vehicle = [ 'car', 'truck', 'bicycle', 'motorcycle' ]
Animal = [ 'dog', 'cat', 'bird', 'bear' ]
Person = [ 'person' ]
TARGETS = Vehicle + Animal + Person

PutDate = False
PutDetect = False				# put cropped image of detection (e.g. 'vehicle.jpg')

MDET = 0.1                                      # motion detect trigger level (linear 0-1) (was .1)
MTHRESH = 1                                     # motion detect noise reject threshold (per raw pixel) (was 4)

hide_labels = False
hide_conf = True

Weights='models/yolov8s.pt'                        # model.pt path(s)
Model = YOLO(Weights, task='detect')               # Load model
Names = Model.names

# ---------------------------------------------------------------------------------------------------------------------
# @torch.no_grad()
def detect(image, previmage):

    #data='data/coco128.yaml'                    # dataset.yaml path
    imgsz=(640, 640) 	 				# inference size (height, width)
    conf_thres=0.5  					# confidence threshold
    #device = 'cpu'

    #results = model.predict(source=camdev, device=device, conf=conf_thres, max_det=10, agnostic_nms=True, stream=True, verbose=False, vid_stride=1) 
    results = Model.predict(source=image, conf=conf_thres, max_det=10, agnostic_nms=True, verbose=False, imgsz=imgsz) 

    for pred in results:   		# with YOLO rtsp, always new results (or blocks)
        thisimage = pred.orig_img

        # note: yolov8 sorts in descending confidence-order, thus on-break have best-target...
        for box in pred.boxes:

            cname = Names[int(box[0].cls)]
            if(cname in TARGETS):

                # check for motion this detection box vs prior
                xyxy = box.xyxy[0]  
                if(previmage is not None):
                    bd = boxdiff(thisimage, previmage, xyxy, cname)
                else:
                    bd = MDET    # fake it

                #print(f"bd: {bd}, MDET: {MDET}")
                if(bd >= MDET):
                    if(PutDetect):
                        bname = 'vehicle' if(cname in Vehicle) else 'animal' if(cname in Animal) else 'person' if(cname in Person) else cname
                        imzoom = TargetBox(xyxy, thisimage, imgsz, False)
                        #CTput(camLoc + '/' + bname, imzoom, None, thisTime)            # save zoomed targetBox image

                    # Add bbox to image
                    bconf = box.conf[0]
                    status = f"{cname} d={bconf:.2f} m={bd:.2f}"
                    label = None if hide_labels else (cname if hide_conf else status)
                    newimage = box_label(thisimage.copy(), xyxy, label)  	# this clobbers prior box_label if any
                    #cv2.imwrite('jiffycamDetect.jpg', newimage)
                    #print('detect: '+status+', newimage: '+ str(newimage.shape))
                    return newimage  
                 
    return None  
        
    #print('BuhBye')
    
# ---------------------------------------------------------------------------------------------------------------------

def TargetBox(xyxy, im0s, imgsz, togcrop):
    iwidth = imgsz[1]

    if(xyxy != None):
        x1 = int(max(0,xyxy[0]))
        y1 = int(max(0,xyxy[1]))
        x2 = int(max(0,xyxy[2]))
        y2 = int(max(0,xyxy[3]))
        wx = int(1.2*(x2 - x1))                 # pad
        wy = int(1.2*(y2 - y1))
                    
        h, w, c = im0s.shape
        xmid = (x1 + x2) / 2.
        ymid = (y1 + y2) / 2.
        x1 = int(max(0, xmid - iwidth/2.))
        x1 = int(min(x1, (w - iwidth - 1)))
        x2 = x1+iwidth
        y1 = int(max(0, ymid - iwidth/2.))
        y1 = int(min(y1, (h - iwidth - 1)))
        y2 = y1+iwidth
        targetBox = (x1, y1, x2, y2)
    else:
        h, w, c = im0s.shape
        targetBox = ( 0, 0, w-1, h-1 )
    
    # crop and reshape:
    im = im0s[targetBox[1]:targetBox[3], targetBox[0]:targetBox[2]]
    if(im.shape[1] != iwidth):
        im = cv2.resize(im, imgsz, interpolation=cv2.INTER_LINEAR)

#    return targetBox, im
    return im

# ---------------------------------------------------------------------------------------------------------------------
# add border to target-detected image

def addBorder(img):
    bv = [0,0,255]         # BGR
    bw = 10                 # border width
    h, w, _ = img.shape
    cv2.rectangle(img, [0,0], [w-1, h-1], color=bv, thickness=bw)   # full-pix border

# ---------------------------------------------------------------------------------------------------------------------
# labelTime: put time-label on image
def labelTime(im, thisTime=None):

    if(thisTime == None):
        DateStr = datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
    else:
        DateStr = datetime.fromtimestamp(float(thisTime)).strftime("%Y/%m/%d, %H:%M:%S")

    prec = 1
    if(prec > 0):
        DateStr += ("%.9f" % (time.time() % 1,))[1:2+prec]

#    print(DateStr)
    h, w, c = im.shape
#    fs = max(1,h/1400) 
    fs = max(1.5,h/1000) 
    thick = math.ceil(2 * fs)
    fo = math.ceil(40 * fs) 
    cv2.putText(im, DateStr, (fo, fo), 0, fontScale=fs, color=(240,240,240), thickness=thick, lineType=cv2.FILLED)
    fo = fo + thick 
    cv2.putText(im, DateStr, (fo, fo), 0, fontScale=fs, color=(25,25,25), thickness=thick, lineType=cv2.FILLED)

    return im

# ---------------------------------------------------------------------------------------------------------------------
# boxdiff:  motion detect via pixel-difference this_img detection vs prior
def boxdiff(newimg, previmg, xyxy, cname):

    x1 = int(max(0,xyxy[0]))
    y1 = int(max(0,xyxy[1]))
    x2 = int(max(0,xyxy[2]))
    y2 = int(max(0,xyxy[3]))
    
    h, w, _ = newimg.shape
    minSize = 10                               # minimum pixels (was 5)
    if( ((y2-y1) < minSize) or ((x2-x1)<minSize) ):
        print(cname+": too small: "+str(xyxy))
        return 0
        
    crop1 = newimg[y1:y2, x1:x2]
    crop2 = previmg[y1:y2, x1:x2]
    diff = cv2.absdiff(crop1, crop2)
    diff[diff < MTHRESH] = 0;            # threshold noise
    
    bdiff = diff.sum() / crop1.size / 255     # avg diff (as fraction of max-pix 255), normalized
    return bdiff

# ---------------------------------------------------------------------------------------------------------------------
# add label to detected object
def box_label(im=None, box=None, label='', mycolor=(0,0,255), txt_color=(0,0,0), lw=3):

#    im = imorig.copy()   	# keep box labels out of original image
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, p1, p2, mycolor, thickness=lw, lineType=cv2.LINE_AA)
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h - 3 >= 0  # label fits outside box
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(im, p1, p2, mycolor, -1, cv2.LINE_AA)  # filled
    cv2.putText(im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
        
    #addBorder(im)  	# add border frame to highlight target-hit

    return im

