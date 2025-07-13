# ---------------------------------------------------------------------------------------------------------------------
"""
JiffyDetect:  YOLOV8 version
Cycronix
1/29/2024
"""

# ---------------------------------------------------------------------------------------------------------------------
import time
from datetime import datetime
import math
import cv2
from ultralytics import YOLO
import numpy as np

import torch

# ---------------------------------------------------------------------------------------------------------------------
# rtsp-camera fetch

PutDate = False
PutDetect = False				# put cropped image of detection (e.g. 'vehicle.jpg')

#MDET = 0.15                                      # motion detect trigger level (linear 0-1) (was .1)
MTHRESH = 1                                     # motion detect noise reject threshold (per raw pixel) (was 4)

# Tiling configuration
TILE_THRESHOLD = 1280                           # minimum image size to trigger tiling
TILE_OVERLAP = 0.15                             # overlap between tiles (15%)
TILE_NMS_THRESHOLD = 0.5                        # NMS threshold for merging tile detections
TILE_BATCH_SIZE = 8                             # process tiles in batches for better GPU utilization (optimal for MPS)
TILE_SIZE = (640, 640)                          # tile size
#TILE_SIZE = (1280, 1280)                          # tile size

hide_labels = False
hide_conf = True

# Global YOLO model instance
Model = None
Names = None

previmage = None
prevxyxy = None

TARGETS = None
MDET = None
CONF = None

MPS_AVAILABLE = None

# ---------------------------------------------------------------------------------------------------------------------
mps_available = None
def device_type():
    global mps_available
    if(mps_available == None):
        mps_available = torch.backends.mps.is_available()
        print(f"mps_available: {mps_available}")
    return 'mps' if mps_available else 'cpu'

# ---------------------------------------------------------------------------------------------------------------------
def setTargets(weights_path='models/yolov8l.pt'):
    global TARGETS, MDET, CONF
    global Vehicle, Animal, Person
    
    OIV7 = "oiv7".lower() in weights_path
    #print(f"OIV7: {OIV7}")

    if(OIV7):
        Vehicle = [ 'Boat', 'Truck', 'Bicycle', 'Motorcycle', 'Bus', 'Canoe', 'Snowplow', 'Vehicle' ]
        Animal = [ 'Dog', 'Cat', 'Bird', 'Bear', 'Bat (Animal)', 'Butterfly', 'Deer', 'Dragonfly', 'Duck', \
                'Eagle', 'Falcon', 'Goose', 'Moths and butterflies', 'Mouse', 'Otter', 'Porcupine', 'Rabbit', \
                    'Racoon', 'Raven', 'Reptile', 'Skunk', 'Snake', 'Sparrow', 'Spider', 'Squirrel', 'Turkey', 'Turtle', 'Woodpecker' ]
        Person = [ 'Person' ]
        MDET = 0.15
        CONF = 0.25
    else:
        Vehicle = [ 'car', 'truck', 'bicycle', 'motorcycle' ]
        Animal = [ 'dog', 'cat', 'bird', 'bear' ]
        Person = [ 'person' ]
        MDET = 0.15
        CONF = 0.5

    TARGETS = Vehicle + Animal + Person

# ---------------------------------------------------------------------------------------------------------------------
# Tiling utility functions

def generate_tiles(image, tile_size=TILE_SIZE, overlap_ratio=TILE_OVERLAP):
    """
    Generate overlapping tiles from a large image
    Returns list of (tile_image, x_offset, y_offset) tuples
    """
    h, w = image.shape[:2]
    tile_h, tile_w = tile_size
    
    # Calculate step size with overlap
    step_h = int(tile_h * (1 - overlap_ratio))
    step_w = int(tile_w * (1 - overlap_ratio))
    
    tiles = []
    
    for y in range(0, h, step_h):
        for x in range(0, w, step_w):
            # Calculate tile boundaries
            x1 = x
            y1 = y
            x2 = min(x + tile_w, w)
            y2 = min(y + tile_h, h)
            
            # Extract tile
            tile = image[y1:y2, x1:x2]
            
            # Pad tile to exact size if needed (for edge tiles)
            if tile.shape[0] != tile_h or tile.shape[1] != tile_w:
                padded_tile = cv2.copyMakeBorder(
                    tile,
                    0, tile_h - tile.shape[0],
                    0, tile_w - tile.shape[1],
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0)
                )
                tile = padded_tile
            
            tiles.append((tile, x1, y1))
    
    return tiles

def merge_detections(detections, nms_threshold=TILE_NMS_THRESHOLD):
    """
    Merge detections from multiple tiles using Non-Maximum Suppression
    detections: list of (xyxy, conf, cls, cname) tuples
    """
    if not detections:
        return []
    
    import numpy as np
    
    # Convert to numpy arrays for NMS
    boxes = np.array([det[0] for det in detections])
    scores = np.array([det[1] for det in detections])
    classes = np.array([det[2] for det in detections])
    
    # Apply NMS using OpenCV
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        CONF,
        nms_threshold
    )
    
    # Return filtered detections
    if len(indices) > 0:
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        return [detections[i] for i in indices]
    
    return []

def detect_on_tiles(image, weights_path='models/yolov8l.pt'):
    """
    Perform detection on image tiles using batch processing for better GPU utilization
    Returns list of merged detections or None if no detections
    """
    global Model, Names
    
    # Set targets and configuration based on model type
    setTargets(weights_path)
    
     # Generate all tiles
    tiles = generate_tiles(image)

    # Initialize model if not already done
    if Model is None:
        Model = YOLO(weights_path, task='detect')
        Names = Model.names
        print(f"Generated {len(tiles)} tiles for image shape {image.shape}")
    
    all_detections = []
    
    # Process tiles in batches
    for batch_start in range(0, len(tiles), TILE_BATCH_SIZE):
        batch_end = min(batch_start + TILE_BATCH_SIZE, len(tiles))
        batch_tiles = tiles[batch_start:batch_end]
        
        # Extract tile images for batch processing
        tile_images = [tile[0] for tile in batch_tiles]
        
        #print(f"Processing tile batch {batch_start//TILE_BATCH_SIZE + 1}/{(len(tiles)-1)//TILE_BATCH_SIZE + 1} ({len(tile_images)} tiles)")
        
        # Run detection on batch of tiles
        results = Model.predict(
            source=tile_images,
            conf=CONF,
            max_det=10,
            agnostic_nms=True,
            verbose=False,
            imgsz=(640, 640),
            device=device_type()
        )
        
        # Process results from batch
        for i, pred in enumerate(results):
            # Get the corresponding tile info
            tile, x_offset, y_offset = batch_tiles[i]
            
            # Process detections for this tile
            for box in pred.boxes:
                cname = Names[int(box[0].cls)]
                if cname in TARGETS:
                    # Transform coordinates back to original image space
                    xyxy = box.xyxy[0].cpu().numpy()
                    xyxy[0] += x_offset  # x1
                    xyxy[1] += y_offset  # y1
                    xyxy[2] += x_offset  # x2
                    xyxy[3] += y_offset  # y2
                    
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box[0].cls)
                    
                    all_detections.append((xyxy, conf, cls, cname))
                    #print(f"detect: {cname} conf: {conf}")  

    # Merge overlapping detections
    merged_detections = merge_detections(all_detections)
    
    return merged_detections

# ---------------------------------------------------------------------------------------------------------------------
# @torch.no_grad()
def detect(image, weights_path='models/yolov8l.pt', enable_tiling=False):
    global Model, Names, previmage, prevxyxy
    
    # Check if image is large enough for tiling (NEW TILING LOGIC)
    h, w = image.shape[:2]
    use_tiling = enable_tiling and (h > TILE_THRESHOLD or w > TILE_THRESHOLD)
    setTargets(weights_path)  # set TARGETS, MDET, CONF

    if use_tiling:
        #print(f"Using tiling for large image: {w}x{h}")
        detections = detect_on_tiles(image, weights_path)
        
        if detections:
            # Process the best detection (highest confidence)
            best_detection = max(detections, key=lambda x: x[1])
            xyxy, conf, cls, cname = best_detection
            
            # Convert numpy array to tensor-like format for compatibility
            xyxy_tensor = torch.tensor(xyxy)
            
            # Check for motion (use first valid detection for motion comparison)
            if previmage is not None and prevxyxy is not None:
                bd = boxdiff(image, previmage, xyxy_tensor, prevxyxy)
            else:
                bd = MDET  # fake it for first detection
            
            previmage = image.copy()
            prevxyxy = xyxy_tensor
            
            if bd >= MDET:
                if PutDetect:
                    bname = 'vehicle' if(cname in Vehicle) else 'animal' if(cname in Animal) else 'person' if(cname in Person) else cname
                    imzoom = TargetBox(xyxy_tensor, image, (640, 640), False)
                    # CTput(camLoc + '/' + bname, imzoom, None, thisTime)  # save zoomed targetBox image
                
                # Add bbox to image (only the best detection to match original behavior)
                status = f"{cname} d={conf:.2f} m={bd:.2f}"
                label = None if hide_labels else (cname if hide_conf else status)
                newimage = box_label(image.copy(), xyxy_tensor, label)
                print('detect: '+status+', image: '+ str(newimage.shape), flush=True)
                return newimage
        
        return None
    
    # ORIGINAL LOGIC RESTORED EXACTLY AS IT WAS:
    # Initialize model if not already done
    if Model is None:
        Model = YOLO(weights_path, task='detect')  # Load model
        Names = Model.names

    #data='data/coco128.yaml'                    # dataset.yaml path
    imgsz=(640, 640) 	 				# inference size (height, width)

    #results = model.predict(source=camdev, device=device, conf=conf_thres, max_det=10, agnostic_nms=True, stream=True, verbose=False, vid_stride=1) 
    results = Model.predict(source=image, conf=CONF, max_det=10, agnostic_nms=True, verbose=False, imgsz=imgsz, device=device_type()) 

    for pred in results:   		# with YOLO rtsp, always new results (or blocks)
        thisimage = pred.orig_img

        # note: yolov8 sorts in descending confidence-order, thus on-break have best-target...
        for box in pred.boxes:

            cname = Names[int(box[0].cls)]
            if(cname in TARGETS):

                # check for motion this detection box vs prior
                xyxy = box.xyxy[0]  
                if(previmage is not None):
                    bd = boxdiff(thisimage, previmage, xyxy, prevxyxy)
                else:
                    bd = MDET    # fake it

                previmage = thisimage
                prevxyxy = xyxy

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
                    print('detect: '+status+', newimage: '+ str(newimage.shape), flush=True)
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
def boxdiff(newimg, previmg, xyxy, prevxyxy):

    x1 = int(max(0,xyxy[0]))
    y1 = int(max(0,xyxy[1]))
    x2 = int(max(0,xyxy[2]))
    y2 = int(max(0,xyxy[3]))
    xycenter = (x1 + x2) / 2, (y1 + y2) / 2
    xysize = ( (x2 - x1) + (y2 - y1) ) / 2

    x1p = int(max(0,prevxyxy[0]))
    y1p = int(max(0,prevxyxy[1]))
    x2p = int(max(0,prevxyxy[2]))
    y2p = int(max(0,prevxyxy[3]))  
    prevxycenter = (x1p + x2p) / 2, (y1p + y2p) / 2
    prevxysize = ( (x2p - x1p) + (y2p - y1p) ) / 2
    xysize_avg = (xysize + prevxysize) / 2

    minSize = 10                               # minimum pixels (was 5)
    if( ((y2-y1) < minSize) or ((x2-x1)<minSize) ):
        #print(cname+": too small: "+str(xyxy))
        return 0
        
    crop1 = newimg[y1:y2, x1:x2]
    crop2 = previmg[y1:y2, x1:x2]
    diff = cv2.absdiff(crop1, crop2)
    diff[diff < MTHRESH] = 0;            # threshold noise
    bdiff = diff.sum() / crop1.size / 255     # avg diff (as fraction of max-pix 255), normalized

    xydist = math.dist(xycenter, prevxycenter) / xysize_avg
    #print(f"boxdiff: xydist: {xydist}, bdiff: {bdiff}")

    return min(bdiff, xydist)

# ---------------------------------------------------------------------------------------------------------------------
# add label to detected object
def box_label(im=None, box=None, label='', mycolor=(0,0,255), txt_color=(0,0,0), lw=2):

#    im = imorig.copy()   	# keep box labels out of original image
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, p1, p2, mycolor, thickness=lw, lineType=cv2.LINE_AA)
    tf = max(lw, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=1, thickness=tf)[0]  # text width, height
    outside = p1[1] - h - 3 >= 0  # label fits outside box
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(im, p1, p2, mycolor, -1, cv2.LINE_AA)  # filled
    cv2.putText(im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),0, 1, txt_color, thickness=tf, lineType=cv2.LINE_AA)
        
    addBorder(im)  	# add border frame to highlight target-hit

    return im

