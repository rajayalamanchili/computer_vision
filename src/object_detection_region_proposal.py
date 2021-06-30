# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 08:16:00 2021

@author: raja
"""

import cv2
import random
import torch
import numpy as np
import torchvision.models as models


def preprocess_cv2image(input_image):
    
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, (224, 224))
    input_image = input_image.astype("float32") / 255.0
    
    input_image -= [0.485, 0.456, 0.406]
    input_image /= [0.229, 0.224, 0.225]
    input_image = np.transpose(input_image, (2, 0, 1))
    input_image = np.expand_dims(input_image, 0)
    
    return input_image

def non_maxima_suppression(bboxes, probs=None, overlap_threshold=0.3):
    
    if len(bboxes) == 0:
        return []
    bboxes = np.array(bboxes)
    bboxes = bboxes.astype("float")
    
    bboxes_filtered = []
    
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    
    bboxes_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idx = np.argsort(y2)
    
    if probs is not None:
        idx = np.argsort(probs)
    
    filtered_idx = []
    
    while len(idx) > 0:
        last = len(idx) - 1
        i = idx[last]
        filtered_idx.append(i)
        
        xx1 = np.maximum(x1[i], x1[idx[:last]])
        yy1 = np.maximum(y1[i], y1[idx[:last]])
        xx2 = np.minimum(x2[i], x2[idx[:last]])
        yy2 = np.minimum(y2[i], y2[idx[:last]])
        
        width = np.maximum(0, (xx2 - xx1 + 1))
        height = np.maximum(0, (yy2 - yy1 + 1))
        
        overlap_ratio = (width * height) / bboxes_area[:last]
        
        idx = np.delete(idx, np.concatenate(([last], np.where( overlap_ratio > overlap_threshold)[0])))
        
        
    bboxes_filtered = bboxes[filtered_idx].astype("int")
        
    return bboxes_filtered

def selective_search_regions(input_image):
    
    debug = False
    
    print("Info: selective search processing ...")
    search_params = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    search_params.setBaseImage(input_image)
    search_params.switchToSelectiveSearchFast()
    
    search_boxes = search_params.process()
    
    visualize_boxes(input_image, search_boxes, title="Boxes from Selective search")
    
    if debug == True:
        print("Debug: graph segmentation processing")
        search_segmented = cv2.ximgproc.segmentation.createGraphSegmentation()
        segmented_image = search_segmented.processImage(input_image)
        segmented_image = segmented_image.astype("uint8")
        cv2.imshow("segmented image", cv2.applyColorMap(segmented_image, 
                                                        cv2.COLORMAP_HSV))
    
    print("Info: Applying image classifier to selective search boxes...")
    labels_boxes = {}
    
    (image_height, image_width) = input_image.shape[:2]
    
    print("Info: Loading model")
    model = models.resnet50(pretrained=True)
    model.eval()
    
    imagenetLabels = dict(enumerate(open("../data/ilsvrc2012_wordnet_lemmas.txt")))
    
    # create tensors
    print("Info: Applying model and extracting labels")
    for (x, y, w, h) in search_boxes:
        
        # exclude boxes with height or width less 10% of image size
        if ( (w / float(image_width) < 0.1) or 
            (h / float(image_height) < 0.1) ):
            continue
        
        # convert to RGB channel, range [0, 1],
        # resize to (224, 224) as required by pretrained model
        roi = input_image[ y:y+h, x:x+w]
        roi = preprocess_cv2image(roi)
        roi = torch.from_numpy(roi)
        
        # apply pretrained model
        logits = model(roi)
        
        # extract label and probabilities
        probabilities = torch.nn.Softmax(dim=-1)(logits)
        (label, prob) = (imagenetLabels[probabilities.argmax().item()].strip(),
                         probabilities.max().item())
        
        # if probability is less than 90% ignore the box
        if prob < 0.9:
            continue
        L = labels_boxes.get(label, [])
        L.append(((x, y, w, h), prob))
        labels_boxes[label] = L
        
    
    # visualize results
    print("Info: display results")
    
    for label in labels_boxes.keys():
        
        bboxes = []
        probs = []
        
        for (bbox, prob) in labels_boxes[label]:
            bboxes.append(bbox)
            probs.append(prob)
            
        visualize_boxes(input_image, bboxes, box_color=(0, 255, 0),
                        title="Boxes for label '{}'".format(label))
        
        final_boxes = non_maxima_suppression(bboxes, probs)
                               
        visualize_boxes(input_image, final_boxes.tolist(), box_color=(0, 255, 0),
                        title="Boxes for label '{}' after suppression".format(label))
        
        
        
    
    
def visualize_boxes(image, bounding_boxes, box_color=(), title=""):
    
    input_image = image.copy()
    
    for bbox in bounding_boxes:
        
        if len(box_color) != 3:
            color = [random.randint(0, 255) for ch in range(0,3)]
        else:
            color = box_color
            
        cv2.rectangle(input_image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[0]+bbox[2]), color, 2)
        cv2.putText(input_image, "Box Count: " + str(len(bounding_boxes))
                    , (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cv2.imshow(title,input_image)
    

if __name__ == "__main__":
    # read images
    print("Info: Reading images")
    input_image = cv2.imread("../data/images/beagle.png")
    
    
    # process images
    selective_search_regions(input_image)
    
    
"""
credits
Adrian Rosebrock, OpenCV Selective Search for Object Detection, PyImageSearch, 
https://www.pyimagesearch.com/2020/06/29/opencv-selective-search-for-object-detection/, 
accessed on 29 June 2021

"""