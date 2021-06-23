# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:54:02 2021

@author: raja

"""

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

def single_template_match(input_image, template_image):
    
    # convert images to gray scalre
    input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    image_template_gray = cv2.cvtColor(image_template, cv2.COLOR_BGR2GRAY)
    
    # apply template matching
    print("Info: seraching template in image")
    match_result = cv2.matchTemplate(input_image_gray, image_template_gray, 
                                     cv2.TM_CCOEFF_NORMED)
    
    # compute bounding box
    (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(match_result)
    
    (startX, startY) = max_loc
    
    endX = startX + image_template.shape[1]
    endY = startY + image_template.shape[0]
    
    # display results
    fig, ax = plt.subplots(2, 2, figsize=(30,12), sharex=True, sharey=True)
    ax[0,0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    ax[0,0].title.set_text("Image")
    
    ax[0,1].imshow(cv2.cvtColor(image_template, cv2.COLOR_BGR2RGB))
    ax[0,1].title.set_text("Template")
    
    ax[1,0].imshow(match_result)
    ax[1,0].title.set_text("Template Correlation")
    
    cv2.rectangle(input_image, (startX, startY), (endX, endY), (255, 0, 0), 5)
    ax[1,1].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    ax[1,1].title.set_text("Match Result")
    
    print("Info: save result")
    fig.suptitle("Single Template Match")
    fig.savefig("../results/template_matching" + str(int(time.time())) + ".png")

def non_maxima_suppression(bboxes, overlap_threshold=0.3):
    
    if len(bboxes) == 0:
        return []
    bboxes = bboxes.astype("float")
    
    bboxes_filtered = []
    
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    
    bboxes_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idx = np.argsort(y2)
    
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
    
def multiple_template_match(input_image, image_template, threshold=0.8):
    
    # convert images to gray scalre
    input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    image_template_gray = cv2.cvtColor(image_template, cv2.COLOR_BGR2GRAY)
    
    # apply template matching
    print("Info: seraching template in image")
    match_result = cv2.matchTemplate(input_image_gray, image_template_gray, 
                                     cv2.TM_CCOEFF_NORMED)
    
    # compute bounding box
    (ys, xs) = np.where( match_result >= threshold)
    template_width = image_template.shape[1]
    template_height = image_template.shape[0]
    
    bboxes = [[x, y, x + template_width, y + template_height] for x,y in zip(xs, ys)]
    
    bboxes_nms = non_maxima_suppression(np.array(bboxes))
    
    # display results
    fig, ax = plt.subplots(3, 2, figsize=(30,12), sharex=True, sharey=True)
    ax[0,0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    ax[0,0].title.set_text("Image")
    
    ax[0,1].imshow(cv2.cvtColor(image_template, cv2.COLOR_BGR2RGB))
    ax[0,1].title.set_text("Template")
    
    ax[1,0].imshow(match_result)
    ax[1,0].title.set_text("Template Correlation")
    
    input_copy = input_image.copy()
    for bbox in bboxes:
        cv2.rectangle(input_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 5)
    ax[1,1].imshow(cv2.cvtColor(input_copy, cv2.COLOR_BGR2RGB))
    ax[1,1].title.set_text("Match Result: " + str(len(bboxes)))
    
    for bbox in bboxes_nms:
        cv2.rectangle(input_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 5)
    ax[2,0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    ax[2,0].title.set_text("Match Result: " + str(len(bboxes_nms)))
    
    print("Info: save result")
    fig.suptitle("Multiple Template Match")
    fig.savefig("../results/template_matching" + str(int(time.time())) + ".png")
    
    
    

if __name__ == "__main__":

    # read images
    print("Info: Reading images")
    input_image = cv2.imread("../data/images/8_diamonds.png")
    image_template = cv2.imread("../data/images/diamonds_template.png") 
    
    
    # process images
    
    #single_template_match(input_image, image_template)
    multiple_template_match(input_image, image_template)


"""
credits
Adrian Rosebrock, OpenCV Face Recognition, PyImageSearch, 
https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/, 
accessed on 21 June 2021

"""