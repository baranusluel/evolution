import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("refImg", type=str, help="path to input reference image")
ap.add_argument("altImg", type=str, help="path to image to be altered/transformed")
args = ap.parse_args()

if __name__ == '__main__' :
    # Load images
    refImg = cv2.imread(args.refImg)
    altImg = cv2.imread(args.altImg)
    
    # Downscale the first image
    scale = 600 / refImg.shape[1]
    # Resize reference image to 600 width for convenience, preserving aspect ratio
    refImg = cv2.resize(refImg, (int(scale*refImg.shape[1]), int(scale*refImg.shape[0])))
    
    # Downscale the second image
    scale = 600 / altImg.shape[1]
    # Resize reference image to 600 width for convenience, preserving aspect ratio
    altImg = cv2.resize(altImg, (int(scale*altImg.shape[1]), int(scale*altImg.shape[0])))
    
    # Select region of interest bounding box
    roi = cv2.selectROI(refImg)
    # Crop the template from the reference image
    template = refImg[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    cv2.imshow('', template)
    cv2.waitKey(0)
 
