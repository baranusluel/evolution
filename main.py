import argparse
import cv2
import numpy as np

from os import listdir
from os.path import isfile, join

ap = argparse.ArgumentParser()
ap.add_argument("reference", type=str, help="path to input reference image")
ap.add_argument("target", type=str, help="path to directory containing images to transform")
args = ap.parse_args()

MIN_MATCH_COUNT = 4
IMG_WIDTH = 600

def load_image(imgName):
    # Load image
    img = cv2.imread(imgName)
    # Downscale the image
    img = rescale_image(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
   
def rescale_image(img, width=IMG_WIDTH):
    scale = width / img.shape[1]
    # Resize to specified width, preserving aspect ratio
    img = cv2.resize(img, (int(scale*img.shape[1]), int(scale*img.shape[0])))
    return img

if __name__ == '__main__' :
    # Load reference image
    img1 = load_image(args.reference)
    
    # Select region of interest bounding box
    roi = cv2.selectROI(img1)
    # Crop the template from the reference image
    template = img1[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    
    # Initiate SURF detector
    surf = cv2.xfeatures2d.SIFT_create()
    # find keypoints and descriptors for reference image
    kp1, des1 = surf.detectAndCompute(template, None)
    
    
    # Get paths of target images
    targetImgs = [join(args.target, f) for f in listdir(args.target) if isfile(join(args.target, f))]
    
    for imgName in targetImgs:
        img2 = load_image(imgName)
        kp2, des2 = surf.detectAndCompute(img2, None)
        
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        # Find best two matches for each descriptor
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test to find good matches
        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append(m)
                
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            
            #M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            matchesMask = mask.ravel().tolist()
            
            draw_params = dict(matchColor = (0,255,0), singlePointColor=-1, matchesMask = matchesMask, flags = 0)
            imgMatches = cv2.drawMatches(template, kp1, img2, kp2, good, None, **draw_params)
            cv2.imshow('', imgMatches)
            cv2.waitKey(0)
            
            img2_aligned = cv2.warpAffine(img2, M, (template.shape[1], template.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
            
            #cv2.imshow("Template", template)
            cv2.imshow("Aligned Image", img2_aligned)
            cv2.waitKey(0)
            
        else:
            print("Not enough matches, %d/%d" % (len(good), MIN_MATCH_COUNT))
            matchesMask = None
        