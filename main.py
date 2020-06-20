import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("refImg", type=str, help="path to input reference image")
ap.add_argument("altImg", type=str, help="path to image to be altered/transformed")
args = ap.parse_args()

MIN_MATCH_COUNT = 10
IMG_WIDTH = 600

def load_images(imgName1, imgName2):
    # Load images
    img1 = cv2.imread(imgName1)
    img2 = cv2.imread(imgName2)
    
    # Downscale the images
    img1 = rescale_image(img1)
    img2 = rescale_image(img2)
    
    return (img1, img2)
   
def rescale_image(img, width=IMG_WIDTH):
    scale = width / img.shape[1]
    # Resize to specified width, preserving aspect ratio
    img = cv2.resize(img, (int(scale*img.shape[1]), int(scale*img.shape[0])))
    return img

if __name__ == '__main__' :
    refImg, altImg = load_images(args.refImg, args.altImg)
    
    # Select region of interest bounding box
    roi = cv2.selectROI(refImg)
    # Crop the template from the reference image
    template = refImg[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    
    # Initiate SURF detector
    surf = cv2.xfeatures2d.SIFT_create()
    
    # find keypoints and descriptors
    kp1, des1 = surf.detectAndCompute(template, None)
    kp2, des2 = surf.detectAndCompute(altImg, None)
    
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
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        
    else:
        print("Not enough matches, %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
    
    draw_params = dict(matchColor = (0,255,0), singlePointColor=-1, matchesMask = matchesMask, flags = 0)
    imgMatches = cv2.drawMatches(template, kp1, altImg, kp2, good, None, **draw_params)
    
    cv2.imshow('', imgMatches)
    cv2.waitKey(0)
    
    