import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("img1", type=str, help="path to input reference image")
ap.add_argument("img2", type=str, help="path to image to be altered/transformed")
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
    
    # Convert to grayscale?
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    return (img1, img2)
   
def rescale_image(img, width=IMG_WIDTH):
    scale = width / img.shape[1]
    # Resize to specified width, preserving aspect ratio
    img = cv2.resize(img, (int(scale*img.shape[1]), int(scale*img.shape[0])))
    return img
    
def get_gradient(im) :
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)
    grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)
    
    cv2.imshow("gradx",grad_x)
    cv2.waitKey(0)
    
    #cv2.imshow("grady",grad_y)
    #cv2.waitKey(0)

    # Combine the two gradients
    grad = np.sqrt(np.square(grad_x) + np.square(grad_y))
    grad = np.array(grad / np.max(grad) * 255).astype(int)
    #cv2.imshow("grad",grad)
    #print("grad min: %d, max: %d" % (np.min(grad), np.max(grad)))
    #print(grad)
    #cv2.waitKey(0)
    
    return grad_x

if __name__ == '__main__' :
    img1, img2 = load_images(args.img1, args.img2)
    
    # Select region of interest bounding box
    roi = cv2.selectROI(img1)
    # Crop the template from the reference image
    template = img1[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    sz = template.shape
    
    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrix = np.eye(3, 3, dtype=np.float32)
    
    num_iter = 5000;
    termination_eps = 1e-1;
    
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iter, termination_eps)
    
    grad1 = get_gradient(template)
    grad2 = get_gradient(img2)
    
    (cc, warp_matrix) = cv2.findTransformECC(grad1, grad2, warp_matrix, warp_mode, criteria)
    print(warp_matrix)
    img2_aligned = cv2.warpPerspective(img2, warp_matrix, (img2.shape[1], img2.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    
    cv2.imshow("Template", template)
    cv2.imshow("Image 2", img2)
    cv2.imshow("Aligned Image 2", img2_aligned)
    cv2.waitKey(0)
    
    