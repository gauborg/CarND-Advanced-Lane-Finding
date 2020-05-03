import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import natsort
import pickle

import moviepy
import imageio
from moviepy.editor import VideoFileClip

dist_pickle = pickle.load( open( "pickle/test_images_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]



# Thresholding functions
# since we have evaludated earlier that HLS gives good image filtering results
# only included the relevant thresholding functions from "advanced_lanelines.py"

def lightness_select(img, thresh = (120,255)):
    
    # 1. Convert to hls colorspace
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2. Apply threshold to s channel
    l_channel = hls[:,:,1]
    # 3. Create empty array to store the binary output and apply threshold
    lightness_image = np.zeros_like(l_channel)
    lightness_image[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1

    return lightness_image


def saturation_select(img, thresh = (100,255)):

    # 1. convert to hls colorspace
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2. apply threshold to s channel
    s_channel = hls[:,:,2]
    # 3. create empty array to store the binary output and apply threshold
    sat_image = np.zeros_like(s_channel)
    sat_image[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1

    return sat_image


def abs_sobel_thresh(img, orient = 'x', sobel_kernel = 3, thresh = (0,255)):

    # 1. Applying the Sobel depending on x or y direction and getting the absolute value
    if (orient == 'x'):
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobel_kernel))
    if (orient == 'y'):
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobel_kernel))
    # 2. Scaling to 8-bit and converting to np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 3. Create mask of '1's where the sobel magnitude is > thresh_min and < thresh_max
    sobel_image = np.zeros_like(scaled_sobel)
    sobel_image[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return sobel_image


### Combined Thresholding Function
def combined_threshold(img):

    # convert to hls format and extract channels
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    l_channel = hls[:,:,1]

    # convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # applying thresholding and storing different filtered images
    l_binary = lightness_select(img, thresh = (120, 255))
    s_binary = saturation_select(img, thresh = (100, 255))

    ksize = 9
    gradx = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(20, 100))

    # creating an empty binary image
    combined_binary = np.zeros_like(s_binary)
    combined_binary[((gradx == 1) | (s_binary == 1)) & ((l_binary == 1) & (s_binary == 1))] = 1

    # apply region of interest mask
    height, width = combined_binary.shape
    mask = np.zeros_like(combined_binary)
    region = np.array([[0, height-1], [int(width/2), int(height/2)], [width-1, height-1]], dtype=np.int32)
    cv2.fillPoly(mask, [region], 1)

    masked_binary = cv2.bitwise_and(combined_binary, mask)

    return masked_binary


def perspective_view(img):

    img_size = (img.shape[1], img.shape[0])

    # image points extracted from image approximately
    bottom_left = [210, 720]
    bottom_right = [1100, 720]
    top_left = [570, 470]
    top_right = [720, 470]

    src = np.float32([bottom_left, bottom_right, top_right, top_left])
    pts = np.array([bottom_left, bottom_right, top_right, top_left])
    pts = pts.reshape((-1, 1, 2))

    # choose four points in warped image so that the lines should appear as parallel
    bottom_left_dst = [320, 720]
    bottom_right_dst = [920, 720]
    top_left_dst = [320, 1]
    top_right_dst = [920, 1]

    dst = np.float32([bottom_left_dst, bottom_right_dst, top_right_dst, top_left_dst])
    # apply perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    # compute inverse perspective transform
    Minv = cv2.getPerspectiveTransform(dst, src)
    # warp the image using perspective transform M
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, Minv



'''
# ------------------------------------- PIPELINE STARTS HERE ON A TEST IMAGE -------------------------------------- #

from find_lanelines_1 import LaneLines

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
img = mpimg.imread('test_images/test1.jpg')


# image processing started here ...
thresholded = combined_threshold(img)
warped_image = perspective_view(thresholded)

# for storing previously detected laneline details
previous_left_fit = []
previous_right_fit = []

# draw lanelines

'''






# this function returns our thresholded and perspective transformed images...
def lanelines(img):

    binary = combined_threshold(img)
    warped_image = perspective_view(img)

    return warped_image

video_binary_output = 'persp_view_lanees_video-output.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(lanelines) # NOTE: this function expects color images!!
white_clip.write_videofile(video_binary_output, audio=False)

