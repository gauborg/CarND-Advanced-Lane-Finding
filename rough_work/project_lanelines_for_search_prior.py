import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

import moviepy
import imageio
from moviepy.editor import VideoFileClip


dist_pickle = pickle.load( open( "pickle/test_images_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
M = dist_pickle["M"]
Minv = dist_pickle["Minv"]

# Thresholding functions
# since we have evaludated earlier that HLS gives good image filtering results
# only included the relevant thresholding functions from "pipeline.ipynb"
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

# function for applying perspective view on the masked thresholded images
def perspective_view(img):
    img_size = (img.shape[1], img.shape[0])
    # image points extracted from image approximately
    bottom_left = [220, 720]
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
    return warped, M, Minv


def advanced_lanelines(img):

    # undistort the original image using stored values from pickle
    undist_original = cv2.undistort(img, mtx, dist, None, mtx)
    # apply perspective view on the image
    warped_original, M, Minv = perspective_view(undist_original)

    # apply combined threshold
    threshold = combined_threshold(img)
    # undistort the thresholded image
    undist_thresholded = cv2.undistort(threshold, mtx, dist, None, mtx)
    # apply perspective transform on the thresholded image
    warped, M, Minv = perspective_view(undist_thresholded)

    '''
    plt.figure(1)
    plt.imshow(warped, cmap = "gray")
    plt.show()
    '''

    # these will be empty for the first iteration and they will store the values of lane fits from previous iterations
    # declaring lane fits as global variables so that they can be modified from anywhere in the code

    global previous_left_fit
    global previous_right_fit
    global previous_detection
    global prev_left_fits
    global prev_right_fits
    global average_left_fit
    global average_right_fit

    global i

    # initialize the lanelines class by giving inputs from previous iteration
    binary_warped = LaneLines(warped, previous_left_fit, previous_right_fit, previous_detection, average_left_fit, average_right_fit)

    # calculate the left and right lane fits
    out_img, leftfit, rightfit, detected = binary_warped.find_lane_pixels()

    print("Output Left Fit - ", leftfit)
    print("Output Right Fit - ", rightfit)

    # we convert our previous left and right fits from shape (3,) to shape (1,3) to append it to out list
    previous_left_fit_array = np.array([leftfit])
    previous_right_fit_array = np.array([rightfit])
    previous_detection = detected

    # we add fits from previous detections to our list of previous fits
    prev_left_fits = np.append(prev_left_fits, previous_left_fit_array, axis = 0)
    prev_right_fits = np.append(prev_right_fits, previous_right_fit_array, axis = 0)

    # we ensure that the list doesn't take into account more than 10 previous measurements
    if (prev_left_fits.shape[0] > 10):
        prev_left_fits = np.delete(prev_left_fits, 0, axis = 0)
    if(prev_right_fits.shape[0] > 10):
        prev_right_fits = np.delete(prev_right_fits, 0, axis = 0)

    print("shape of left - ", prev_left_fits.shape)
    print("shape of right - ", prev_right_fits.shape)
    print("frame number - ", i)
    
    i = i + 1


    # compute average of past 10 best fits and pass them over to next iteration
    average_left_fit = np.mean(prev_left_fits, axis = 0)
    average_right_fit = np.mean(prev_right_fits, axis = 0)

    print("Avg Left Fit - ", average_left_fit)
    print("Avg Right Fit - ", average_right_fit)


    # get the left and right lane radii
    center_offset, left_radius, right_radius = binary_warped.measure_curvature_pixels()

    road_radius = round(0.5*(left_radius+right_radius), 2)
    center_offset = round(center_offset, 2)
    road_curvature = "Road Curvature = " + str(road_radius) + "m"
    center_offset = "Center Offset = " + str(center_offset) + "m"

    # print("Left = ", left_radius)
    # print("Right = ", right_radius)
    print("Road Curvature = ", road_radius)

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([binary_warped.left_fitx, binary_warped.ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([binary_warped.right_fitx, binary_warped.ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    unwarped = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undist_original, 1, unwarped, 0.3, 0)
    # this prints the value of road curvature onto the output image
    cv2.putText(result, road_curvature, (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=2)
    cv2.putText(result, center_offset, (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=2)

    # this line prints the frame number
    cv2.putText(result, str(i-1), (1200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=2)

    # we assign the leftfit and right values here to pass them onto the next frame
    previous_left_fit = leftfit
    previous_right_fit = rightfit

    print("Previous left fit - ", leftfit)
    print("Previous right fit - ", rightfit)
    print()

    return result


# define variables needed in the global scope
previous_left_fit = None
previous_right_fit = None
previous_detection = False

# for calculating the average data of past 10 frames
prev_left_fits = np.empty([1,3])
prev_right_fits = np.empty([1,3])

# intitialize the average left and right fits to empty lists - these will be updated in every iteration

average_left_fit = []
average_right_fit = []

# import the lanelines class
from class_lanelines_w_prior_search_algorithm import LaneLines

i = 1

# video pipeline
video_binary_output = 'project-video-output.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(advanced_lanelines) # NOTE: this function expects color images!!
white_clip.write_videofile(video_binary_output, audio=False)


# image pipeline - run for two successive images"
'''
test_image = mpimg.imread("test_images/my_test.jpg")
lane_image = advanced_lanelines(test_image)
plt.imshow(lane_image)
plt.show()

test_image2 = mpimg.imread("test_images/test1.jpg")
lane_image2 = advanced_lanelines(test_image2)
plt.imshow(lane_image2)
plt.show()
'''

