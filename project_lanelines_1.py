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

    # these will be empty for the first iteration and they will store the values of lane fits from previous iterations
    # declaring lane fits as global variables so that they can be modified from anywhere in the code
    
    # list storing left and right lanefit values from previous frames
    global previous_left_fit
    global previous_right_fit
    
    global prev_left_fits
    global prev_right_fits

    # average of previous 10 lanefits
    global average_left_fit
    global average_right_fit

    global frame

    global past_radii


    # initialize the lanelines class by giving inputs from previous iteration
    binary_warped = LaneLines(warped, average_left_fit, average_right_fit, previous_left_fit, previous_right_fit)

    # calculate the left and right lane fits
    out_img, leftfit, rightfit = binary_warped.find_lane_pixels()

    # we convert our left and right fits from shape (3,) to an array of shape (1,3) to append it to our lists
    previous_left_fit_array = np.array([leftfit])
    previous_right_fit_array = np.array([rightfit])

    # print("frame = ", frame)
    # print("left fit = ", leftfit)
    # print("right fit = ", rightfit)

    # we add fits from previous detections to our list of previous fits
    prev_left_fits = np.append(prev_left_fits, previous_left_fit_array, axis = 0)
    prev_right_fits = np.append(prev_right_fits, previous_right_fit_array, axis = 0)

    # we ensure that the list doesn't take into account more than 15 previous measurements
    # we delete the initial element of the array if it does, i.e. - earliest element in the array
    if (prev_left_fits.shape[0] > 10):
        prev_left_fits = np.delete(prev_left_fits, 0, axis = 0)
    if(prev_right_fits.shape[0] > 10):
        prev_right_fits = np.delete(prev_right_fits, 0, axis = 0)

    # compute average of past 10 best fits and pass them over to next iteration
    average_left_fit = np.mean(prev_left_fits, axis = 0)
    average_right_fit = np.mean(prev_right_fits, axis = 0)

    # print("Average Left Fit = ", average_left_fit)
    # print("Average Right Fit = ", average_right_fit)

    # get the left and right lane radii
    center_offset, left_radius, right_radius = binary_warped.measure_curvature()

    # START OF RADIUS CALCULATIONS
    # calculation of road curvature
    current_road_radius = 0.5*(left_radius+right_radius)

    # Storing past five frame's radii 
    past_radii.append(current_road_radius)
    # calculate weighted average of radii to reduce noisy measurements
    road_radius = sum(past_radii)/len(past_radii)

    # if no outliers are detected then, delete the oldest value
    if (len(past_radii) > 5):
        past_radii.pop(0)

    # calculate mean radius
    road_radius = sum(past_radii)/len(past_radii)
    road_radius = round(road_radius)

    center_offset = round(center_offset, 2)
    road_curvature = "Road Curvature = " + str(road_radius) + "m"
    center_offset = "Center Offset = " + str(center_offset) + "m"

    # print("Left = ", left_radius)
    # print("Right = ", right_radius)
    # print("Road Curvature = ", road_radius)

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
    cv2.putText(result, road_curvature, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), thickness=2)

    # this prints the value of center offset onto the output image
    cv2.putText(result, center_offset, (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), thickness=2)

    # this line prints the frame number
    cv2.putText(result, ("Frame: "+str(frame)), (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), thickness=1)

    # overlay images
    img2_resize = cv2.resize(out_img, None, fx = 0.25, fy = 0.25, interpolation = cv2.INTER_CUBIC)
    result[15:15+img2_resize.shape[0],945:945+img2_resize.shape[1]] = img2_resize


    # VISUALIZATION
    # THIS SECTION SHOULD BE COMMENTED OUT WHEN RUNNING ON VIDEO STREAM
    '''
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(25, 10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(threshold, cmap = 'gray')
    ax2.set_title('Original Thresholded', fontsize=20)
    ax3.imshow(warped_original, cmap = 'gray')
    ax3.set_title('Warped Perspective', fontsize=20)
    ax4.imshow(out_img)
    ax4.set_title('Sliding Boxes', fontsize=20)
    plt.show()
    '''

    previous_left_fit = leftfit
    previous_right_fit = rightfit

    frame = frame + 1

    return result


# define variables needed in the global scope
previous_left_fit = None
previous_right_fit = None

# initialize empty 1*3 empty arrays for calculating the storing lane fit data of previous frames
prev_left_fits = np.empty([1,3])
prev_right_fits = np.empty([1,3])

# intitialize the average left and right fit empty lists - these will be updated in every iteration
average_left_fit = []
average_right_fit = []

# initialize list for storing past radius measurements
past_radii = [0, 0, 0, 0, 0]

# frame number of the video stream (not necessary)
frame = 1

# import the lanelines class
from class_lanelines_1 import LaneLines


# COMMENT OUT THIS SETION TO RUN ON IMAGES
# video pipeline
video_binary_output = 'project-video-output.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(advanced_lanelines) # NOTE: this function expects color images!!
white_clip.write_videofile(video_binary_output, audio=False)


# UNCOMMENT THIS SECTION TO RUN ON IMAGES
'''
# image pipeline - run for two successive images"

test_image = mpimg.imread("test_images/test2.jpg")
lane_image = advanced_lanelines(test_image)
plt.imshow(lane_image)
plt.show()

test_image2 = mpimg.imread("test_images/test6.jpg")
lane_image2 = advanced_lanelines(test_image2)
plt.imshow(lane_image2)
plt.show()
'''
