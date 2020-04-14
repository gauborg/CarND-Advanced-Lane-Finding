import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# function to calibrate the camera over different images

# get the list of images
images = glob.glob('camera_cal/calibration*.jpg')

# creating arrays to store object and image points from all the images
objpoints = []      # 3D object points in real space
imgpoints = []      # 2D points in image space

# prepare object points
objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)    # x, y cor-ordinates

# convert image to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# iterate through the images
i = 1
for fname in images:

    # read file
    img = cv2.imread(fname)

    # convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    print(ret)

    # If found, we will get image coordinates
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        # draw and display the corners
        img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        mpimg.imsave(("camera_cal/identified_corners/calibration-output"+str(i)+".jpg"), img)
        # plt.imshow(img)
        # plt.show()
    
    i+=1

print(len(objpoints))



'''
def calibrate_camera(image):

    # function to calibrate the camera over different images first

    # prepare object points
    nx = 9 #TODO: enter the number of inside corners in x
    ny = 6 #TODO: enter the number of inside corners in y

    # Make a list of calibration images
    
    img = cv2.imread(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret == True:
        # Draw and display the corners
        print("Found Corners!!!")
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        plt.imshow(img)
        mpimg.imsave((directory + "identified_corners/" + fname + "-output-corners.png"), img)
        # plt.show()

    # return drawn chessboard corner images
    return img

directory = './camera_cal/'

for fname in directory:

    img = mpimg.imread(fname)
    calibrate_camera(img)
    # mpimg.imsave((directory + "identified_corners/" + fname + "-output-corners.png"), corners)

'''

'''
def correct_distortion():

    # function to correct image distortion of input images
    pass

def apply_threshold():

    # function to apply appropriate color thresholds to images
    pass

def apply_perspective_transform():

    # applying perspective transforms to images
    pass

'''