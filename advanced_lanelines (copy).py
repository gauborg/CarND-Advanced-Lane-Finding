import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


def cal_undistort(img, object_points, image_points):
    # this function saves undistorted images

    # Use cv2.calibrateCamera() and cv2.undistort()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img.shape[1::-1], None, None)
    undistorted_image = cv2.undistort(img, mtx, dist, None, mtx)

    return undistorted_image



def calibrate_camera(nx, ny):
    # function to calibrate the camera over different chessboard images and input the x and y corners

    # get the list of images
    images = glob.glob('camera_cal/calibration*.jpg')

    # creating arrays to store object and image points from all the images
    objpoints = []      # 3D object points in real space
    imgpoints = []      # 2D points in image space

    # prepare object points
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)    # x, y cor-ordinates


    i = 1
    for fname in images:

        # read file
        img = cv2.imread(fname)

        # convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        print(ret)

        # If found, we will get image coordinates
        if ret == True:
            # add the corners and objectpoints to our lists
            imgpoints.append(corners)
            objpoints.append(objp)

            # draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

            # save the drawn chessboard corners
            mpimg.imsave(("camera_cal_outputs/calibration"+str(i)+"-corners.jpg"), img)

            # calculate and save undistorted image
            undist = cal_undistort(img, objpoints, imgpoints)
            mpimg.imsave(("camera_cal_outputs/calibration"+str(i)+"-undistorted.jpg"), undist)
        
        i+=1

    print(len(imgpoints))


calibrate_camera(9,6)






'''
# arrays to store image and object points
imgpoints = []
objpoints = []

# prepare object points like (0,0,0), (1,0,0) ... (7,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)    # x, y cor-ordinates

fname = "camera_cal/calibration3.jpg"
# read file
img = cv2.imread(fname)

# convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
print(ret)

# iterate through the images
i = 1

# If found, draw corners
if ret == True:

    imgpoints.append(corners)
    objpoints.append(objp)

    # draw and display the corners
    img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
    mpimg.imsave(("camera_cal_outputs/calibration-corners.jpg"), img)


print(len(imgpoints))





def calibrate_camera(nx, ny):
    # function to calibrate the camera over different chessboard images and input the x and y corners

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
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        print(ret)

        # If found, we will get image coordinates
        if ret == True:
            # add the corners and objectpoints to our lists
            imgpoints.append(corners)
            objpoints.append(objp)

            # draw and display the corners
            img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)

            # save the drawn chessboard corners
            mpimg.imsave(("camera_cal/identified_corners/calibration-corners"+str(i)+".jpg"), img)

            # perform camera calibration, given object points, image points, and the shape of the grayscale image
            # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

            # plt.imshow(img)
            # plt.show()
        
        i+=1
    # print(len(imgpoints))



calibrate_camera(9, 6)

'''