import numpy as np
import os
import glob
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import time
import datetime

# class to store the characteristics of every laneline
class LaneLines():

    # constructor
    def __init__(self, binary_warped, prev_left_fit, prev_right_fit, previous_detection, prev_avg_left_fit, prev_avg_right_fit):
        # incoming binary image
        self.binary_warped = binary_warped
        # was the line detected in the last iteration?
        self.detected = previous_detection
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # creating the array for binary
        self.ploty = np.linspace(0, self.binary_warped.shape[0]-1, self.binary_warped.shape[0])
        # no of windows
        # previous left and right fits which worked
        self.left_fit = prev_left_fit
        self.right_fit = prev_right_fit
        # average of past 10 fits
        self.avg_left_fit = prev_avg_left_fit
        self.avg_right_fit = prev_avg_right_fit


    # function for detecting lanelines manually using the sliding windows approach
    def sliding_windows(self):

        # This part creates the histogram if lanelines are not detected in the previous iteration
        # Take a histogram of the bottom half of the image
        histogram = np.sum(self.binary_warped[self.binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((self.binary_warped, self.binary_warped, self.binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 10
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(self.binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self.binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.binary_warped.shape[0] - (window+1)*window_height
            win_y_high = self.binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # VISUALIZATION
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
                    
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
                    
            # If we find > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        print("leftx size", leftx.size)
        print("rightx size", rightx.size)

        '''
        # for first frame, we assume that we will find some pixels...
        # we check if either avg_left_fit or avg_right_fit lists are empty
        '''

        # only for first frame, when the average left and right fit lists are empty
        if((len(self.avg_left_fit) == 0) or (len(self.avg_right_fit) == 0)):
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)
            print(self.left_fit)
            print(self.right_fit)

        if((len(self.avg_left_fit) != 0) or (len(self.avg_right_fit) != 0)):
            if ((leftx.size < 50) or (lefty.size < 50)):
                temp_l_fit = self.avg_left_fit
                self.left_fit = temp_l_fit
                # print("avg left fit = ", self.left_fit)
                self.right_fit = np.polyfit(righty, rightx, 2)
                print('Reverting to average of previous estimates for left lane')
            
            # if right laneline is not detected ...
            elif ((rightx.size < 50) or (righty.size < 50)):
                
                # If a laneline is still not detected in the current iteration, we compute
                # right lane equation using the left laneline equation
                
                print('Reverting to average of previous estimates for right lane ...')

                # calculate current left fit
                current_left_fit = np.polyfit(lefty, leftx, 2)
                print("current calculated left fit = ", current_left_fit)

                # compute an offset fit from current left fit
                offset_r_fit = current_left_fit

                print("temp_r_fit[2] = ", offset_r_fit[2])
                offset_r_fit[2] = offset_r_fit[2] + 700.0
                print("new temp_r_fit[2] = ", offset_r_fit[2])

                print("offset r fit = ", offset_r_fit)
                # print("previous frame right fit = ", prev_estimate)
                # print("prev average right fit = ", prev_avg)

                # use 80% of offset fit and 20% of previous average fit
                self.right_fit = offset_r_fit
                # self.left_fit = current_left_fit    # DOESN'T SEEM TO WORK, GRABS THE VALUE OF offset_r_fit
                self.left_fit = np.polyfit(lefty, leftx, 2)

                print("self.left_fit in function = ", self.left_fit)

            else:
                self.left_fit = np.polyfit(lefty, leftx, 2)
                self.right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        self.left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]

        '''
        ## Visualization ##
        ## Uncommment only when running on test images"
        
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        # Plots the left and right polynomials on the lane lines
        plt.plot(self.left_fitx, self.ploty, color='yellow')
        plt.plot(self.right_fitx, self.ploty, color='yellow')
        plt.imshow(out_img)
        plt.show()
        '''
        
        return out_img, self.left_fit, self.right_fit

    
    # for searching from a prior region
    def search_from_prior(self):

        out_img = np.dstack((self.binary_warped, self.binary_warped, self.binary_warped))

        # HYPERPARAMETER
        search_margin = 100

        # Grab activated pixels
        nonzero = self.binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Here we set the area of search based on activated x-values within the +/- margin of our polynomial function

        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + 
                        self.left_fit[2] - search_margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + 
                        self.left_fit[1]*nonzeroy + self.left_fit[2] + search_margin)))
        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + 
                        self.right_fit[2] - search_margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + 
                        self.right_fit[1]*nonzeroy + self.right_fit[2] + search_margin)))
        
        # Again, extract left and right line pixel positions

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        
        print("prior section - leftx size: ", leftx.size)
        print("prior section - rightx size: ", rightx.size)

        # check if the arrays are empty, i.e. no pixels are detected
        if ((leftx.size == 0) | (lefty.size == 0)):
            detection_in_current = False
        elif ((rightx.size == 0) | (righty.size == 0)):
            detection_in_current = False
        else:
            detection_in_current = True
        
        # if above condition is true, then we calculate lanelines based on above x and y, else we execute function sliding widows
        if (detection_in_current):
            # print("detection in current = ", detection_in_current)
            
            # calculate current fits if lanelines are detected
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)

            try:
                self.left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
                self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
            except IndexError:
                # Avoids an error if `left` and `right_fit` are still none or incorrect
                print('The function failed to fit a line!')
                ## self.left_fitx = 1*self.ploty**2 + 1*self.ploty
                ## self.right_fitx = 1*self.ploty**2 + 1*self.ploty

        else:
            # if no lanelines are found using search from prior option, use sliding windows functionality
            print("sliding windows was called from search prior...")
            out_img, self.left_fit, self.right_fit = self.sliding_windows()
        
        
        ## Visualization ##
        # IMPORTANT - This should be commented out for the video section, else it will show an image for every frame of the video
        '''
        # Colors in the left and right lane regions
        out_img = np.dstack((self.binary_warped, self.binary_warped, self.binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([self.left_fitx-search_margin, self.ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx+search_margin, 
                                self.ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([self.right_fitx-search_margin, self.ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx+search_margin, 
                                self.ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.figure(1)
        plt.imshow(out_img)
        plt.show()
        '''
        self.detected = detection_in_current

        return out_img, self.left_fit, self.right_fit


    def find_lane_pixels(self):

        out_img = np.dstack((self.binary_warped, self.binary_warped, self.binary_warped))

        if (self.detected):
            out_img, self.left_fit, self.right_fit = self.search_from_prior()
            # print("Search from prior from find_pixels executed!")
        else:
            out_img, self.left_fit, self.right_fit = self.sliding_windows()
            self.detected = True
            print("Sliding window executed!")

        return out_img, self.left_fit, self.right_fit, self.detected


    def measure_curvature_pixels(self):
        
        # Calculates the curvature of polynomial functions in pixels.
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.ploty)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        left_fit_cr = np.polyfit(self.ploty*ym_per_pix, self.left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.ploty*ym_per_pix, self.right_fitx*xm_per_pix, 2)
        
        ##### TO-DO: Implement the calculation of R_curve in pixels (radius of curvature) #####
        left_curverad = (1 + ((2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])  ## Implement the calculation of the left line here
        right_curverad = (1 + ((2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        lane_center = (self.right_fitx[self.binary_warped.shape[0]-1]-self.left_fitx[self.binary_warped.shape[0]-1])/2
        offset_in_pixels = abs(lane_center - (self.binary_warped.shape[0]/2))
        offset = offset_in_pixels * xm_per_pix

        return offset, left_curverad, right_curverad


    # function to remove outliers from an array
    def rmv_outliers(original_array, no_of_std_deviations):

        # calculate average of all elements
        mean = np.mean(original_array)

        # get the standard deviation
        std_deviation = np.std(original_array)

        # calculate standard deviation of all elements
        distance_from_mean = abs(original_array - mean)

        # no of standard deviations allowed
        no_of_std_deviations = 2

        # not outliers
        not_outlier = distance_from_mean < max_deviations * std_deviation

        # new array without the outliers
        new_array = an_array[not_outlier]

        return new_array

