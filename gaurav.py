import os
import numpy as np
# Thresholding functions
# since we have evaludated earlier that HLS gives good image filtering results

def s_select(img, thresh=(100, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    s_channel = hls[:,:,2]
    # 3) Return a binary image of threshold result
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
#     binary_output = np.copy(img) # placeholder line
    return binary_output

def l_select(img, thresh=(120, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    l_channel = hls[:,:,1]
    # 3) Return a binary image of threshold result
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
#     binary_output = np.copy(img) # placeholder line
    return binary_output

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img,cv2.CV_64F,1,0))
    else:
        abs_sobel = np.absolute(cv2.Sobel(img,cv2.CV_64F,0,1))
        
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    sobelx = cv2.Sobel(img,cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(img,cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag>=mag_thresh[0]) & (gradmag<=mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    sobelx = cv2.Sobel(img,cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(img,cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    absgraddir = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary









def comb_threshold(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    s_channel = hls[:,:,2]
    l_channel = hls[:,:,1]
    
    # img to s threshold
    s_binary = s_select(img,thresh=(100, 255))
    
    # img to l threshold
    l_binary = l_select(img,thresh=(120, 255))   
    
    # schannel to x threshold
    ksize = 9  # Choose a larger odd number to smooth gradient measurements
    gradx = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    
    # combined
    combined = np.zeros_like(s_binary)
    combined[((gradx == 1) | (s_binary == 1)) & ((l_binary == 1) & (s_binary == 1))] = 1

    # apply the region of interest mask
    mask = np.zeros_like(combined)
    region_of_interest_vertices = np.array([[0,height-1], [width/2, int(0.5*height)], [width-1, height-1]], dtype=np.int32)
    cv2.fillPoly(mask, [region_of_interest_vertices], 1)
    thresholded = cv2.bitwise_and(combined, mask)


    mpimg.imsave(('output_images/test_images_l_binary/l_binary-' + i), l_binary)
    mpimg.imsave(('output_images/test_images_s_binary/s_binary-' + i), s_binary)
    mpimg.imsave(('output_images/test_images_gradx_binary/gradx-' + i), gradx)
    mpimg.imsave(('output_images/test_images_thresholded/comb-' + i), thresholded)

    
    return thresholded



# Let us test our functions on given test images
directory = os.listdir("test_images/")
# print(directory)


for i in directory:

    img = mpimg.imread(os.path.join("test_images/",i))

    thresholded = comb_threshold(img)
    # undist = cv2.undistort(thresholded, mtx, dist, None, mtx)
    # mpimg.imsave(('output_images/test_images_output/thresholded-', i),undist)
    #warped,img_pts,presp_M,persp_M_inv = persp_view(undist)
    #mpimg.imsave(os.path.join("output_images/warped/",i),warped)