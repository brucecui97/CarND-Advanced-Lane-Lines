#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from helper import abs_sobel_thresh, mag_thresh, dir_threshold,hist, find_lane_pixels, fit_polynomial,fit_poly

images = glob.glob(
    "./data/test_images/test6.jpg")
img_read = cv2.cvtColor(cv2.imread(images[0]), cv2.COLOR_BGR2RGB)


dist_coef, dist_mat = np.load("./data/camera_calibration.npy", allow_pickle=True)
persp_transform_mat = np.load("./data/mat_persp_transform.npy", allow_pickle=True)
persp_transform_mat_inv = np.load("./data/mat_persp_transform_back.npy", allow_pickle=True)

img_undistort = cv2.undistort(img_read, dist_mat, dist_coef, None, dist_mat)
img_size = (img_undistort.shape[1], img_undistort.shape[0])

#do operations on image

#first create a better version to extract lines than gray image
hls = cv2.cvtColor(img_undistort, cv2.COLOR_RGB2HLS)
h = hls[:,:,0]
l = hls[:,:,1]
s = hls[:,:,2]


binary_sobelx = abs_sobel_thresh(s, 'x', 20, 100)
binary_sobely = abs_sobel_thresh(s, 'y', 20, 100)
binary_direction = dir_threshold(s, 3, (-np.pi / 2.5, np.pi / 2.5))

criteria1=(binary_sobelx == 1) & (binary_direction == 1) & (l>100)
criteria2=(binary_sobely == 1) & (binary_direction == 1) & (l>100)
criteria3=(h>=140) &(h<=180)& (l>=190)&(l<=240)&(s<=120)
criteria4=(s<=200)

img_binary = np.zeros_like(binary_sobelx)
img_binary[(criteria1 | criteria2 | criteria3) & criteria4] = 1

plt.imshow(img_binary)

img_bird_eye_warped = cv2.warpPerspective(
    img_binary, persp_transform_mat, img_size)

#plt.imshow(img_bird_eye_warped)


histogram = np.sum(
    img_bird_eye_warped[img_bird_eye_warped.shape[0] // 2:, :], axis=0)
# Create an output image to draw on and visualize the result
out_img = np.dstack(
    (img_bird_eye_warped,
     img_bird_eye_warped,
     img_bird_eye_warped)) * 255
# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
midpoint = np.int(histogram.shape[0] // 2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# HYPERPARAMETERS
# Choose the number of sliding windows
nwindows = 9
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50

window_height = np.int(img_bird_eye_warped.shape[0] // nwindows)


nonzero = img_bird_eye_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

leftx_current = leftx_base
rightx_current = rightx_base

left_lane_inds = []
right_lane_inds = []





out_img, left_fit, right_fit, ploty ,left_fitx, right_fitx = fit_polynomial(img_bird_eye_warped)

plt.imshow(out_img)




def search_around_poly(binary_warped):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0] *
                                   (nonzeroy**2) +
                                   left_fit[1] *
                                   nonzeroy +
                                   left_fit[2] -
                                   margin)) & (nonzerox < (left_fit[0] *
                                                           (nonzeroy**2) +
                                                           left_fit[1] *
                                                           nonzeroy +
                                                           left_fit[2] +
                                                           margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] *
                                    (nonzeroy**2) +
                                    right_fit[1] *
                                    nonzeroy +
                                    right_fit[2] -
                                    margin)) & (nonzerox < (right_fit[0] *
                                                            (nonzeroy**2) +
                                                            right_fit[1] *
                                                            nonzeroy +
                                                            right_fit[2] +
                                                            margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(
        binary_warped.shape, leftx, lefty, rightx, righty)

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array(
        [np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array(
        [np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)


    return result, left_fitx, right_fitx, ploty


result, left_fitx, right_fitx, ploty = search_around_poly(img_bird_eye_warped)

def measure_curvature_real():
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the
    # image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix +
                           left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix +
                            right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])

    return left_curverad, right_curverad

def measure_lane_offset():
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    img_num_x_pixels=1280
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    perfect_lane_x_pos_m=img_num_x_pixels*xm_per_pix

    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the
    # image
    left_lane_location_x = np.polyval(left_fit_cr, np.max(ploty)*ym_per_pix)
    right_lane_location_x=np.polyval(right_fit_cr, np.max(ploty)*ym_per_pix)


    return (left_lane_location_x+right_lane_location_x)/2-perfect_lane_x_pos_m/2


left_curverad, right_curverad = measure_curvature_real()

print(left_curverad, 'm', right_curverad, 'm')

#
img_bird_eye_warped_zero = np.zeros_like(img_bird_eye_warped).astype(np.uint8)
color_warp = np.dstack((img_bird_eye_warped_zero, img_bird_eye_warped_zero, img_bird_eye_warped_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, persp_transform_mat_inv, (img_read.shape[1], img_read.shape[0])) 
# Combine the result with the original image
result = cv2.addWeighted(img_undistort, 1, newwarp, 0.3, 0)
plt.imshow(result)

left_fitx_list=np.array([left_fitx,left_fitx,left_fitx])
right_fitx_list=np.array([right_fitx,right_fitx,right_fitx])

def process_image(img_read):
    
    global out_img, left_fit, right_fit, ploty ,left_fitx, right_fitx ,left_fitx_list ,right_fitx_list

    
    img_undistort = cv2.undistort(img_read, dist_mat, dist_coef, None, dist_mat)
    img_size = (img_undistort.shape[1], img_undistort.shape[0])
    
    #do operations on image
    
    #first create a better version to extract lines than gray image
    hls = cv2.cvtColor(img_undistort, cv2.COLOR_RGB2HLS)
    h = hls[:,:,0]
    l = hls[:,:,1]
    s = hls[:,:,2]
    
   
    binary_sobelx = abs_sobel_thresh(s, 'x', 20, 100)
    binary_sobely = abs_sobel_thresh(s, 'y', 20, 100)
    binary_direction = dir_threshold(s, 3, (-np.pi / 2.5, np.pi / 2.5))
    
    criteria1=(binary_sobelx == 1) & (binary_direction == 1) & (l>100)
    criteria2=(binary_sobely == 1) & (binary_direction == 1) & (l>100)
    criteria3=(h>=140) &(h<=180)& (l>=190)&(l<=240)&(s<=120)
    criteria4=(s<=200)
    
    img_binary = np.zeros_like(binary_sobelx)
    img_binary[(criteria1 | criteria2 | criteria3) & criteria4] = 1
    
    #plt.imshow(img_binary)
    
    img_bird_eye_warped = cv2.warpPerspective(
        img_binary, persp_transform_mat, img_size)

    histogram = np.sum(
        img_bird_eye_warped[img_bird_eye_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack(
        (img_bird_eye_warped,
         img_bird_eye_warped,
         img_bird_eye_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    
    window_height = np.int(img_bird_eye_warped.shape[0] // nwindows)
    
    
    nonzero = img_bird_eye_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    left_lane_inds = []
    right_lane_inds = []
    
    try:
        result, left_fitx1, right_fitx1, ploty = search_around_poly(img_bird_eye_warped)
        
        left_fitx_list=np.append(left_fitx_list,np.array([left_fitx1]),axis=0)
        right_fitx_list=np.append(right_fitx_list,np.array([right_fitx1]),axis=0)
        
        left_fitx_list=left_fitx_list[1:,:]
        right_fitx_list=right_fitx_list[1:,:]
    
       
    except:
        pass
    
    left_fitx,right_fitx=(left_fitx_list.mean(axis=0),right_fitx_list.mean(axis=0))
    
    img_bird_eye_warped_zero = np.zeros_like(img_bird_eye_warped).astype(np.uint8)
    color_warp = np.dstack((img_bird_eye_warped_zero, img_bird_eye_warped_zero, img_bird_eye_warped_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, persp_transform_mat_inv, (img_read.shape[1], img_read.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img_undistort, 1, newwarp, 0.3, 0)
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,100)
    fontScale              = 2
    fontColor              = (0,0,0)
    lineType               = 2
    
    curvature_left=int(measure_curvature_real()[0])
    curvature_right=int(measure_curvature_real()[1])
    
    cv2.putText(result,'Curvature (left,right) (m)= ('+str(curvature_left)+','+str(curvature_right)+')', 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    
    
    cv2.putText(result,f'Lane Offset (left-,right+) (m)= {measure_lane_offset():.2f}' , 
        (10,300), 
        font, 
        fontScale,
        fontColor,
        lineType)
    return result

result=process_image(img_read)



plt.imshow(result)




