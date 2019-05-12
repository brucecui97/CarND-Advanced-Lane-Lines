#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from gradient_threshold import abs_sobel_thresh, mag_thresh, dir_threshold

images = glob.glob(
    "/home/bruce/Education/Udacity/CarND-Advanced-Lane-Lines/test_images/*.jpg")
img_read = cv2.cvtColor(cv2.imread(images[7]), cv2.COLOR_BGR2RGB)

dist_coef, dist_mat = np.load("camera_calibration.npy", allow_pickle=True)
persp_transform_mat = np.load("mat_persp_transform.npy", allow_pickle=True)
persp_transform_mat_inv = np.load("mat_persp_transform_back.npy", allow_pickle=True)

img_undistort = cv2.undistort(img_read, dist_mat, dist_coef, None, dist_mat)
img_size = (img_undistort.shape[1], img_undistort.shape[0])

#do operations on image

#first create a better version to extract lines than gray image
hls = cv2.cvtColor(img_undistort, cv2.COLOR_RGB2HLS)
h = hls[:,:,0]
l = hls[:,:,1]
s = hls[:,:,2]

# =============================================================================
# f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
# ax1.set_title('Standardized image')
# ax1.imshow(img_undistort)
# ax2.set_title('H channel')
# ax2.imshow(h, cmap='gray')
# ax3.set_title('L channel')
# ax3.imshow(l, cmap='gray')
# ax4.set_title('S channel')
# ax4.imshow(s, cmap='gray')
#
# =============================================================================

binary_sobelx = abs_sobel_thresh(s, 'x', 30, 180)
binary_sobely = abs_sobel_thresh(s, 'y', 20, 100)
binary_direction = dir_threshold(s, 3, (-np.pi / 2.5, np.pi / 2.5))

img_binary = np.zeros_like(binary_sobelx)
img_binary[(binary_sobelx == 1) & (binary_direction == 1) | ( (binary_sobelx == 1) & (l>200))] = 1

#plt.imshow(img_binary)

img_bird_eye_warped = cv2.warpPerspective(
    img_binary, persp_transform_mat, img_size)

#plt.imshow(img_bird_eye_warped)

def hist(img):
    # Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0] // 2:, :]

    # Sum across image pixels vertically - make sure to set an `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)

    return histogram


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


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
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

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
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
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = (
            (nonzeroy >= win_y_low) & (
                nonzeroy < win_y_high) & (
                nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = (
            (nonzeroy >= win_y_low) & (
                nonzeroy < win_y_high) & (
                nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean
        # position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of
    # pixels)
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

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + \
            right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty**2 + 1 * ploty
        right_fitx = 1 * ploty**2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
# =============================================================================
#     plt.plot(left_fitx, ploty, color='yellow')
#     plt.plot(right_fitx, ploty, color='yellow')
# =============================================================================

    return out_img, left_fit, right_fit, ploty


out_img, left_fit, right_fit, ploty = fit_polynomial(img_bird_eye_warped)

plt.imshow(out_img)


def fit_poly(img_shape, leftx, lefty, rightx, righty):
     ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx, ploty


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

    # Plot the polynomial lines onto the image
# =============================================================================
#     plt.plot(left_fitx, ploty, color='yellow')
#     plt.plot(right_fitx, ploty, color='yellow')
# =============================================================================
    ## End visualization steps ##

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
newwarp = cv2.warpPerspective(color_warp, persp_transform_mat_inv, (img.shape[1], img.shape[0])) 
# Combine the result with the original image
result = cv2.addWeighted(img_undistort, 1, newwarp, 0.3, 0)
plt.imshow(result)

# =============================================================================
# b=np.dstack((img_bird_eye_warped,img_bird_eye_warped,img_bird_eye_warped))
#
# img_bird_eye_warped[ploty_pix,left_pix]=[0]
# =============================================================================