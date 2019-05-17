import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


images = glob.glob(
    "/home/bruce/Education/Udacity/CarND-Advanced-Lane-Lines/camera_cal/calibration*.jpg")

objpoints = []
imgpoints = []
objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

for fname in images:
    img = cv2.imread(fname)
    # plt.imshow(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)

    # plt.imshow(img)

    if ret:
        imgpoints.append(corners)
        objpoints.append(objp)
        print("entered")
    print("ran")

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

undistored = cv2.undistort(img, mtx, dist, None, mtx)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('undistored')
ax1.imshow(img_undistort)
ax2.set_title('original distorted')
ax2.imshow(img_read, cmap='gray')


np.save("camera_calibration", [dist, mtx])
