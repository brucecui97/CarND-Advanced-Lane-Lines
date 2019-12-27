import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

images=glob.glob("/home/bruce/Education/Udacity/CarND-Advanced-Lane-Lines/frame0.jpg")
img=cv2.cvtColor(cv2.imread(images[0]),cv2.COLOR_BGR2RGB)
img_size=(img.shape[1],img.shape[0])

dist_coef,dist_mat=np.load("camera_calibration.npy",allow_pickle=True)
img=cv2.undistort(img,dist_mat,dist_coef,None,dist_mat)

src=np.array([[250,685],[566,460],[755,460],[1110,685]],np.float32)

plt.imshow(img)
plt.plot(src[0][0],src[0][1],'.')
plt.plot(src[1][0],src[1][1],'.')
plt.plot(src[2][0],src[2][1],'.')
plt.plot(src[3][0],src[3][1],'.')

offset=40
dst=np.array([
        [offset,img_size[1]-offset],
        [offset,offset],
        [img_size[0]-offset,offset],
        [img_size[0]-offset,img_size[1]-offset]
        ], np.float32)

mat_persp_transform = cv2.getPerspectiveTransform(src, dst)
mat_persp_transform_back = cv2.getPerspectiveTransform(dst, src)

warped = cv2.warpPerspective(img, mat_persp_transform, img_size)
#plt.imshow(warped)

np.save("mat_persp_transform",mat_persp_transform)
np.save("mat_persp_transform_back",mat_persp_transform_back)



