import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from CarND_Advanced_Lane_Lines.core import process_image

images = glob.glob(
    "./data/test_images/test6.jpg")
img_read = cv2.cvtColor(cv2.imread(images[0]), cv2.COLOR_BGR2RGB)

result=process_image(img_read)
plt.imshow(result)
plt.show()