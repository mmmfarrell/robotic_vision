import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread("/home/michael/gradschool/Winter18/robotic_vision/datasets/dataset/sequences/00/image_0/000000.png")
imgR = cv2.imread("/home/michael/gradschool/Winter18/robotic_vision/datasets/dataset/sequences/00/image_1/000000.png")
# imgR = cv2.imread("/home/michael/gradschool/Winter18/robotic_vision/datasets/dataset/sequences/00/image_0/000004.png")

imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

stereo = cv2.StereoBM_create()
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity, 'gray')
# plt.imshow(imgL, 'gray')
plt.show()
