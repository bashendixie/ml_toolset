import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("D:\\Project\\ip_opencvsharp\\SmartCoalApplication\\bin\\x64\\Debug\\pictures\\house.jpg").astype(np.float)

# Display histogram
plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("out.png")
plt.show()