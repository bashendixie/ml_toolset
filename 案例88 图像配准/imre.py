import numpy as np
import imutils
import cv2
#from google.colab.patches import cv2_imshow

img_align = cv2.imread("image-1.jpeg")
img_temp = cv2.imread("image.jpeg")
img1 = cv2.cvtColor(img_align, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
height, width = img2.shape

orb_detector = cv2.ORB_create(5000)
kp1, d1 = orb_detector.detectAndCompute(img1, None)
kp2, d2 = orb_detector.detectAndCompute(img2, None)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matches = matcher.match(d1, d2)

#matches.sorted(key = lambda x: x.distance)
matches = matches[:int(len(matches)*0.9)]
no_of_matches = len(matches)
p1 = np.zeros((no_of_matches, 2))
p2 = np.zeros((no_of_matches, 2))
for i in range(len(matches)):
  p1[i, :] = kp1[matches[i].queryIdx].pt
  p2[i, :] = kp2[matches[i].trainIdx].pt
homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

transformed_img = cv2.warpPerspective(img_align, homography, (width, height))
matchedVis =cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
matchedVis = imutils.resize(matchedVis, width=1000)
cv2.imwrite('C:\\Users\\zyh\\Desktop\\123.jpg', matchedVis)
cv2.waitKey(0)