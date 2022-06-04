# loop over each of the individual class IDs in the image
for classID in np.unique(classMap):
	# build a binary mask for the current class and then use the mask
	# to visualize all pixels in the image belonging to the class
	print("[INFO] class: {}".format(CLASSES[classID]))
	classMask = (mask == COLORS[classID]).astype("uint8") * 255
	classMask = classMask[:, :, 0]
	classOutput = cv2.bitwise_and(image, image, mask=classMask)
	classMask = np.hstack([image, classOutput])


	# show the output class visualization
	cv2.imshow("Class Vis", classMask)
	cv2.waitKey(0)