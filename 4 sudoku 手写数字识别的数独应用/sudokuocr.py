# import the necessary packages
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sudoku import Sudoku
import tensorflow as tf


def find_puzzle(image, debug=False):
    # convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    # apply adaptive thresholding and then invert the threshold map
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    # check to see if we are visualizing each step of the image
    # processing pipeline (in this case, thresholding)
    if debug:
        cv2.imshow("Puzzle Thresh", thresh)
        cv2.waitKey(0)

    # find contours in the thresholded image and sort them by size in
    # descending order
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # initialize a contour that corresponds to the puzzle outline
    puzzleCnt = None
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we can
        # assume we have found the outline of the puzzle
        if len(approx) == 4:
            puzzleCnt = approx
            break

    # if the puzzle contour is empty then our script could not find
    # the outline of the Sudoku puzzle so raise an error
    if puzzleCnt is None:
        raise Exception(("Could not find Sudoku puzzle outline. "
                         "Try debugging your thresholding and contour steps."))
    # check to see if we are visualizing the outline of the detected
    # Sudoku puzzle
    if debug:
        # draw the contour of the puzzle on the image and then display
        # it to our screen for visualization/debugging purposes
        output = image.copy()
        cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline", output)
        cv2.waitKey(0)

    # apply a four point perspective transform to both the original
    # image and grayscale image to obtain a top-down bird's eye view
    # of the puzzle
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))
    # check to see if we are visualizing the perspective transform
    if debug:
        # show the output warped image (again, for debugging purposes)
        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)
    # return a 2-tuple of puzzle in both RGB and grayscale
    return (puzzle, warped)



def extract_digit(cell, debug=False):
    # apply automatic thresholding to the cell and then clear any
    # connected borders that touch the border of the cell
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    # check to see if we are visualizing the cell thresholding step
    if debug:
        cv2.imshow("Cell Thresh", thresh)
        cv2.waitKey(0)

    # find contours in the thresholded cell
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # if no contours were found than this is an empty cell
    if len(cnts) == 0:
        return None
    # otherwise, find the largest contour in the cell and create a
    # mask for the contour
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    # compute the percentage of masked pixels relative to the total
    # area of the image
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    # if less than 3% of the mask is filled then we are looking at
    # noise and can safely ignore the contour
    if percentFilled < 0.03:
        return None
    # apply the mask to the thresholded cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    # check to see if we should visualize the masking step
    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)
    # return the digit to the calling function
    return digit


def loadmodelandocr():
    # load the digit classifier from disk
    #model = load_model('D:\\Project\\DeepLearn\\2trainlog\\handwriting\\mnist.h5')
    model = load_model('C:\\Users\\zyh\\Desktop\\mnist.h5')
    # load the input image from disk and resize it
    image = cv2.imread('C:\\Users\\zyh\\Desktop\\sudoku.jpg')
    image = imutils.resize(image, width=600)
    # find the puzzle in the image and then
    (puzzleImage, warped) = find_puzzle(image, False)
    # initialize our 9x9 Sudoku board
    board = np.zeros((9, 9), dtype="int")
    # a Sudoku puzzle is a 9x9 grid (81 individual cells), so we can
    # infer the location of each cell by dividing the warped image
    # into a 9x9 grid
    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9
    # initialize a list to store the (x, y)-coordinates of each cell
    # location
    cellLocs = []
    na = 0;
    # loop over the grid locations
    for y in range(0, 9):
        # initialize the current list of cell locations
        row = []
        for x in range(0, 9):
            # compute the starting and ending (x, y)-coordinates of the
            # current cell
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY
            # add the (x, y)-coordinates to our cell locations list
            row.append((startX, startY, endX, endY))
            # crop the cell from the warped transform image and then
            # extract the digit from the cell
            cell = warped[startY:endY, startX:endX]
            digit = extract_digit(cell, False)
            # verify that the digit is not empty
            if digit is not None:
                #cv2.imwrite('C:/Users/zyh/Desktop/1/' +str(na)+'.jpg', digit)
                #na += 1
                # resize the cell to 28x28 pixels and then prepare the
                # cell for classification
                roi = cv2.resize(digit, (28, 28))
                #roi = img_to_array(roi)
                #roi = roi.astype("float32")
                roi = tf.reshape(roi, (1, 28 * 28))
                # roi = img_to_array(roi)
                # roi = np.expand_dims(roi, axis=0)
                # classify the digit and update the Sudoku board with the
                # prediction
                pred = model.predict(roi).argmax(axis=1)[0]
                board[y, x] = pred
            # add the row to our cell locations
        cellLocs.append(row)

    # construct a Sudoku puzzle from the board
    print("[INFO] OCR'd Sudoku board:")
    puzzle = Sudoku(3, 3, board=board.tolist())
    puzzle.show()
    # solve the Sudoku puzzle
    print("[INFO] solving Sudoku puzzle...")
    solution = puzzle.solve()
    solution.show_full()

    # loop over the cell locations and board
    for (cellRow, boardRow) in zip(cellLocs, solution.board):
        # loop over individual cell in the row
        for (box, digit) in zip(cellRow, boardRow):
            # unpack the cell coordinates
            startX, startY, endX, endY = box
            # compute the coordinates of where the digit will be drawn
            # on the output puzzle image
            textX = int((endX - startX) * 0.33)
            textY = int((endY - startY) * -0.2)
            textX += startX
            textY += endY
            # draw the result digit on the Sudoku puzzle image
            cv2.putText(puzzleImage, str(digit), (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    # show the output image
    cv2.imshow("Sudoku Result", puzzleImage)
    cv2.waitKey(0)



loadmodelandocr()








