# USAGE
# python test_grader.py --image images/test_01.png

# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to the input image")
# args = vars(ap.parse_args())

# load the image
image = cv2.imread('images/pad7.jpg')

# TESTING
# rotate the image
# grab the dimensions of the image and calculate the center
# of the image
(h, w) = image.shape[:2]
center = (w / 2, h / 2)

# rotate the image by 180 degrees
M = cv2.getRotationMatrix2D(center, 180, 1.0)
image = cv2.warpAffine(image, M, (w, h))
# cv2.imshow("rotated", image)
# cv2.waitKey(0)

# Save image
# cv2.imwrite('images/pad4_rotated.png',image)

# resize image
# we need to keep in mind aspect ratio so the image does
# not look skewed or distorted -- therefore, we calculate
# the ratio of the new image to the old image
r = 1000.0 / image.shape[1]
dim = (1000, int(image.shape[0] * r))

# perform the actual resizing of the image and show it
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# convert it to grayscale, blur it
# slightly, then find edges
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

# For testing
# cv2.imshow("Resized", edged)
# cv2.waitKey(0)

# find contours in the edge map, then initialize
# the contour that corresponds to the document
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

# ensure that at least one contour was found
if len(cnts) > 0:
    # sort the contours according to their size in
    # descending order
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # loop over the sorted contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points,
        # then we can assume we have found the paper
        if len(approx) == 4:
            docCnt = approx
            break

# apply a four point perspective transform to both the
# original image and grayscale image to obtain a top-down
# birds eye view of the paper
paper = four_point_transform(resized, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))

# For testing
# cv2.imshow("Transformed_gray", warped)
# cv2.imshow("Transformed_color", paper)
# cv2.waitKey(0)

# apply Otsu's thresholding method to binarize the warped
# piece of paper
thresh = cv2.threshold(warped, 0, 255,
                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# For testing
# cv2.imshow("Otsu", thresh)
# cv2.waitKey(0)

# find contours in the thresholded image, then initialize
# the list of contours that correspond to wells
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

new_paper = paper.copy()

# loop over the contours
for c in cnts:
    # Find the contour that is the box (used to rotate paper if necessary)
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # compute the bounding box of the contour, then use the
    # bounding box to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    # if our approximated contour has four points,
    # then we can assume we have found the box
    if len(approx) == 4:
        docCnt = approx
        cv2.drawContours(new_paper, [c], -1, 255, 3)
        # rotate image until box is at bottom of image

        # grab the dimensions of the image and calculate the center
        # of the image
        (h, w) = thresh.shape[:2]
        center = (w / 2, h / 2)

        # box is at top of image
        if center[1] - y > 150:
            # rotate the image by 180 degrees
            M = cv2.getRotationMatrix2D(center, 180, 1.0)
            thresh = cv2.warpAffine(thresh, M, (w, h))
            new_paper = cv2.warpAffine(new_paper, M, (w, h))
            warped = cv2.warpAffine(warped, M, (w, h))

        # box is at right of image
        elif x - center[0] > 150:
            # rotate the image by 90 degrees
            M = cv2.getRotationMatrix2D(center, 270, 1.0)
            thresh = cv2.warpAffine(thresh, M, (w, h))
            new_paper = cv2.warpAffine(new_paper, M, (w, h))
            warped = cv2.warpAffine(warped, M, (w, h))

        # box is at left of image
        elif center[0] - x > 150:
            # rotate the image by 270 degrees
            M = cv2.getRotationMatrix2D(center, 90, 1.0)
            thresh = cv2.warpAffine(thresh, M, (w, h))
            new_paper = cv2.warpAffine(new_paper, M, (w, h))
            warped = cv2.warpAffine(warped, M, (w, h))

        # box is at bottom of image
        elif y - center[1] > 150:
            # don't rotate the image
            continue

        else:
            print('where the heck is the box?')

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
wellCnts = []
# loop over the contours
for c in cnts:
    # Find the contour that is the box (used to rotate paper if necessary)
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # compute the bounding box of the contour, then use the
    # bounding box to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    # If the contour isn't the box then it could be a well
    # in order to label the contour as a well, region
    # should be sufficiently wide, sufficiently tall, and
    # have an aspect ratio approximately equal to 1
    if len(approx) != 4:
        if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
            wellCnts.append(c)
            cv2.drawContours(new_paper, [c], -1, (0, 0, 255), 3)

print('Num circles found:', len(wellCnts))

# For testing
cv2.imshow("Old Paper", paper)
cv2.imshow("New Paper", new_paper)
cv2.waitKey(0)

# Save images
cv2.imwrite('images/pad4_result.png', new_paper)
