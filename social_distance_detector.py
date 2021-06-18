# import the necessary packages
import config   # the file config.py
from people_detection import P_detector
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2

def SD_detector(net, ln, personIdx, frame):
    # resize the frame and then detect people (and only people) in it
    frame = imutils.resize(frame, width=1000)
    # results = list of [confidence, bounding box coordinates, centeroid]
    results = P_detector(frame, net, ln, personIdx)

    # initialize the set of indexes that violate the minimum social distance
    violate = set()
    # initalize the list of images of social distance violators
    images = []

    # ensure there are at least two people detections (required in
    # order to compute our pairwise distance maps)
    if len(results) >= 2:
        # extract all centroids from the results and compute the
        # Euclidean distances between all pairs of the centroids
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        # loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                # check to see if the distance between any two
                # centroid pairs is less than the configured number of pixels
                if D[i, j] < config.MIN_DISTANCE:
                    # update the violation set with the indexes of the centroid pairs
                    violate.add(i)
                    violate.add(j)

    # loop over the results
    for (i, (prob, bbox, centroid)) in enumerate(results):
        # extract the bounding box and centroid coordinates, then initialize the color
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        # if the index pair exists within the violation set, then update the color and
        if i in violate:
            color = (0, 0, 255)
            # cut the pictures of social distance violators from the frame and add it to the list
            images.append(frame[startY:endY, startX:endX])

        # draw bounding box around the person and centroid on the person
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)

    # draw the total number of social distancing violations on the output frame
    text = "Social Distancing Violations: {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0] - 25),
    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

    return [frame, images]
