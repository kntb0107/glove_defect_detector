
# LIBRARIES
import cv2
import pandas as pd
import numpy as np
import imutils
import mediapipe as mp
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from google.protobuf.json_format import MessageToDict

# BUILT-IN WEBCAM INITIALISATION
webcam = cv2.VideoCapture(0)
# IP CAMERA
# webcam=cv2.VideoCapture("http://192.168.1.102:6677")

# MODEL INITIALISATION
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)

# TRACKBAR INITIALISATION
def tbar(x):
    pass

# ERROR HANDLING
def safe_div(x, y):
    if y == 0: return 0
    return x / y

if not webcam.isOpened():
    print("THE VIDEO CAMERA CANNOT BE ACCESSED. PLEASE TRY AGAIN.")
    exit()

# RESIZING THE LIVE FEED WINDOW
def resize_window(frame, percent=80):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

livefeed_name = "GLOVE DEFECT DETECTION & MEASUREMENT"

cv2.namedWindow(livefeed_name)

# SLIDERS IN WINDOWS - ADJUSTING ACCORDING TO THE VIDEO FEED
cv2.createTrackbar("THRESHOLD", livefeed_name, 75, 255, tbar)
cv2.createTrackbar("KERNAL", livefeed_name, 5, 30, tbar)
cv2.createTrackbar("ITERATIONS", livefeed_name, 1, 10, tbar)


is_live = True
while (is_live):

    ret, frame = webcam.read()
    if not ret:
        print("VIDEO FRAME CANNOT BE CAPTURED. PLEASE TRY AGAIN.")
        exit()

    feed = cv2.flip(frame, 1)
    # ORIGINAL VIDEO FEED
    window_resize = resize_window(feed)

    # BGR --> RGB
    feed_rgb = cv2.cvtColor(window_resize, cv2.COLOR_BGR2RGB)
    # PROCESSING THE IMAGE
    processed_feed = hands.process(feed_rgb)

    # IF HANDS ARE PRESENT ON VIEWFINDER
    if processed_feed.multi_hand_landmarks:

        # BOTH HANDS ARE PRESENT
        if len(processed_feed.multi_handedness) == 2:
            cv2.putText(window_resize, 'PLEASE PUT UP ONE HAND ONLY.', (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (93, 173, 226), 2)

        # ONE OF THE HANDS ARE PRESENT
        else:
            for i in processed_feed.multi_handedness:

                hand_orientation = MessageToDict(i)[
                    'classification'][0]['label']

                if hand_orientation == 'Left':
                    # DISPLAY 'LEFT' HAND
                    cv2.putText(window_resize, 'ORIENTATION: LEFT', (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (93, 173, 226), 2)

                if hand_orientation == 'Right':
                    # DISPLAY 'RIGHT' HAND
                    cv2.putText(window_resize, 'ORIENTATION: RIGHT',(20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (93, 173, 226), 2)


    th = cv2.getTrackbarPos("THRESHOLD", livefeed_name)
    ret, th1 = cv2.threshold(window_resize, th, 255, cv2.THRESH_BINARY)

    k = cv2.getTrackbarPos("KERNAL", livefeed_name)
    k1 = np.ones((k, k), np.uint8)  # square image kernel used for erosion

    itr = cv2.getTrackbarPos("ITERATIONS", livefeed_name)
    feed_dilation = cv2.dilate(th1, k1, iterations=itr)
    feed_erosion = cv2.erode(feed_dilation, k1, iterations=itr)  # refines all edges in the binary image

    feed_opening = cv2.morphologyEx(feed_erosion, cv2.MORPH_OPEN, k1)
    feed_closing = cv2.morphologyEx(feed_opening, cv2.MORPH_CLOSE, k1)
    feed_closing = cv2.cvtColor(feed_closing, cv2.COLOR_BGR2GRAY)

    # SEARCH AND FIND THE CONTOURS IN THE IMAGE
    contours, hierarchy = cv2.findContours(feed_closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    feed_closing = cv2.cvtColor(feed_closing, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(feed_closing, contours, -1, (128, 255, 0), 1)

    # focus on only the largest outline by area
    list_areas = []  # list to hold all areas

    for contour in contours:
        a = cv2.contourArea(contour)
        list_areas.append(a)

    max_area = max(list_areas)
    max_area_index = list_areas.index(max_area)  # index of the list element with largest area

    c = contours[max_area_index - 1]  # largest area contour is usually the viewing window itself, why?

    cv2.drawContours(feed_closing, [c], 0, (0, 0, 255), 1)

    # COMPUTING ROTATED BOUNDING BOX OF THE CONTOUR
    original_feed = window_resize.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # ORDER OF THE POINTS IN THE CONTOUR
    # DIRECTION: TOP: LEFT AND RIGHT, AND BOTTOM: LEFT AND RIGHT
    # DRAW OUTLINE
    # BOX
    box = perspective.order_points(box)
    cv2.drawContours(original_feed, [box.astype("int")], -1, (0, 255, 0), 1)

    # LOOP OVER ORI POINTS AND DRAW BOX
    for (x, y) in box:
        cv2.circle(original_feed, (int(x), int(y)), 5, (0, 0, 255), -1)


    # UNPACK THE ORDERED BOUNDING BOX
    (tl, tr, br, bl) = box
    # COMPUTE MIDPOINT
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # DRAW MIDPOINTS
    cv2.circle(original_feed, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(original_feed, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(original_feed, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(original_feed, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # CONNET THE MIDPOINT TOWARDS THE EXTREMITIES
    cv2.line(original_feed, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 1)
    cv2.line(original_feed, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 1)
    cv2.drawContours(original_feed, [c], 0, (0, 0, 255), 1)

    # COMPUTE EUCLIDEAN DIST BETWEEN THE MIDPOINTS
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # COMPUTE SIZE OF OBJECT compute the size of the object
    pixelsPerMetric = 1  # more to do here to get actual measurements that have meaning in the real world
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    # DRAWING OBJ SIZE
    cv2.putText(original_feed, "{:.1f}mm".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2)
    cv2.putText(original_feed, "{:.1f}mm".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2)

    # COMPUTE CENTER OF CONTOUR
    M = cv2.moments(c)
    cX = int(safe_div(M["m10"], M["m00"]))
    cY = int(safe_div(M["m01"], M["m00"]))

    # DRAW CONTOUR AND CENTER
    cv2.circle(original_feed, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(original_feed, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow(livefeed_name, original_feed)
    cv2.imshow('', feed_closing)
    if cv2.waitKey(30) >= 0:
        showLive = False

webcam.release()
cv2.destroyAllWindows()