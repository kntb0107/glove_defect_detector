# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# LIBRARIES - DETECT.PY
import argparse
import os
import sys
import seaborn
import torch
import torchvision
import torch.backends.cudnn as cudnn
from pathlib import Path

# IMPORTED LIBRARIES - YOLO DIRECTORY
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


# LIBRARIES - MEASUREMENT
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


# MODEL INITIALISATION
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)

# INITIALISATION 
@torch.no_grad()
def run(
        weights=ROOT / 'E:/Study/Intake Jan 2022/MVI - Machine Mission and Intelligence/Assignment MVI/yolov5/content/yolov5/runs/train/yolov5s_results/weights/best.pt',  # model.pt path(s)
        source=ROOT / '0',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'E:/Study/Intake Jan 2022/MVI - Machine Mission and Intelligence/Assignment MVI/yolov5/content/yolov5/glove-defect-measurement-11/data.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # DIRECTORIES
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # LOAD MODEL
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    myvideo = cv2.VideoCapture(0)
    ret, myframe = myvideo.read()
    # DATALOADER
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    livefeed_name = "GLOVE DEFECT DETECTION & MEASUREMENT"
    cv2.namedWindow(livefeed_name)

    # SLIDERS IN WINDOWS - ADJUSTING ACCORDING TO THE VIDEO FEED
    cv2.createTrackbar("THRESHOLD", livefeed_name, 61, 255, tbar)
    cv2.createTrackbar("KERNAL", livefeed_name, 5, 27, tbar)
    cv2.createTrackbar("ITERATIONS", livefeed_name, 0, 10, tbar)

    # RUN INFERENCE
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    #ret, myframe = myvideo.read()
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # EXPAND FOR BATCH DIMENTIONS
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3


        # PROCESS PREDICTIONS 
        for i, det in enumerate(pred):  # per image

            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            feed = im0.copy()
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
                                (170, 51, 106), 2)

                # ONE OF THE HANDS ARE PRESENT
                else:
                    for i in processed_feed.multi_handedness:

                        hand_orientation = MessageToDict(i)[
                            'classification'][0]['label']

                        if hand_orientation == 'Left':
                            # DISPLAY 'LEFT' HAND
                            cv2.putText(window_resize, 'ORIENTATION: LEFT', (20, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (170, 51, 106), 2)

                        if hand_orientation == 'Right':
                            # DISPLAY 'RIGHT' HAND
                            cv2.putText(window_resize, 'ORIENTATION: RIGHT', (20, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (170, 51, 106), 2)

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

            # FOCUS ON ONLY THE LARGEST OUTLINE BY AREA
            list_areas = []  # HOLD ALL AREAS

            for contour in contours:
                a = cv2.contourArea(contour)
                list_areas.append(a)

            max_area = max(list_areas)
            max_area_index = list_areas.index(max_area)  # INDEX OF THE LARGEST AREA

            c = contours[max_area_index - 1]  

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
            cv2.putText(original_feed, "{:.1f}Width mm".format(dimA), (int(tltrX - 100), int(tltrY - 100)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255,165,0), 2)
            cv2.putText(original_feed, "{:.1f}Height mm".format(dimB), (int(trbrX + 100), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255,165,0), 2)

            # COMPUTE CENTER OF CONTOUR
            M = cv2.moments(c)
            cX = int(safe_div(M["m10"], M["m00"]))
            cY = int(safe_div(M["m01"], M["m00"]))

            # DRAW CONTOUR AND CENTER
            cv2.circle(original_feed, (cX, cY), 5, (255, 255, 255), -1)
            cv2.putText(original_feed, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow(livefeed_name, original_feed)
            #cv2.imshow('', feed_closing)
            if cv2.waitKey(30) >= 0:
                showLive = False

        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'E:/Study/Intake Jan 2022/MVI - Machine Mission and Intelligence/Assignment MVI/yolov5/content/yolov5/runs/train/yolov5s_results/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / '0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'E:/Study/Intake Jan 2022/MVI - Machine Mission and Intelligence/Assignment MVI/yolov5/content/yolov5/glove-defect-measurement-11/data.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

# TRACKBAR INITIALISATION
def tbar(x):
    pass

# ERROR HANDLING
def safe_div(x, y):
    if y == 0: return 0
    return x / y

# RESIZING THE LIVE FEED WINDOW
def resize_window(frame, percent=80):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)





