from __future__ import division  #### It has to be imported at the beginning of the file
from utils.app_utils import *
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random
import pickle as pkl
import argparse
# import ConnectionServer ## Import it if you are using raspberrypi or any thrid party camera to detect object
import os,sys,time,json
import math
import requests
import win32com.client as wincl       #### Python's Text-to-speech (tts) engine for windows
speak = wincl.Dispatch("SAPI.SpVoice")    #### This initiates the tts engine


def get_test_input(input_dim, CUDA):
    """
        Test the performance of the model on a image
    """
    img = cv2.imread("pias.png")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

labels = []

def write(x, img, classes, colors):
    """
        Draws the bounding box in every frame over the objects that the model detects
    """
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    # print(label)
    labels.clear()
    labels.insert(0, label)
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    return img

labels.clear()

def print_labels():
    """
    Print the labels from the labels list
    The
    """
    return labels

def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.25)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="160", type=str)
    return parser.parse_args()


def draw_detections(img, rects, thickness = 1):
    """
        INPUT :
            img  : Gets the input frame
            rect : Number from the regression layer (x0,y0,width,height)
        OUTPUT:
            count: Number of objects in a given frame
            distance : Calculates the distance from the rect value

    """

    count = 0
    distancei = 0.0
    for x, y, w, h in rects:
        print(len(rects))

        if len(rects) >= 0: ### Increase the value of count if there are more than one rectangle in a given frame
            count += 1
        distancei = (2 * 3.14 * 180) / (w + h * 360) * 1000 + 3 ### Distance measuring in Inch
        # print(distancei)
        # distance = distancei * 2.54
        # print(distance)
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)
    return count, distancei


class Camera(object):
    def __init__(self,file_path):
        self.M = None
        self.width = 400 # Set Phone width and height here
        self.height = 400
        self.image = None
        self.file_path = file_path
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        self.video = cv2.VideoCapture(file_path)
    
    def __del__(self):
        self.video.release()

    def set_rect(self, pts):
        pts1 = np.float32(pts)
        pts2 = np.float32([[0, 0], [self.width, 0], [0, self.height], [self.width, self.height]])
        self.M = cv2.getPerspectiveTransform(pts1, pts2)
    
    def get_frame(self):
        cfgfile = "cfg/yolov3.cfg"
        weightsfile = "yolov3.weights"
        args = arg_parse()
        confidence = float(args.confidence)
        nms_thesh = float(args.nms_thresh)
        start = 0
        num_classes = 80

        CUDA = torch.cuda.is_available()

        bbox_attrs = 5 + num_classes

        # print("Loading network.....")
        model = Darknet(cfgfile)
        model.load_weights(weightsfile)
        # print("Network successfully loaded")

        model.net_info["height"] = args.reso
        inp_dim = int(model.net_info["height"])
        assert inp_dim % 32 == 0
        assert inp_dim > 32

        if CUDA:
            model.cuda()
        success, image = self.video.read()
        count = 0
        start = time.time()
        while self.video.isOpened():
            if success:
                img, orig_im, dim = prep_image(image, inp_dim)  #### Pre-processing part of every frame that came from the source
                im_dim = torch.FloatTensor(dim).repeat(1,2)

                if CUDA:                            #### If you have a gpu properly installed then it will run on the gpu
                    im_dim = im_dim.cuda()
                    img = img.cuda()

                with torch.no_grad():               #### Set the model in the evaluation mode
                    output = model(Variable(img), CUDA)
                output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)  #### Localize the objects in a frame
                output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim
                im_dim = im_dim.repeat(output.size(0), 1)
                output[:, [1, 3]] *= image.shape[1]
                output[:, [2, 4]] *= image.shape[0]
                classes = load_classes('data/coco.names')
                colors = pkl.load(open("pallete", "rb"))

                list(map(lambda x: write(x, orig_im, classes, colors), output))
                l = print_labels()[0]
                print(l)
                hog = cv2.HOGDescriptor()
                hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                found,w = hog.detectMultiScale(image, winStride=(8,8), padding=(32,32), scale=1.05)
                get_number_of_object, get_distance= draw_detections(image,found)
                if get_number_of_object >=1 and get_distance!=0:
                    feedback = ("{}".format(get_number_of_object)+ " " +l+" at {}".format(round(get_distance))+"Inches")
                    speak.Speak(feedback)
                    print(feedback)
                else:
                    feedback = ("{}".format("1")+ " " +l)
                    speak.Speak(feedback)
                    print(feedback)
                self.image = image
                if self.M is not None:
                    image = cv2.warpPerspective(image, self.M, (self.width, self.height))
                # We are using Motion JPEG, but OpenCV defaults to capture raw images,
                # so we must encode it into JPEG in order to correctly display the
                # video stream.
                
                ret, jpeg = cv2.imencode('.jpg', image)
                return jpeg.tostring(),l,feedback
            else:
                self.video = cv2.VideoCapture(self.file_path)
                return open('outputs/temp.jpg', 'rb').read()

if __name__ == '__main__':
    vc = Camera()
    vc.get_frame()