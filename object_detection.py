from __future__ import division  #### It has to be imported at the beginning of the file
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
from zeep import Client
import math

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
    distance = 0.0
    for x, y, w, h in rects:
        print(len(rects))

        if len(rects) >= 0:
            count += 1
            # person = ("There is {}".format(count)+" Person")
            # print("Person found: ", count)
        distancei = (2 * 3.14 * 180) / (w + h * 360) * 1000 + 3

        #        distance = distancei *2.54
        distance = math.floor(distancei / 2)
        # print(count, distance)

        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)
    return count, distance

def object_detection():
    """
        Will load the pre-trained weight file and the cfg file which has knowledge of 80 different objects 
        Using the arg_parse function it will compare the confidence and threshold value of every object in a given frame

    """

    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    num_classes = 80

    CUDA = torch.cuda.is_available()

    bbox_attrs = 5 + num_classes

    print("Loading network.....")
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    #### Test the performance of the model on a Static Image
    # model(get_test_input(inp_dim, CUDA), CUDA)
    # model.eval()
    ####

    #### Test the performance of the model on any video file
    videofile = 'video3.avi'
    ####

    #### If you are using any thrird party camera access using IP address you can use this part of the code
    # address = ConnectionServer.connect()
    # address = 'http://' + address[0] + ':8000/stream.mjpg'
    # print("Fetching Video from", address)
    ####
    cap = cv2.VideoCapture(0)   #### If you are using your default webcam then use 0 as a source and for usbcam use 1

    assert cap.isOpened(), 'Cannot capture source'   #### If camera is not found assert this message
    count = 0
    frames = 0
    start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img, orig_im, dim = prep_image(frame, inp_dim)  #### Pre-processing part of every frame that came from the source
            im_dim = torch.FloatTensor(dim).repeat(1,2)

            if CUDA:                            #### If you have a gpu properly installed then it will run on the gpu
                im_dim = im_dim.cuda()
                img = img.cuda()

            with torch.no_grad():               #### Set the model in the evaluation mode
                output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)  #### Localize the objects in a frame

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                cv2.imshow("Object Detection Window", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            #im_dim = im_dim.repeat(output.size(0), 1)
            #scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)

            output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim
            im_dim = im_dim.repeat(output.size(0), 1)
            output[:, [1, 3]] *= frame.shape[1]
            output[:, [2, 4]] *= frame.shape[0]

            #output[:,1:5] /= scaling_factor

            # for i in range(output.shape[0]):
            #     output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
            #     output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])

            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))

            list(map(lambda x: write(x, orig_im, classes, colors), output))


            cv2.imshow("Object Detection Window", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1

            # print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            l = print_labels()[0]
            print(l)
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found,w = hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)
            # time.sleep(2)
            # print(found)
            # print(len(found))
            # draw_detections(frame, found)
            get_number_of_object, get_distance= draw_detections(frame,found)
            person = ("{}".format(get_number_of_object)+ " " +l+" at {}".format(get_distance)+" centimeter")
            print(person)
            
if __name__ == "__main__":
    object_detection()