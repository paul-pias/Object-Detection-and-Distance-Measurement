import torch,cv2,random,os,time
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pickle as pkl
import argparse
import threading, queue
from torch.multiprocessing import Pool, Process, set_start_method
from util import write_results, load_classes
from preprocess import letterbox_image
from darknet import Darknet
from imutils.video import WebcamVideoStream,FPS
# from camera import write
import win32com.client as wincl       #### Python's Text-to-speech (tts) engine for windows, multiprocessing
speak = wincl.Dispatch("SAPI.SpVoice")    #### This initiates the tts engine

torch.multiprocessing.set_start_method('spawn', force=True)

##  Setting up torch for gpu utilization
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True 
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

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

labels = {}
b_boxes = {}
def write(bboxes, img, classes, colors):
    """
        Draws the bounding box in every frame over the objects that the model detects
    """
    class_idx = bboxes
    bboxes = bboxes[1:5]
    bboxes = bboxes.cpu().data.numpy()
    bboxes = bboxes.astype(int)
    b_boxes.update({"bbox":bboxes.tolist()})
    # bboxes = bboxes + [150,100,200,200] # personal choice you can modify this to get distance as accurate as possible
    bboxes = torch.from_numpy(bboxes)
    cls = int(class_idx[-1])
    label = "{0}".format(classes[cls])
    labels.update({"Current Object":label})
    color = random.choice(colors)

    ## Put text configuration on frame
    text_str = '%s' % (label) 
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.6
    font_thickness = 1
    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
    text_pt = (bboxes[0], bboxes[1] - 3)
    text_color = [255, 255, 255]

    
    ## Distance Meaasurement for each bounding box
    x, y, w, h = bboxes[0], bboxes[1], bboxes[2], bboxes[3]
    ## item() is used to retrieve the value from the tensor
    distance = (2 * 3.14 * 180) / (w.item()+ h.item() * 360) * 1000 + 3 ### Distance measuring in Inch 
    feedback = ("{}".format(labels["Current Object"])+ " " +"is"+" at {} ".format(round(distance))+"Inches")
    # # speak.Speak(feedback)     # If you are running this on linux based OS kindly use espeak. Using this speaking library in winodws will add unnecessary latency 
    print(feedback)
    
    cv2.putText(img, str("{:.2f} Inches".format(distance)), (text_w+x,y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0,255,0), font_thickness, cv2.LINE_AA)
    cv2.rectangle(img, (bboxes[0],bboxes[1]),(bboxes[2] + text_w -30,bboxes[3]), color, 2)
    cv2.putText(img, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return img

class ObjectDetection:
    def __init__(self, id): 
        # self.cap = cv2.VideoCapture(id)
        self.cap = WebcamVideoStream(src = id).start()
        self.cfgfile = "cfg/yolov3.cfg"
        # self.cfgfile = 'cfg/yolov3-tiny.cfg'
        self.weightsfile = "yolov3.weights"
        # self.weightsfile = 'yolov3-tiny.weights'
        self.confidence = float(0.6)
        self.nms_thesh = float(0.8)
        self.num_classes = 80
        self.classes = load_classes('data/coco.names')
        self.colors = pkl.load(open("pallete", "rb"))
        self.model = Darknet(self.cfgfile)
        self.CUDA = torch.cuda.is_available()
        self.model.load_weights(self.weightsfile)
        self.model.net_info["height"] = 160
        self.inp_dim = int(self.model.net_info["height"])
        self.width = 1280 #640#1280
        self.height = 720 #360#720
        print("Loading network.....")
        if self.CUDA:
            self.model.cuda()
        print("Network successfully loaded")
        assert self.inp_dim % 32 == 0
        assert self.inp_dim > 32
        self.model.eval()

    def main(self):
        q = queue.Queue()
        while True:
            def frame_render(queue_from_cam):
                frame = self.cap.read() # If you capture stream using opencv (cv2.VideoCapture()) the use the following line
                # ret, frame = self.cap.read()
                frame = cv2.resize(frame,(self.width, self.height))
                queue_from_cam.put(frame)
            cam = threading.Thread(target=frame_render, args=(q,))
            cam.start()
            cam.join()
            frame = q.get()
            q.task_done()
            fps = FPS().start() 
            try:
                img, orig_im, dim = prep_image(frame, self.inp_dim)
                im_dim = torch.FloatTensor(dim).repeat(1,2)
                if self.CUDA:                            #### If you have a gpu properly installed then it will run on the gpu
                    im_dim = im_dim.cuda()
                    img = img.cuda()
                # with torch.no_grad():               #### Set the model in the evaluation mode
                output = self.model(Variable(img), self.CUDA)
                output = write_results(output, self.confidence, self.num_classes, nms = True, nms_conf = self.nms_thesh)  #### Localize the objects in a frame
                output = output.type(torch.half)
                
                if list(output.size()) == [1,86]:
                    print(output.size())
                    pass
                else:
                    output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(self.inp_dim))/self.inp_dim
                
        #            im_dim = im_dim.repeat(output.size(0), 1)
                    output[:,[1,3]] *= frame.shape[1]
                    output[:,[2,4]] *= frame.shape[0]
                    list(map(lambda boxes: write(boxes, frame, self.classes, self.colors),output))
                    
            except:
                pass
            
            fps.update()
            fps.stop()
            print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
            print("[INFO] approx. FPS: {:.1f}".format(fps.fps()))
            cv2.imshow("Object Detection Window", frame)
             
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
            torch.cuda.empty_cache()
            

if __name__ == "__main__":
    id = 0
    ObjectDetection(id).main()
