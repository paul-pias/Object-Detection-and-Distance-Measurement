# Object Detection and Distance Measurement

[![N|Solid](http://muizzer07.pythonanywhere.com/media/files/YOLO-m-ram-copy_RQByeS4.jpg)](https://pjreddie.com/darknet/yolo/?style=centerme)


## Introduction
 This repo contains object_detection.py which is able to perform the following task -
 - Object detection from live video frame, in any video file or in a image
 - Counting the number of objects in a frame
 - Measuring the distance of object using depth information

### Instalation
        $ pip install requirements.txt
            or
        $ pip install opencv_python
        $ pip install numpy
        $ pip install pandas
        $ pip install matplotlib
        $ pip install Pillow
<hr>

#### For the installation of torch using "pip" 
    $pip3 install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
    or please follow the instruction from [Pytorch](https://pytorch.org/)

<hr>

#### Theory
In a traditional image classification approach for object detection there are two well-known strategies.
For single object in a image there are two scenarios.
- Classification
- Localization
For multiple objects in a image there are two scenarios.
- Object detection and localization
- Object segmentation
<p align="center"> 
 <b> For Single Objects </b>
    <img src ="http://muizzer07.pythonanywhere.com/media/files/puppy-1903313__340.jpg?style=centerme">
</p> 

<p align="center"> 
<b> For Multiple Objects </b>
    <img src ="http://muizzer07.pythonanywhere.com/media/files/pexels-photo-1108099.jpeg?style=centerme">
</p> 

## Distance Measurement
![N|Multiple Object](http://muizzer07.pythonanywhere.com/media/files/Ultrasonic-Sensor.jpg?style=centerme)
<hr>
Traditionally we measure distance of any object using Ultrasonic sensors such as HC-sr04 or other any high frquency devices which generate sound waves to calculates the distance it traverse.
However, when you are working with a embedded device to make a compact design which has functionalities such as 

- Object detection (with camera) and 
- Distance measurement 

you don't always want to make your device heavier by adding unnnecessary hardware modules. To avoid such cases you can follow a more convinent and feasible apporoach. As you have already integrated a camera for object detection, you can use the depth information that camera uses to draw the bounding boxes for localizing objects to calculate the distance of that object from the camera.

### How it works?
From the initial part we understood that, we need to measure distance from an image we to localize it first to get the depth information.
<b> Now, how actually localization works?</b>
#### Localize objects with regression
   Regression is about returning a number istead of a class. The number can be represented as (x0,y0,width,height) which are related to a bounding box. In the images illustrated above for single object if you want to only classify  the object type then we don't need to draw the bounding box around that object that's why this part is known as <b> Classification </b>.
   
   However, if we are interested to know where does this object locates in the image then we need to know that 4 numbers that a regreesion layer will return. As you can see there is a black rectangle shape box in the image of white dog which was drawn using the regression layer. What happens here is that after the final convolutional layer + Fully connected layers instead of asking for class scores to compare with some offsets a regression layer is introduced. Regression layer is nothing but some rectangular box which represents individual objects. So prior to the training phase of a neural network some pre-defined rectangular boxes that represents some objects are given to the network to train with. So when a image is gone through the network, after the fully connected layer the trained model tries to match predefined boxes to objects on that image by using Intersection over Union (IoU) algorith to completely tied. If the comparison crosses some threshold the model tries to draw the bounding box over the object. For example, in the case of the picture of white dog, the model knows what is the coordinates of the box of the dog object and when the image classification is done the model uses L2 distance to calculate the loss between the actual box coordinates that was predefined and the coordinate that the model gave so that it can perfectly draw the bounding box over the object on that image.
