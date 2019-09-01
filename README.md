# Object Detection and Distance Measurement

[![N|Solid](http://muizzer07.pythonanywhere.com/media/files/YOLO-m-ram-copy_RQByeS4.jpg)](https://pjreddie.com/darknet/yolo/?style=centerme)


## Introduction
    In a traditional image classification approach for object detection there are two well-known 
    strategies.
    For single object in a image there are two scenarios.
        - Classification
        - Localization
    For multiple objects in a image there are two scenarios.
        - Object detection and localization
        - Object segmentation
<p align="center"> 
For Single Objects
    <img src ="http://muizzer07.pythonanywhere.com/media/files/puppy-1903313__340.jpg?style=centerme">
</p> 

<p align="center"> 
For Multiple Objects
    <img src ="http://muizzer07.pythonanywhere.com/media/files/pexels-photo-1108099.jpeg?style=centerme">
</p> 

## Distance Measurement
![N|Multiple Object](http://muizzer07.pythonanywhere.com/media/files/Ultrasonic-Sensor.jpg?style=centerme)
<hr>
Traditionally we measure distance of any object using Ultrasonic sensors such as HC-sr04 or other any high frquency devices which generates sound waves to calculates the distance it covers.
However, when you are working with a embedded device to make a compact design which has functionalities such as 
    - object detection (with camera) and 
    - distance measurement 
you don't always want to make your device heavier by adding unnnecessary hardware modules. To avoid such cases you can follow another more convinent and feasible apporoach. As you have already integrated a camera for object detection you can use the depth information that camera uses to draw the bounding boxes for localizing objects.

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
