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
![N|Single Object](http://muizzer07.pythonanywhere.com/media/files/puppy-1903313__340.jpg?style=centerme)
<p align="center"> 
For Single Objects
</p> 
![N|Single Object](http://muizzer07.pythonanywhere.com/media/files/puppy-1903313__340.jpg?style=centerme)
<p align="center"> 
For Multiple Objects
</p> 
## Distance Measurement
![N|Multiple Object](http://muizzer07.pythonanywhere.com/media/files/Ultrasonic-Sensor.jpg?style=centerme)
Traditionally we measure distance of any object using Ultrasonic sensors such as HC-sr04 or other any high frquency devices which generates sound waves to calculates the distance it covers.
However, when you are working with a embedded device to make a compact design which has functionalities such as object detection (with camera) and distance measurement you don't always want to make your device heavier. To avoid such cases you can follow another apporoach. As you have already integrated a camera for object detection you can use the depth information that you camera uses to draw the bounding boxes for localizing objects.
