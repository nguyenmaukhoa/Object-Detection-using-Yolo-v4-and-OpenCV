# Object-Detection-using-YOLO-v4

YOLO (You only look once) is unique because it detects objects by looking at the overall scene (image or video) as a whole, instead of analyzing individual regions at a time. This results in some level of tradeoff for speed over precision. Despite this, YOLO is one of the most powerful realtime object detection architectures today, and is ideal to power the vision behind driverless cars.

When trained, YOLO learns a generalized representation of the class of objects - meaning it is likely to perform well when presented with an unfamiliar representation of the object (shadowed, eclipsed, etc), and not get thrown off when there are other unexpected objects in the scene. This is because, by definition, it was trained to identify objects when considering the scene has a whole.

## Why Use OpenCV for YOLO

**Easy integration with an OpenCV application:** If your application already uses OpenCV and you simply want to use YOLOv4, you don’t have to worry about compiling and building the extra Darknet code.

**OpenCV CPU version is 9x faster:** OpenCV’s CPU implementation of the DNN module is astonishingly fast. For example, Darknet when used with OpenMP takes about 2 seconds on a CPU for inference on a single image. In contrast, OpenCV’s implementation runs in a mere 0.22 seconds! Check out table below.

**Python support:** Darknet is written in C, and it does not officially support Python. In contrast, OpenCV does. There are python ports available for Darknet though.

## More about YOLO
https://github.com/AlexeyAB/darknet

## Output Sample

<img width="1438" alt="Screen Shot 2022-01-14 at 9 35 12 PM" src="https://user-images.githubusercontent.com/42128166/149610951-4f0ff631-fdc2-4c9b-b360-4b65570915ca.png">

(Video Source: https://www.pexels.com/video/vehicles-on-the-road-5274079/)

