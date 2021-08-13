# YoloTutorial

## prerequisite

* Anaconda
* darknet: https://github.com/AlexeyAB/darknet

## Instruction
* Makefile 

    In the Makefile
GPU=1
CUDNN=1
OPENCV=1  # for speeding up image augmentation

    At line 117:
ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/

    replace "/usr/local/cuda/include/" with  /usr/local/cuda-11.0/include/
    
* Download yolov4-csp.conv.142 pretrained weight for training:  https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-csp.conv.142 
