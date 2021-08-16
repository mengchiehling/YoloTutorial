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


## How to demo it

1. In the command line move to the directory YoloTutorial.
2. Put some images in YoloTutorial/data/images
3. Put your cfg file in the YoloTutorial/tutorial/cfg folder
4. Put your yolov3 model weight in YoloTutorial/trained_models/yolov3
5. In command line type python -m tutorial.run --cfg_code=\<your cfg file> --model_code=\<your weight file> --filename=\<image file name in the images folder>

## Example:
Suppose we have cfg file template-yolov3.cfg and weights file template-yolov3_4000.weights
Then the command line is 
python -m tutorial.run --cfg_code=template-yolov3.cfg --model_code=template-yolov3_4000.weights --filename=\<image file name in the images folder>