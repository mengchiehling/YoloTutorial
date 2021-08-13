#!/bin/bash
set -e

d=yolov4-2021-08-14
cfg=template-yolov4-csp.cfg
data=template.data
pretrained_model=yolov4-csp.conv.142

cd ~/data
find $PWD -name '*train*.jpg' > train.txt
find $PWD -name '*val*.jpg' > test.txt
cd ~/darknet
aws s3 cp s3://c24-fa-ds-object-detection/trained_models/YOLO/${pretrained_model} ./
#make
#./darknet detector train ./cfg/${data} ./cfg/${cfg} ./${pretrained_model} -dont_show > ${d}/train.log
#./darknet detector map cfg/${data} ./cfg/${cfg} ../models/template-yolov4-csp_final.weights > ${d}/evluation_report.txt
