#!/bin/bash
mkdir data
cd data
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
unzip data_object_image_2.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
unzip data_object_label_2.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_3.zip
unzip data_object_label_2.zip

mkdir kitti
mkdir kitti/images
mkdir kitti/labels
mkdir kitti/right_images
mv training/image_2 kitti/images/train 
mv testing/image_2 kitti/images/test 
mv training/image_3 kitti/right_images/train 
mv testing/image_3 kitti/right_images/test 
mv training/label_2 kitti/labels/train

cd kitti/images/train && ls -d $PWD/* > ../../train.txt
cd ..