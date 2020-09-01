mkdir data
cd data
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
unzip data_object_image_2.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
unzip data_object_label_2.zip

mkdir kitti
mkdir kitti/images
mkdir kitti/labels
mv training/image_2 kitti/images/train 
mv testing/image_2 kitti/images/test 

mv training/label_2 kitti/labels/train 
mv testing/label_2 kitti/labels/test 
cd kitti/images/train && ls -d $PWD/* > ../../train.txt
cd kitti/images/test && ls -d $PWD/* > ../../test.txt
cd ..