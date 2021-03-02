# SSD-Object-Detection

## MobileNets
MobileNets, as the name suggests, are neural networks constructed for the purpose of running very efficiently (high FPS, low memory footprint) on mobile and embedded devices. MobileNets achieve this with 3 techniques:
Perform a depthwise convolution followed by a 1x1 convolution rather than a standard convolution. The 1x1 convolution is called a pointwise convolution if it's following a depthwise convolution. The combination of a depthwise convolution followed by a pointwise convolution is sometimes called a separable depthwise convolution.
Use a "width multiplier" - reduces the size of the input/output channels, set to a value between 0 and 1.
Use a "resolution multiplier" - reduces the size of the original input, set to a value between 0 and 1.
These 3 techniques reduce the size of cummulative parameters and therefore the computation required. Of course, generally models with more paramters achieve a higher accuracy. MobileNets are no silver bullet, while they perform very well larger models will outperform them. MobileNets are designed for mobile devices, NOT cloud GPUs. The reason we're using them in this lab is automotive hardware is closer to mobile or embedded devices than beefy cloud GPUs.

### MobileNet SSD
In this section you'll use a pretrained MobileNet SSD model to perform object detection. You can download the MobileNet SSD and other models from the TensorFlow detection model zoo (note: we'll provide links to specific models further below). Paper describing comparing several object detection models.

### Single Shot Detection (SSD)
Many previous works in object detection involve more than one training phase. For example, the Faster-RCNN architecture first trains a Region Proposal Network (RPN) which decides which regions of the image are worth drawing a box around. RPN is then merged with a pretrained model for classification (classifies the regions). The image below is an RPN:
### Faster-RCNN Visual
The SSD architecture is a single convolutional network which learns to predict bounding box locations and classify the locations in one pass. Put differently, SSD can be trained end to end while Faster-RCNN cannot. The SSD architecture consists of a base network followed by several convolutional layers:
### SSD Visual
NOTE: In this lab the base network is a MobileNet (instead of VGG16.)

### Result
| 10s of driving|10sec detected result |
|--|--|
|![Driving](sample.gif) | ![Result](detection.gif)|

<!--
<iframe width="1280" height="720" src="https://www.youtube.com/embed/E5Jg4Wm9b7o" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
--> 
### trained model Download  
* [SSD_Mobilenet v11.6.17](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz)
* [RFCN_ResNet101 v11.6.17](http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_11_06_2017.tar.gz)
* [Faster_RCNN_Inception_ResNet v11.6.17](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz)

### Reference 
* [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
* [Deep Lab](https://github.com/tensorflow/models/tree/master/research/deeplab)
* [CityScapes DataSet](https://www.cityscapes-dataset.com/)
