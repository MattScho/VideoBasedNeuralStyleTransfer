# Video Neural Style Transfer with Instance Segmentation

This project combines the cutting edge advances in computer vision of instance segmentation and neural style transfer
to generate artificial intelligence based artistic videos.

See this link for example output videos:
https://drive.google.com/drive/folders/1Vk9r2kuplR8nkkL-MWOoVvAnHk-PK4C-?usp=sharing
## Installation
I have provided a requirements.txt file for easy installation using the command:
pip install -r requirements.txt.

I recommend opening the program within PyCharm using the top level folder of VideoNeuralStyleTransfer. 
There are many sub-datasets and models that are neccessary for the project see text files with links placed in their respective directories


## Usage
Settings for running the application can be set using the config.ini file. 

Configuration Settings:
- input_video this is the path to the video to style, this video must exist inside the directory raw_data/video_style_transfer/base_videos
- style_transfer_models this is a list of paths to style model weights, these models must exist in models/style_transfer
- segmenter_mdoel this is the path to the instance segmentation model that YOLACT should use, this model must exist in models/instance_segmentation/yolact_models
- resolution this is a tuple representing the resolution to output the video as
- use_cuda this is a boolean whether to use cuda optimization for pytorch

To run the program use the command: "start.py -c config.ini" from the directory containing both the start.py script and the config.ini file.
An output video will be generated in the outputs directory.

To train a style transfer model use the script style_transfer/training/train.py. 
Within this script there are settings for linking to your dataset.

## Architecture

### Style Transfer Approach
I utilize an approach demonstrated in pytorch's example documentation (https://github.com/pytorch/examples/tree/master/fast_neural_style) based on Johnson et al..
They utilize a pre-trained VGG-16 convolutional neural network, a popular network for object detection, for feature extraction as each of its layers preforms well at iteratively isolating features in an image.
This pre-trained model extracts high level features from the content image, to preserve its larger shapes and structures from the image.
As well, the pre-trained model is used to extract low-level features from the style image, such as colors, textures and small shapes.
First, a target style image is passed through the VGG-16 network and its activations at each layer are saved.
Next, a content image is passed through the VGG-16 network and its activations at each layer are also saved.
Finally, the content image is passed through a transformer network and a loss is calculated using the saved activations weighted between the style and content images.
The result is a transformer network that given an input content image will output a version stylized based on the original style image.


### Instance Segmentation Approach
I utilize the YOLACT model (https://github.com/dbolya/yolact) created by Bolya et al.. YOLACT stands for You Only Look At Coefficients 
a play on the similar object detection model family of YOLO You Only Look Once. YOLACT is able to produce bounding boxes around instances
in an image, similar to R-CNNs and YOLO models. YOLACT goes one step further and also produces masks for the objects that are detected 
within their respective boxes. This is achieved through the use of two specialized branches in their network, one to produce "prototype masks" and
one to produce "mask co-efficients". The prototype masks generation branch creates a set of k prototype masks that are within the detected bounding box 
of the object. The mask co-efficients branch outputs k co-efficent matrices that each correspond to a mask. The co-efficient matrices and the corresponding 
prototype masks are multiplied together to generate a final output mask.

## References
Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A Neural Algorithm of Artistic Style. CoRR, abs/1508.06576. From http://arxiv.org/abs/1508.06576

Ruder, M., Dosovitskiy, A., & Brox, T. (2016). Artistic style transfer for videos. CoRR, abs/1604.08610. From http://arxiv.org/abs/1604.08610

Johnson J., Alahi A., Fei-Fei L. (2016). Perceptual Losses for Real-Time Style Transfer and Super-Resolution, abs/1603.08155. From https://arxiv.org/pdf/1603.08155.pdf

Bolya D., Zhou C., Xiao F., & Lee Y. (2019). YOLACT: Real-time Instance Segmentation. 
    See https://github.com/dbolya/yolact

Image segmentation - Tensorflow Core. TensorFlow. (n.d.). Retrieved November 7, 2021, from ttps://www.tensorflow.org/tutorials/images/segmentation.

VGG16 – Convolutional Network for Classification and Detection. (November 20, 2018). Retrieved November 20, 2021, from https://neurohive.io/en/popular-networks/vgg16/.

fast-neural-style. (n.d.). Retrieved November 23, 2021, from https://github.com/pytorch/examples/tree/master/fast_neural_style.
