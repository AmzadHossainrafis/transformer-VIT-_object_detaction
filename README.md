# transformer-VIT-_object_detaction
In the past few years, there has been a rapid increase in the use of 
computer vision technology in many applications such as security, surveillance and machine learning. 
One of the applications that have seen significant growth in the past few years is object detection. 
Object detection is the process of detection process if detective objects such as
the face,  vehicles or pedestrians in an image and video frame and then labelling each detected object. 
Vision transformers are a new invention that is introduced in deep learning for computer vision for object detection.
In this project, I implemented a VIT model from scratch, built an aircraft detection, and reached more than 80% accuracy.


# vision transformer model architecture 

The vision transformer is a deep neural network that is designed to process images. It is based on the transformer architecture, which is a type of neural network that is particularly well suited for processing sequential data. The vision transformer has been designed to specifically take advantage of the properties of images that make them easier to process than other types of data. In particular, the vision transformer is able to take advantage of the fact that images are typically two-dimensional and have a regular structure. This enables the vision transformer to learn to identify patterns in images and to learn to process them in a way that is similar to how humans process images . 

There are mainly 3 part in transformer 
 1.The Encoder: The encoder is responsible for taking in an image and representing it as a set of features. The encoder is typically a convolutional neural network (CNN).
 2.The Attention Module(Transformer):The attention module is responsible for computing the relationships between the features that have been extracted by the encoder. The attention module is typically a self-attention module.
 3.The Decoder:The decoder is responsible for taking the features that have been extracted by the encoder and reconstructing the original image. The decoder is typically  a deconvolutional neural network (DECONV).
 
images are divided into small patches then the patches are embedded and passed to the encoder 
