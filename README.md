# Face_Recognition_on_videos
This repository is an inferencing implementation of FaceNet(Inception-ResNet) which also uses MTCNN for detection on still images as well as Videos.
FaceNet used here is not from the original paper, it is a modified structure which uses Inception-ResNet as the backbone network and softmax as the loss.

## Preparing the Dataset:
* Clone this repository.
* Create directories as Face_Recognition/Outputs/Detection_Recognition_outputs
* Create another directory as Face_Recognition/Dataset.
* Create three directories inside Face_Recognition(i.e. Face_Recognition/Dataset/)
  *train_raw
  *train_aligned
  *videos
* Fill the train_raw directory with data in the following format:
  * Identity 1/
    * 1_Identity1img.jpg
    * 1_Identity2img.jpg...
    
  * Identity 2/
    * 2_Identity1img.jpg
    * 2_Identity2img.jpg...
    
* All the identities should have a directory containing all their images inside the train_raw directory.
* Load a test video with the Identities from the train_raw directory.(as Face_Recognition/videos/vid1.mp4)

## HOW TO RUN?

* Install the required libraries from requirements.txt using the command **pip install requirements.txt** 
* Now open the ipynb located at **Face_Recognition/src/TRAIN_CLASSIFIER.ipynb** and run all the cells.
* Open the ipynb  located at **Face_Recognition/src/MAIN_VIDEO.ipynb** and run the cells
* Find the output video at **Face_Recognition/Outputs/Detection_Recognition_outputs/**
