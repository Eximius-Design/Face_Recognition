import tensorflow as tf
import cv2
import os
from imutils import paths
from Detection import Detection
import numpy as np


def align_data(input_path,output_path,Detection_model_path):
    #reteiving image paths from the train_raw folder of all identities
    #creating folders of identities in train_aligned path
    detection = Detection(Detection_model_path)
    identities = []
    imagePaths = list(paths.list_images(input_path))
    for (i, imagePath) in enumerate(imagePaths):
        name = imagePath.split(os.path.sep)[-2]
        identities.append(name)
    identities = list(set(identities))
    print(identities)
    for identity in identities:
        dir_path = output_path+identity
        os.mkdir(dir_path)
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
        img_name = (imagePath.split(os.path.sep)[-1]).split(".")[-2]
        image = cv2.imread(imagePath)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #Inferencing the Detection model
        bbox, scores, landmarks = detection.detect(rgb_image)
        j=0
        for box in bbox:
            box = box.astype('int32')
            box_w = box[3] - box[1]
            box_h = box[2] - box[0]
            box_a = box_w*box_h

            #Cropping the face out of the image and writing it in the aligned path

            face = np.copy(rgb_image[box[0]:box[2] , box[1]:box[3]])
            if face.shape[0] != 0 and face.shape[1]!= 0 and face.shape[2] !=0:
                write_path = output_path+name+"/"+img_name+"_"+str(j)+".jpg" 
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                print(write_path)
                cv2.imwrite(write_path,face)
                j=j+1