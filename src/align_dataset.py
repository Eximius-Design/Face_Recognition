import tensorflow as tf
import cv2
import os
from imutils import paths
import numpy as np
from tqdm import tqdm_notebook as tqdm
import math
import imutils


def align_data(input_path,output_path,detection):
    #reteiving image paths from the train_raw folder of all identities
    #creating folders of identities in train_aligned path
    identities = []
    sess = tf.Session()
    imagePaths = list(paths.list_images(input_path))
    for (i, imagePath) in enumerate(imagePaths):
        name = imagePath.split(os.path.sep)[-2]
        identities.append(name)
    identities = list(set(identities))
    print(identities)
    for identity in identities:
        dir_path = output_path+identity
        os.mkdir(dir_path)
    for (i, imagePath) in enumerate(tqdm(imagePaths)):
        # extract the person name from the image path
        #print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
        img_name = (imagePath.split(os.path.sep)[-1]).split(".")[-2]
        image = cv2.imread(imagePath)
        img_h, img_w = image.shape[:2]
        img_a = img_h*img_w
        rgb_image = image
        #rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #Inferencing the Detection model
        bbox, scores, pts = detection.detect(rgb_image)
        #print("NUMBER OF BOXES",len(bbox))
        j=0
        #margin = 30
        for box,pts in zip(bbox,pts):
            box = box.astype('int32')
            box_w = box[3] - box[1]
            box_h = box[2] - box[0]
            box_a = box_w*box_h
            percent = box_a*100/img_a
            pts = pts.astype('int32')
            x0 = pts[5]
            y0 = pts[0]
            x1 = pts[6]
            y1 = pts[1]
            m = (y1-y0)/(x1-x0)
            deg = math.degrees(math.atan(m))
            rgb_image = imutils.rotate(rgb_image, (deg))
            
            #Cropping the face out of the image and writing it in the aligned path

            face = rgb_image[box[0]:box[2] , box[1]:box[3]]           
            if percent > 3.0 and face.shape[0] != 0 and face.shape[1]!= 0 and face.shape[2] !=0:
                write_path = output_path+name+"/"+img_name+"_"+str(j)+".jpg" 
                #face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                #print(write_path)
                cv2.imwrite(write_path,face)
                j=j+1
