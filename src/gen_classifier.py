import tensorflow as tf
import numpy as np
import pickle
import os
from imutils import paths
import cv2
from sklearn.svm import SVC
import preprocess
from Recognition import Recognition


def rec_gen_classifier(train_aligned_path , recognition ,output_classifier_path):
    #training the classifier using sklearn.svm 
    #fitting the SVC classifier with embeddings and labels
    labels = []
    emb = []
    imagePaths = list(paths.list_images(train_aligned_path))
    for (i, imagePath) in enumerate(imagePaths):
        print("[INFO] Processing face", i)
        name = imagePath.split(os.path.sep)[-2]
        img_name = (imagePath.split(os.path.sep)[-1]).split(".")[-2]
        face = cv2.imread(imagePath)
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        process_face = preprocess.prewhiten(rgb_face)
        embedding = recognition.recognize(face = process_face)
        labels.append(name)
        emb.append(embedding)
    emb_arrays = np.array(emb).reshape(np.array(emb).shape[0], np.array(emb).shape[2])
    labels = np.array(labels)
    classnames = list(set(labels))
    model = SVC(kernel='linear', probability=True)
    print(model,classnames)
    model.fit(emb_arrays, labels)
    with open(output_classifier_path, 'wb') as outfile:
        pickle.dump((model,classnames), outfile)