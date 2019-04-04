import tensorflow as tf
import numpy as np
import pickle
import os
from imutils import paths
import cv2
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
import preprocess
from Recognition import Recognition
from tqdm import tqdm_notebook as tqdm

def rec_gen_classifier(train_aligned_path , recognition ,output_classifier_path):
    #training the classifier using sklearn.svm 
    #fitting the SVC classifier with embeddings and labels
    labels = []
    emb = []
    imagePaths = list(paths.list_images(train_aligned_path))
    for (i, imagePath) in enumerate(tqdm(imagePaths)):
        name = imagePath.split(os.path.sep)[-2]
        img_name = (imagePath.split(os.path.sep)[-1]).split(".")[-2]
        face = cv2.imread(imagePath)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        process_face = preprocess.prewhiten(face)
        embedding = recognition.recognize(face = process_face)
        labels.append(name)
        emb.append(embedding)
    emb_arrays = np.array(emb).reshape(np.array(emb).shape[0], np.array(emb).shape[2])
    labels = np.array(labels)
    classnames = list(set(labels))
    #model = SGDClassifier(loss="log", penalty="l2", max_iter=5)
    model = SVC(kernel='linear', probability=True)
    model.fit(emb_arrays, labels)
    np.save("EMB",emb)
    np.save("labels",labels)
    print(model,classnames)
    with open(output_classifier_path, 'wb') as outfile:
        pickle.dump((model,classnames), outfile)