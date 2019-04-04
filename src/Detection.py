import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


class Detection:
    
    def __init__(self, model_path, min_size=40, factor=0.709, thresholds=[0.6, 0.7, 0.7]):
        self.min_size = min_size
        self.factor = factor
        self.thresholds = thresholds
        

        graph = tf.Graph()
        with graph.as_default():
            with open(model_path, 'rb') as f:
                graph_def = tf.GraphDef.FromString(f.read())
                tf.import_graph_def(graph_def, name='')
        self.graph = graph
        config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=4)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=config)
        print("Detection Model Graph Initialized")

       
    def detect(self, img):
        img_size = img.shape[:2]
        feeds = {
            self.graph.get_operation_by_name('input').outputs[0]: img,
            self.graph.get_operation_by_name('min_size').outputs[0]: self.min_size,
            self.graph.get_operation_by_name('thresholds').outputs[0]: self.thresholds,
            self.graph.get_operation_by_name('factor').outputs[0]: self.factor
        }
        fetches = [self.graph.get_operation_by_name('prob').outputs[0],
                  self.graph.get_operation_by_name('landmarks').outputs[0],
                  self.graph.get_operation_by_name('box').outputs[0]]
        prob, landmarks, box = self.sess.run(fetches, feeds)
        margin = 30
        for b in box:
            if b[0]-margin > 0 and b[1]-margin > 0:
                b[0] = b[0]-margin
                b[1] = b[1]-margin
            if b[2]+margin < img_size[1] and b[3]+margin < img_size[0]:
                b[2] = b[2]+margin
                b[3] = b[3]+margin
        return box, prob, landmarks
