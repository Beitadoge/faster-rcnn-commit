#coding=utf-8

import tensorflow as tf
from nets.network import Network
from nets.vgg16 import vgg16
from ipdb import set_trace
from layer_utils.snippets import generate_anchors_pre, generate_anchors_pre_tf
import math
from layer_utils.generate_anchors import generate_anchors,_mkanchors
import numpy as np
from utils.cython_bbox import bbox_overlaps



if __name__ == '__main__':
    # a = np.array([[1,2,3,4],[5,6,7,8],[1,1,1,1]])
    # b = np.array([[1,1,1,1],[1,2,3,4],[5,6,7,8]])
    a = np.array([[1,2,3,4],[5,6,7,8],[1,2,3,4],[1,2,3,4]])
    b = np.array([[3,4,5,6],[5,6,7,8]])
    c = np.array([0.9,0.8,0.7,0.6])
    indices = tf.image.non_max_suppression(a, c, max_output_size=10, iou_threshold=0.8)
    overlaps = bbox_overlaps(
    np.ascontiguousarray(a, dtype=np.float),
    np.ascontiguousarray(b, dtype=np.float))
    set_trace()


    pass