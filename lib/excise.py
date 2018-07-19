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
import xml.etree.ElementTree as ET


if __name__ == '__main__':
    filename = "../data/VOCdevkit2007/VOC2007/Annotations/000001.xml"
    tree = ET.parse(filename)
    objs = tree.findall('object')
    non_diff_objs = [
        obj for obj in objs if int(obj.find('difficult').text) == 0]
    set_trace()


    pass