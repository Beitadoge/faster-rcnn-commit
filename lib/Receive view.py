#coding=utf-8

import tensorflow as tf
from nets.network import Network
from nets.vgg16 import vgg16
from ipdb import set_trace
from layer_utils.snippets import generate_anchors_pre, generate_anchors_pre_tf
import math


# convnet =   [[11,4,0],[3,2,0],[5,1,2],[3,2,0],[3,1,1],[3,1,1],[3,1,1],[3,2,0],[6,1,0], [1, 1, 0]]
convnet =  [[3,1,1],[3,1,1],[2,2,0],[3,1,1],[3,1,1],[2,2,0],[3,1,1],[3,1,1],[3,1,1],[2, 2, 0],[3,1,1],[3,1,1],[3,1,1], [2, 2, 0],[3,1,1],[3,1,1],[3,1,1],[3,1,1]]
layer_names = ['conv1_1','conv1_2','pool1','conv2_1','conv2_2','pool2','conv3_1','conv3_2','conv3_3', 'pool3','conv4_1','conv4_2','conv4_3', 'pool4','conv5_1','conv5_2','conv5_3','conv5_4']
imsize = 227

def outFromIn(conv, layerIn):
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]
    k = conv[0]
    s = conv[1]
    p = conv[2]
    
    n_out = math.floor((n_in - k + 2*p)/s) + 1
    actualP = (n_out-1)*s - n_in + k 
    pR = math.ceil(actualP/2)
    pL = math.floor(actualP/2)
    
    j_out = j_in * s
    r_out = r_in + (k - 1)*j_in
    start_out = start_in + ((k-1)/2 - pL)*j_in
    return n_out, j_out, r_out, start_out

def printLayer(layer, layer_name):
    print(layer_name + ":")
    print("\t n features: %s \n \t jump: %s \n \t receptive size: %s \t start: %s " % (layer[0], layer[1], layer[2], layer[3]))

layerInfos = []
if __name__ == '__main__':
    #first layer is the data layer (image) with n_0 = image size; j_0 = 1; r_0 = 1; and start_0 = 0.5
    print ("-------Net summary------")
    currentLayer = [imsize, 1, 1, 0.5]
    printLayer(currentLayer, "input image")
    for i in range(len(convnet)):
        currentLayer = outFromIn(convnet[i], currentLayer)
        layerInfos.append(currentLayer)
        printLayer(currentLayer, layer_names[i])
    print ("------------------------")
    layer_name = raw_input ("Layer name where the feature in: ")
    layer_idx = layer_names.index(layer_name)
    idx_x = int(raw_input ("index of the feature in x dimension (from 0)"))
    idx_y = int(raw_input ("index of the feature in y dimension (from 0)"))
    
    n = layerInfos[layer_idx][0]
    j = layerInfos[layer_idx][1]
    r = layerInfos[layer_idx][2]
    start = layerInfos[layer_idx][3]
    assert(idx_x < n)
    assert(idx_y < n)
    
    print ("receptive field: (%s, %s)" % (r, r))
    print ("center: (%s, %s)" % (start+idx_x*j, start+idx_y*j))
    # anchors, anchor_length=generate_anchors_pre_tf(1000, 800, feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2))
    # sess = tf.InteractiveSession()
    # set_trace()
    # pass