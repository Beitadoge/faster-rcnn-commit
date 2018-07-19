#coding=utf-8
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from model.config import cfg
import numpy as np
import numpy.random as npr
from utils.cython_bbox import bbox_overlaps
from model.bbox_transform import bbox_transform

'''Same as the anchor target layer in original Fast/er RCNN'''
def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
  '''
  rpn_cls_score : = <tf.Tensor 'vgg_16_1/rpn_cls_score/BiasAdd:0' shape=(1, ?, ?, 18) dtype=float32> 假设是(1,h,w,18)
  gt_boxes = (gt_size,5):gt_size为一幅图中所有框的个数.每行元素为[x1,y1,x2,y2,gt_box_class]其中(x1,y1)为左上角的坐标.(x2,y2)为右下角的坐标,,gt_box_class为该框里面的物体的类别
  im_info : (3,)
  all_anchors : (anchor_size,4):anchor_size=h*w*9
  num_anchors : 9
  '''
  A = num_anchors #9
  total_anchors = all_anchors.shape[0] #h*w*9
  K = total_anchors / num_anchors#特征图的大小:h*w

  # allow boxes to sit over the edge by a small amount
  _allowed_border = 0

  # map of shape (..., H, W)
  height, width = rpn_cls_score.shape[1:3]

  '''
  ind_inside这段程序没看懂,,但是知道ind_inside保存着符合条件的anchor的行的索引号
  假设ind_inside=(in_anchor_size,),,in_anchor_size:在图片内部的anchor的数量
  '''
  # only keep anchors inside the image
  inds_inside = np.where(
    (all_anchors[:, 0] >= -_allowed_border) &
    (all_anchors[:, 1] >= -_allowed_border) &
    (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
    (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
  )[0]

  '''保留在图片内部的框,并存在anchors中'''
  anchors = all_anchors[inds_inside, :] #(in_anchor_size,4)

  '''初始化anchors的标签'''
  # label: 1 is positive, 0 is negative, -1 is dont care
  labels = np.empty((len(inds_inside),), dtype=np.float32) #(k,)
  labels.fill(-1)

  '''计算每一个anchor与所有gt_box的IOU值'''
  '''
  overlaps between the anchors and the gt boxes
  overlaps : 二维数组,其中每个元素是IOU值.(in_anchor_size,gt_size)
  若A = [a,b,c,d] B = [d,b,c] ,其中a b c d的维度是(1,4):(X左上,Y左上,X右下,Y右下)并且它们之间没有交集
  C = bbox_overlaps(A,B),,那么C结果为[[0,0,0],[0,1,0],[0,0,1],[1,0,0]]
  '''
  overlaps = bbox_overlaps(
    np.ascontiguousarray(anchors, dtype=np.float),
    np.ascontiguousarray(gt_boxes, dtype=np.float))

  '''筛选出符合条件的框'''
  '''
  argmax_overlaps[i]:表示在所有gt_box中，与anchors[i]的IOU值最大的gt_box的索引号
  max_overlaps[i]:表示在anchors[i]与gt_box[argmax_overlaps[i]]的IOU值
  gt_argmax_overlaps[i]:表示在所有anchors中，与gt_box[i]的IOU值最大的那个anchor的索引号
  gt_max_overlaps[i]:gt_box[i]与anchors[gt_argmax_overlaps[i]]的IOU值
  '''
  argmax_overlaps = overlaps.argmax(axis=1)#(in_anchor_zize, )
  max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]#(in_anchor_zize , )
  gt_argmax_overlaps = overlaps.argmax(axis=0)#(gt_size, )
  gt_max_overlaps = overlaps[gt_argmax_overlaps,
                             np.arange(overlaps.shape[1])]#(gt_size, )

  '''

  '''
  gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

  #cfg.TRAIN.RPN_CLOBBER_POSITIVES=False
  if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
    
    '''assign bg labels first so that positive labels can clobber them first set the negatives'''
    #cfg.TRAIN.RPN_NEGATIVE_OVERLAP=0.3
    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

  '''选取postive的框,并把对应的标签设置为1>>>labels[i]=1'''
  #the anchor with the highest IOUwith a ground-truth box is postive
  labels[gt_argmax_overlaps] = 1
  # an anchor that has an IOU overlap higer than 0.7 with any ground-truth box is postive
  # cfg.TRAIN.RPN_POSITIVE_OVERLAP=0.7
  labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

  #cfg.TRAIN.RPN_CLOBBER_POSITIVES=False
  if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
    # assign bg labels last so that negative labels can clobber positives
    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
  
  
  '''下面这段程序就是判断正样本的数量是否超过128个'''
  # cfg.TRAIN.RPN_FG_FRACTION=0.5 cfg.TRAIN.RPN_BATCHSIZE=256
  num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE) #128
  fg_inds = np.where(labels == 1)[0]
  if len(fg_inds) > num_fg:#如果正样本的个数大于128
    disable_inds = npr.choice(
      fg_inds, size=(len(fg_inds) - num_fg), replace=False)#随机选取多余的正样本
    labels[disable_inds] = -1 #把多余的正样本的标签设置为为dont care

  '''下面这段程序就是判断负样本的数量是否大于128个'''
  num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1) #理论上负样本的个数
  bg_inds = np.where(labels == 0)[0] #实际负样本的个数
  if len(bg_inds) > num_bg: #如果实际负样本的个数大于理论上负样本的个数
    disable_inds = npr.choice(
      bg_inds, size=(len(bg_inds) - num_bg), replace=False) #选出多余的负样本
    labels[disable_inds] = -1 #把多余的负样本的标签设置为为dont care

  '''计算anchors的每一个anchor与其对应的gt_box之间的[dx,dy,dw,dh]'''
  bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32) #(in_anchor_size,4)
  bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :]) #(in_anchor_size,4)

  '''bbox_inside_weights:每行值为：[0,0,0,0]或者[1,1,1,1]'''
  bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)#(in_anchor_size,4)
  # only the positive ones have regression targets cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS=[1,1,1,1]
  bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

  '''bbox_outside_weights每行值为：[0,0,0,0]或者[1/256,1/256,1/256,1/256]'''
  bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)#(in_anchor_size,4)

  #cfg.TRAIN.RPN_POSITIVE_WEIGHT=-0.1
  if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
    # uniform weighting of examples (given non-uniform sampling)
    '''要注意labels中，元素为1与-1的总数为256'''
    num_examples = np.sum(labels >= 0) #256
    positive_weights = np.ones((1, 4)) * 1.0 / num_examples #1/256
    negative_weights = np.ones((1, 4)) * 1.0 / num_examples
  else:
    assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
            (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
    positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                        np.sum(labels == 1))
    negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                        np.sum(labels == 0))
  bbox_outside_weights[labels == 1, :] = positive_weights #[1/256,1/256,1/256,1/256]
  bbox_outside_weights[labels == 0, :] = negative_weights #[1/256,1/256,1/256,1/256]

  # map up to original set of anchors
  labels = _unmap(labels, total_anchors, inds_inside, fill=-1)#labels从(in_anchor_size, )>>>(w*h*9,)
  bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)#bbox_targets从(in_anchor_size,4)>>>(w*h*9,4)
  bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)#同上
  bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)#同上

  # labels
  labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
  labels = labels.reshape((1, 1, A * height, width))
  rpn_labels = labels

  # bbox_targets
  bbox_targets = bbox_targets \
    .reshape((1, height, width, A * 4))
  rpn_bbox_targets = bbox_targets

  # bbox_inside_weights
  bbox_inside_weights = bbox_inside_weights \
    .reshape((1, height, width, A * 4))
  rpn_bbox_inside_weights = bbox_inside_weights

  # bbox_outside_weights
  bbox_outside_weights = bbox_outside_weights \
    .reshape((1, height, width, A * 4))
  rpn_bbox_outside_weights = bbox_outside_weights

  '''
  rpn_labels : (1,1,h*9,w)
  rpn_bbox_targets : (1,h,w,36)
  rpn_bbox_inside_weights : (1,h,w,36)
  rpn_bbox_outside_weights : (1,h,w,36)
  '''
  return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):
  """ Unmap a subset of item (data) back to the original set of items (of
  size count) """
  if len(data.shape) == 1:
    ret = np.empty((count,), dtype=np.float32)
    ret.fill(fill)
    ret[inds] = data
  else:
    ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
    ret.fill(fill)
    ret[inds, :] = data
  return ret


def _compute_targets(ex_rois, gt_rois):
  """Compute bounding-box regression targets for an image."""

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 5

  return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
