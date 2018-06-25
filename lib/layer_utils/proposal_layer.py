#coding=utf-8
# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from model.config import cfg
from model.bbox_transform import bbox_transform_inv, clip_boxes, bbox_transform_inv_tf, clip_boxes_tf
from model.nms_wrapper import nms

def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  """A simplified version compared to fast/er RCNN
     For details please see the technical report
  """
  if type(cfg_key) == bytes:
      cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
  scores = scores.reshape((-1, 1))
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
  proposals = clip_boxes(proposals, im_info[:2])

  # Pick the top region proposals
  order = scores.ravel().argsort()[::-1]
  if pre_nms_topN > 0:
    order = order[:pre_nms_topN]
  proposals = proposals[order, :]
  scores = scores[order]

  # Non-maximal suppression
  keep = nms(np.hstack((proposals, scores)), nms_thresh)

  # Pick th top region proposals after NMS
  if post_nms_topN > 0:
    keep = keep[:post_nms_topN]
  proposals = proposals[keep, :]
  scores = scores[keep]

  # Only support single image as input
  batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
  blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

  return blob, scores


def proposal_layer_tf(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  """
  rpn_cls_prob = Tensor("vgg_16_1/rpn_cls_prob/transpose_1:0", shape=(1, ?, ?, 18), dtype=float32)
  rpn_bbox_pred = Tensor("vgg_16_1/rpn_bbox_pred/BiasAdd:0", shape=(1, ?, ?, 36), dtype=float32)
  im_info = Tensor("Placeholder_1:0", shape=(3,), dtype=float32)
  cfg_key = TRAIN
  _feat_stride = [16]
  anchors = Tensor("vgg_16_1/ANCHOR_default/Cast:0", shape=(?, 4), dtype=float32)
  num_anchors = 9
  """
  if type(cfg_key) == bytes:#True
    cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N   #12000
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N #2000
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH #0.7

  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  scores = tf.reshape(scores, shape=(-1,))
  rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))

  proposals = bbox_transform_inv_tf(anchors, rpn_bbox_pred)#得到修正后的框
  proposals = clip_boxes_tf(proposals, im_info[:2])#去除不合格的预测框

  # Non-maximal suppression,,得到非极大值抑制后的框的索引信息
  indices = tf.image.non_max_suppression(proposals, scores, max_output_size=post_nms_topN, iou_threshold=nms_thresh)

  #根据刷选出来的索引号,得到对应的框
  '''
  a = tf.gather([[1,2],[4,5],[6,7],[8,9]],[0,2,3])
  a.eval()
  >>>[[1,2],[6,7],[8,9]]
  '''
  boxes = tf.gather(proposals, indices)
  boxes = tf.to_float(boxes)
  scores = tf.gather(scores, indices)
  scores = tf.reshape(scores, shape=(-1, 1))

  # Only support single image as input
  '''
  假设经过'Non-maximal suppression'后,留下了2k个框,那么boxes的维度就是(2k,4)
  下面语句的作用就是往boxes的第二维度添加个0,boxes的维度变为(2k,5),如[[1,2,3,4],[5,6,7,8]]>>>[[0,1,2,3,4],[0,5,6,7,8]]
  '''
  batch_inds = tf.zeros((tf.shape(indices)[0], 1), dtype=tf.float32)
  blob = tf.concat([batch_inds, boxes], 1)#

  '''
  blob:(2k,5):2k代表非极大值抑制后剩余的框
  scores:(2k,1)
  '''
  return blob, scores


