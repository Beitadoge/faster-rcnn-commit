#coding=utf-8
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick, Sean Bell and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from model.config import cfg
from model.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps


def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes):
  """
  Assign object detection proposals to ground-truth targets. Produces proposal
  classification labels and bounding-box regression targets.
  """
  # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
  # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
  all_rois = rpn_rois
  all_scores = rpn_scores

  # Include ground-truth boxes in the set of candidate rois
  # cfg.TRAIN.USE_GT=False 
  '''
  为什么可以用gt_box去用作训练?????
  '''
  if cfg.TRAIN.USE_GT:
    zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
    all_rois = np.vstack(
      (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
    )
    # not sure if it a wise appending, but anyway i am not using it
    all_scores = np.vstack((all_scores, zeros))

  num_images = 1
  rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images  #cfg.TRAIN.BATCH_SIZE=256
  fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)  #cfg.TRAIN.FG_FRACTION=0.25

  # Sample rois with classification labels and bounding box regression
  # targets
  labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(
    all_rois, all_scores, gt_boxes, fg_rois_per_image,
    rois_per_image, _num_classes)

  rois = rois.reshape(-1, 5)
  roi_scores = roi_scores.reshape(-1)
  labels = labels.reshape(-1, 1)
  bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
  bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
  bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

  '''
  [labels] : 大小为[256, ],保存着rois中所有positive的proposal的类别,negative的proposal的类别为0
  [rois] : 大小为[256,5],每行元素为[0,x1,y1,x2,y2]
  [roi_scores] : 大小为[256, ]保存着rois中每一个proposal的得分
  [bbox_targets] :[256 , num_class×4] 每行元素为［0,0,0,0,0,dx,dy,dw,dh,0,0,0,...］
  [bbox_inside_weights] : [256 , num_class×4] 每行元素为［0,0,0,0,0,1,1,1,1,0,0,0,...］
  '''
  return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


def _get_bbox_regression_labels(bbox_target_data, num_classes):
  """Bounding-box regression targets (bbox_target_data) are stored in a
  compact form N x (class, tx, ty, tw, th)

  This function expands those targets into the 4-of-4*K representation used
  by the network (i.e. only one class has non-zero targets).

  Returns:
      bbox_target (ndarray): N x 4K blob of regression targets
      bbox_inside_weights (ndarray): N x 4K blob of loss weights
  """

  clss = bbox_target_data[:, 0]
  bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
  bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
  inds = np.where(clss > 0)[0]
  for ind in inds:
    cls = clss[ind]
    start = int(4 * cls)
    end = start + 4
    bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
    bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS#cfg.TRAIN.BBOX_INSIDE_WEIGHTS=[1,1,1,1]
  return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
  """Compute bounding-box regression targets for an image."""
  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 4

  #targets:是anchor距离gt_box的 :[dx, dy，dw, dh]
  targets = bbox_transform(ex_rois, gt_rois)
  #cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED=True
  if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    # Optionally normalize targets by a precomputed mean and stdev
    '''
    下面这条语句的作用不懂
    '''
    targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
               / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
  return np.hstack(
    (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


'''
该函数作用就是从all_rois中的anchor中，随机筛选rois_per_image个anchor作为proposal,并且在
这些proposal中，前景一共有fg_rois_per_image个．并返回：
[labels] : 大小为[256, ],保存着rois中所有positive的proposal的类别,negative的proposal的类别为0
[rois] : 大小为[256,5],每行元素为[0,x1,y1,x2,y2]
[roi_scores] : 大小为[256, ]保存着rois中每一个proposal的得分
[bbox_targets] :[256 , num_class×4] 每行元素为［0,0,0,0,0,dx,dy,dw,dh,0,0,0,...］
[bbox_inside_weights] : [256 , num_class×4] 每行元素为［0,0,0,0,0,1,1,1,1,0,0,0,...］
'''
def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
  '''
  [all_rois] : 大小为[anchor_size,5],,anchor_size大约为2k,,每行元素为[0,x1,y1,x2,y2],其中(x1,y1)为左上角的坐标.(x2,y2)为右下角的坐标
	[all_scores] : 大小为[anchor_size, ],,保存着每个框所得到的分数
	[gt_boxes] : 大小为[gt_size,5],,gt_size为一幅图中所有框的个数.每行元素为[x1,y1,x2,y2,gt_box_class]其中(x1,y1)为左上角的坐标.(x2,y2)为右下角的坐标,,gt_box_class为该框里面的物体的类别
	[fg_rois_per_image] : 每张图片中前景的个数
	[rois_per_image] : 每张图片中索要提取的proposal数
	[num_classes] : 需要分类的类别总数
  '''

  #overlaps: (anchors_size,gt_box_size)
  overlaps = bbox_overlaps(
    np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
    np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
  gt_assignment = overlaps.argmax(axis=1)#大小为[anchors_size , ]， gt_assignment[i]表示与all_rois[i]IOU值最大的gt_boxes的索引号
  max_overlaps = overlaps.max(axis=1)#大小为[anchors_size , ], max_overlaps[i]表示：与all_rois[i]与gt_boxes[gt_assignment[i]]的IOU值
  labels = gt_boxes[gt_assignment, 4]#大小为[anchors_size , ]，labels[i]表示:all_rois[i]这个框里面物体的具体类别，如猫，狗

  # Select foreground RoIs as those with >= FG_THRESH overlap
  # cfg.TRAIN.FG_THRESH=0.5
  fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
  # Guard against the case when an image has fewer than fg_rois_per_image
  # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
  # cfg.TRAIN.BG_THRESH_HI=0.5  cfg.TRAIN.BG_THRESH_LO=0
  bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                     (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]

  # Small modification to the original version where we ensure a fixed number of regions are sampled
  if fg_inds.size > 0 and bg_inds.size > 0:
    fg_rois_per_image = min(fg_rois_per_image, fg_inds.size)
    #replacement 代表的意思是抽样之后还放不放回去，如果是False的话，那么出来的三个数都不一样
    #如果是True的话， 有可能会出现重复的，因为前面的抽的放回去了。
    fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_image), replace=False)
    bg_rois_per_image = rois_per_image - fg_rois_per_image
    to_replace = bg_inds.size < bg_rois_per_image
    bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace)
  elif fg_inds.size > 0:
    to_replace = fg_inds.size < rois_per_image
    fg_inds = npr.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
    fg_rois_per_image = rois_per_image
  elif bg_inds.size > 0:
    to_replace = bg_inds.size < rois_per_image
    bg_inds = npr.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
    fg_rois_per_image = 0
  else:
    import pdb
    pdb.set_trace()

  # The indices that we're selecting (both fg and bg)
  keep_inds = np.append(fg_inds, bg_inds)
  # Select sampled values from various arrays:
  labels = labels[keep_inds]
  # Clamp labels for the background RoIs to 0
  labels[int(fg_rois_per_image):] = 0
  rois = all_rois[keep_inds]
  roi_scores = all_scores[keep_inds]

  bbox_target_data = _compute_targets(
    rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

  bbox_targets, bbox_inside_weights = \
    _get_bbox_regression_labels(bbox_target_data, num_classes)

  return labels, rois, roi_scores, bbox_targets, bbox_inside_weights
