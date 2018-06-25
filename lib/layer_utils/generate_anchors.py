#coding=utf-8
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from ipdb import set_trace


# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
  """
  Generate anchor (reference) windows by enumerating aspect ratios X
  scales wrt a reference (0, 0, 15, 15) window.
  scales=(8,16,32)
  ratios=[0.5, 1, 2]
  """
  base_anchor = np.array([1, 1, base_size, base_size]) - 1 #[0,0,15,15]
  ratio_anchors = _ratio_enum(base_anchor, ratios)
  anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                       for i in range(ratio_anchors.shape[0])])
  return anchors

#输入:一个anchor的左上角与右下角坐标/***/输出:这个anchor的中心坐标以及wide height
def _whctrs(anchor):
  """
  anchor=[0,0,15,15]
  Return width, height, x center, and y center for an anchor (window).
  """

  w = anchor[2] - anchor[0] + 1#16
  h = anchor[3] - anchor[1] + 1#16
  x_ctr = anchor[0] + 0.5 * (w - 1)#7.5
  y_ctr = anchor[1] + 0.5 * (h - 1)#7.5
  return w, h, x_ctr, y_ctr

#输入:wide height 中心坐标
#输出:对应的左上角及右下角的anchor
def _mkanchors(ws, hs, x_ctr, y_ctr):
  """
  ws=[23,16,11]
  hs=[12,16,22]
  x_ctr=7.5 y_ctr=7.5
  Given a vector of widths (ws) and heights (hs) around a center
  (x_ctr, y_ctr), output a set of anchors (windows).
  """

  ws = ws[:, np.newaxis]#[[23],[16],[11]]
  hs = hs[:, np.newaxis]#[[12],[16],[22]]
  anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                       y_ctr - 0.5 * (hs - 1),
                       x_ctr + 0.5 * (ws - 1),
                       y_ctr + 0.5 * (hs - 1)))
  """
  array([[ -3.5,   2. ,  18.5,  13. ],
       [  0. ,   0. ,  15. ,  15. ],
       [  2.5,  -3. ,  12.5,  18. ]])
  """
  return anchors

#枚举一个anchor的各种宽高比，以anchor[0 0 15 15]为例,ratios[0.5,1,2]
#即输出面积大小一定(16*16),,宽高比为1:2 1:1 2:1的anchors
def _ratio_enum(anchor, ratios):
  """
  anchor=[0,0,15,15]
  ratios=[0.5,1,2]
  Enumerate a set of anchors for each aspect ratio wrt an anchor.
  """

  w, h, x_ctr, y_ctr = _whctrs(anchor)#16,16,7.5,7.5
  size = w * h #256
  size_ratios = size / ratios#[512,256,128]
  ws = np.round(np.sqrt(size_ratios))#[23,16,11]
  hs = np.round(ws * ratios)#[12,16,22]
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  """
  array([[ -3.5,   2. ,  18.5,  13. ],
       [  0. ,   0. ,  15. ,  15. ],
       [  2.5,  -3. ,  12.5,  18. ]])
  """
  return anchors

#枚举一个anchor的各种尺度，以anchor[0 0 15 15]为例,scales[8 16 32]  
def _scale_enum(anchor, scales):
  """
  anchor=[-3.5,2,18.5,13]
  scales=[8,16,32]
  Enumerate a set of anchors for each scale wrt an anchor.
  """

  w, h, x_ctr, y_ctr = _whctrs(anchor)
  ws = w * scales
  hs = h * scales
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors


if __name__ == '__main__':
  import time

  t = time.time()
  a = generate_anchors()
  print(time.time() - t)
  print(a)
  from IPython import embed; embed()
