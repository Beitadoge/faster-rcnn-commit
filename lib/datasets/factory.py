#coding=utf-8
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco

import numpy as np

# Set up voc_<year>_<split> 
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}_diff'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, use_diff=True))

# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

#Get an imdb (image database) by name.
def get_imdb(name):
  '''
  eg:
  name = "voc_2007_trainval"
  '''
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]() #返回的就是类pascal_voc的一个实例对象pascal_voc(trainval,2007)


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
