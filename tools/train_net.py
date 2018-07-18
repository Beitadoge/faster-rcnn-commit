#coding=utf-8
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.train_val import get_training_roidb, train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from datasets.factory import get_imdb
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1
from ipdb import set_trace


import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)
  parser.add_argument('--weight', dest='weight',
                      help='initialize with pretrained model weights',
                      type=str)
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to train on',
                      default='voc_2007_trainval', type=str)
  parser.add_argument('--imdbval', dest='imdbval_name',
                      help='dataset to validate on',
                      default='voc_2007_test', type=str)
  parser.add_argument('--iters', dest='max_iters',
                      help='number of iterations to train',
                      default=70000, type=int)
  parser.add_argument('--tag', dest='tag',
                      help='tag of the model',
                      default=None, type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152, mobile',
                      default='res50', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args



'''Combine multiple roidbs'''
def combined_roidb(imdb_names):
  '''
  输入变量：imdb_names = 'voc_2007_trainval'
  '''
  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD) #这句是什么意思？？？？
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))  ##cfg.TRAIN.PROPOSAL_METHOD='gt'
    roidb = get_training_roidb(imdb) 

    '''
    roidb是一个列表,长度为10022,(10022,),,每个元素是一个字典
    imdb.roidb[0]:
    {
      'gt_classes': array([9, 9, 9], dtype=int32), 'max_classes': array([9, 9, 9]), 
      'image': '/home/beitadoge/.../000005.jpg', 'flipped': False, 'width': 500, 
      'boxes': array([[262, 210, 323, 338],[164, 263, 252, 371],[240, 193, 294, 298]], dtype=uint16), 
      'max_overlaps': array([ 1.,  1.,  1.], dtype=float32), 'height': 375, 
      'seg_areas': array([ 7998.,  9701.,  5830.], dtype=float32), 
      'gt_overlaps': <3x21 sparse matrix of type '<type 'numpy.float32'>'with 3 stored elements in Compressed Sparse Row format>
    }
    '''
    return roidb
  
  roidbs = [get_roidb(s) for s in imdb_names.split('+')] #imdb_names.split('+')=['voc_2007_trainval']
  roidb = roidbs[0]#是一个列表,(10022,)
  if len(roidbs) > 1: #len(roidbs)=1
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names) #返回的就是类pascal_voc的一个实例对象pascal_voc(trainval,2007)
  return imdb, roidb


if __name__ == '__main__':
  args = parse_args()
  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  np.random.seed(cfg.RNG_SEED) #cfg.RNG_SEED=3

  # train set
  imdb, roidb = combined_roidb(args.imdb_name)#args.imdb_name = 'voc_2007_trainval'
  print('{:d} roidb entries'.format(len(roidb)))

  # output directory where the models are saved
  output_dir = get_output_dir(imdb, args.tag)
  print('Output will be saved to `{:s}`'.format(output_dir))

  # tensorboard directory where the summaries are saved during training
  tb_dir = get_output_tb_dir(imdb, args.tag)
  print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

  # also add the validation set, but with no flipping images
  orgflip = cfg.TRAIN.USE_FLIPPED
  cfg.TRAIN.USE_FLIPPED = False
  _, valroidb = combined_roidb(args.imdbval_name)
  print('{:d} validation roidb entries'.format(len(valroidb)))
  cfg.TRAIN.USE_FLIPPED = orgflip

  #选择基础网络结构
  if args.net == 'vgg16':
    net = vgg16()
  elif args.net == 'res50':
    net = resnetv1(num_layers=50)
  elif args.net == 'res101':
    net = resnetv1(num_layers=101)
  elif args.net == 'res152':
    net = resnetv1(num_layers=152)
  elif args.net == 'mobile':
    net = mobilenetv1()
  else:
    raise NotImplementedError
  
  train_net(net, imdb, roidb, valroidb, output_dir, tb_dir,
            pretrained_model=args.weight,
            max_iters=args.max_iters)
