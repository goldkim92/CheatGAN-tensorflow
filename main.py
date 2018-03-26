#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 15:39:31 2018

@author: jm
"""

import argparse
import os
import tensorflow as tf
from model import dcgan

# argument parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase',          type=str,   default='train',    help='train or test')
parser.add_argument('--gpu_number',     type=str,   default='0')
parser.add_argument('--data_dir',       type=str,   default=os.path.join('.','MNIST_data'))
parser.add_argument('--log_dir',        type=str,   default='log') # in assets/ directory
parser.add_argument('--ckpt_dir',       type=str,   default='checkpoint') # in assets/ directory
parser.add_argument('--sample_dir',     type=str,   default='sample') # in assets/ directory
parser.add_argument('--test_dir',       type=str,   default='test') # in assets/ directory
parser.add_argument('--assets_dir',     type=str,   default=None,   required=True) # if assets_dir='aa' -> assets_dir='./assets/aa'
parser.add_argument('--epoch',          type=int,   default=20)
parser.add_argument('--batch_size',     type=int,   default=16)
parser.add_argument('--image_size',     type=int,   default=28)
parser.add_argument('--input_size',     type=int,   default=32)
parser.add_argument('--image_c',        type=int,   default=1)
parser.add_argument('--z_dim',          type=int,   default=100)
parser.add_argument('--nf',             type=int,   default=128) # number of filters
parser.add_argument('--lambda_gp',      type=int,   default=10)
parser.add_argument('--lambda_ae',      type=int,   default=0.001) # reconstruction loss weight
parser.add_argument('--loss_type',      type=str,   default='WGAN') # GAN or WGAN
parser.add_argument('--lr',             type=float, default=0.0001) # learning_rate
parser.add_argument('--beta1',          type=float, default=0.9)
#parser.add_argument('--continue_train', type=bool,  default=False)
parser.add_argument('--sample_step',    type=int,   default=300) 
parser.add_argument('--log_step',       type=int,   default=1) 
parser.add_argument('--ckpt_step',      type=int,   default=500) 

args = parser.parse_args()

def main(_):
    # setting
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_number
    tf.reset_default_graph()
    
    assets_dir = os.path.join('.','assets',args.assets_dir)
    args.log_dir = os.path.join(assets_dir, args.log_dir)
    args.ckpt_dir = os.path.join(assets_dir, args.ckpt_dir)
    args.sample_dir = os.path.join(assets_dir, args.sample_dir)
    args.test_dir = os.path.join(assets_dir, args.test_dir)
    
    # make directory if not exist
    try: os.makedirs(args.log_dir)
    except: pass
    try: os.makedirs(args.ckpt_dir)
    except: pass
    try: os.makedirs(args.sample_dir)
    except: pass
    try: os.makedirs(args.test_dir)
    except: pass

    # run session
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = dcgan(sess,args)
        model.train() if args.phase == 'train' else model.test()

# run main function
if __name__ == '__main__':
    tf.app.run()