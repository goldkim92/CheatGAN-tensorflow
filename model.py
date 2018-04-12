import os
import numpy as np
import random
import scipy.misc as scm
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from collections import namedtuple
from tqdm import tqdm
from glob import glob

from module import generator, discriminator, gan_loss, wgan_gp_loss
from util import make3d

class dcgan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.phase = args.phase
        self.data_dir = args.data_dir
        self.log_dir = args.log_dir
        self.ckpt_dir = args.ckpt_dir
        self.sample_dir = args.sample_dir
        self.test_dir = args.test_dir
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.input_size = args.input_size
        self.image_c = args.image_c
        self.z_dim = args.z_dim
        self.nf = args.nf
        self.lr = args.lr
        self.beta1 = args.beta1
        self.lambda_gp = args.lambda_gp
        self.loss_type = args.loss_type
        self.sample_step = args.sample_step
        self.log_step = args.log_step
        self.ckpt_step = args.ckpt_step

        # hyper parameter for building the module
        OPTIONS = namedtuple('options', ['batch_size', 'image_c', 'nf', 'z_dim'])
        self.options = OPTIONS(self.batch_size, self.image_c, self.nf, self.z_dim)
        
        # build model & make checkpoint saver
        self.build_model()
        self.saver = tf.train.Saver()
        
    def build_model(self):
        # placeholder
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='noise')
        self.real = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, self.image_c], name='real')
        
        # generate image
        self.fake = generator(self.z, self.options, False, name='gen')
        
        # discrimate image
        self.real_d = discriminator(self.real, self.options, False, name='disc')
        self.fake_d = discriminator(self.fake, self.options, True, name='disc')

            
        # loss : discriminator loss
        if self.loss_type == 'GAN':
            d_real_loss = gan_loss(self.real_d, tf.ones_like(self.real_d))
            d_fake_loss = gan_loss(self.fake_d, tf.zeros_like(self.fake_d))
            self.d_loss = d_real_loss + d_fake_loss
        else: # 'WGAN'
            gp_loss = wgan_gp_loss(self.real, self.fake, self.options)
            self.d_loss = tf.reduce_mean(self.fake_d) - tf.reduce_mean(self.real) + self.lambda_gp * gp_loss
        
        # loss : generator loss
        if self.loss_type == 'GAN':
            self.g_loss = gan_loss(self.fake_d, tf.ones_like(self.fake_d))
        else: # 'WGAN'
            self.g_loss = -tf.reduce_mean(self.fake_d)
            
        # trainable variables
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'disc' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]
        
        # optimizer
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(self.g_loss, var_list=self.g_vars)
        
    def train(self):
        # summary setting
        self.summary()
        
        # load train data list
        mnist = input_data.read_data_sets(self.data_dir, one_hot=True)
        
        # random seed for sampling test during training
        z_rand_sample = np.random.normal(0,1,size=(self.batch_size, self.z_dim)).astype(np.float32)
        
        # variable initialize
        self.sess.run(tf.global_variables_initializer())
        
        batch_idxs = mnist.train.num_examples // self.batch_size
        count_idx = 0
        #train
        for epoch in range(self.epoch):
            print('Epoch[{}/{}]'.format(epoch+1, self.epoch))
            for idx in tqdm(range(batch_idxs)):
                # get batch images and labels
                images, labels = mnist.train.next_batch(self.batch_size)
                images = images.reshape([self.batch_size, self.image_size, self.image_size])
                images = self.zero_padding(images) # 28*28 --> 32*32*1
                
                # get z value (random noise) 
                z_value = np.random.normal(0,1,size=(self.batch_size, self.z_dim)).astype(np.float32)
#                feed = {self.z: z_value}
#                fake = self.sess.run(self.fake, feed_dict=feed)
                
                # update D network
                feed = {self.real: images, self.z: z_value}
                _, d_summary = self.sess.run([self.d_optim, self.d_sum], feed_dict=feed)
                
                # update G network
                feed = {self.z: z_value}
                _, g_summary = self.sess.run([self.g_optim, self.g_sum], feed_dict=feed)
                
                count_idx += 1

                # log step (summary)
                if count_idx % self.log_step == 0:
                    self.writer.add_summary(d_summary, count_idx)
                    self.writer.add_summary(g_summary, count_idx)
                    
                # checkpoint step
                if count_idx % self.ckpt_step == 0:
                    self.checkpoint_save(count_idx)
                
                # sample step
                if count_idx % self.sample_step == 0:
                    feed = {self.z: z_rand_sample}
                    fake_sample, img_summary = self.sess.run([self.fake, self.img_sum], feed_dict=feed)
                    fake_sample = make3d(fake_sample, int(np.sqrt(self.batch_size)), int(np.sqrt(self.batch_size)) )
                    
                    scm.imsave(os.path.join(self.sample_dir, str(count_idx)+'.png'), np.squeeze(fake_sample))
                    self.writer.add_summary(img_summary, count_idx)
                    
        
    def summary(self):
        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        
        # session: discriminator
        tf.summary.scalar('loss/d', self.d_loss, collections=['disc'])
        tf.summary.scalar('val/d_real', tf.reduce_mean(self.real_d), collections=['disc'])
        tf.summary.scalar('val/d_fake', tf.reduce_mean(self.fake_d), collections=['disc'])
#        self.d_sum = tf.summary.merge([sum_d_loss, sum_d_real_val, sum_d_fake_val])
        self.d_sum = tf.summary.merge_all('disc')
        
        # session: generator
        self.g_sum = tf.summary.scalar('loss/g', self.g_loss)
#        self.g_sum = tf.summary.merge([sum_g])
    
        # session: image
        self.img_sum = tf.summary.image('sample image', make3d(self.fake[:12,:,:,:],4,3))
    
    def checkpoint_save(self, count):
        model_name = 'dcgan.model'
        self.saver.save(self.sess, os.path.join(self.ckpt_dir, model_name), global_step=count)
        
    
    def zero_padding(self, images):
        pad_imgs = np.zeros([self.batch_size, self.input_size, self.input_size]) # 32*32
        pad_imgs[:,2:-2,2:-2] = images
        return np.expand_dims(pad_imgs,axis=3) # 32*32*1
        