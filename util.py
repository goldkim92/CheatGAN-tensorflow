import numpy as np
import tensorflow as tf
import os
from glob import glob
from collections import namedtuple
import scipy.misc as scm

#%%
def unstack(img, axis):
    d =img.shape[axis]
    arr = [np.squeeze(a,axis=axis) for a in np.split(img, d, axis=axis)]
    return arr

def make3d(img, nx,ny):
    '''
    img:[b,h,w,c]--> img:[ny*h,nx*w,c]
    '''
    if type(img) is np.ndarray:
        b,h,w,c= img.shape
        black = np.zeros([nx*ny-b,h,w,c],dtype=img.dtype)
        img = np.concatenate([img,black],axis=0)
        img = np.reshape(img, [ny,nx,h,w,c])
        img = unstack(img,axis=0) # ny *[nx,h,w,c]
        img = np.concatenate(img, axis=1) # [nx, ny*h,w,c]
        img = unstack(img, axis=0) # nx*[ny*h, w,c]
        img = np.concatenate(img, axis=1) # [ny*h, nx*w,c]
        return img
    else: # tf.Tensor
        b,h,w,c = img.get_shape().as_list()
        print(img.get_shape().as_list())
#        black = tf.zeros([nx*ny-b,h,w,c],dtype=img.dtype)
#        img = tf.concat([img,black],axis=0)
        img = tf.reshape(img, [ny,nx,h,w,c])
        img = tf.unstack(img,axis=0) # ny *[nx,h,w,c]
        img = tf.concat(img, axis=1) # [nx, ny*h,w,c]
        img = tf.unstack(img, axis=0) # nx*[ny*h, w,c]
        img = tf.concat(img, axis=1) # [ny*h, nx*w,c]
        img = tf.stack([img])
        return img


#%% 
'''
cifar10 preprocessing
'''
        
def one_hot(labels):
    onehot = np.zeros([labels.shape[0],10])
    for i, label in enumerate(labels):
        onehot[i,label] = 1
    return onehot

def next_batch(data, batch_size, idx):
    if idx == 0:
        np.random.shuffle(data)
        
    batch = data[idx*batch_size:(idx+1)*batch_size,:,:,:]
    return batch
    

#%% 
'''
celebA preprocessing & post-processing
'''
    
def load_data_list(data_dir):
    path = os.path.join(data_dir, 'train', '*')
    file_list = glob(path)
    return file_list

def preprocess_image(file_list, input_size, phase='train'):
    imgA = [get_image(img_path, input_size, phase=phase) for img_path in file_list]
    return np.array(imgA)

def get_image(img_path, input_size, phase='train'):
    img = scm.imread(img_path) # 218*178*3
    img_crop = img[34:184,14:164,:] #188*160*3
    img_resize = scm.imresize(img_crop,[input_size,input_size,3])
    img_resize = img_resize/127.5 - 1.
    
    if phase == 'train' and np.random.random() >= 0.5:
        img_resize = np.flip(img_resize,1)
    
    return img_resize

def inverse_image(img):
    img = (img + 1.) * 127.5
    img[img > 255] = 255.
    img[img < 0] = 0.
    return img.astype(np.uint8)


#def save_images(realA, realB, fake_B, image_size, sample_file, num=10):
#    # [5,6] with the seequnce of (realA, realB, fakeB), total 10 set save
#    if np.array_equal(realA, realB): # for test
#        img = np.concatenate((realA[:5,:,:,:],fake_B[:5,:,:,:],
#                          realA[5:,:,:,:],fake_B[5:,:,:,:]), axis=0)
#        img = make3d(img, image_size, row=5, col=4)
#    else: # for sample while training
#        img = np.concatenate((realA[:5,:,:,:],realB[:5,:,:,:],fake_B[:5,:,:,:],
#                          realA[5:,:,:,:],realB[5:,:,:,:],fake_B[5:,:,:,:]), axis=0)
#        img = make3d(img, image_size, row=5, col=6)
#    img = inverse_image(img)
#    scm.imsave(sample_file, img)
