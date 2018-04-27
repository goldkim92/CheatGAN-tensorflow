import numpy as np
import tensorflow as tf

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
