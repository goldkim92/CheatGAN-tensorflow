import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc as scm
import os

plt.switch_backend('agg')

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
#        black = tf.zeros([nx*ny-b,h,w,c],dtype=img.dtype)
#        img = tf.concat([img,black],axis=0)
        img = tf.reshape(img, [ny,nx,h,w,c])
        img = tf.unstack(img,axis=0) # ny *[nx,h,w,c]
        img = tf.concat(img, axis=1) # [nx, ny*h,w,c]
        img = tf.unstack(img, axis=0) # nx*[ny*h, w,c]
        img = tf.concat(img, axis=1) # [ny*h, nx*w,c]
        img = tf.stack([img])
        return img

# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)    
    
def save_scattered_image(latent_prior, latent_real, each_len, epoch, test_dir):
    ''' 1000 latent vectors for image-encoded-latent and prior respectively '''
    
    z = np.concatenate([latent_prior, latent_real], axis=0)
    colors = [0]*each_len + [1]*each_len
    N = 2
    
    # plot
    plt.figure(figsize=(10,8))
    plt.scatter(z[:,0], z[:,1], c=colors, marker='o', edgecolor='none', cmap=discrete_cmap(N,'jet'))
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    axes.set_xlim([-6,6])
    axes.set_ylim([-6,6])
    plt.grid(True)
    plt.savefig(os.path.join(test_dir,'tmp_{}.png'.format(str(epoch))))
    
    plot_img = scm.imread(os.path.join(test_dir,'tmp_{}.png'.format(str(epoch))))
    plot_img = np.expand_dims(plot_img, axis=0)
    plt.close()
    return plot_img
    
