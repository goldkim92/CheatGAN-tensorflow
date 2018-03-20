import numpy as np


def unstack(img, axis):
    d =img.shape[axis]
    arr = [np.squeeze(a,axis=axis) for a in np.split(img, d, axis=axis)]
    return arr

def make3d(img, nx,ny):
    '''
    img:[b,h,w,c]--> img:[ny*h,nx*w,c]
    '''
    b,h,w,c= img.shape
    black = np.zeros([nx*ny-b,h,w,c],dtype=img.dtype)
    img = np.concatenate([img,black],axis=0)
    img = np.reshape(img, [ny,nx,h,w,c])
    img = unstack(img,axis=0) # ny *[nx,h,w,c]
    img = np.concatenate(img, axis=1) # [nx, ny*h,w,c]
    img = unstack(img, axis=0) # nx*[ny*h, w,c]
    img = np.concatenate(img, axis=1) # [ny*h, nx*w,c]
    return img