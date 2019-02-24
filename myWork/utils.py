import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import numpy as np

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=48, hrg=48, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[12, 12], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def ssim_loss_fn(real, fake, val=2.0):

#   real_ = (real + 1.) * 127.5 
#   fake_ = (fake + 1.) * 127.5 
    real = tf.image.rgb_to_grayscale(real)
    fake = tf.image.rgb_to_grayscale(fake)
#    real = tf.image.resize_images(real, (72, 72), method=tf.image.ResizeMethod.BILINEAR)
#    fake = tf.image.resize_images(fake, (72, 72), method=tf.image.ResizeMethod.BILINEAR)
    ssim = tf.ones_like(tf.image.ssim(real, fake, max_val=val)) - tf.image.ssim(real, fake, max_val=val)


    return tf.reduce_mean(ssim)
def ms_ssim_loss_fn(real, fake, val=2.0):

#    real_ = (real + 1.) * 127.5 
#    fake_ = (fake + 1.) * 127.5 
    real = tf.image.rgb_to_grayscale(real)
    fake = tf.image.rgb_to_grayscale(fake)
    ms_ssim = tf.image.ssim_multiscale(real, fake, max_val=val, power_factors=(0.2, 0.8))
    ms_ssim_loss = tf.ones_like(ms_ssim) - ms_ssim

    return tf.reduce_mean(ms_ssim_loss)

#def gmsd_loss_fn(real, fake):

#    return tf.convert_to_tensor(metric.gmsd(real_, fake_, max_val=val)) 
def L1_loss_fn(labels, predictions):

    return tf.reduce_mean(tf.losses.absolute_difference(labels, predictions))

def vgg_loss_fn(labels, predictions):
    

    min = tf.cond(tf.reduce_min(labels) >= tf.reduce_min(predictions), lambda: tf.reduce_min(predictions), lambda: tf.reduce_min(labels)) 
    max = tf.cond(tf.reduce_max(labels) >= tf.reduce_max(predictions), lambda: tf.reduce_max(labels), lambda: tf.reduce_max(predictions))
    
    labels = (labels - min) / (max - min)
    predictions = (predictions - min) / (max - min)
    ssim_feature = tf.image.ssim(labels, predictions, max_val=1.0)
    return tf.reduce_mean(tf.ones_like(ssim_feature) - ssim_feature)





def gmsd(vref, vcmp, rescale=True, returnMap=False):
    if rescale:
        scl = (255.0/tf.reduce_max(vref))
    else:
        scl = np.float32(1.0)

    T = 170.0
    dwn = 2
    dx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])/3.0
    dy = dx.T

    ukrn = np.ones((2, 2))/4.0
    aveY1 = signal.convolve2d(scl*vref, ukrn, mode='same', boundary='symm')
    aveY2 = signal.convolve2d(scl*vcmp, ukrn, mode='same', boundary='symm')
    Y1 = aveY1[0::dwn, 0::dwn]
    Y2 = aveY2[0::dwn, 0::dwn]

    IxY1 = signal.convolve2d(Y1, dx, mode='same', boundary='symm')
    IyY1 = signal.convolve2d(Y1, dy, mode='same', boundary='symm')
    grdMap1 = np.sqrt(IxY1**2 + IyY1**2)

    IxY2 = signal.convolve2d(Y2, dx, mode='same', boundary='symm')
    IyY2 = signal.convolve2d(Y2, dy, mode='same', boundary='symm')
    grdMap2 = np.sqrt(IxY2**2 + IyY2**2)

    quality_map = (2*grdMap1*grdMap2 + T) / (grdMap1**2 + grdMap2**2 + T)
    score = np.std(quality_map)

    if returnMap:
        return (score, quality_map)
    else:
        return score

def gradient_penalty(real, fake, f):
    def interpolate(a, b):
        shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
        alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
        inter = a + alpha * (b - a)
        inter.set_shape(a.get_shape().as_list())
        return inter

    x = interpolate(real, fake)
    _, pred = f(x, is_train=False, reuse=True)
    gradients = tf.gradients(pred, x)[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
    gp = tf.reduce_mean((slopes - 1.)**2)
    return gp