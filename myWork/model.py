
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

def SRGAN_g(t_image, is_train=False, reuse=False):

    size = t_image.get_shape().as_list()
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:

        tl.layers.set_name_reuse(reuse)
        n_init = InputLayer(t_image, name='in')
        n_3 = Conv2d(n_init, 128, (3, 3), (1, 1), act=tf.nn.elu, padding='SAME', W_init=w_init, name='k3_n128s1/c')
        n_5 = Conv2d(n_init, 128, (5, 5), (1, 1), act=tf.nn.elu, padding='SAME', W_init=w_init, name='k5_n128s1/c')
        n_7 = Conv2d(n_init, 128, (7, 7), (1, 1), act=tf.nn.elu, padding='SAME', W_init=w_init, name='k7_n128s1/c')
        n = ConcatLayer([n_init, n_3, n_5, n_7], concat_dim=3, name='concat_layer')
        n = Conv2d(n, 128, (1, 1), (1, 1), act=tf.nn.elu, padding='SAME', W_init=w_init, name='k1_f_n128s1/c')

        temp = n

        # B residual blocks
        for i in range(16):
            nn = Conv2d(n, 128, (3, 3), (1, 1), act=tf.nn.elu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/ck3_1/%s' % i)
            nn = Conv2d(nn, 128, (3, 3), (1, 1), act=tf.nn.elu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/ck3_2/%s' % i)            
            nn = Conv2d(nn, 128, (3, 3), (1, 1), act=tf.nn.elu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/ck3_3/%s' % i)
            nn = Conv2d(nn, 128, (3, 3), (1, 1), act=tf.nn.elu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/ck3_4/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
            n = nn
        n = ElementwiseLayer([n, temp], tf.add, name='add3')
        n = ConcatLayer([n_init, n], concat_dim=3, name='ref')
        n = Conv2d(n, 128, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=b_init, name='up1/conv2d')
        n = UpSampling2dLayer(n, size=[size[1]*2, size[2]*2], is_scale=False, method=1, align_corners=False, name='up1/upsample2d_nn')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='up1/batch_norm')
        n = Conv2d(n, 128, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=b_init, name='up2/conv2d')
        n = UpSampling2dLayer(n, size=[size[1]*4, size[2]*4], is_scale=False, method=1, align_corners=False, name='up2/upsample2d_nn')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='up2/batch_norm')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.elu, padding='SAME', W_init=w_init, b_init=b_init, name='rck3_1')
        n = Conv2d(n, 16, (1, 1), (1, 1), act=tf.nn.elu, padding='SAME', W_init=w_init, b_init=b_init, name='shrink')
        n_7 = Conv2d(n, 3, (7, 7), (1, 1), act=tf.nn.elu, padding='SAME', W_init=w_init, name='reduction_1')
        n_5 = Conv2d(n, 3, (5, 5), (1, 1), act=tf.nn.elu, padding='SAME', W_init=w_init, name='reduction_2')
        n_3 = Conv2d(n, 3, (3, 3), (1, 1), act=tf.nn.elu, padding='SAME', W_init=w_init, name='reduction_3')
        n = ConcatLayer([n_3, n_5, n_7], concat_dim=3, name='concat_layer_rec')
        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, b_init=b_init, name='shrink_rec')
        n_rf_1 = Conv2d(n, 3, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='ft')
        n_rf_2 = Conv2d(n_rf_1, 3, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='ft_2')
        n = ElementwiseLayer([n, n_rf_2], combine_fn=tf.add, act=tf.nn.tanh, name='out')

        return n


def SRGAN_d2(t_image, is_train=False, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x : tl.act.lrelu(x, 0.2)
    with tf.variable_scope("SRGAN_d", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='n64s1/c')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s2/b')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s1/b')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s2/b')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n256s1/b')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n256s2/b')
        n = Conv2d(n, 512, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n512s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n512s1/b')
        n = Conv2d(n, 512, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n512s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n512s2/b')
        n = FlattenLayer(n, name='f')
        n = DenseLayer(n, n_units=1024, act=lrelu, name='d1024')
        n = DenseLayer(n, n_units=1, name='out')
        logits = n.outputs
        n.outputs = tf.nn.sigmoid(n.outputs)

        return n, logits

def SRGAN_d(input_images, is_train=True, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope("SRGAN_d", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='input/images')
        net_h0 = Conv2d(net_in, df_dim, (3, 3), (1, 1), act=lrelu,
                padding='SAME', W_init=w_init, name='h0/c')
        net_h1 = Conv2d(net_h0, df_dim*2, (3, 3), (1, 1), act=lrelu,
                padding='SAME', W_init=w_init, b_init=b_init, name='h1/c')
        net_h2 = Conv2d(net_h1, df_dim*4, (3, 3), (1, 1), act=lrelu,
                padding='SAME', W_init=w_init, b_init=b_init, name='h2/c')
        net_h3 = Conv2d(net_h2, df_dim*8, (3, 3), (1, 1), act=lrelu,
                padding='SAME', W_init=w_init, b_init=b_init, name='h3/c')
        net_h7 = Conv2d(net_h3, df_dim*4, (1, 1), (1, 1), act=lrelu,
                padding='SAME', W_init=w_init, b_init=b_init, name='h7/c')
        net = Conv2d(net_h7, df_dim*2, (1, 1), (1, 1), act=lrelu,
                padding='SAME', W_init=w_init, b_init=b_init, name='res/c')
        net = Conv2d(net, df_dim*2, (3, 3), (1, 1), act=lrelu,
                padding='SAME', W_init=w_init, b_init=b_init, name='res/c2')
        net = Conv2d(net, df_dim*4, (3, 3), (1, 1), act=lrelu,
                padding='SAME', W_init=w_init, b_init=b_init, name='res/c3')
        net_h8 = ElementwiseLayer(layer=[net_h7, net],
                combine_fn=tf.add, name='res/add')
        net_h8.outputs = tl.act.lrelu(net_h8.outputs, 0.2)
        net_ho = FlattenLayer(net_h8, name='ho/flatten')
        net_ho = DenseLayer(net_ho, n_units=1, act=tf.identity,
                W_init = w_init, name='ho/dense')
        logits = net_ho.outputs

    return net_ho, logits

def Vgg19_simple_api(rgb, reuse):
   
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope("VGG19", reuse=reuse) as vs:
        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else:

            red, green, blue = tf.split(rgb_scaled, 3, 3)
        assert red.get_shape().as_list()[1:] == [56, 56, 1]
        assert green.get_shape().as_list()[1:] == [56, 56, 1]
        assert blue.get_shape().as_list()[1:] == [56, 56, 1]
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], axis=3)
        assert bgr.get_shape().as_list()[1:] == [56, 56, 3]

        """ input layer """
        net_in = InputLayer(bgr, name='input')
        """ conv1 """
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_2') 
        conv_1 = network       
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool1')
#        conv_1 = network
        """ conv2 """
        network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv2_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool2')
#        conv_2 = network
        """ conv3 """
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool3')
#        conv_3 = network
        """ conv4 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool4')                               # (batch_size, 14, 14, 512)
        conv_4 = network
        """ conv5 """

#        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
#                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_1')
#        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
#                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_2')
#        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
#                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_3')
#        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
#                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_4')
#        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
#                    padding='SAME', name='pool5')                               # (batch_size, 7, 7, 512)
                                                   # (batch_size, 7, 7, 512)
#        conv_5 = network
        """ fc 6~8 """
#        network = FlattenLayer(network, name='flatten')
#        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
#        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
#        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
        print("build model finished: %fs" % (time.time() - start_time))
        return network, conv_1, conv_4

