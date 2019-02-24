

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *
from config import config, log_config

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
logdir = config.VALID.logdir
initial_epoch = 5001
ni = int(np.sqrt(batch_size))

def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
#        print('read %d from %s' % (len(imgs), path))
    return imgs

def train():
    ## create folders to save result images and trained model

    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    save_dir_gan_valid = 'samples/valid_gan'
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
#    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
#    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
#    train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    # valid_lr_imgs = read_all_imgs(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = read_all_imgs(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, 12, 12, 3], name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [batch_size, 48, 48, 3], name='t_target_image')

    net_g = SRGAN_g(t_image, is_train=True, reuse=False)
    net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    _,     logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)

    net_g.print_params(False)
    net_d.print_params(False)

    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_56 = tf.image.resize_images(t_target_image, size=[56, 56], method=0, align_corners=False) # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_56 = tf.image.resize_images(net_g.outputs, size=[56, 56], method=0, align_corners=False) # resize_generate_image_for_vgg
#    net_vgg, vgg_target_emb_c1, vgg_target_emb_c4 = Vgg19_simple_api((t_target_image+1)/2, reuse=False)
#    _, vgg_predict_emb_c1, vgg_predict_emb_c4 = Vgg19_simple_api((net_g.outputs+1)/2, reuse=True)
    net_vgg, vgg_target_emb_c1, vgg_target_emb_c4 = Vgg19_simple_api((t_target_image_56+1)/2, reuse=False)
    _, vgg_predict_emb_c1, vgg_predict_emb_c4 = Vgg19_simple_api((t_predict_image_56+1)/2, reuse=True)
#    net_vgg, vgg_target_emb_c1, vgg_target_emb_c4 = Vgg19_simple_api((t_target_image+1)/2, reuse=False)
#    _, vgg_predict_emb_c1, vgg_predict_emb_c4 = Vgg19_simple_api((net_g.outputs+1)/2, reuse=True)

    ## test inference
    net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###
    # d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    # d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')

    # d_loss1 = tl.cost.cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    # d_loss2 = tl.cost.cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    #
    # d_loss = d_loss1 + d_loss2

    # Wasserstein GAN Loss
    with tf.name_scope('w_loss/WARS_1'):
#        d_loss =  tf.reduce_mean(logits_fake) - tf.reduce_mean(logits_real)
        wd = tf.reduce_mean(logits_real) - tf.reduce_mean(logits_fake)
        gp = gradient_penalty(t_target_image, net_g.outputs, SRGAN_d)
        d_loss = -wd + gp * 10.0
        tf.summary.scalar('w_loss', d_loss)

    merged = tf.summary.merge_all()
    # loss_writer = tf.summary.FileWriter('/home/ubuntu/huzhihao/WARS/log/', sess.graph)
    # g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    g_gan_loss = - 1e-3 * tf.reduce_mean(logits_fake)
#    ssim_loss = ssim_loss_fn(t_target_image, net_g.outputs, val=2)
     
    content_loss = tl.cost.mean_squared_error(t_target_image, net_g.outputs, is_mean=True)
        
    content_loss_ssim = 1e-1 * ssim_loss_fn(t_target_image, net_g.outputs, val=2.0)
#    vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)
#    L1_loss = L1_loss_fn(t_target_image, net_g.outputs)
#    gmsd_loss = gmsd_loss_fn(t_target_image.eval(session=tf.Session()), net_g.outputs.eval(session=tf.Session()))
#    vgg_loss = 1.3e-3 * L1_loss_fn(vgg_target_emb.outputs, vgg_predict_emb.outputs)
    vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_target_emb_c4.outputs, vgg_predict_emb_c4.outputs, is_mean=True)
#        vgg_loss = vgg_loss_fn(vgg_target_emb_c1.outputs, vgg_predict_emb_c1.outputs)
#        vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_target_emb_c4.outputs, vgg_predict_emb_c4.outputs, is_mean=True)

#    vgg_loss = 4e-3 * (0.7 * vgg_loss_fn(vgg_target_emb_c1.outputs, vgg_predict_emb_c1.outputs) + 0.3 * L1_loss_fn(vgg_target_emb_c4.outputs, vgg_predict_emb_c4.outputs))
#    gmsd_loss = gmsd_loss(vgg_predict_emb.outputs, vgg_target_emb.outputs)
    g_loss_ssim = content_loss_ssim + vgg_loss + g_gan_loss
    g_loss = content_loss + vgg_loss + g_gan_loss
    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    ## Pretrain
    # g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
#    g_optim_init = tf.train.RMSPropOptimizer(lr_v).minimize(ssim_loss, var_list=g_vars)
    g_optim_init = tf.train.RMSPropOptimizer(lr_v).minimize(content_loss, var_list=g_vars)
    ## SRGAN
    # g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    # d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

    g_optim = tf.train.RMSPropOptimizer(lr_v).minimize(g_loss, var_list=g_vars)
    g_optim_ssim = tf.train.RMSPropOptimizer(lr_v).minimize(g_loss_ssim, var_list=g_vars)
    d_optim = tf.train.RMSPropOptimizer(lr_v).minimize(d_loss, var_list=d_vars)

    # clip op
#    clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]


    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    loss_writer = tf.summary.FileWriter(logdir, sess.graph)
    tl.layers.initialize_global_variables(sess)
    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}.npz'.format(tl.global_flag['mode']), network=net_g) is False:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/d_{}.npz'.format(tl.global_flag['mode']), network=net_d)

    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()
    npz.pop('fc6', None)
    npz.pop('fc7', None)
    npz.pop('fc8', None)
    npz.pop('conv5_1', None)
    npz.pop('conv5_2', None)
    npz.pop('conv5_3', None)
    npz.pop('conv5_4', None)
#    npz.pop('conv4_1', None)
#    npz.pop('conv4_2', None)
#    npz.pop('conv4_3', None)
#    npz.pop('conv4_4', None)
#    npz.pop('conv3_1', None)
#    npz.pop('conv3_2', None)
#    npz.pop('conv3_3', None)
#    npz.pop('conv3_4', None)
#    npz.pop('conv2_1', None)
#    npz.pop('conv2_2', None)    
    params = []
    for val in sorted( npz.items() ):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)
    # net_vgg.print_params(False)
    # net_vgg.print_layers()

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    sample_imgs = read_all_imgs(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path, n_threads=32)
    valid_imgs = read_all_imgs(valid_hr_img_list[0:batch_size], path=config.VALID.hr_img_path, n_threads=32)
    
    valid_imgs_48 = tl.prepro.threading_data(valid_imgs, fn=crop_sub_imgs_fn, is_random=False)
    sample_imgs_48 =  tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
#   sample_imgs = read_all_imgs(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path, n_threads=32) # if no pre-load train set


    print('sample HR sub-image:', sample_imgs_48.shape, sample_imgs_48.min(), sample_imgs_48.max())
    sample_imgs_12 = tl.prepro.threading_data(sample_imgs_48, fn=downsample_fn)
    print('sample LR sub-image:', sample_imgs_12.shape, sample_imgs_12.min(), sample_imgs_12.max())
#   valid_imgs_48 = tl.prepro.threading_data(valid_imgs, fn=None)
    print('validation HR image:', valid_imgs_48.shape, valid_imgs_48.min(), valid_imgs_48.max())
    valid_imgs_12 = tl.prepro.threading_data(valid_imgs_48, fn=downsample_fn)
    print('validation LR image:', valid_imgs_12.shape, valid_imgs_12.min(), valid_imgs_12.max())
    tl.vis.save_images(sample_imgs_12, [ni, ni], save_dir_ginit+'/_train_sample_12.png')
    tl.vis.save_images(valid_imgs_12, [ni, ni], save_dir_ginit+'/_valid_sample_12.png')
    tl.vis.save_images(sample_imgs_48, [ni, ni], save_dir_ginit+'/_train_sample_48.png')
    tl.vis.save_images(valid_imgs_48, [ni, ni], save_dir_ginit+'/_valid_sample_48.png')
    tl.vis.save_images(sample_imgs_12, [ni, ni], save_dir_gan+'/_train_sample_12.png')
    tl.vis.save_images(valid_imgs_12, [ni, ni], save_dir_gan_valid+'/_valid_sample_12.png')
    tl.vis.save_images(sample_imgs_48, [ni, ni], save_dir_gan+'/_train_sample_48.png')
    tl.vis.save_images(valid_imgs_48, [ni, ni], save_dir_gan_valid+'/_valid_sample_48.png')

    ###========================= initialize G ====================###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)
    for epoch in range(0, n_epoch_init+1):
        epoch_time = time.time()
        total_content_loss, n_iter = 0, 0

        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        random.shuffle(train_hr_img_list)
        for idx in range(0, len(train_hr_img_list), batch_size):
            step_time = time.time()
            b_imgs_list = train_hr_img_list[idx : idx + batch_size]
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
            b_imgs_48 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
            b_imgs_12 = tl.prepro.threading_data(b_imgs_48, fn=downsample_fn)
#            b_imgs_12 = tl.prepro.threading_data(b_imgs_12, fn=upsample_fn)

        ## If your machine have enough memory, please pre-load the whole train set.
        #for idx in range(0, len(train_hr_imgs), batch_size):
        #    step_time = time.time()
        #    b_imgs_48 = tl.prepro.threading_data(
        #            train_hr_imgs[idx : idx + batch_size],
        #            fn=crop_sub_imgs_fn, is_random=True)
        #    b_imgs_12 = tl.prepro.threading_data(b_imgs_48, fn=downsample_fn)
            ## update G
            errM, _ = sess.run([content_loss, g_optim_init], {t_image: b_imgs_12, t_target_image: b_imgs_48})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, content: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
            total_content_loss += errM
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, content: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_content_loss/n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            out = sess.run(net_g_test.outputs, {t_image: sample_imgs_12})#; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_ginit+'/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)

    ###========================= train GAN (SRGAN) =========================###

    # clipping method
    # clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, self.clip_values[0], self.clip_values[1])) for
    #                                      var in self.discriminator_variables]


    for epoch in range(initial_epoch, n_epoch+1):
        ## update learning rate

        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0

        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        random.shuffle(train_hr_img_list)
        for idx in range(0, len(train_hr_img_list), batch_size):
            step_time = time.time()
            b_imgs_list = train_hr_img_list[idx : idx + batch_size]
            b_imgs = read_all_imgs(b_imgs_list, path=config.TRAIN.hr_img_path, n_threads=32)
            b_imgs_48 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
            b_imgs_12 = tl.prepro.threading_data(b_imgs_48, fn=downsample_fn)
#            b_imgs_12 = tl.prepro.threading_data(b_imgs_48, fn=upsample_fn)
        ## If your machine have enough memory, please pre-load the whole train set.
        #for idx in range(0, len(train_hr_imgs), batch_size):
        #    step_time = time.time()
        #    b_imgs_48 = tl.prepro.threading_data(
        #            train_hr_imgs[idx : idx + batch_size],
        #            fn=crop_sub_imgs_fn, is_random=True)
        #    b_imgs_12 = tl.prepro.threading_data(b_imgs_48, fn=downsample_fn)
            ## update D
            
            errD, summary, _ = sess.run([d_loss, merged, d_optim], {t_image: b_imgs_12, t_target_image: b_imgs_48})
            loss_writer.add_summary(summary, idx)
            # d_vars = sess.run(clip_discriminator_var_op)
            ## update G
            if epoch>=0.04*(n_epoch):
                errG, errM, errV, errA, _ = sess.run([g_loss_ssim, content_loss_ssim, vgg_loss, g_gan_loss, g_optim_ssim],{t_image: b_imgs_12 , t_target_image: b_imgs_48})

                print("Epoch [%2d/%2d] %4d time: %4.4fs, W_loss: %.8f g_loss: %.8f (content_ssim: %.6f vgg: %.6f adv: %.6f)"
                      % (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))
                total_d_loss += errD
                total_g_loss += errG
                n_iter += 1
            else:
                errG, errM, errV, errA, _ = sess.run([g_loss, content_loss, vgg_loss, g_gan_loss, g_optim],{t_image: b_imgs_12 , t_target_image: b_imgs_48})

                print("Epoch [%2d/%2d] %4d time: %4.4fs, W_loss: %.8f g_loss: %.8f (contente: %.6f vgg: %.6f adv: %.6f)"
                      % (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))
                total_d_loss += errD
                total_g_loss += errG
                n_iter += 1                

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss/n_iter, total_g_loss/n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            out = sess.run(net_g_test.outputs, {t_image: sample_imgs_12})#; print('gen sub-image:', out.shape, out.min(), out.max())
            out_valid = sess.run(net_g_test.outputs, {t_image: valid_imgs_12})
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_gan+'/train_%d.png' % epoch)
            tl.vis.save_images(out_valid, [ni, ni], save_dir_gan_valid+'/valid_%d.png' % epoch)
        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
#            if ckt_flag:
                tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_{}.npz'.format(tl.global_flag['mode']), sess=sess)
                tl.files.save_npz(net_d.all_params, name=checkpoint_dir+'/d_{}.npz'.format(tl.global_flag['mode']), sess=sess)
#                ckt_flag = not ckt_flag
#            else:
#                tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_{}_1.npz'.format(tl.global_flag['mode']), sess=sess)
#                tl.files.save_npz(net_d.all_params, name=checkpoint_dir+'/d_{}_1.npz'.format(tl.global_flag['mode']), sess=sess)
#                ckt_flag = not ckt_flag



def testing():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.jpg', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.jpg', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs = read_all_imgs(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = read_all_imgs(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    
    imid = 0 # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    valid_lr_img = valid_lr_imgs[imid]
    valid_hr_img = valid_hr_imgs[imid]
    #img_name = '0010_120.jpg'
    #valid_lr_img = get_imgs_fn(img_name, '/home/ubuntu/dataset/sr_test/testing/')  # if you want to test your own image
    valid_lr_img = (valid_lr_img / 127.5) - 1   # rescale to ［－1, 1]
    # print(valid_lr_img.min(), valid_lr_img.max())

    size = valid_lr_img.shape
    t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image')
    # t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')

    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_srgan.npz', network=net_g)

    ###======================= EVALUATION =============================###
    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
    print("took: %4.4fs" % (time.time() - start_time))

    print("LR size: %s /  generated HR size: %s" % (size, out.shape)) # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    #tl.vis.save_image(out[0], save_dir+ '/gen_' + img_name[:-4] + '.png')
    tl.vis.save_image(out[0], save_dir + '/valid_gen.png')
    #tl.vis.save_image(valid_lr_img, save_dir+'/valid_lr.png')
    #tl.vis.save_image(valid_hr_img, save_dir+'/valid_hr.png')

    out_bicu = scipy.misc.imresize(valid_lr_img, [size[0]*4, size[1]*4], interp='bicubic', mode=None)
    #tl.vis.save_image(out_bicu, save_dir + '/bicubic_' + img_name[:-4] + '.png')
    tl.vis.save_image(out_bicu, save_dir + '/valid_bicubic.png')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='esrgan', help='esrgan, testing')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'esrgan':
        train()
    elif tl.global_flag['mode'] == 'testing':
        testing()
    else:
        raise Exception("Unknow --mode")
