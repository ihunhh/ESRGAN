from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 16
config.TRAIN.lr_init = 0.8e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 0
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 10000
config.TRAIN.lr_decay = 0.5
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
config.TRAIN.hr_img_path = 'data2017/96x96/'
config.TRAIN.lr_img_path = 'data2017/24x24/'
#config.TRAIN.hr_img_path = 'data2017/DIV2K_train_HR/'
#config.TRAIN.lr_img_path = '/data2017/DIV2K_train_HR'


config.VALID = edict()
## test set location
config.VALID.hr_img_path = 'data2017/valid_96/'
config.VALID.lr_img_path = 'data2017/valid_lr/'
#config.VALID.hr_img_path = '/home/ubuntu/dataset/image_tag/srgan_all_jpg/val_hr/'
#config.VALID.lr_img_path = '/home/ubuntu/dataset/image_tag/srgan_all_jpg/val_lr/'

config.VALID.logdir = 'SRGAN_Wasserstein/log/'
def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
