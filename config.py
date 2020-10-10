'''
本文件是配置文件
'''

# 一些公共配置
class BaseConfig:
    #
    # batch_size = 8
    # shuffle = True
    # num_workers = 4
    # rnn_hidden = 256
    # embedding_dim = 256
    # num_layers = 2
    # share_embedding_weights = False
    # debug_file = '/tmp/debugc'
    pass

###################################################################################
# 本次配置是 for mnist_testing
class Config_mnist_testing(BaseConfig):
    #
    plot_every = 10
    # set_1
    # model_ckpt = 'checkpoints/'  # 每次从某个断点加载模型
    # output_file = 'ucsd_ped1.txt'
    # set_2
    model_ckpt = 'checkpoints/mnist_0606_1305.pkl'
    output_file = 'mnist_0606_1305.txt'
    #
    video_folder = 'data/MNIST'
    shuffle = False
    epoch = 1  #
    batch_size = 1  #
    num_workers = 1
    device_idx = "0"  # GPU:1
    cpd_channels = 100
    LAM = 0.1

###################################################################################
# 本次配置是 for cifar10_testing
class Config_cifar10_testing(BaseConfig):
    #
    plot_every = 10
    # set_1
    # model_ckpt = 'checkpoints/'  # 每次从某个断点加载模型
    # output_file = 'ucsd_ped1.txt'
    # set_2
    model_ckpt = 'checkpoints/ucsd_ped1_0527_0721.pkl'
    output_file = 'ucsd_ped1_0527_0721.txt'
    #
    video_folder = 'data/CIFAR10'
    shuffle = False
    epoch = 1  # 经过验证 1200 it(即batch) / epoch， 16.52s/it，所以~ 5 h/epoch
    batch_size = 1  #
    num_workers = 1
    device_idx = "0"  # GPU:1
    cpd_channels = 100
    LAM = 0.1

###################################################################################

# 本次配置是 for ped1_testing
class Config_ped1_testing(BaseConfig):
    #
    plot_every = 10
    save_ckpt_every = 2000
    prefix = "checkpoints/"
    dataset_name = "ucsd_ped1"
    # set_1
    # model_ckpt = 'checkpoints/'  # 每次从某个断点加载模型
    # output_file = 'ucsd_ped1.txt'
    # set_2
    model_ckpt = 'checkpoints/ucsd_ped1_0527_0721.pkl'
    output_file = 'ucsd_ped1_0527_0721.txt'
    #
    video_folder = 'data/UCSD_Anomaly_Dataset.v1p2'
    shuffle = False
    epoch = 1  # 经过验证 1200 it(即batch) / epoch， 16.52s/it，所以~ 5 h/epoch
    batch_size = 1  #
    num_workers = 1
    device_idx = "3"  # GPU:1
    LR = 0.001
    cpd_channels = 100
    LAM = 0.1
    # TIME_STEPS = 16  # 这个论文代码在代码中写死了，TODO
    # new_weight, new_height = 225, 225  # 这个作者代码也写死了
    # channel = 1  # avenue是RGB image，但是论文转化为Gray，这个作者也写死了

####################################################################################
# 本次配置是 for ped2_testing
class Config_ped2_testing(BaseConfig):
    #
    plot_every = 10
    save_ckpt_every = 1000
    prefix = "checkpoints/"
    dataset_name = "ucsd_ped2"
    # set_1
    model_ckpt = 'checkpoints/ucsd_ped2_0626_2251.pkl'  # 每次从某个断点加载模型
    output_file = 'ucsd_ped2_0626_2251.txt'
    # set_2
    # model_ckpt = 'checkpoints/ucsd_ped2.pkl'
    # output_file = 'ucsd_ped2.txt'
    #
    video_folder = 'data/UCSD_Anomaly_Dataset.v1p2'
    shuffle = False
    epoch = 1  # 经过验证 1200 it(即batch) / epoch， 16.52s/it，所以~ 5 h/epoch
    batch_size = 1  #
    num_workers = 1
    device_idx = "1"  # GPU:1
    #
    # lam_rec, lam_svdd = 1, 0  # for rec loss
    lam_rec, lam_svdd = 0, 1  # for deepSVDD loss
    objective = "one-class"
    nu = 0.1
    # TIME_STEPS = 16  # 这个论文代码在代码中写死了，TODO
    # new_weight, new_height = 225, 225  # 这个作者代码也写死了
    # channel = 1  # avenue是RGB image，但是论文转化为Gray，这个作者也写死了

####################################################################################

class Config_ShanghaiTech_testing(BaseConfig):
    #
    plot_every = 10
    save_ckpt_every = 2000
    prefix = "checkpoints/"
    dataset_name = "ucsd_ped2"
    # model_ckpt = 'checkpoints/ucsd_ped2_0523_1531.pkl'  # 每次从某个断点加载模型
    # output_file = 'ucsd_ped2_0523_1531.txt'
    model_ckpt = 'checkpoints/shanghaitech_0525_1012.pkl'
    output_file = 'shanghaitech_0525_1012.txt'
    #
    video_folder = 'data/shanghaitech'
    shuffle = False
    epoch = 1  # 经过验证 1200 it(即batch) / epoch， 16.52s/it，所以~ 5 h/epoch
    batch_size = 1  #
    num_workers = 1
    device_idx = "3"  # GPU:1
    LR = 0.001
    cpd_channels = 100
    LAM = 0.1
    # TIME_STEPS = 16  # 这个论文代码在代码中写死了，TODO
    # new_weight, new_height = 225, 225  # 这个作者代码也写死了
    # channel = 1  # avenue是RGB image，但是论文转化为Gray，这个作者也写死了

####################################################################################
# 本次配置是 for mnist_training
class Config_mnist_training(BaseConfig):
    #
    plot_every = 10
    save_ckpt_every = 1000
    prefix = "checkpoints/"
    dataset_name = "mnist"
    model_ckpt = 'checkpoints/xxx'  # 每次从某个断点加载模型
    output_file = 'xxx.txt'
    #
    video_folder = 'data/MNIST'
    shuffle = True
    epoch = 100 #
    batch_size = 256  #
    num_workers = 4
    device_idx = "0"  # GPU:1
    normal_or_dist = "normal"
    LR = 0.0001
    cpd_channels = 100
    LAM = 1
    # TIME_STEPS = 16  # 这个论文代码在代码中写死了，TODO
    # new_weight, new_height = 225, 225  # 这个作者代码也写死了
    # channel = 1  # avenue是RGB image，但是论文转化为Gray，这个作者也写死了

#################################################################################
# 本次配置是 for cifar10_training
class Config_cifar10_training(BaseConfig):
    #
    plot_every = 10
    save_ckpt_every = 1000
    prefix = "checkpoints/"
    dataset_name = "cifar10"
    model_ckpt = 'checkpoints/xxx'  # 每次从某个断点加载模型
    output_file = 'xxx.txt'
    #
    video_folder = 'data/CIFAR10'
    shuffle = True
    epoch = 100 #
    batch_size = 256  #
    num_workers = 4
    device_idx = "1"  # GPU:1
    normal_or_dist = "normal"
    LR = 0.001
    cpd_channels = 100
    LAM = 0.1
    # TIME_STEPS = 16  # 这个论文代码在代码中写死了，TODO
    # new_weight, new_height = 225, 225  # 这个作者代码也写死了
    # channel = 1  # avenue是RGB image，但是论文转化为Gray，这个作者也写死了

####################################################################################
# 本次配置是 for ped1_training
class Config_ped1_training(BaseConfig):
    #
    plot_every = 10
    save_ckpt_every = 2000
    prefix = "checkpoints/"
    dataset_name = "ucsd_ped1"
    model_ckpt = 'checkpoints/xxx'  # 每次从某个断点加载模型
    output_file = 'ucsd_ped1.txt'
    #
    video_folder = 'data/UCSD_Anomaly_Dataset.v1p2'
    shuffle = True
    epoch = 10 # 经过验证 1200 it(即batch) / epoch， 16.52s/it，所以~ 5 h/epoch
    batch_size = 2  #
    num_workers = 4
    device_idx = "0"  # GPU:1
    normal_or_dist = "normal"
    LR = 0.001
    cpd_channels = 100
    LAM = 0.1
    # TIME_STEPS = 16  # 这个论文代码在代码中写死了，TODO
    # new_weight, new_height = 225, 225  # 这个作者代码也写死了
    # channel = 1  # avenue是RGB image，但是论文转化为Gray，这个作者也写死了

#################################################################################

# 本次配置是 for ped2_training
class Config_ped2_training(BaseConfig):
    #
    plot_every = 10
    save_ckpt_every = 1000
    prefix = "checkpoints/"
    dataset_name = "ucsd_ped2"
    model_ckpt = 'checkpoints/ucsd_ped2_0626_0851.pkl'  # 每次从某个断点加载模型
    # model_ckpt = 'checkpoints/xxx'  # 每次从某个断点加载模型
    output_file = 'xxx.txt'
    #
    video_folder = 'data/UCSD_Anomaly_Dataset.v1p2'
    shuffle = True
    epoch = 10 # 经过验证 1200 it(即batch) / epoch， 16.52s/it，所以~ 5 h/epoch
    batch_size = 2  #
    num_workers = 4
    device_idx = "0"  # GPU:1
    normal_or_dist = "normal"
    LR = 0.001
    # lam_rec, lam_svdd = 1, 0    # for rec loss
    lam_rec, lam_svdd = 0, 1 # for deepSVDD loss
    # lam_rec, lam_svdd = 0.8, 0.2 # joint learn
    warm_up_n_steps = 300
    objective = "one-class"
    nu = 0.1
    # TIME_STEPS = 16  # 这个论文代码在代码中写死了，TODO
    # new_weight, new_height = 225, 225  # 这个作者代码也写死了
    # channel = 1  # avenue是RGB image，但是论文转化为Gray，这个作者也写死了

####################################################################################

# 本次配置是 for shanghaitech_training
class Config_shanghaitech_training(BaseConfig):
    #
    plot_every = 10
    save_ckpt_every = 2000 # 这个是我自己观察tensorboard发现2000 step就有不错的结果
    prefix = "checkpoints/"
    dataset_name = "shanghaitech"
    model_ckpt = 'checkpoints/shanghaitech.pkl'  # 每次从某个断点加载模型
    output_file = 'shanghaitech.txt'
    #
    device_ids = [0,1]
    device_idx = "0"  # GPU:1
    video_folder = 'data/shanghaitech'
    shuffle = True
    num_workers = 4
    epoch = 1 # 经过验证 ？ batch / epoch
    batch_size = 4  # 3是 gpu_num
    # LR = 0.0005
    LR = 0.001
    cpd_channels = 100
    LAM = 1
    # TIME_STEPS = 16  # 这个论文代码在代码中写死了，TODO
    # new_weight, new_height = 225, 225  # 这个作者代码也写死了
    # channel = 1  # avenue是RGB image，但是论文转化为Gray，这个作者也写死了

#############################################################################################
