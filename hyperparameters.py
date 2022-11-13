# @Time    : 2022/8/9 20:15
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : hyperparameters
# @Project Name :code

class Hyperparameters:


    #data proprecess
    data_dir = './datasets/'
    eth_dir = 'eth/'
    hotel_dir = 'hotel/'
    univ_dir = 'univ/'
    zara1_dir = 'zara1/'
    zara2_dir = 'zara2/'
    train_dir = 'train/'
    test_dir = 'test/'
    val_dir = 'val/'
    obs_len = 8
    pred_len = 12
    seq_len = obs_len+pred_len
    skip=1
    delim='\t'
    norm_lap_matr = True
    sigma = 0.1
    eplision = 0.5
    kernel_name = 'sim_dis'
    poly_threshold = 0.002
    max_peds_in_frame = 0
    min_ped = 1
    norm_lap_matr = True
    reconstruct = False
    padding_type = 'rel'#to,tobs,rel
    store_frac=0.8#帧填充后，完整轨迹节点需要保留的上限


    #training setting
    num_units=64
    drop_rate = 0.1
    kernel_size =3
    num_heads = 4
    num_se_blocks = 2
    num_blocks = 3
    out_units = 5
    lr=0.001
    batch_size = 32
    num_epochs = 150
    if_grad_clip = True
    grad_clip = 10
    logdir = 'logdir'
    ckpt_path = './ckpt/weight'
    if_tma_dense =True
    if_sma_dense =True  
    #validation setting
    if_val = True
    val_samples = 450
    val_ksteps = 10
    init_ade=9999

    #testing setting
    ksteps = 20
