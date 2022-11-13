# @Time    : 2022/8/12 21:29
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : train
# @Project Name :code

import random

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time
from hyperparameters import Hyperparameters as hp
from utilss import *
from model import *


# np.random.seed(42) 
# tf.set_random_seed(42)
#get data
train_data = read_from_pickle(data_type=hp.hotel_dir,data_used=hp.train_dir)
valid_data = read_from_pickle(data_type=hp.hotel_dir,data_used=hp.val_dir)
# valid_data  = read_from_pickle(data_type=hp.hotel_dir,data_used=hp.test_dir)
print('The datasets {} is used!'.format(hp.hotel_dir))
# train_data[6] = train_data[6]+valid_data_[6]
# train_data[7] = train_data[7]+valid_data_[7]
# train_data[8] = train_data[8]+valid_data_[8]
# train_data[9] = train_data[9]+valid_data_[9]
v_obs_ = train_data[6]
a_obs_ = train_data[7]
v_pred_ = train_data[8]
a_pred_ = train_data[9]
print("一共训练数据为-->{}".format(len(v_obs_)))
#Define placehodel
v_obs = tf.placeholder(dtype=tf.float32, shape=[None, 8, None, 2])
a_obs = tf.placeholder(dtype=tf.float32, shape=[None, 8, None, None])
v_pred = tf.placeholder(dtype=tf.float32, shape=[None, 12, None, 2])
a_pred = tf.placeholder(dtype=tf.float32, shape=[None, 12, None, None])
# padding_mask = tf.placeholder(dtype=tf.float32, shape=[None, 12, None])#to elimate padding part
if_trainng = tf.placeholder(tf.bool)
#train data setting
# train_log,train_ped_tuple,padding_mask_train = get_inform(train_data[6],train_data[5])
#validation data prosessing
val_log,val_ped_tuple,_= get_inform(valid_data[6],valid_data[5])
#model setting
model = DGFomer(v_obs,a_obs,v_pred,a_pred,if_trainng)
memory, src_mask = model.encoder()
pred = model.decoder(memory, src_mask)
loss = graph_loss(tf.squeeze(pred),tf.squeeze(v_pred))
global_step = tf.train.get_or_create_global_step()
lr = tf.train.exponential_decay(hp.lr, global_step ,
decay_steps=5 * (len(v_obs_)//hp.batch_size), decay_rate=0.7, staircase=True)
#clip gradiants
optimizer = tf.train.RMSPropOptimizer(lr)
# optimizer = tf.train.AdamOptimizer(lr)
#add
# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
if hp.if_grad_clip:
    #add
    # with tf.control_dependencies(update_ops):
    tvars = tf.trainable_variables()
    gradients = tf.gradients(loss, tvars)
    grads, _ = tf.clip_by_global_norm(gradients, hp.grad_clip)
    train_op = optimizer.apply_gradients(zip(grads, tvars))
else:
    train_op = optimizer.minimize(loss,global_step=global_step)
#summary
tf.summary.scalar('lr', lr)
tf.summary.scalar("loss", loss)
tf.summary.scalar("global_step", global_step)
summaries = tf.summary.merge_all()
#if bn used https://blog.csdn.net/TeFuirnever/article/details/93457816
# var_list = tf.trainable_variables()
# g_list = tf.global_variables()
# bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
# bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
# var_list += bn_moving_vars
# saver = tf.train.Saver(var_list=var_list,max_to_keep=3)
saver = tf.train.Saver(max_to_keep=3)
# gpu_options = tf.GPUOptions(allow_growth=True)
#open a session
# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
init_ade = hp.init_ade
with tf .Session() as sess:
    writer = tf.summary.FileWriter(hp.logdir, sess.graph)
    sess.run(tf.global_variables_initializer())
    #future training
    # saver.restore(sess,tf.train.latest_checkpoint('ckpt'))#future training
    for e in range(hp.num_epochs):
        for idx,datas in enumerate(gen_batch(train_data)):
            start = time.time()
            loss_batch = 0
            v_obs_, a_obs_, v_pred_, a_pred_ = datas
            for s in range(len(v_obs_)-1):
                v_obs_batch,a_obs_batch,v_pred_batch,a_pred_batch = v_obs_[s:s+1], a_obs_[s:s+1],v_pred_[s:s+1],a_pred_[s:s+1]
                # pm_batch = np.transpose(pm_batch,(0,2,1))[:,-12:,:]
                # assert len(v_obs_batch.shape)==4,'the shape is incorrect!'
                feed = {v_obs:v_obs_batch,a_obs:a_obs_batch,v_pred:v_pred_batch,
                a_pred:a_pred_batch,if_trainng:True}
                _,_gs,summary,loss_,yhat = sess.run([train_op,global_step,summaries,loss,pred],feed)
                loss_batch+=loss_
                # if loss_<0:
                #     print('损失为{:.2f}，分子为{}，分母为{}，除数为{}，分母的形状为{},分子的形状为{}'.format(
                #         loss_,denom_,print_out_,print_out_/denom_,denom_.shape,print_out_.shape
                #     )+'->'+'输出的形状为{}'.format(yhat.shape))
            end = time.time()
            loss_batch = loss_batch/len(v_obs_)
            print('{}/{},train_loss={:.3f}.time/batch={:.3f}'.format(e,idx,loss_batch,end-start))
            if (e+1)%5==0 and idx%30==0:
            #validation
                if hp.if_val:
                    inform = dict()
                    ksteps=hp.val_ksteps
                    scene = list(set([info[0] for info in val_log]))
                    sc_ids = np.arange(len(scene))#场景的ids
                    # np.random.shuffle(sc_ids)
                    # sc_ids = sc_ids[:samples]
                    for sc_id in sc_ids:#遍历每一个场景，一个场景可能包含多个行人因此出现多次场景
                        sc = scene[sc_id]
                        decoder_v = np.expand_dims(np.zeros_like(valid_data[8][sc]),0)
                        decoder_a = np.expand_dims(np.zeros_like(valid_data[9][sc]),0)
                        encoder_v = np.expand_dims(valid_data[6][sc],0)#->(b,12,n,2)
                        encoder_a = np.expand_dims(valid_data[7][sc],0)#->(b,12,n,n)
                        for j in range(hp.pred_len):
                            _pred = sess.run(pred,feed_dict={v_obs:encoder_v,a_obs:encoder_a,v_pred:decoder_v,a_pred:decoder_a,if_trainng:False})
                            #采样k次，选取ade最低的best-of-n
                            ade = []
                            temp_pred = []
                            for k in range(ksteps):#对每一个时间步长采样k次
                                decoder_v[:,j,:,:]= np.expand_dims(sample_traij(_pred),0)[:,j,:,:]
                                y_hat_ = np.squeeze(decoder_v)#->(12,n,2)
                                ade.append(cal_ade(y_hat_[j,:,:],valid_data[8][sc][j,:,:]))
                                temp_pred.append(y_hat_[j,:,:].copy())#[[n,2],[n,2],...]
                            #选择最优的轨迹
                            best_id = np.argmin(ade)
                            decoder_v[:,j,:,:] = np.expand_dims(temp_pred[best_id],0)
                            decoder_a[:,j,:,:] = np.expand_dims(valid_data[9][sc],0)[:,j,:,:]
                        try:
                            peds = [info[1] for info in val_log if info[0]==sc]
                            for ped in peds:
                                inform[(sc,ped)]=decoder_v[:,:,ped,None,:]#存在重复的场景
                        except:
                            print(sc,ped)
                        #rel_traij->abs_traij
                    scen_id = list(inform.keys())
                    obs_0 = []
                    future_0 = []
                    pred_0 = []
                    ade_0 = []
                    fde_0 = []
                    for id in scen_id:
                        v_x_ro,v_y_ro,v_hat = get_obs_trajs(valid_data[6],valid_data[8],valid_data[0],
                        valid_data[1],np.squeeze(inform[id],axis=0),val_ped_tuple,id)
                        obs_0.append(v_x_ro)
                        future_0.append(v_y_ro)
                        pred_0.append(v_hat)
                    ade = []
                    fde = []
                    for sc in range(len(future_0)):
                        ade.append(cal_ade(future_0[sc],pred_0[sc]))
                        fde.append(cal_fde(future_0[sc],pred_0[sc]))
                    avg_ade = np.mean(ade)
                    avg_fde = np.mean(fde)
                    print('epoch/step-->{}/{}===ade-->{:.2f}===fde-->{:.2f}.'.format(e,idx,avg_ade,avg_fde))
                    if avg_ade<init_ade:
                        init_ade =  avg_ade
                        saver.save(sess=sess, save_path=hp.ckpt_path+'_'+'{:.2f}'.format(avg_ade), global_step=(e + 1))
                        print('The newest ckpt has been saved!')
                        print('*'*30)
                # no validation
                else:
                    saver.save(sess=sess, save_path=hp.ckpt_path, global_step=(e + 1))
                    print("model saved to {}!".format(hp.ckpt_path))
        print('*'*50)
    sess.close()




