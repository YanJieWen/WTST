# @Time    : 2022/9/22 13:48
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : eval
# @Project Name :code


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from utilss import *
from hyperparameters import Hyperparameters as hp

#load data
test_data = read_from_pickle(data_type=hp.eth_dir,data_used=hp.test_dir)
obs_traij = test_data[0]
pred_traij = test_data[1]
obs_traij_rel = test_data[2]
pred_traij_rel = test_data[3]
v_obs = test_data[6]
v_pred = test_data[8]
a_obs = test_data[7]
a_pred = test_data[9]
obs_rel=test_data[2]
non_linear =test_data[4]
loss_mask = test_data[5]


#gen scene list
test_log,ped_tuple,no_mask = get_inform(v_obs,loss_mask)

#Load graph & weight
tf.reset_default_graph()
test_sess = tf.Session()
saver = tf.train.import_meta_graph('./ckpt/eth-eth/weight.meta')
graph = tf.get_default_graph()
# saver.restore(test_sess,tf.train.latest_checkpoint('ckpt'))
saver.restore(test_sess,'./ckpt/eth-eth/weight')
print("The graph and weights have been restored!")

#Define placehodle
v_obs_ = graph.get_tensor_by_name('Placeholder:0')
a_obs_ = graph.get_tensor_by_name('Placeholder_1:0')
v_pred_ = graph.get_tensor_by_name('Placeholder_2:0')
a_pred_ = graph.get_tensor_by_name('Placeholder_3:0')
if_training = graph.get_tensor_by_name('Placeholder_4:0')
y_hat = graph.get_tensor_by_name('Prediction/prediction/Tensordot:0')


#run model for each scene
inform = dict()
ksteps=hp.ksteps
scene = list(set([info[0] for info in test_log]))#所有的场景
for sc in scene:#遍历每一个场景
  decoder_v = np.expand_dims(np.zeros_like(v_pred[sc]),0)
  decoder_a = np.expand_dims(np.zeros_like(a_pred[sc]),0)
  encoder_v = np.expand_dims(v_obs[sc],0)#->(b,12,n,2)
  encoder_a = np.expand_dims(a_obs[sc],0)#->(b,12,n,n)
  
  for j in range(hp.pred_len):
    _pred = test_sess.run(y_hat,feed_dict={v_obs_:encoder_v,a_obs_:encoder_a,v_pred_:decoder_v,a_pred_:decoder_a,if_training:False})
    #采样k次，选取ade最低的best-of-n
    ade = []
    temp_pred = []
    for k in range(ksteps):#对每一个时间步长采样k次
      decoder_v[:,j,:,:]= np.expand_dims(sample_traij(_pred),0)[:,j,:,:]
      y_hat_ = np.squeeze(decoder_v)#->(12,n,2)
      ade.append(cal_ade(y_hat_[j,:,:],v_pred[sc][j,:,:]))
      temp_pred.append(y_hat_[j,:,:].copy())#[[n,2],[n,2],...]
    #选择最优的轨迹
    best_id = np.argmin(ade)
    decoder_v[:,j,:,:] = np.expand_dims(temp_pred[best_id],0)
    decoder_a[:,j,:,:] = np.expand_dims(a_pred[sc],0)[:,j,:,:]
  try:
    peds = [info[1] for info in test_log if info[0]==sc]
    for ped in peds:
      inform[(sc,ped)]=decoder_v[:,:,ped,None,:]#存在重复的场景
  except:
    print(sc,ped)
  # print('The scenc {} trajectories has been recorded to information!'.format(scenc))
  # print('*'*30)

assert len(inform)==len(test_log),"The number of information is not equal!"

#rel_traij->abs_traij
scen_id = list(inform.keys())
obs_0 = []
future_0 = []
pred_0 = []
ade_0 = []
fde_0 = []
for id in scen_id:
  v_x_ro,v_y_ro,v_hat = get_obs_trajs(v_obs,v_pred,obs_traij,
  pred_traij,np.squeeze(inform[id],axis=0),ped_tuple,id)
  obs_0.append(v_x_ro)
  future_0.append(v_y_ro)
  pred_0.append(v_hat)
#将历史轨迹和未来轨迹进行合并生成完整轨迹
gt_traij = []
for scenc in range(len(obs_0)):
  gt_traij.append(np.concatenate([obs_0[scenc],future_0[scenc]],axis=0))
#计算损失
ade = []
fde = []
for sc in range(len(future_0)):
  ade.append(cal_ade(future_0[sc],pred_0[sc]))
  fde.append(cal_fde(future_0[sc],pred_0[sc]))
print("平均ade为{:.2f},平均fde为{:.2f}".format(np.mean(ade),np.mean(fde)))