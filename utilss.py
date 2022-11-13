# @Time    : 2022/8/9 20:15
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : utilss
# @Project Name :code

import numpy as np
import math
import networkx as nx
import pickle
import os
import torch.distributions.multivariate_normal as torchdist
import torch


from hyperparameters import Hyperparameters as hp

def poly_fit(traj, traj_len, threshold):
    """
    chinese:用于区分预测曲线是非线性轨迹还是线性轨迹
    english:To distinguish whetehr the trajectory is the linear or non-linear
    Input:
    - traj: Numpy array of shape (2, T)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0
def read_file(_path, delim='\t'):
    """
    chinese:读取行人数据->(t,id,x,y)
    english:read the pedstrain datasets
    :param _path:such as  './datasets/eth/train/biwi_hotel_train.txt'
    :param delim:' ' or '\t'
    :return: txt data
    """
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)
def anorm(p1,p2,sigma,dis_eplision,kernel_name):
    """
    chinese:计算相同时间下，不同行人之间的距离
    english:the distance between the different pedstrains at the same time 
    :param p1: (x1,y1)
    :param p2: (x2,y2)
    :param sigma: sigma^2->guass_dis
    :param dis_eplision: for guass_dis a threshold
    :param kernel_name: guass_dis or sim_dis
    :return: norm a value 
    """""
    if kernel_name=='guass_dis':
        dis = ((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
        NORM = np.exp(-dis/sigma)
        if NORM >= dis_eplision:
          return NORM
        else:
          return 0
    elif kernel_name=='sim_dis':
        NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
        if NORM ==0:
          return 0
        else:
          return 1/(NORM)
def seq_to_graph(seq_,seq_rel,norm_lap_matr = True):
    """
    chinese:将轨迹序列转为动态图
    english:transfer the trajectory into the dynamic graph
    :param seq_: (N,2,T)
    :param seq_rel: (N,2,T)
    :param norm_lap_matr: boolen
    :return: v->(T,N,2);A->(T,N,N)
    """
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]#view predstrain as  nodes
    V = np.zeros((seq_len,max_nodes,2))
    A = np.zeros((seq_len,max_nodes,max_nodes))
    for s in range(seq_len):
      step_ = seq_[:,:,s]
      step_rel = seq_rel[:,:,s]
      for h in range(len(step_)):
        V[s,h,:] = step_rel[h]
        # V[s, h, :] = step_[h]#abs_taij needed,i think rel_traij too complex when it comes to get abs traij
        A[s,h,h] = 1
        for k in range(h+1,len(step_)):
          l2_norm = anorm(step_rel[h],step_rel[k],hp.sigma,hp.eplision,kernel_name=hp.kernel_name)
          # l2_norm = anorm(step_[h], step_[k], hp.sigma, hp.eplision, kernel_name=hp.kernel_name)
          A[s,h,k] = l2_norm
          A[s,k,h] = l2_norm
      A[s]+= np.identity(len(step_))#The diagonal is set to 1 to ensure information transfer
      if norm_lap_matr:
        G = nx.from_numpy_matrix(A[s,:,:])
        A[s,:,:] = nx.normalized_laplacian_matrix(G).toarray()
    return V,A

def seq_to_nodes(x_traj,ped_tuple,scene_idx):
    '''
    abs_traij2abs_garph
    :param x_traj: (all_ped_num)
    :param ped_tuple: [(0,8),(8,16),...(start_id,end_id)]
    :param scene_idx: correspond the ped_tuple length
    :return: abs_graph->(T,N,2)
    '''
    obs_sc = x_traj[ped_tuple[scene_idx][0]:ped_tuple[scene_idx][1]]
    v_x = np.transpose(obs_sc,(2,0,1))
    return v_x
def nodes_rel_to_nodes_abs(nodes,init_node):
    '''
    To transfer rel_graph2abs_graph->obs_rel_graph,future_rel_graph,and pred_rel_graph
    :param nodes:rel_graph
    :param init_node:abs_graph
    :return:abs_graph->(T,N,2)
    '''
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            if hp.padding_type=='rel':
                nodes_[s,ped,:] = np.sum(nodes[:s+1,ped,:],axis=0) + init_node[ped,:]#the kernel code
            else:
                nodes_[s,ped, :] = nodes[s,ped,:]+init_node[ped,:]
    return nodes_.squeeze()

def get_obs_trajs(v_obs,v_pred,obs_traij,pred_traij,pred,ped_tuple,scene_id):
    '''
    Convert historical and future relative trajectories to absolute trajectories based on historical trajectories
    :param v_obs:a list which each cell include (8,n,2)
    :param v_pred:list which each cell include (12,n,2)
    :param obs_traij:(all_pep,8,2)
    :param pred_traij:(all_pep,12,2)
    :param pred:(12,n,2)
    :param ped_tuple:[(s,e),...,(s,e)]
    :param scene_id:the index of scene
    :return:abs traiject include->history,future,pred->(t,n,2)
    '''
    v_obs_0 = v_obs[scene_id[0]][:,scene_id[1],None,:]#第1个场景的相对观测轨迹
    v_pred_0 = v_pred[scene_id[0]][:,scene_id[1],None,:]#第1场景的相对未来轨迹
    v_x = seq_to_nodes(obs_traij,ped_tuple,scene_idx=scene_id[0])#第1个场景的绝对历史轨迹->(8,n,2)
    v_y = seq_to_nodes(pred_traij,ped_tuple,scene_idx=scene_id[0])#第1个场景的绝对未来轨迹->(12,n,2)
    if hp.padding_type=='rel':
        v_x_ro = nodes_rel_to_nodes_abs(v_obs_0,v_x[0,scene_id[1],None,:])#基于历史场景图初始时刻的历史图
        v_y_ro = nodes_rel_to_nodes_abs(v_pred_0,v_x[-1,scene_id[1],None,:])#基于历史场景图最后时刻的未来图
        v_hat = nodes_rel_to_nodes_abs(pred,v_x[-1,scene_id[1],None,:])#基于历史场景图最后时刻的预测图
        return v_x_ro, v_y_ro, v_hat
    elif hp.padding_type=='tobs':
        v_x_ro = nodes_rel_to_nodes_abs(v_obs_0, v_x[-1,scene_id[1],None,:])
        v_y_ro = nodes_rel_to_nodes_abs(v_pred_0, v_x[-1,scene_id[1],None,:])
        v_hat = nodes_rel_to_nodes_abs(pred, v_x[-1,scene_id[1],None,:])
        return v_x_ro, v_y_ro, v_hat
    elif hp.padding_type=='t0':
        v_x_ro = nodes_rel_to_nodes_abs(v_obs_0, v_x[0,scene_id[1],None,:])
        v_y_ro = nodes_rel_to_nodes_abs(v_pred_0, v_x[0,scene_id[1],None,:])
        v_hat = nodes_rel_to_nodes_abs(pred, v_x[0,scene_id[1],None,:])
        return v_x_ro, v_y_ro, v_hat
    else:
        print("No padding type!")


def cal_ade(y_hat,gt):#计算平均位移误差,输入形状为（t,n,2）
    '''
    Average displacement error(ADE) after filtering out padding
    :param y_hat: （t,n,2）
    :param gt: （t,n,2）
    :return: a value
    '''
    return np.mean(np.abs(y_hat-gt))

def cal_fde(y_hat,gt):#计算最终位移误差
    '''
    Final displacement error(FDE) after filtering out padding
    :param y_hat: （t,n,2）
    :param gt: （t,n,2）
    :return:  a value
    '''
    return np.mean(np.abs(y_hat[-1,:]-gt[-1,:]))

def trajectory_aligen(input_traj,start,end):
    """
    fake algorithm 1
    chinese:在一个片段中（长度为20），并不是该片段内的所有行人的轨迹都是20的完成轨迹，对于缺失的轨迹用0进行补全
    english:In a sequence(the length is 20), Not all pedestrian trajectories in this segment are completed trajectories
    of 20, 0 is used to complete the missing trajectories
    :param input_traj:（t,same_id,x,y）
    :param start:start frame, like 190
    :param end:end frame,like 380
    :return:updated input_traj,m_id,e_id,s_id(the number of miss trajectory, miduem,end,and start)
    """
    ped_id = np.unique(input_traj[:, 1])[0]
    # when it comes to the medium trajectory
    diff = (input_traj[:, 0][1:] - input_traj[:, 0][:-1]) / (10 * hp.skip)
    origin_insert_loc = np.where(diff != 1)[0]  # the origin location where insert into 
    num_inserts = diff[origin_insert_loc] - 1  # the response location should be inserted times
    m_id = 0  # to log the loss times if medium frames ara loss
    if len(origin_insert_loc) != 0:
        for i, loc in enumerate(origin_insert_loc):
            for _ in range(int(num_inserts[i])):
                m_id += 1#to ensure the array index hae been changed
                loc_ = loc + m_id
                m_t = [input_traj[loc_ - 1, 0] + hp.skip * 10, ped_id, 0, 0].copy()
                input_traj = np.insert(input_traj, loc_, m_t, axis=0)
    # when it comes to the end of the frame loss
    e_id = 0
    while input_traj[-1, 0] != end:
        e_id += 1
        input_traj = np.append(input_traj, [[input_traj[-1, 0] + 10, ped_id, 0, 0]], axis=0)
    # when it comes to the start of the frame loss
    s_id = 0
    while input_traj[0, 0] != start:
        s_id += 1
        input_traj = np.insert(input_traj, 0, [input_traj[0, 0] - 10, ped_id, 0, 0], axis=0)
    assert len(input_traj) == (end - start) / 10 + 1, 'Trajectory length does not meet input requirements！'
    # print('There are {} frames loss in total,and m,e,s are {}!'.format(m_id+e_id+s_id,[m_id,e_id,s_id]))
    return input_traj, m_id, e_id, s_id

def ped_pad(traij,pad_type='rel'):
    '''
    chinese:对用0补齐的行人轨迹进行重新填充，[0,1,0,2,0]->[1,1,1,2,2]（相对填充）->[2,1,2,2,2](最后一个观测填充)->[1,1,1,2,1](第一个填充)
    :param traij: [2,20]
    :return:[2,20] after padding with a variable depending on ground truth
    '''
    zero = np.where(np.sum(traij, axis=0) == 0)[0]#为0的序列索引
    no_zero = np.where(np.sum(traij, axis=0) != 0)[0]#不为0的序列索引
    if pad_type=='rel':
        for colum in range(traij.shape[1]):
            if colum in zero&np.all(no_zero>colum):#开端填充
                traij[:,colum]=traij[:,no_zero[0]]
            elif colum in zero&np.all(no_zero<colum):
                traij[:,colum]=traij[:,no_zero[-1]]#结尾填充
            elif colum in zero:
                traij[:,colum] = traij[:,colum-1]#中间填充
    elif pad_type=='tobs':
        if np.any(no_zero<=7):
            traij[:,zero]=traij[:,no_zero[no_zero<=7][-1],None]
        else:
            traij=traij
    elif pad_type=='t0':
        if np.any(no_zero <= 7):
            traij[:, zero] = traij[:, no_zero[no_zero <= 7][0], None]
        else:
            traij=traij
    else:
        print("No padding type!")
    return traij

def reconstruct_trajdata(seq_list,seq_list_rel,non_linear_ped,loss_mask_list,seq_start_end):
    """
    chinese:去掉经过补全的数据，其余所有数组均按照完整的轨迹数据生成
    english:Remove the completed data, and all other arrays are generated according to the complete trajectory data
    :param seq_list:(N,2，20)
    :param seq_list_rel:(N,2，20)
    :param non_linear_ped:（N,）
    :param loss_mask_list:（N,20）
    :param seq_start_end:(start_idx,end_idx)
    :return:a new data format with origin datasets
    """
    idx = np.where(np.sum(loss_mask_list, axis=1) != 0)[0]
    print('The complete trajectories are {}, and the all are {}'.format(len(idx), len(loss_mask_list)))
    seq_origin_list = seq_list[idx]
    seq_origin_list_rel = seq_list_rel[idx]
    non_linear_ped_origin = non_linear_ped[idx]
    loss_mask_list_origin = loss_mask_list[idx]
    origin_num_peds_in_seq = []
    for se in range(len(seq_start_end)):
        count = 0
        start, end = seq_start_end[se]
        for id in idx:
            if id in np.arange(start, end):
                count += 1
            else:
                count += 0
        origin_num_peds_in_seq.append(count)
    return seq_origin_list ,seq_origin_list_rel,non_linear_ped_origin ,loss_mask_list_origin,origin_num_peds_in_seq

def sample_traij(y_hat):
    '''
    Random sampling based on multivariate Gaussian distribution
    :param y_hat:(b,t,n,2)
    :return:(t,n,2)
    '''
    y_hat = np.squeeze(y_hat)
    y_hat = torch.from_numpy(np.float32(y_hat))
    sx = torch.exp(y_hat[:, :, 2])
    sy = torch.exp(y_hat[:, :, 3])
    corr = torch.tanh(y_hat[:, :, 4])
    cov = torch.zeros(y_hat.shape[0], y_hat.shape[1], 2, 2)
    cov[:, :, 0, 0] = sx * sx
    cov[:, :, 0, 1] = corr * sx * sy
    cov[:, :, 1, 0] = corr * sx * sy
    cov[:, :, 1, 1] = sy * sy
    mean = y_hat[:, :, 0:2]
    # if np.any(np.isnan(cov).numpy()==1) or np.any(np.isnan(mean).numpy()==1):
    #   # print("协方差为",cov)
    #   # print("均值为",mean)
    #   mean=torch.zeros_like(mean)
    #   cov = torch.ones_like(cov)
    mvnormal = torchdist.MultivariateNormal(mean, cov)
    return mvnormal.sample().numpy()  # 基于多元高斯分布进行采样

def get_inform(v_obs,loss_mask):
    num_ped = [scene.shape[1] for scene in v_obs]
    cum_ped = np.cumsum(num_ped)
    ped_tuple = [(start, end) for start,
                end in zip([0]+cum_ped.tolist()[:-1],cum_ped.tolist())]
    no_mask = []
    for id,se in enumerate(ped_tuple):
        s_e = np.arange(se[0],se[1])
        no_mask.append(loss_mask[s_e,:])
    log = []
    for sc in range(len(no_mask)):
        for ped in range(no_mask[sc].shape[0]):
            if np.all(no_mask[sc][ped,:]!=0):
                log.append((sc,ped))
    return log,ped_tuple,no_mask#->(sc,ped)


def gen_batch(data):
    '''
    chinese:数据生成器
    english:data generator
    :param data: a data list->[D.obs_traj,D.pred_traj,D.obs_traj_rel,D.pred_traj_rel,D.non_linear_ped,
    D.loss_mask_list,D.v_obs,D.A_obs,D.v_pred,D.A_pred]
    :return:[v_obs,a_obs,v_pred,a_ored]->;the batch_size is 32
    '''
    v_obs = data[6]
    a_obs = data[7]
    v_pred = data[8]
    a_pred = data[9]
    len_inputs = len(v_obs)
    idx = np.arange(len_inputs)
    np.random.shuffle(idx)
    for start_idx in range(0,len_inputs,hp.batch_size):
        end_idx = start_idx+hp.batch_size
        if end_idx>len_inputs:
            end_idx = len_inputs
        slide = idx[start_idx:end_idx]
        yield [get_all_data(v_obs,slide),get_all_data(a_obs,slide),
        get_all_data(v_pred,slide),get_all_data(a_pred,slide)]

def get_all_data(datas,slide):
    '''
    chinese:根据索引返回列表中的数据
    english：according to the index ,return the each data in list
    :param datas: []->a list
    :param slide: [id1,id2,....,idb]
    :return: a list
    '''
    return [datas[idx] for idx in slide]

def write_to_pickle(data,data_type,data_used):
    """
    chinese:将所有处理的数据保存的pkl文件中
    english:Save all processed data in a pkl file
    :param data: a total list
    :param data_type: such as 'eth/'
    :param data_used: such as 'train/'
    :return: None
    """
    folder = os.path.exists(data_type)
    if not folder:
        os.makedirs(data_type)
    file = open(data_type+data_used.strip('/')+'.pkl','wb')#wb to binary
    pickle.dump(data,file)
    file.close()
def read_from_pickle(data_type,data_used):
    """
    chinese:读取pkl文件
    english:read data from .pkl file
    :param data_type: such as 'eth/'
    :param data_used: such as 'train/'
    :return: all datas
    """
    file = open(data_type+data_used.strip('/')+'.pkl','rb')
    return pickle.load(file)

def toarray(data):
    """
    chinese:将一个列表转为数组
    english:list->array
    :param data: a list
    :return: an array
    """
    if isinstance(data,list):
        return np.array(data)