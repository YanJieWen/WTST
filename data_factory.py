# @Time    : 2022/8/9 20:15
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : data_factory
# @Project Name :code

import os
import math
from tqdm import tqdm
import time
from hyperparameters import Hyperparameters as hp
from utilss import *


class Traject_data_factory():
    def __init__(self, data_dir):
        """

        :param data_dir: such as ./datasets/eth/train/
        """
        self.data_dir = data_dir
        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]  # 读取train文件下所有的的txt文件
        num_peds_in_seq = []  # the number of pedstrain in each seuqnece
        seq_list = []  # each save-> [(N0,2,20),...,(Nn,2,20)]
        seq_list_rel = []  # each save relative-> [(N0,2,20),...,(Nn,2,20)]
        loss_mask_list = []  # [(N0,20),...,(Nn,20)]
        non_linear_ped = []  # [1,0],whther the traj is non-linear or linear
        num_m_loss = 0  # the number of  trajectory that medium traj loss
        num_s_loss = 0  # the number of  trajectory that start traj loss
        num_e_loss = 0  # the number of  trajectory that end traj loss
        for path in all_files:
            '''
            chinese:受到环境，摄像头性能的影响，可能存在部分帧不连续，对这部分缺失的总帧进行补全
            english:Influenced by the environment and camera performance, 
            some frames may be discontinuous, and the missing total frames will be completed.
            fake algorithm 2
            '''
            data = read_file(path, hp.delim)
            frames = np.unique(data[:, 0]).tolist()  # 共计587帧
            insert_id = 99999
            frames_new = np.arange(frames[0], frames[-1] + 10 * hp.skip, 10 * hp.skip)  # 补齐丢失的帧为594
            insert_frames = [i for i in frames_new if i not in frames]  # 需要对齐的帧
            for frame in insert_frames:
                data = np.append(data, [[frame, insert_id, 0, 0]], axis=0)
            frames = frames_new.tolist()
            data = data[data[:, 0].argsort()]  # 对第一列时间进行排序
            # ===============================================================================
            frame_data = []  # 用于存储每一个时间帧内所包含的行人
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - hp.seq_len + 1) / hp.skip))  # 共计T个样本点
            d = 0
            for idx in range(0, num_sequences * hp.skip + 1, hp.skip):  # 间隔1个数据帧进行采集
                curr_seq_data = np.concatenate(frame_data[idx:idx + hp.seq_len], axis=0)  # (m,4),超过了则直接到最后一位拼接
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                max_peds_in_frame = max(hp.max_peds_in_frame, len(peds_in_curr_seq))
                seq_frames = np.unique(curr_seq_data[:, 0])
                if len(seq_frames) != hp.seq_len:
                    continue
                start_frame = seq_frames[0]
                end_frame = seq_frames[-1]
                # 保存相对位置
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, hp.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, hp.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), hp.seq_len))
                num_peds_considered = 0  # 统计一个片段里面有多少个满足要求的行人轨迹
                # num_full_peds = 0 #统计
                _non_linear_ped = []  # 一个片段内部非线性行人轨迹

                for _, ped_id in enumerate(peds_in_curr_seq):  # 开始遍历每一个seq下的每一个行人轨迹
                    if ped_id != insert_id:  # 过滤掉insert_id否则会对insert_id进行补全
                        curr_seq_data_temp = curr_seq_data.copy()
                        if insert_id in np.unique(curr_seq_data_temp[:, 1]):
                            curr_seq_data_temp[
                                np.where(curr_seq_data_temp[:, 1] == insert_id), 1] = ped_id  # 更新insert_id,用ped_id替换它
                        curr_ped_seq = curr_seq_data_temp[curr_seq_data_temp[:, 1] == ped_id, :]
                        curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                        # copy_curr = curr_ped_seq.copy()#用于判定该行人轨迹是否为目标代理
                        curr_ped_seq, m_id, e_id, s_id = trajectory_aligen(curr_ped_seq, start_frame,
                                                                           end_frame)  # 调用轨迹对齐处理每一个行人的轨迹
                        if len([i for i in insert_frames if
                                i in np.unique(curr_ped_seq[:, 0])]) != 0:  # 过滤掉因为插入丢失帧触发的中间轨迹丢失冗余计算
                            m_id = 0
                        num_m_loss += m_id  # 中间可能包含如：某一个时间序列内的一个行人只有最后一个轨迹，但是通过插入丢失的帧数形成了类似中间轨迹丢失的序列
                        num_e_loss += e_id
                        num_s_loss += s_id
                        # 绝对轨迹的处理->转换维度->统计填充的轨迹多少->进行填充转换->非线性轨迹统计
                        pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                        pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                        curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])  # (2,T)
                        _idx = num_peds_considered
                        if 0 in curr_ped_seq[0, :] or 0 in curr_ped_seq[1, :]:  # 筛选出经过补全的轨迹
                            curr_loss_mask[_idx, pad_front:pad_end] = 0
                        else:
                            curr_loss_mask[_idx, pad_front:pad_end] = 1
                        curr_ped_seq = ped_pad(curr_ped_seq,hp.padding_type)#相对轨迹填充，obs填充，第一个位置为原点的填充，核心代码
                        curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                        _non_linear_ped.append(poly_fit(curr_ped_seq, hp.pred_len, hp.poly_threshold))
                        # 制作相对轨迹
                        rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                        if hp.padding_type=='rel':
                            rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                        elif hp.padding_type=='tobs':
                            rel_curr_ped_seq = curr_ped_seq - curr_ped_seq[:, hp.obs_len-1:hp.obs_len]
                        elif hp.padding_type=='t0':
                            rel_curr_ped_seq = curr_ped_seq - curr_ped_seq[:, 0:1]
                        curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                        num_peds_considered += 1
                        # ff =np.sum(copy_curr[:,-2:],axis=-1)#->[0,1,0,...,0,0]
                        # if copy_curr.shape[0]==hp.obs_len+hp.pred_len and len(ff[ff!=0])>int((hp.obs_len+hp.pred_len)*hp.store_frac):#完整轨迹以及过滤帧填充轨迹（必须保存百分之80的非帧填充）
                        #     num_full_peds+=1

                if num_peds_considered > hp.min_ped:  # 场景下要有行人交互行为，即目标行人数目>1,这个值的存在可能导致提取的行人轨迹数量出现偏差
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        self.loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        self.non_linear_ped = np.asarray(non_linear_ped)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        if hp.reconstruct:
            seq_list, seq_list_rel, self.non_linear_ped, \
            self.loss_mask_list, num_peds_in_seq = reconstruct_trajdata(seq_list, seq_list_rel, self.non_linear_ped,
                                                                        self.loss_mask_list, seq_start_end)
        # convert array into the graph
        self.obs_traj = seq_list[:, :, :hp.obs_len]
        self.pred_traj = seq_list[:, :, hp.obs_len:]
        self.obs_traj_rel = seq_list_rel[:, :, :hp.obs_len]
        self.pred_traj_rel = seq_list_rel[:, :, hp.obs_len:]
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        self.v_obs = []
        self.A_obs = []
        self.v_pred = []
        self.A_pred = []
        print("Processing Data .....")
        pbar = tqdm(total=len(seq_start_end))
        for ss in range(len(seq_start_end)):
            try:
                pbar.update(1)
                start, end = seq_start_end[ss]
                v_, a_ = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], hp.norm_lap_matr)
                self.v_obs.append(v_.copy())
                self.A_obs.append(a_.copy())
                v_, a_ = seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :], hp.norm_lap_matr)
                self.v_pred.append(v_.copy())
                self.A_pred.append(a_.copy())
            except:
                # print(self.obs_traj[start:end, :].shape)
                print("continue!")#可能过滤后某些场景没有轨迹
        pbar.close()


def main():
    data_type = hp.hotel_dir  # need to change
    data_used = hp.test_dir  # need to change
    data_dir = hp.data_dir + data_type + data_used
    D = Traject_data_factory(data_dir)
    all_ = [D.obs_traj, D.pred_traj, D.obs_traj_rel, D.pred_traj_rel, D.non_linear_ped,
            D.loss_mask_list, D.v_obs, D.A_obs, D.v_pred, D.A_pred]
    write_to_pickle(all_, data_type, data_used)
    print('The {} has been writed!'.format(data_used))


if __name__ == '__main__':
    main()