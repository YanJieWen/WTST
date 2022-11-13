# @Time    : 2022/8/11 17:02
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : model
# @Project Name :code


from layers import *
from hyperparameters import Hyperparameters as hp
from utilss import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior


class DGFomer():
    def __init__(self,v_obs,a_obs,v_pred,a_pred,if_training):
        self.v_obs = v_obs
        self.a_obs = a_obs
        self.v_pred = v_pred
        self.a_pred = a_pred
        self.if_training=if_training
    def encoder(self):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            #encoder_embeeding->src_mask,fc,se_moudle,positional_embdding
            src_mask = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(self.v_obs), axis=-1)), axis=-1)
            hidden_1 = tf.layers.dense(self.v_obs, hp.num_units, use_bias=True)
            se_out = se_layer(hidden_1, hp.num_se_blocks,if_training=self.if_training)
            pe_out = positional_embedding(se_out)
            pe_out *= src_mask
            enc = pe_out
            enc = tf.layers.dropout(enc, hp.drop_rate, training=self.if_training)
            #encoer kernel multi-head attention
            for i in range(hp.num_blocks):
                with tf.variable_scope('num_blocks{}'.format(i), reuse=tf.AUTO_REUSE):
                    enc = TMA(qureis=enc,keys=enc,values=enc,src_mask=src_mask,scope='TSA',if_training=self.if_training
                              ,if_keymask=True,if_casualmask=False)
                    enc = SMA(qureis=enc,keys=enc,values=enc,a_obs=self.a_obs,src_mask=src_mask,scope='SMA'
                              ,if_training=self.if_training,if_keymask=True,if_socialmask=True)
                    #feed forward layers
                    enc = feed_forward(enc)
        enc_memory = enc
        return enc_memory,src_mask

    def decoder(self,memory,src_mask):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            # decoder_embeeding->right_shifted,tgt_mask,fc,se_moudle,positional_embdding
            #right-shifted
            v_pred_copy = tf.concat((self.v_obs[:,-1:,:,:],self.v_pred[:,:-1,:,:]),axis=1)
            a_pred_copy = tf.concat((self.a_obs[:,-1:,:,:],self.a_pred[:,:-1,:,:]),axis=1)
            tgt_mask = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(v_pred_copy), axis=-1)), axis=-1)
            hidden_1 = tf.layers.dense(v_pred_copy, hp.num_units, use_bias=True)
            se_out = se_layer(hidden_1, hp.num_se_blocks,if_training=self.if_training)
            pe_out = positional_embedding(se_out)
            pe_out *= tgt_mask
            dec = pe_out
            dec = tf.layers.dropout(dec, hp.drop_rate, training=self.if_training)
            # decoder kernel multi-head attention
            for i in range(hp.num_blocks):
                with tf.variable_scope('num_blocks{}'.format(i),reuse=tf.AUTO_REUSE):
                    dec = TMA(qureis=dec,keys=dec,values=dec,src_mask=tgt_mask,scope='CAUSALTSA',if_training=self.if_training
                              ,if_keymask=True,if_casualmask=True)
                    dec = SMA(qureis=dec,keys=dec,values=dec,a_obs=a_pred_copy,src_mask=tgt_mask,scope='SMA'
                              ,if_training=self.if_training,if_keymask=True,if_socialmask=True)
                    dec = TMA(qureis=dec,keys=memory,values=memory,src_mask=src_mask,scope='TISA',if_training=self.if_training
                              ,if_keymask=True,if_casualmask=False)
                    # feed forward layers
                    dec = feed_forward(dec)
        with tf.variable_scope('Prediction',reuse=tf.AUTO_REUSE):
            y_hat = tf.layers.dense(dec, hp.out_units, use_bias=False,name='prediction')
        return y_hat


def graph_loss(v_pred,v_gt):
    """
    chinese:双变量高斯分布损失，5维度分别代表ux,uy,sigmax,sigmay，corr.
    ux:mean of the distribution in x
    uy:mean of the distribution in y
    sigmax:std dev of the distribution in x
    sigmay:std dev of the distribution in y
    corr:Correlation factor of the distribution
    english:Bivariate Gaussian Loss
    :param v_pred: [1,t,n,5]->tf.squeeze(v_pred)
    :param v_gt: [1,t,n,2]
    :param padding_mask: [t,n]
    :return:-logz loss
    """
    # is_target = tf.cast(tf.sign(tf.reduce_sum(tf.abs(v_gt), axis=-1)),dtype=v_gt.dtype)
    # padding_mask = tf.cast(tf.not_equal(is_target, 0),dtype=v_gt.dtype)
    normx = v_gt[:,:,0]-v_pred[:,:,0]
    normy = v_gt[:,:,1]-v_pred[:,:,1]
    sx = tf.exp(v_pred[:,:,2])
    sy = tf.exp(v_pred[:,:,3])
    corr = tf.tanh(v_pred[:,:,4])
    sxsy=sx*sy
    z = (normx/sx)**2+(normy/sy)**2-2*((corr*normx*normy)/sxsy)
    negrhp = 1-corr**2
    result = tf.exp(-z/(2*negrhp))
    denom = 2*np.pi*(sxsy*tf.sqrt(negrhp))
    result=result/denom#a difficult if result>according to the social-stgcnn the -loss seems correct......
    eplision = 1e-20#to avoid nan
    # result = -tf.log(tf.maximum(result, eplision))*padding_mask
    result = -tf.log(tf.maximum(result, eplision))
    result = tf.reduce_mean(result)
    # result = tf.reduce_sum(result)/(tf.reduce_sum(padding_mask)+1e-7)#avoid nan
    return result

# def main():
#     v_obs = tf.placeholder(dtype=tf.float32, shape=[1, 8, 5, 2])
#     a_obs = tf.placeholder(dtype=tf.float32, shape=[1, 8, 5, 5])
#     v_pred = tf.placeholder(dtype=tf.float32, shape=[1, 12, 5, 2])
#     a_pred = tf.placeholder(dtype=tf.float32, shape=[1, 12, 5, 5])
#     model = DGFomer(v_obs,a_obs,v_pred,a_pred)
#     memory, src_mask = model.encoder(if_training=True)
#     pred = model.decoder(memory, src_mask)
#     print(pred)

# if __name__ == '__main__':
#     main()



