# @Time    : 2022/8/11 17:02
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : layers
# @Project Name :code

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import keras
import numpy as np
import math
from hyperparameters import Hyperparameters as hp

def se_layer(input_,n_blocks,if_training):
    '''
    chinese:挤压模块，实现通道注意力se-resnet.采用挤压模块对原始轨迹进行编码，并引入了残差连接
    english:The original trajectory is embedding with an extrusion module and a residual connection is introduce
    :param input_: the origin shape (b,t,n,32)
    :param n_blocks: the number of residual blocks

    :return: A tensor (b,t,n,d)
    '''
    with tf.variable_scope('SE_moudle',reuse=tf.AUTO_REUSE):
        res_unit = input_
        for _ in range(n_blocks):
            # hidden_0 = tf.layers.conv2d(res_unit,filters=hp.num_units,kernel_size=[hp.kernel_size,hp.kernel_size],strides=(1,1),padding='SAME',use_bias=True,activation='linear')
            # bn_0=tf.layers.batch_normalization(hidden_0,training=if_training)#->result in nan when inference？
            # res_unit = tf.nn.relu(bn_0)
            hidden_0 = tf.layers.dense(res_unit,hp.num_units//4,activation='linear')
            hidden_1 = tf.layers.dense(hidden_0,hp.num_units,activation='linear')
            res_unit+=hidden_1
        global_avg_pool = keras.layers.GlobalAvgPool2D()(res_unit)
        extrusion_layer = tf.layers.dense(global_avg_pool,hp.num_units//16,activation=tf.nn.relu)
        out_ = tf.layers.dense(extrusion_layer,hp.num_units,activation=tf.nn.sigmoid)
        oout = tf.einsum('bd,btnd->btnd',out_,res_unit)
        oout+=input_
        out = tf.nn.relu(oout)
        return out 

def positional_embedding(input_):#位置嵌入，在时间维度上
    '''
    chinese:采用正弦对齐进行位置编码
    english:Positional embedding using sinusoidal alignment
    :param input_: A tensor (b,t,n,d)
    :return: A tensor (b,t,n,d)
    '''
    with tf.variable_scope('Positinal_embedding',reuse=tf.AUTO_REUSE):
        b,t,n,d = get_shape(input_)
        PE = np.array([[pos / np.power(10000, (i-i%2)/hp.num_units) for i in range(d)] for pos in range(t)])
        PE[:, 0::2] = np.sin(PE[:, 0::2])
        PE[:, 1::2] = np.cos(PE[:, 1::2])
        pe = tf.convert_to_tensor(PE)
        pe = tf.cast(pe,'float32')
        pe = tf.tile(tf.expand_dims(pe,0),[b,1,1])
        pe = tf.tile(tf.expand_dims(pe,2),[1,1,n,1])
        return input_+pe

def Tcn(inputs_,scope,if_training=True):
    '''
    chinese:在时间注意力模块中替换全连接操作q,k,v进行特征提取，因为全连接层实际上是时间节点共享权重矩阵
    english:Replace the full connection operations Q, K, V in the temporal attention module for feature extraction,
    because the full connection layer is actually a time node sharing weight matrix
    :param inputs_:
    :param scope: Q,K,V to avoid rename
    :param if_training:a boolen
    :return:A tensor (b,t,n,d)
    '''
    with tf.variable_scope('TCN'+'_'+scope,reuse=tf.AUTO_REUSE):
        bn = keras.layers.BatchNormalization(axis=-1)(inputs_)
        pr = tf.nn.leaky_relu(bn)
        conv = tf.layers.conv2d(pr,filters=hp.num_units,kernel_size=[hp.kernel_size,1],strides=(1,1),padding='SAME')
        bn_ = keras.layers.BatchNormalization(axis=-1)(conv)
        dr = tf.layers.dropout(bn_, hp.drop_rate, training=if_training)
    return dr

def Gcn(inputs_,adj_matrix,scope):
    '''
    chiense:采用图卷积捕获社会特征
    english: gcn adopted for  social features
    :param inputs_: a tensor (b,t,n,d)
    :param adj_matrix: adjance matrix ()
    :param scope: variable name space
    :return:a tensor (b,t,n,d)
    '''
    with tf.variable_scope('GCN'+'_'+scope,reuse=tf.AUTO_REUSE):
        out_ = tf.layers.conv2d(inputs_,filters=hp.num_units,kernel_size=[1,1],strides=(1,1),padding='SAME',use_bias=True)
        out_ = tf.transpose(out_,[0,3,1,2])
        out_ = tf.einsum('bdtn,btnn->bdtn',out_,adj_matrix)
        return tf.transpose(out_,[0,2,3,1])



def TMA(qureis,keys,values,src_mask,scope,if_training=True,if_keymask=True,if_casualmask=True):
    '''
    chinese:核心模块：用于完成时间多头注意力机制，它根据keys的不同可以分为时间多头自注意力机制和时间多头交互式注意力机制
    english:used to complete the temporal multi head attention mechanism.
    It can be divided into temporal multi head self attention mechanism and temporal multi head interactive attention mechanism
    according to the different keys
    :param qureis: (b,t_q,n,d)
    :param keys: (b,t_k,n,d)
    :param values: (b,t_k,n,d)
    :param src_mask: (b,t,n,1)
    :param scope: string, name scope
    :param if_training: dropout parameter
    :param if_keymask:a boolen for key mask
    :param if_casualmask:a boolen for future mask
    :return:(b,t_q,n,d)
    '''
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        b,t,n,d = get_shape(qureis)
        b_,t_,n_,d_ = get_shape(keys)
        if hp.if_tma_dense:
            Q = tf.layers.dense(qureis, hp.num_units, use_bias=False,name='quries')
            K = tf.layers.dense(keys, hp.num_units, use_bias=False, name='keys')
            V = tf.layers.dense(values, hp.num_units, use_bias=False, name='values')
        #tcn
        else:
            Q = Tcn(qureis, 'Q')#->(b,t,n,d)
            K = Tcn(keys, 'K')#->(b,t,n,d)
            V = Tcn(values, 'V')#->(b,t,n,d)
        Q_ = tf.reshape(tf.transpose(tf.concat(tf.split(Q, hp.num_heads, axis=-1), axis=0), [0, 2, 1, 3]),
                        [-1, t, d // hp.num_heads])#->(bhn,T,d/h)
        K_ = tf.reshape(tf.transpose(tf.concat(tf.split(K, hp.num_heads, axis=-1), axis=0), [0, 2, 1, 3]),
                        [-1, t_, d_ // hp.num_heads])#->(bhn,T,d/h)
        V_ = tf.reshape(tf.transpose(tf.concat(tf.split(V, hp.num_heads, axis=-1), axis=0), [0, 2, 1, 3]),
                        [-1, t_, d_ // hp.num_heads])#->(bhn,T,d/h)
        #SDPA for attention score matrix
        with tf.variable_scope('scaled_dot_product_attention', reuse=tf.AUTO_REUSE):
            d_k = Q_.get_shape().as_list()[-1]
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))#->(bhn,T,T)
            outputs /= d_k ** 0.5
            padding_num = -2 ** 32 + 1
            if if_keymask:
                key_mask = src_mask
                key_mask_ = tf.tile(tf.transpose(
                    tf.tile(tf.reshape(tf.transpose(key_mask, [0, 2, 1, 3]), [-1, t_, 1]), [hp.num_heads, 1, 1]),
                    [0, 2, 1]), [1, t, 1])
                paddings = tf.ones_like(outputs) * padding_num
                outputs = tf.where(tf.equal(key_mask_, 0), paddings, outputs)
            if if_casualmask:
                diag_vals = tf.ones_like(outputs[0, :, :])
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
                causal_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])
                paddings = tf.ones_like(causal_masks) * padding_num
                outputs = tf.where(tf.equal(causal_masks, 0), paddings, outputs)
            outputs = tf.nn.softmax(outputs)
            attention = tf.transpose(outputs, [0, 2, 1])
            tf.summary.image("attention", tf.expand_dims(attention[:1], -1))
            outputs = tf.layers.dropout(outputs, rate=hp.drop_rate, training=if_training)
            outputs = tf.matmul(outputs, V_)#->(bhn,T,d/h)
        outputs = tf.concat(tf.split(outputs, hp.num_heads, axis=0), axis=-1)#->(bn,T,d)
        outputs = tf.layers.dense(outputs, hp.num_units)
        outputs = tf.layers.dropout(outputs, rate=hp.drop_rate, training=if_training)
        outputs += tf.reshape(tf.transpose(qureis, [0, 2, 1, 3]), [-1, t, d])
        outputs = ln(outputs)
        outputs = tf.transpose(tf.reshape(outputs, [-1, n, t, d]), [0, 2, 1, 3])
        return outputs

def SMA(qureis,keys,values,a_obs,src_mask,scope,if_training=True,if_keymask=True,if_socialmask=True):
    """
    chinese:核心模块：基于动态图的社会多头注意力机制，采用了两个掩码技术：轨迹掩码和社会掩码。轨迹掩码掩盖掉补齐的轨迹，社会掩码掩盖掉不相关的行人对
    english:The core module: a graph-based social multi-head attention mechanism employs two masking techniques:
    trajectory masking and social masking.Trajectory masks mask out padded trajectories, and social masks mask irrelevant pedestrian pairs
    :param qureis:a tensor (b,t,n,d)
    :param keys:a tensor (b,t,n,d) the same as qureies
    :param values:a tensor (b,t,n,d) the same as qureies
    :param a_obs:a tensor (b,t,n,n)
    :param src_mask:(b,t,n,1)
    :param scope:name space,'SMA'
    :param if_training: a boolen,when test False
    :param if_keymask:a boolen, defalut--True
    :param if_socialmask: a boolen, defalut--True
    :return: a tensor (b,t,n,d)
    """
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        b, t, n, d = get_shape(qureis)
        if hp.if_tma_dense:
            Q = tf.layers.dense(qureis, hp.num_units, use_bias=False,name='qureis')
            K = tf.layers.dense(keys, hp.num_units, use_bias=False, name='keys')
            V = tf.layers.dense(values, hp.num_units, use_bias=False, name='values')
        else:
            # gcn
            Q = Gcn(qureis, a_obs, 'Q')
            K = Gcn(keys, a_obs, 'K')
            V = Gcn(values, a_obs, 'V')
        # split&concat
        Q_ = tf.reshape(tf.concat(tf.split(Q, hp.num_heads, axis=-1), axis=0), [-1, n, d // hp.num_heads])#->(bth,n,d/h)
        K_ = tf.reshape(tf.concat(tf.split(K, hp.num_heads, axis=-1), axis=0), [-1, n, d // hp.num_heads])#->(bth,n,d/h)
        V_ = tf.reshape(tf.concat(tf.split(V, hp.num_heads, axis=-1), axis=0), [-1, n, d // hp.num_heads])#->(bth,n,d/h)
        # SDPA for attention score matrix
        with tf.variable_scope('scaled_dot_product_attention', reuse=tf.AUTO_REUSE):
            d_k = Q_.get_shape().as_list()[-1]
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
            outputs /= d_k ** 0.5
            padding_num = -2 ** 32 + 1
            if if_keymask:
                key_mask = src_mask
                key_mask_ = tf.tile(
                    tf.transpose(tf.tile(tf.reshape(key_mask, [-1, n, 1]), [hp.num_heads, 1, 1]), [0, 2, 1]), [1, n, 1])
                paddings = tf.ones_like(outputs) * padding_num
                outputs = tf.where(tf.equal(key_mask_, 0), paddings, outputs)
            if if_socialmask:
                adj_ = tf.reshape(a_obs, [-1, n, n])
                zeros_a = tf.zeros_like(adj_)
                paddings_a = tf.ones_like(adj_) * padding_num
                social_mask = tf.where(tf.equal(adj_, 0), paddings_a, zeros_a)
                outputs = tf.add(outputs, tf.tile(social_mask,[hp.num_heads,1,1]))
            outputs = tf.nn.softmax(outputs) * tf.tile(adj_,[hp.num_heads,1,1])
            # #query mask becuase there is no interative attention in SMA
            # query_masks = tf.tile(tf.reshape(src_mask,[-1,n,1]),[hp.num_heads,1,n])
            # outputs *=query_masks
            #**********************************************************************
            attention = tf.transpose(outputs, [0, 2, 1])
            tf.summary.image("attention", tf.expand_dims(attention[:1], -1))
            outputs = tf.layers.dropout(outputs, rate=hp.drop_rate, training=if_training)
            outputs = tf.matmul(outputs, V_)
        outputs = tf.concat(tf.split(outputs, hp.num_heads, axis=0), axis=2)
        out = tf.layers.dense(outputs, hp.num_units)
        out = tf.layers.dropout(out, rate=hp.drop_rate, training=if_training)
        out += tf.reshape(qureis, [-1, n, d])
        out = ln(out)
        out = tf.reshape(out, [-1, t, n, d])
        return out

def feed_forward(inputs):
    """

    :param inputs: (B,T,N,d)
    :return: (B,T,N,,d)
    """
    with tf.variable_scope('FFN_layers',reuse=tf.AUTO_REUSE):
        outputs = tf.layers.dense(inputs,hp.num_units//4,activation=tf.nn.relu)
        outputs = tf.layers.dense(outputs,hp.num_units)
        outputs+=inputs
        outputs = ln(outputs)
    return outputs


def ln(inputs, epsilon=1e-8):
    '''
    Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    :param inputs: A tensor (b,t,n,d)
    :param epsilon:A floating number
    :return:A tensor (b,t,n,d)
    '''
    with tf.variable_scope('ln', reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer(), )
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

def get_shape(input_):
    """
    :param input_: origin tensor->(b,t,n,d)
    :return: 4 shape value
    """
    b = tf.shape(input_)[0]
    t = input_.get_shape().as_list()[1]
    n = tf.shape(input_)[2]
    d = input_.get_shape().as_list()[3]
    return b,t,n,d