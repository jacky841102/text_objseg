import tensorflow as tf
from tensorflow.python.ops.nn import dropout as drop
import util.rnn as rnn
from util.cnn import conv_layer as conv
from util.cnn import conv_relu_layer as conv_relu
from util.cnn import deconv_layer as deconv
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu
from models import vgg_net, deeplab, lstm_net
from models.processing_tools import *

def recurrent_multimodal(text_seq_batch, imcrop_batch, num_vocab, embed_dim,
    lstm_dim, mlp_hidden_dims, feature_vis_dropout, mlp_dropout):

    _, feat_langs, embedded_seq = lstm_net.lstm_net(text_seq_batch, num_vocab, embed_dim, lstm_dim)

    # feat_vis = vgg_net.vgg_fc8_full_conv(imcrop_batch, 'vgg_local', apply_dropout=vgg_dropout)
    feat_vis = deeplab.deeplab_fc8_full_conv(imcrop_batch, 'deeplab', output_dim=1000)

    featmap_H, featmap_W = feat_vis.get_shape().as_list()[1:3]

    # Reshape and tile feat_langs, embedded_seq
    T, N, D_text = embedded_seq.get_shape().as_list()
    feat_langs = [tf.tile(tf.reshape(feat_lang, [N, 1, 1, D_text]),
        [1, featmap_H, featmap_W, 1]) for feat_lang in feat_langs]

    embedded_seq = [tf.tile(tf.reshape(_embedded_seq, (N, 1, 1, embed_dim)),
        [1, featmap_H, featmap_W, 1]) for _embedded_seq in tf.split(embedded_seq, T, 0)]

    # L2-normalize the features (except for spatial_batch)
    # and concatenate them along axis 3 (channel dimension)
    spatial_batch = tf.convert_to_tensor(generate_spatial_batch(N, featmap_H, featmap_W))

    #concat all features
    feat_alls = []
    for i in range(T):
        feat_alls.append(tf.concat([feat_langs[i], embedded_seq[i], feat_vis, spatial_batch], 3))
    
    feat_all = tf.stack(feat_alls, 3)
    feat_all = tf.transpose(feat_all, [0, 3, 1, 2, 4])
    print(feat_all.shape)

    #mlstm
    print(tf.get_variable_scope().reuse)
    mlstm_top = rnn.mlstm_layer('mlstm', feat_all, None, 500)[0]
    print(tf.get_variable_scope().reuse)

    #MLP classfier
    with tf.variable_scope('classifier'):
        mlp_l1 = conv('mlp_l1', mlstm_top, kernel_size=1, stride=1,
            output_dim=1)
        # if mlp_dropout: mlp_l1 = drop(mlp_l1, 0.5)
        # mlp_l2 = conv('mlp_l2', mlp_l1, kernel_size=1, stride=1, output_dim=1)

    return mlp_l1



    