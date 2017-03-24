import tensorflow as tf
from tensorflow.python.ops.nn import dropout as drop
import util.rnn as rnn
from util.cnn import conv_layer as conv
from util.cnn import conv_relu_layer as conv_relu
from util.cnn import deconv_layer as deconv
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu
from models import vgg_net, lstm_net
from models.processing_tools import *

def recurrent_multimodal(text_seq_batch, imcrop_batch, num_vocab, embed_dim,
    lstm_dim, mlp_hidden_dims, vgg_dropout, mlp_dropout):

    _, feat_langs, embedded_seq = lstm(text_seq_batch, num_vocab, embed_dim, lstm_dim)

    feat_vis = vgg_net.vgg_fc8(imcrop_batch, 'vgg_local', apply_dropout=vgg_dropout)

    featmap_H, featmap_W = feat_vis.get_shape().as_list()[1:3]

    # Reshape and tile feat_langs, embedded_seq
    N, D_text = feat_lang[0].get_shape().as_list()
    feat_langs = [tf.tile(tf.reshape(feat_lang, [N, 1, 1, D_text]),
        [1, featmap_H, featmap_W, 1]) for feat_lang in feat_langs]

    embedded_seq = [tf.tile(tf.reshape(_embedded_seq, (N, 1, 1, embed_dim)),
        [1, featmap_H, featmap_W, 1]) for _embedded_seq in embedded_seq]

    feat_lang_all = tf.concat([feat_lang, embedded_seq], 3)

    # L2-normalize the features (except for spatial_batch)
    # and concatenate them along axis 3 (channel dimension)
    spatial_batch = tf.convert_to_tensor(generate_spatial_batch(N, featmap_H, featmap_W))

    #concat all features
    feat_all = tf.concat([feat_lang_all, feat_vis, spatial_batch], 3)

    #mlstm
    mlstm_top = rnn.mlstm('mlstm', feat_all, None, 500)

    #MLP classfier
    with tf.variable_scope('classfier'):
        mlp_l1 = conv_relu('mlp_l1', mlstm_top, kernel_size=1, stride=1,
            output_dim=mlp_hidden_dims)
        if mlp_dropout: mlp_l1 = drop(mlp_l1, 0.5)
        mlp_l2 = conv('mlp_l2', mlp_l1, kernel_size=1, stride=1, output_dim=1)

    return mlp_l2



    