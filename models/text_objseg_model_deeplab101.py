from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

from tensorflow.python.ops.nn import dropout as drop
from util.cnn import conv_layer as conv
from util.cnn import conv_relu_layer as conv_relu
from util.cnn import deconv_layer as deconv
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu
from models import vgg_net, lstm_net, deeplab
from models.processing_tools import *
from deeplab_resnet import model as deeplab101

def text_objseg_full_conv(text_seq_batch, imcrop_batch, num_vocab, embed_dim,
    lstm_dim, mlp_hidden_dims, deeplab_dropout, mlp_dropout, is_training):

    # Language feature (LSTM hidden state)
    feat_lang = lstm_net.lstm_net(text_seq_batch, num_vocab, embed_dim, lstm_dim)[0]

    #deeplab101
    net = deeplab101.DeepLabResNetModel({'data': imcrop_batch}, is_training=is_training)
    feat_vis = net.layers['fc1_voc12']

    # # Local image feature
    # feat_vis = deeplab.deeplab_fc8_full_conv(imcrop_batch, 'deeplab',
    #     apply_dropout=deeplab_dropout)

    # Reshape and tile LSTM top
    featmap_H, featmap_W = feat_vis.get_shape().as_list()[1:3]
    N, D_text = feat_lang.get_shape().as_list()
    feat_lang = tf.tile(tf.reshape(feat_lang, [N, 1, 1, D_text]),
        [1, featmap_H, featmap_W, 1])

    # L2-normalize the features (except for spatial_batch)
    # and concatenate them along axis 3 (channel dimension)
    spatial_batch = tf.convert_to_tensor(generate_spatial_batch(N, featmap_H, featmap_W))
    feat_all = tf.concat(axis=3, values=[tf.nn.l2_normalize(feat_lang, 3),
                             tf.nn.l2_normalize(feat_vis, 3),
                             spatial_batch])

    # MLP Classifier over concatenate feature
    with tf.variable_scope('classifier'):
        mlp_l1 = conv_relu('mlp_l1', feat_all, kernel_size=1, stride=1,
            output_dim=mlp_hidden_dims)
        if mlp_dropout: mlp_l1 = drop(mlp_l1, 0.5)
        mlp_l2 = conv('mlp_l2', mlp_l1, kernel_size=1, stride=1, output_dim=1)

    return mlp_l2

def text_objseg_upsample8s(text_seq_batch, imcrop_batch, num_vocab, embed_dim,
    lstm_dim, mlp_hidden_dims, deeplab_dropout, mlp_dropout, is_training):

    mlp_l2 = text_objseg_full_conv(text_seq_batch, imcrop_batch, num_vocab,
        embed_dim, lstm_dim, mlp_hidden_dims, deeplab_dropout, mlp_dropout, is_training)

    # MLP Classifier over concatenate feature
    with tf.variable_scope('classifier'):
        # bilinear upsampling (no bias)
        #upsample32s = deconv('upsample32s', mlp_l2, kernel_size=64,
        #    stride=32, output_dim=1, bias_term=False)
        upsample8s = deconv('upsample8s', mlp_l2, kernel_size=16,
            stride=8, output_dim=1, bias_term=False)
        #upsample32s = deconv('upsample32s', upsample8s, kernel_size=8,
        #    stride=4, output_dim=1, bias_term=False)

    return upsample8s

