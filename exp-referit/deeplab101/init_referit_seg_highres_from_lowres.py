from __future__ import absolute_import, division, print_function

import tensorflow as tf

from models import text_objseg_model as segmodel
from models import text_objseg_model_deeplab101 as segmodel_deeplab
from six.moves import cPickle

################################################################################
# Parameters
################################################################################

low_model = './exp-referit/tfmodel/deeplab101/referit_fc8_seg_lowres_lstm_iter_30000.tfmodel'
high_model = './exp-referit/tfmodel/deeplab101/referit_fc8_seg_highres_init.tfmodel'

# Model Params
T = 20
N = 1

num_vocab = 8803
embed_dim = 1000
lstm_dim = 1000
mlp_hidden_dims = 500

text_seq_batch = tf.placeholder(tf.int32, [T, N])  # one batch per sentence
imcrop_batch = tf.placeholder(tf.float32, [N, 512, 512, 3])

_ = segmodel_deeplab.text_objseg_upsample8s(text_seq_batch, imcrop_batch,
        num_vocab, embed_dim, lstm_dim, mlp_hidden_dims,
        deeplab_dropout=False, mlp_dropout=False, is_training=False)

load_var = {var.op.name: var for var in tf.global_variables()
        if not var.op.name.startswith('classifier/upsample')}
snapshot_loader = tf.train.Saver(load_var)
with tf.variable_scope('classifier', reuse=True):
    upsample8s_w = tf.get_variable('upsample8s/weights')
    init_upsample8s_w = tf.assign(upsample8s_w, segmodel.generate_bilinear_filter(8))
    #upsample32s_w = tf.get_variable('upsample32s/weights')
    #init_upsample32s_w = tf.assign(upsample32s_w, segmodel.generate_bilinear_filter(4))

snapshot_saver = tf.train.Saver()
with tf.Session() as sess:
    snapshot_loader.restore(sess, low_model)
    sess.run(init_upsample8s_w)
    #sess.run(init_upsample32s_w)
    snapshot_saver.save(sess, high_model)
