from __future__ import absolute_import, division, print_function

import tensorflow as tf

from models import text_objseg_model_deeplab101 as segmodel101
from models import text_objseg_model as segmodel
from deeplab_resnet import model as deeplab101
from six.moves import cPickle

################################################################################
# Parameters
################################################################################

fcn_seg_model = './exp-referit/tfmodel/referit_fc8_seg_lowres_iter30000.tfmodel'
seg_model = './exp-referit/tfmodel/deeplab101/referit_fc8_seg_lowres_init.ckpt'
pretrained_params = './models/convert_caffemodel/params/deeplab_resnet_init.ckpt'

# Model Params
T = 20
N = 1

num_vocab = 8803
embed_dim = 1000
lstm_dim = 1000
mlp_hidden_dims = 500

################################################################################
# low resolution segmentation network
################################################################################

# Inputs
text_seq_batch = tf.placeholder(tf.int32, [T, N])  # one batch per sentence
imcrop_batch = tf.placeholder(tf.float32, [N, 320, 320, 3])


print('Loading deeplab101 weights')

#output
net = deeplab101.DeepLabResNetModel({'data': imcrop_batch}, is_training=True)

#pretrained fc1_voc12 is 21 channel, we need 1000 channel
restored_var = [var for var in tf.global_variables() if 'fc1_voc12' not in var.name]


# Load pretrained model
snapshot_loader = tf.train.Saver(var_list=restored_var)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    snapshot_loader.restore(sess, pretrained_params)
    variable_dict = {var.name:var.eval(session=sess) for var in tf.global_variables()}

print("done")

# Clear the graph
tf.reset_default_graph()

print('Loading fcn segmodel weights')

_ = segmodel.text_objseg_full_conv(text_seq_batch, imcrop_batch,
    num_vocab, embed_dim, lstm_dim, mlp_hidden_dims,
    vgg_dropout=False, mlp_dropout=False)

with tf.Session() as sess:
    for var in tf.global_variables():
        if var.name.startswith('vgg'):
            continue
        variable_dict[var.name] = var.eval(session=sess) 

print('done')

tf.reset_default_graph()

print('Saving deeplab101 segmodel weights')

_ = segmodel101.text.text_objseg_full_conv(text_seq_batch, imcrop_batch,
    num_vocab, embed_dim, lstm_dim, mlp_hidden_dims,
    vgg_dropout=False, mlp_dropout=False, is_training=True)

# Assign outputs
assign_ops = []
for var in tf.global_variables():
    assign_ops.append(tf.assign(var, variable_dict[var.name].reshape(var.get_shape().as_list())))

# Save segmentation model initialization
snapshot_saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.group(*assign_ops))
    snapshot_saver.save(sess, seg_model)

print('done')
