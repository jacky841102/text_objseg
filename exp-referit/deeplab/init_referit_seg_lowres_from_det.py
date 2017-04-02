from __future__ import absolute_import, division, print_function

import tensorflow as tf

from models import text_objseg_model as segmodel
from models import text_objseg_model_deeplab as segmodel_deeplab
from six.moves import cPickle

################################################################################
# Parameters
################################################################################

# det_model = './exp-referit/tfmodel/referit_fc8_det_iter_25000.tfmodel'
fcn_seg_model = './exp-referit/tfmodel/referit_fc8_seg_lowres_init.tfmodel'
seg_model = './exp-referit/tfmodel/deeplab/referit_fc8_seg_lowres_init.tfmodel'
convnet_params = './models/convert_caffemodel/params/deeplab_weights.ckpt'

# Model Params
T = 20
N = 1

num_vocab = 8803
embed_dim = 1000
lstm_dim = 1000
mlp_hidden_dims = 500

################################################################################
# detection network
################################################################################

# Inputs
text_seq_batch = tf.placeholder(tf.int32, [T, N])  # one batch per sentence
imcrop_batch = tf.placeholder(tf.float32, [N, 512, 512, 3])

# Language feature (LSTM hidden state)
_ = segmodel.text_objseg_full_conv(text_seq_batch, imcrop_batch,
    num_vocab, embed_dim, lstm_dim, mlp_hidden_dims,
    vgg_dropout=False, mlp_dropout=False)

# Load pretrained detection model and fetch weights
snapshot_loader = tf.train.Saver()
with tf.Session() as sess:
    snapshot_loader.restore(sess, fcn_seg_model)
    variable_dict = {var.name:var.eval(session=sess) for var in tf.global_variables()}

################################################################################
# low resolution segmentation network
################################################################################

# Clear the graph
tf.reset_default_graph()

# Inputs
text_seq_batch = tf.placeholder(tf.int32, [T, N])  # one batch per sentence
imcrop_batch = tf.placeholder(tf.float32, [N, 512, 512, 3])

_ = segmodel_deeplab.text_objseg_full_conv(text_seq_batch, imcrop_batch,
    num_vocab, embed_dim, lstm_dim, mlp_hidden_dims,
    deeplab_dropout=False, mlp_dropout=False)
    
# deeplab layers
convnet_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
                  'conv3_1', 'conv3_2', 'conv3_3',
                  'conv4_1', 'conv4_2', 'conv4_3',
                  'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7']#, 'fc8_voc12']

# Assign outputs
assign_ops = []
for var in tf.global_variables():
    if var.name.startswith('deeplab'):
        continue
    assign_ops.append(tf.assign(var, variable_dict[var.name].reshape(var.get_shape().as_list())))

with open(convnet_params, 'r') as f:
    processed_params = cPickle.load(f)

with tf.variable_scope('deeplab', reuse=True):
    for l_name in convnet_layers:
        assign_W = tf.assign(tf.get_variable(l_name + '/weights'), processed_params[l_name + '/w'])
        assign_B = tf.assign(tf.get_variable(l_name + '/biases'), processed_params[l_name + '/b'])
        assign_ops += [assign_W, assign_B]

# Save segmentation model initialization
snapshot_saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.group(*assign_ops))
    snapshot_saver.save(sess, seg_model)
