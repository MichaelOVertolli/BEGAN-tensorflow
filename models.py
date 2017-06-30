import numpy as np
import tensorflow as tf
from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util
slim = tf.contrib.slim

def GeneratorCNN(z, hidden_num, output_num, repeat_num, data_format, reuse):
    with tf.variable_scope("G", reuse=reuse) as vs:
        num_output = int(np.prod([7, 7, hidden_num]))
        x = slim.fully_connected(z, num_output, activation_fn=None)
        x = reshape(x, 7, 7, hidden_num, data_format)
        
        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            #if idx < repeat_num - 1:
            x = upscale(x, 2, data_format)

        out = slim.conv2d(x, 1, 3, 1, activation_fn=None, data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables


def GeneratorRCNN(x, input_channel, z_num, repeat_num, hidden_num, data_format):
    with tf.variable_scope("G_r") as vs:
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)

        prev_channel_num = hidden_num
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            #if idx < repeat_num + 2:
            x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)

        x = tf.reshape(x, [-1, np.prod([7, 7, channel_num])], data_format)
        z = slim.fully_connected(x, z_num, activation_fn=None)

    variables = tf.contrib.framework.get_variables(vs)
    return z, variables


def GeneratorGrpCNN(z, hidden_num, output_num, repeat_num, reuse):
    w_init = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope("GG", reuse=reuse) as vs:
        num_output = int(np.prod([7, 7, hidden_num]))
        x = slim.fully_connected(z, num_output, activation_fn=None)
        x = reshape(x, 7, 7, hidden_num, 'NHWC')
        gconv_i, gconv_s, w_s = gconv2d_util('Z2', 'D4', hidden_num, hidden_num, 3)
        x = gconv2d(x, w_init(w_s), [1, 1, 1, 1], 'SAME', gconv_indices=gconv_i,
                    gconv_shape_info=gconv_s, data_format='NHWC')
        #x = tf.nn.relu(x)
        gconv_i, gconv_s, w_s = gconv2d_util('D4', 'D4', hidden_num, hidden_num, 3)
        #x = gconv2d(x, w_init(w_s), [1, 1, 1, 1], 'SAME', gconv_indices=gconv_i,
        #            gconv_shape_info=gconv_s, data_format='NHWC')
        x = upscale(x, 2, 'NHWC')
        #x = tf.nn.relu(x)
        for idx in range(repeat_num - 1):
            x = gconv2d(x, w_init(w_s), [1, 1, 1, 1], 'SAME', gconv_indices=gconv_i,
                        gconv_shape_info=gconv_s, data_format='NHWC')
            #x = tf.nn.relu(x)
            #x = gconv2d(x, w_init(w_s), [1, 1, 1, 1], 'SAME', gconv_indices=gconv_i,
            #            gconv_shape_info=gconv_s, data_format='NHWC')
            #if idx < repeat_num - 1:
            x = upscale(x, 2, 'NHWC')
            #x = tf.nn.relu(x)

        out = slim.conv2d(x, 1, 3, 1, activation_fn=None, data_format='NHWC')
        #gconv_i, gconv_s, w_s = gconv2d_util('D4', 'D4', hidden_num, 1, 3)
        #out = gconv2d(x, w_init(w_s), [1, 1, 1, 1], 'SAME', gconv_indices=gconv_i,
                      #gconv_shape_info=gconv_s, data_format='NHWC')

    out = nhwc_to_nchw(out)
    variables = tf.contrib.framework.get_variables(vs)
    return out, variables


def GeneratorGrpRCNN(x, input_channel, z_num, repeat_num, hidden_num):
    w_init = tf.contrib.layers.xavier_initializer()
    x = nchw_to_nhwc(x)
    with tf.variable_scope("GG_r") as vs:
        # Encoder
        gconv_i, gconv_s, w_s = gconv2d_util('Z2', 'D4', 1, hidden_num, 3)
        x = gconv2d(x, w_init(w_s), [1, 1, 1, 1], 'SAME', gconv_indices=gconv_i,
                    gconv_shape_info=gconv_s, data_format='NHWC')

        prev_channel_num = hidden_num
        gconv_s2 = None
        for idx in range(repeat_num):
            channel_num = prev_channel_num * (idx + 1)
            gconv_i, gconv_s, w_s = gconv2d_util('D4', 'D4', prev_channel_num, channel_num, 3)
            gconv_i2, gconv_s2, w_s2 = gconv2d_util('D4', 'D4', channel_num, channel_num, 3)
            x = gconv2d(x, w_init(w_s), [1, 1, 1, 1], 'SAME', gconv_indices=gconv_i,
                        gconv_shape_info=gconv_s, data_format='NHWC')
            #x = gconv2d(x, w_init(w_s2), [1, 1, 1, 1], 'SAME', gconv_indices=gconv_i2,
            #            gconv_shape_info=gconv_s2, data_format='NHWC')
            #if idx < repeat_num:
            x = gconv2d(x, w_init(w_s2), [1, 2, 2, 1], 'SAME', gconv_indices=gconv_i2,
                        gconv_shape_info=gconv_s2, data_format='NHWC')
            prev_channel_num = channel_num


        x = tf.reshape(x, [-1, np.prod(7*7*gconv_s2[0]*gconv_s2[1])], 'NHWC')
        z = slim.fully_connected(x, z_num, activation_fn=None)
    variables = tf.contrib.framework.get_variables(vs)
    return z, variables


def DiscriminatorGrpCNN(x, input_channel, z_num, repeat_num, hidden_num):
    w_init = tf.contrib.layers.xavier_initializer()
    x = nchw_to_nhwc(x)
    with tf.variable_scope("DG") as vs:
        # Encoder
        gconv_i, gconv_s, w_s = gconv2d_util('Z2', 'D4', 1, hidden_num, 3)
        x = gconv2d(x, w_init(w_s), [1, 1, 1, 1], 'SAME', gconv_indices=gconv_i,
                    gconv_shape_info=gconv_s, data_format='NHWC')

        prev_channel_num = hidden_num
        gconv_s2 = None
        for idx in range(repeat_num):
            channel_num = prev_channel_num * (idx + 1)
            gconv_i, gconv_s, w_s = gconv2d_util('D4', 'D4', prev_channel_num, channel_num, 3)
            gconv_i2, gconv_s2, w_s2 = gconv2d_util('D4', 'D4', channel_num, channel_num, 3)
            x = gconv2d(x, w_init(w_s), [1, 1, 1, 1], 'SAME', gconv_indices=gconv_i,
                        gconv_shape_info=gconv_s, data_format='NHWC')
            #x = gconv2d(x, w_init(w_s2), [1, 1, 1, 1], 'SAME', gconv_indices=gconv_i2,
            #            gconv_shape_info=gconv_s2, data_format='NHWC')
            #if idx < repeat_num:
            x = gconv2d(x, w_init(w_s2), [1, 2, 2, 1], 'SAME', gconv_indices=gconv_i2,
                        gconv_shape_info=gconv_s2, data_format='NHWC')
            prev_channel_num = channel_num


        x = tf.reshape(x, [-1, np.prod(7*7*gconv_s2[0]*gconv_s2[1])], 'NHWC')
        z = x = slim.fully_connected(x, z_num, activation_fn=None)

        # Decoder
        num_output = int(np.prod([7, 7, hidden_num]))
        x = slim.fully_connected(x, num_output, activation_fn=None)
        x = reshape(x, 7, 7, hidden_num, 'NHWC')
        gconv_i, gconv_s, w_s = gconv2d_util('Z2', 'D4', hidden_num, hidden_num, 3)
        x = gconv2d(x, w_init(w_s), [1, 1, 1, 1], 'SAME', gconv_indices=gconv_i,
                    gconv_shape_info=gconv_s, data_format='NHWC')
        #x = tf.nn.relu(x)
        gconv_i, gconv_s, w_s = gconv2d_util('D4', 'D4', hidden_num, hidden_num, 3)
        #x = gconv2d(x, w_init(w_s), [1, 1, 1, 1], 'SAME', gconv_indices=gconv_i,
        #            gconv_shape_info=gconv_s, data_format='NHWC')
        x = upscale(x, 2, 'NHWC')
        #x = tf.nn.relu(x)
        for idx in range(repeat_num - 1):
            x = gconv2d(x, w_init(w_s), [1, 1, 1, 1], 'SAME', gconv_indices=gconv_i,
                        gconv_shape_info=gconv_s, data_format='NHWC')
            #x = tf.nn.relu(x)
            #x = gconv2d(x, w_init(w_s), [1, 1, 1, 1], 'SAME', gconv_indices=gconv_i,
            #            gconv_shape_info=gconv_s, data_format='NHWC')
            #if idx < repeat_num - 1:
            x = upscale(x, 2, 'NHWC')
            #x = tf.nn.relu(x)

        out = slim.conv2d(x, 1, 3, 1, activation_fn=None, data_format='NHWC')
        #gconv_i, gconv_s, w_s = gconv2d_util('D4', 'D4', hidden_num, 1, 3)
        #out = gconv2d(x, w_init(w_s), [1, 1, 1, 1], 'SAME', gconv_indices=gconv_i,
                      #gconv_shape_info=gconv_s, data_format='NHWC')

    out = nhwc_to_nchw(out)
    variables = tf.contrib.framework.get_variables(vs)
    return out, z, variables
    

def DiscriminatorCNN(x, input_channel, z_num, repeat_num, hidden_num, data_format):
    with tf.variable_scope("D") as vs:
        # Encoder
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)

        prev_channel_num = hidden_num
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            if idx < repeat_num + 2:
                x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
                #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

        x = tf.reshape(x, [-1, np.prod([7, 7, channel_num])])
        z = x = slim.fully_connected(x, z_num, activation_fn=None)

        # Decoder
        num_output = int(np.prod([7, 7, hidden_num]))
        x = slim.fully_connected(x, num_output, activation_fn=None)
        x = reshape(x, 7, 7, hidden_num, data_format)
        
        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            #if idx < repeat_num - 1:
            x = upscale(x, 2, data_format)

        out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out, z, variables

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor, data_format):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    if data_format == 'NCHW':
        return [shape[0], shape[2], shape[3], shape[1]]
    elif data_format == 'NHWC':
        return shape

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def reshape(x, h, w, c, data_format):
    if data_format == 'NCHW':
        x = tf.reshape(x, [-1, c, h, w])
    else:
        x = tf.reshape(x, [-1, h, w, c])
    return x

def resize_nearest_neighbor(x, new_size, data_format):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)
        x = tf.image.resize_nearest_neighbor(x, new_size)
        x = nhwc_to_nchw(x)
    else:
        x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale, data_format):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, (h*scale, w*scale), data_format)
