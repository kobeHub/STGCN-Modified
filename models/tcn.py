import tensorflow as tf


def gated_tcn_layer(x, Kt, c_in, c_out):
    """
    Gated TCN layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Kt: int, kernel size of temporal convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, time_step-Kt+1, n_route, c_out].
    """
    # frames, number of stations
    _, T, n, _ = x.get_shape().as_list()

    if c_in > c_out:
        w_input = tf.get_variable(
            "tcn_input", shape=[1, 1, c_in, c_out], dtype=tf.float32
        )
        tf.add_to_collection(name="weight_decay", value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding="SAME")
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x

    # For residual connection.
    residual = x_input[:, Kt - 1 : T, :, :]

    # TCN-a part
    filter_wt = tf.get_variable(
        name="filter_wt", shape=[Kt, 1, c_in, c_out], dtype=tf.float32
    )
    tf.add_to_collection(name="weight_decay", value=tf.nn.l2_loss(filter_wt))
    filter_bt = tf.get_variable(
        name="filter_bt", initializer=tf.zeros([c_out], dtype=tf.float32)
    )
    filter_conv = (
        tf.nn.conv2d(x, filter_wt, strides=[1, 1, 1, 1], padding="VALID") + filter_bt
    )

    # TCN-b part
    gated_wt = tf.get_variable(
        name="gated_wt", shape=[Kt, 1, c_in, c_out], dtype=tf.float32
    )
    tf.add_to_collection(name="weight_decay", value=tf.nn.l2_loss(gated_wt))
    gated_bt = tf.get_variable(
        name="gated_bt", initializer=tf.zeros([c_out], dtype=tf.float32)
    )
    gated_conv = (
        tf.nn.conv2d(x, gated_wt, strides=[1, 1, 1, 1], padding="VALID") + gated_bt
    )

    # tanh(filter) * sigm(gated)
    return tf.nn.tanh(filter_conv + residual) * tf.nn.sigmoid(gated_conv)
