from models.layers import st_conv_block, output_layer
from os.path import join as pjoin
import tensorflow.compat.v1 as tf

import os
import pathlib


def build_model(inputs, n_his, Ks, Kt, blocks, keep_prob, is_modified):
    """
    Build the base model.
    :param inputs: placeholder.
    :param n_his: int, size of historical records for training.
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param blocks: list, channel configs of st_conv blocks.
    :param keep_prob: placeholder.
    :param is_modified: whether use the modified TCN.
    """
    # shape: [len_seq, n_frame, n_route, C]
    # Get the input frames.
    x = inputs[:, 0:n_his, :, :]

    # Ko>0: kernel size of temporal convolution in the output layer.
    Ko = n_his
    # ST-Block
    for i, channels in enumerate(blocks):
        x = st_conv_block(
            x,
            Ks,
            Kt,
            channels,
            i,
            keep_prob,
            is_modified=is_modified,
            act_func="GLU",
        )
        Ko -= 2 * (Kt - 1)

    # Output Layer
    if Ko > 1:
        y = output_layer(x, Ko, "output_layer")
    else:
        raise ValueError(
            f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".'
        )

    tf.add_to_collection(
        name="copy_loss",
        value=tf.nn.l2_loss(
            inputs[:, n_his - 1 : n_his, :, :] - inputs[:, n_his : n_his + 1, :, :]
        ),
    )
    train_loss = tf.nn.l2_loss(y - inputs[:, n_his : n_his + 1, :, :])
    single_pred = y[:, 0, :, :]
    tf.add_to_collection(name="y_pred", value=single_pred)
    return train_loss, single_pred


def model_save(saver, sess, global_steps, model_name, save_path):
    """
    Save the checkpoint of trained model.
    :param sess: tf.Session().
    :param global_steps: tensor, record the global step of training in epochs.
    :param model_name: str, the name of saved model.
    :param save_path: str, the path of saved model.
    :return:
    """
    if not os.path.exists(save_path):
        pathlib.Path(save_path).mkdir(parents=True)
    prefix_path = saver.save(
        sess, pjoin(save_path, model_name), global_step=global_steps
    )
    print(f"<< Saving model to {prefix_path} ...")
