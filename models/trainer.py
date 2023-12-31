from data_loader.data_utils import gen_batch
from models.tester import model_inference
from models.base_model import build_model, model_save
from os.path import join as pjoin

import tensorflow as tf
import numpy as np
import time
import os
import csv


def model_train(
    inputs,
    blocks,
    args,
    sum_path=pjoin(pjoin(os.getcwd(), "output"), "tensorboard"),
    model_save_dir=pjoin(pjoin(os.getcwd(), "output"), "models"),
):
    """
    Train the base model.
    :param inputs: instance of class Dataset, data source for training.
    :param blocks: list, channel configs of st_conv blocks.
    :param args: instance of class argparse, args for training.
    """
    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    Ks, Kt = args.ks, args.kt
    batch_size, epoch, inf_mode, opt = (
        args.batch_size,
        args.epoch,
        args.inf_mode,
        args.opt,
    )

    is_modified = args.struct == "tcn"
    if is_modified:
        print("Train model with new temporal layer TCN")
        sum_path = pjoin(sum_path, "modified")
        model_save_dir = pjoin(model_save_dir, "modified")
    else:
        print("Train origin STGCN model")
        sum_path = pjoin(sum_path, "original")
        model_save_dir = pjoin(model_save_dir, "original")

    # Placeholder for model training
    x = tf.placeholder(tf.float32, [None, n_his + 1, n, 1], name="data_input")
    # Prob of keep elements on dropout layer.
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    # Define model loss
    train_loss, pred = build_model(x, n_his, Ks, Kt, blocks, keep_prob, is_modified)
    tf.summary.scalar("train_loss", train_loss)
    copy_loss = tf.add_n(tf.get_collection("copy_loss"))
    tf.summary.scalar("copy_loss", copy_loss)

    # Learning rate settings
    global_steps = tf.Variable(0, trainable=False)
    len_train = inputs.get_len("train")
    if len_train % batch_size == 0:
        epoch_step = len_train / batch_size
    else:
        epoch_step = int(len_train / batch_size) + 1
    # Learning rate decay with rate 0.7 every 5 epochs.
    lr = tf.train.exponential_decay(
        args.lr,
        global_steps,
        decay_steps=5 * epoch_step,
        decay_rate=0.7,
        staircase=True,
    )
    tf.summary.scalar("learning_rate", lr)
    step_op = tf.assign_add(global_steps, 1)
    # A context manager to ensure the step advance will
    # be executed before the minimize operation is one session
    with tf.control_dependencies([step_op]):
        if opt == "RMSProp":
            train_op = tf.train.RMSPropOptimizer(lr).minimize(train_loss)
        elif opt == "ADAM":
            train_op = tf.train.AdamOptimizer(lr).minimize(train_loss)
        else:
            raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')

    merged = tf.summary.merge_all()

    # Restore model
    saver = tf.train.Saver(max_to_keep=3)
    saved_name = "STGCN-TCN" if is_modified else "STGCN"

    # Infernece metrics
    MAPE_metrics = [
        tf.placeholder(tf.float32, shape=(2,), name=f"Inference_MAPE_seq_{i*3}")
        for i in range(1, 4)
    ]
    MAPE_summary = [
        tf.summary.scalar(f"Inference_MAPE_seq_{i*3}", MAPE_metrics[i - 1][0])
        for i in range(1, 4)
    ]
    MAE_metrics = [
        tf.placeholder(tf.float32, shape=(2,), name=f"Inference_MAE_seq_{i*3}")
        for i in range(1, 4)
    ]
    MAE_summary = [
        tf.summary.scalar(f"Inference_MAE_seq_{i*3}", MAE_metrics[i - 1][0])
        for i in range(1, 4)
    ]
    RMSE_metrics = [
        tf.placeholder(tf.float32, shape=(2,), name=f"Inference_RMPE_seq_{i*3}")
        for i in range(1, 4)
    ]
    RMSE_summary = [
        tf.summary.scalar(f"Inference_RMSE_seq_{i*3}", RMSE_metrics[i - 1][0])
        for i in range(1, 4)
    ]
    # MAE_metrics = tf.placeholder(tf.float32, shape=(2,), name="Inference_MAE")
    # MAE_summary = tf.summary.scalar("Inference_MAE", MAE_metrics[0])
    # RMSE_metrics = tf.placeholder(tf.float32, shape=(2,), name="Inference_RMSE")
    # RMSE_summary = tf.summary.scalar("Inference_RMSE", RMSE_metrics[0])
    csv_file_path = pjoin(
        pjoin(".", "output"), "modified.csv" if is_modified else "original.csv"
    )
    csv_file = open(csv_file_path, "w")
    csv_wr = csv.writer(csv_file)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(pjoin(sum_path, "train"), sess.graph)
        latest_ckp = tf.train.latest_checkpoint(model_save_dir)
        if latest_ckp:
            saver.restore(sess, latest_ckp)
            global_step_value = tf.train.global_step(sess, global_steps)
            print(
                f"Using saved model to train: {latest_ckp},"
                f" global steps: {global_step_value}"
            )
        else:
            sess.run(tf.global_variables_initializer())
            print("Training from scratch")

        if inf_mode == "sep":
            # for inference mode 'sep', the type of step index is int.
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
            min_val = min_va_val = np.array([4e1, 1e5, 1e5])
        elif inf_mode == "merge":
            # for inference mode 'merge', the type of step index is np.ndarray.
            step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
            min_val = min_va_val = np.array([4e1, 1e5, 1e5] * len(step_idx))
        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')

        for i in range(epoch):
            start_time = time.time()
            for j, x_batch in enumerate(
                gen_batch(
                    inputs.get_data("train"),
                    batch_size,
                    dynamic_batch=True,
                    shuffle=True,
                )
            ):
                summary, _ = sess.run(
                    [merged, train_op],
                    feed_dict={x: x_batch[:, 0 : n_his + 1, :, :], keep_prob: 1.0},
                )
                writer.add_summary(summary, i * epoch_step + j)
                if j % 50 == 0:
                    loss_value = sess.run(
                        [train_loss, copy_loss],
                        feed_dict={x: x_batch[:, 0 : n_his + 1, :, :], keep_prob: 1.0},
                    )
                    print(
                        f"Epoch {i:2d}, Step {j:3d}: [Train Loss: {loss_value[0]:.3f}, Copy Loss: {loss_value[1]:.3f}]"
                    )
            print(f"Epoch {i:2d} Training Time {time.time() - start_time:.3f}s")

            start_time = time.time()
            min_va_val, min_val = model_inference(
                sess,
                pred,
                inputs,
                batch_size,
                n_his,
                n_pred,
                step_idx,
                min_va_val,
                min_val,
            )

            csv_row = [epoch_step * i]
            for k, ix in enumerate(tmp_idx):
                va, te = min_va_val[ix - 2 : ix + 1], min_val[ix - 2 : ix + 1]
                summary1, summary2, summary3 = sess.run(
                    [MAPE_summary[k], MAE_summary[k], RMSE_summary[k]],
                    feed_dict={
                        MAPE_metrics[k]: [va[0], te[0]],
                        MAE_metrics[k]: [va[1], te[1]],
                        RMSE_metrics[k]: [va[2], te[2]],
                    },
                )
                writer.add_summary(summary1, i * epoch_step)
                writer.add_summary(summary2, i * epoch_step)
                writer.add_summary(summary3, i * epoch_step)
                print(
                    f"Time Step {ix + 1}: "
                    f"MAPE {va[0]:7.3%}, {te[0]:7.3%}; "
                    f"MAE  {va[1]:4.3f}, {te[1]:4.3f}; "
                    f"RMSE {va[2]:6.3f}, {te[2]:6.3f}."
                )
                csv_row += [va[0], te[0], va[1], te[1], va[2], te[2]]
            csv_wr.writerow(csv_row)

            print(f"Epoch {i:2d} Inference Time {time.time() - start_time:.3f}s")

            if (i + 1) % args.save == 0:
                model_save(saver, sess, global_steps, saved_name, model_save_dir)
        writer.close()
        csv_file.close()
    print("Training model finished!")
