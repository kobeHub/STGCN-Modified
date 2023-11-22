import os
from os.path import join as pjoin

import tensorflow as tf

from utils import math_graph
from data_loader import data_utils
from models.trainer import model_train
from models.tester import model_test

import argparse

# tf.disable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

parser = argparse.ArgumentParser()
# The number of stations. 228 or 1026
parser.add_argument(
    "--n_route",
    type=int,
    help="The number of observation stations (228 or 1026).",
    default=228,
)
# The input length of the sequence, number of frames. 12 frames
# contain 1h data, 9 frames contain 45min data.
parser.add_argument(
    "--n_his",
    type=int,
    help="The length of input frames, each frame is sampling in an interval of 5min (default=12).",
    default=12,
)
parser.add_argument(
    "--n_pred",
    type=int,
    help="The length of predicted frames, 9 frames contains 45min data (default=9).",
    default=9,
)
parser.add_argument(
    "--batch_size", type=int, help="Training batch size (default=50).", default=50
)
parser.add_argument(
    "--epoch", type=int, help="Global training epochs (default=50).", default=50
)
# The epoch number to save model
parser.add_argument(
    "--save", type=int, help="The gap of epochs to save model (default=5).", default=5
)
# Kernel size for spatial conv
parser.add_argument(
    "--ks",
    type=int,
    help="Kernel size for the spatial conv layer (default=3).",
    default=3,
)
# Kernal size for tempora conv
parser.add_argument(
    "--kt",
    type=int,
    help="Kernel size for the temporal conv layer (default=3).",
    default=3,
)
parser.add_argument(
    "--lr", type=float, help="Learning rate (default=1e-3).", default=1e-3
)
parser.add_argument(
    "--opt",
    type=str,
    help="Optimizer used to minimize the loss (default=RMSProp)",
    default="RMSProp",
)
# The file name of customed adjacency matrix
parser.add_argument(
    "--graph",
    type=str,
    help="The file name of adjacency matrix file",
    default="default",
)
parser.add_argument("--inf_mode", type=str, default="merge")
# The model structure
parser.add_argument(
    "--struct",
    type=str,
    help="The model structure used for experiment (defalut=tcn)",
    default="tcn",
)
parser.add_argument(
    "--train", type=bool, help="If it needs training (default=False)", default=False
)

args = parser.parse_args()
print(f"Training configs: {args}")

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt
# blocks: settings of channel size in st_conv_blocks / bottleneck design
blocks = [[1, 32, 64], [64, 32, 128]]

dataset_path = pjoin(os.getcwd(), "dataset")
# Load wighted adjacency matrix W
if args.graph == "default":
    W = math_graph.weight_matrix(pjoin(dataset_path, f"PeMSD7_W_{n}.csv"))
else:
    # load customized graph weight matrix
    W = math_graph.weight_matrix(pjoin(dataset_path, args.graph))

# Calculate graph kernel
L = math_graph.scaled_laplacian(W)
# Alternative approximation method: 1st approx - first_approx(W, n).
Lk = math_graph.cheb_poly_approx(L, Ks, n)
# Add the LK matrix to collection.
tf.add_to_collection(name="graph_kernel", value=tf.cast(tf.constant(Lk), tf.float32))

# Data Preprocessing
# The matrix shape: e.g. 12672 * 228; [len_seq, num_road]; there're
# 288 slots in each day, data from 44 days formed to 12672 seqs.
data_file = f"PeMSD7_V_{n}.csv"
# The number of dates for traininh, validation and test; 34 days'
# data is used for training.
n_train, n_val, n_test = 34, 5, 5
PeMS = data_utils.data_gen(
    pjoin(dataset_path, data_file), (n_train, n_val, n_test), n, n_his + n_pred
)
print(f">> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}")

if __name__ == "__main__":
    if args.train:
        model_train(PeMS, blocks, args)
    else:
        print("Only do inference!")
    model_test(PeMS, PeMS.get_len("test"), n_his, n_pred, args.inf_mode)
