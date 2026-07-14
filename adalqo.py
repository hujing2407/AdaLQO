import argparse
import time
import os

import numpy as np
import pandas as pd
import torch
import sys
import AdaLQO.utils as utils
from AdaLQO.utils import plot_res
from AdaLQO.utils import prediction
from AdaLQO.shift_detector import mmd, ws, ks_values_pca
import AdaLQO.replay_buffer as re_buf
from AdaLQO.utils import sle

sys.path.insert(0, 'bao_server')
import bao_server.model as bao_model
import copy
import random

from config import Config

logger = Config.setup_logging()


# PHASE_NUM = 3

def train_and_predict(data, buffer, buffer_size, num_tasks=6, concentration=1e-4,
                      tradeoff=0.5, seed=0):
    # Load training and validation data
    print("buffer: {}".format(buffer))
    print("seed: {}".format(seed))
    print("concentration: {}".format(concentration))
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    model = bao_model.BaoRegression(have_cache_data=False, verbose=False)
    # model.fit_feature_extractor(X, y)
    # model.fit_model(X, y, seed=42, ada_size=False)

    # Initial replay buffer
    latest_buffer = []
    if buffer == 'lwp':
        handler_buffer = re_buf.summarizer(buffer_limit=buffer_size, loss_ada=True,
                                           concentration=concentration,
                                           is_move=False)
    else:
        handler_buffer = re_buf.summarizer(buffer_limit=buffer_size, loss_ada=False,
                                           concentration=concentration,
                                           is_move=False)
    num_queries_seen_far = 0

    for task_id in range(len(data[0])):
        # first replay old queries
        replay_list = []
        replay_queries_tmp, _ = handler_buffer.get_all_samples()
        for (_, y_i, plan, _, _) in replay_queries_tmp:
            replay_list.append((plan, y_i))

        (x_train, y_train) = utils.get_training_data(data[0][task_id])

        current_bs = len(x_train)
        x_train_all = copy.deepcopy(x_train)
        y_train_all = copy.deepcopy(y_train)

        for (plan, y_i) in replay_list:
            x_train_all.append(plan)
            y_train_all.append(y_i)

        ada_size = False
        if buffer.lower() == 'lwp':
            ada_size = True

        if tradeoff != 0 and len(replay_list) > 0:
            model.fit_feature_extractor(x_train_all, y_train_all)
            idx_list, current_losses, replay_idx_list, replay_losses_list = (
                model.fit_model(x_train_all, y_train_all, tradeoff=tradeoff, ada_size=ada_size,
                                size_current_batch=current_bs, seed=seed))
        else:
            model.fit_feature_extractor(x_train_all, y_train_all)
            idx_list, current_losses, replay_idx_list, replay_losses_list = (
                model.fit_model(x_train_all, y_train_all, seed=seed, ada_size=ada_size))

        # update buffer size based on loss
        if buffer == 'lwp' and len(replay_losses_list):
            norm_losses_list = [None] * handler_buffer.buffer_size
            for (q_id, loss) in zip(replay_idx_list, replay_losses_list):
                norm_losses_list[q_id] = loss[0]
            handler_buffer.update_losses(norm_losses_list)

        # predict the queries in current task
        # TODO://

        # add new queries to the replay buffer
        # (x_train, y_train) = train_list[task_id]
        experience_features = model.get_before_features(x_train)
        for i in range(experience_features.shape[0]):
            q_feature = experience_features[i, :]
            if buffer == 'lwp':
                loss_id = idx_list.index(i)
                handler_buffer.process_a_query(q_feature, y_train[i], x_train[i], None, current_losses[loss_id][0])
            else:
                handler_buffer.process_a_query(q_feature, y_train[i], x_train[i])

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="dataset folder name", default='tpc-h_sf10_100-shifting')
    parser.add_argument("--size", help="size of the slide window", type=int, default=20)
    parser.add_argument("--batch", help="batch size (default: 10)", type=int, default=10)
    parser.add_argument("--buffersize", help="buffer size (default: 100)", type=int, default=100)
    args = parser.parse_args()

    ds_name = args.dataset
    window_size = args.size
    batch_size = args.batch

    # concentrations = [1e-2, 1e-1, 1.0, 10, 100]
    concentrations = [1.0]
    # random_seeds = list(range(10))
    random_seeds = [0]

    # 1. Load DataSet
    # 1.1 load all datasets - queries, plans, latecies
    df = pd.read_pickle("dataset/tpc-ds/data_df.pkl")
    BATCH_SIZE = 100
    data = utils.split_dataset(df, batch_size=BATCH_SIZE, random_state=None)

    # 2. Training BAO Model with the first batch of data
    for c in concentrations:
        train_and_predict(data, 'cbp', args.buffersize, concentration=c, seed=random_seeds[0])


if __name__ == "__main__":
    main()
