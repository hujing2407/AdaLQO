import argparse
import time
import os
import copy
import joblib
import random
import numbers

from config import Config
import numpy as np
import pandas as pd
import torch
import sys
import AdaLQO.utils as utils
from AdaLQO.utils import plot_res, pred_single_query
from AdaLQO.shift_detector import mmd, ws, ks_values_pca
import AdaLQO.replay_buffer as re_buf
import AdaLQO.shift_detector as det
from AdaLQO.utils import sle
import AdaLQO.dist_scores as dist_scores

sys.path.insert(0, 'bao_server')
import bao_server.model as bao_model

from config import Config
logger = Config.setup_logging()

INIT_TRAIN_NUM  = 100
TRIGGER_NUM = 100
BUFFER = 'rs'
BUF_SIZEE = 2000
concentration = 0.1

def init_bao_model(train_data):
    X, y = utils.get_training_data(train_data)
    return X,y, train_bao_model(X, y)

def train_bao_model(X, y):
    model = bao_model.BaoRegression(have_cache_data=False, verbose=False)
    model.fit_feature_extractor(X, y)
    idx_list, current_losses, replay_idx_list, replay_losses_list = model.fit_model(X, y, seed=42, ada_size=False)

    return model

def retrain(latest_buffer):
    x_train_all = []
    y_train_all = []
    for (plan, y_i) in latest_buffer:
        x_train_all.append(plan)
        y_train_all.append(y_i)

    print(f"training data with {len(x_train_all)} samples")
    new_model = train_bao_model(x_train_all, y_train_all)
    base_embedding = det.embedding_plans(new_model, x_train_all)
    return new_model, base_embedding

    return model
def main():
    # 1. Load DataSet
    # 1.1 Load all datasets - queries, plans, latecies
    df_all = pd.read_pickle("dataset/tpc-ds/data_df.pkl")
    # df_all = df_all[df_all['latency_list'].map(len) == 13]
    # print(f"filtered_df.shape:{df_all.shape}")

    # 1.2 Get the testing data
    cl_data_idx_df = pd.read_pickle("dataset/tpc-ds/data_X_test_t1.1_test0.4.pkl")
    cl_data_idx = cl_data_idx_df['query_id']
    cl_df = df_all[df_all["query_id"].isin(cl_data_idx)].copy()
    cl_df = cl_df.drop_duplicates(subset=['query_id'])

    # temp for debugging code, TODO:// remove later
    cl_df = cl_df.reset_index(drop=True)

    # 1.3 Load regret classifier
    package = joblib.load("models/classifier_best_model.pkl")
    regret_classifier = package["model"]
    regret_classifier_features = package["feature_names"]
    regret_classifier_features.pop(0)

    # 2. Train BAO Model with the first batch of data
    train_data = df_all[df_all['latency_list'].map(len) == 13][:INIT_TRAIN_NUM]
    X_init_train,y_init_train, bao_ori = init_bao_model(train_data)
    base_embedding = det.embedding_plans(bao_ori, X_init_train)

    # 3. Predict result with bao_init model
    # ori_res_list = []
    # for index, row in cl_df.iterrows():
    #     res = pred_single_query(bao_ori, row)
    #     ori_res_list.append(res)

    # 4. Create buffer
    latest_buffer = []
    # handler_buffer = re_buf.summarizer(buffer_limit=buffer_size, loss_ada=False,
    #                                        concentration=concentration,
    #                                        is_move=False)

    num_queries_seen_far = 0
    # Add init training queries to the replay buffer
    for i in range(len(X_init_train)):
        if len(latest_buffer) < BUF_SIZEE:
            latest_buffer.append((X_init_train[i], y_init_train[i]))
        else:
            random_i = random.uniform(0, 1)
            if random_i < float(len(latest_buffer)) / (num_queries_seen_far + 1):
                latest_buffer.pop(random.randint(0, len(X_init_train)-1))
                latest_buffer.append((X_init_train[i], y_init_train[i]))
        num_queries_seen_far += 1

    # 5. Continue learning
    from sklearn.preprocessing import StandardScaler

    counter = 0
    cl_res_list = []
    x_train_all = []
    y_train_all = []
    cl_model = copy.deepcopy(bao_ori)
    for index, row in cl_df.iterrows():
        X_new = row["plans"]
        y_new = row["latency_list"]

        if counter >= TRIGGER_NUM:
            print(f"Retraining when in index :{index}")
            cl_model, base_embedding = retrain(latest_buffer)
            counter = 0

        cur_embedding = det.embedding_single_query(cl_model, X_new)
        base_embedding = utils.to_numpy(base_embedding)
        scaler = StandardScaler()
        scaler.fit(base_embedding)
        base_scaled = scaler.transform(base_embedding)
        cur_scaled = scaler.transform(cur_embedding)

        scores = dist_scores.record_regret(cl_model, cur_scaled, base_scaled)
        ks_stat, ks_log_pvalue = utils.parse_ks_result(scores["ks_result"])
        scores["ks_stat"] = ks_stat
        scores["ks_log_pvalue"] = -np.log10(ks_log_pvalue)

        selected_scores = { k: scores[k] for k in regret_classifier_features}
        # selected_scores = {
        #     k: v
        #     for k, v in selected_scores.items()
        #     if isinstance(v, numbers.Number)
        # }
        selected_scores = pd.DataFrame([selected_scores])


        is_degrad = regret_classifier.predict(selected_scores)

        if not is_degrad:
            for i in range(len(X_new)):
                if len(latest_buffer) < BUF_SIZEE:
                    latest_buffer.append((X_new[i], y_new[i]))
                else:
                    random_i = random.uniform(0, 1)
                    if random_i < float(len(latest_buffer)) / (num_queries_seen_far + 1):
                        latest_buffer.pop(random.randint(0, len(X_init_train)-1))
                        latest_buffer.append((X_new[i], y_new[i]))
                num_queries_seen_far += 1
            counter += 1
        else:
            latest_buffer.append((X_new[i], y_new[i]))
            if len(latest_buffer) > BUF_SIZEE:
                latest_buffer = latest_buffer[len(latest_buffer) - BUF_SIZEE:]
            counter += 1

        res = pred_single_query(cl_model, row)
        cl_res_list.append(res)

    # ori_res = pd.DataFrame(ori_res_list)
    cl_res = pd.DataFrame(cl_res_list)
    # ori_res.to_csv(f"ori_res.csv")
    cl_res.to_csv(f"cl_res_counter{TRIGGER_NUM}_buf{BUF_SIZEE}_{BUFFER}.csv")


if __name__ == "__main__":
    main()
