import argparse
import time
import os
import numpy as np
import torch
import sys

from utils import *
sys.path.insert(0, 'bao_server')
import bao_server.model as model
import copy
import random

sys.path.append('ShiftHandler')
from replay_buffer import summarizer

# ## 1. Load DataSet
# ### 1.1 load all datasets - queries, plans, latecies

import json
import pandas as pd
import ast

df = pd.read_csv("dataset/tpc-h-shifting/execution_results.csv")
df["latency_list"] = df["latency_list"].apply(ast.literal_eval)

def load_plans(plan_path):
    with open(plan_path, "r") as f:
        return json.load(f)

df["plans"] = df["plan_path"].apply(load_plans)

# Varify the dataframe
row = df.iloc[0]
# print(row["query_id"])
# print(len(row["latency_list"]))
# print(len(row["plans"]))


p0_df = df[df["phase"] == "phase_0"]

X = []
y = []
for _, row in p0_df.iterrows():
    plans = row["plans"]
    latencies = row["latency_list"]
    for plan, latency in zip(plans, latencies):
        X.append(plan)
        y.append(latency)

X_fixed = [
    {"Plan": p}
    for p in X
]

sample = X_fixed[0]

print(sample.keys())

print(sample["Plan"].keys())

print(len(X))
print(len(y))
print(X_fixed[0])


# ## 2. Training BAO Model

reg = model.BaoRegression(have_cache_data=False, verbose=False)
try:
    reg.fit_feature_extractor(X_fixed, y)
except Exception as e:
    print("ERROR:", e)

# ## 2. MMD

# ### MMD simple sample

from torch.autograd import Variable

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n = int(source.size()[0])
    m = int(target.size()[0])

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1,-1)  # K_ss矩阵，Source<->Source
    XY = torch.div(XY, -n * m).sum(dim=1).view(1,-1) # K_st矩阵，Source<->Target

    YX = torch.div(YX, -m * n).sum(dim=1).view(1,-1) # K_ts矩阵,Target<->Source
    YY = torch.div(YY, m * m).sum(dim=1).view(1,-1)  # K_tt矩阵,Target<->Target

    loss = (XX + XY).sum() + (YX + YY).sum()
    return loss

# 样本数量可以不同，特征数目必须相同

# 100和90是样本数量，50是特征数目
data_1 = torch.tensor(np.random.normal(loc=0,scale=10,size=(100,50)))
data_2 = torch.tensor(np.random.normal(loc=10,scale=10,size=(90,50)))
print("MMD Loss:",mmd(data_1,data_2))

data_1 = torch.tensor(np.random.normal(loc=0,scale=10,size=(100,50)))
data_2 = torch.tensor(np.random.normal(loc=0,scale=9,size=(80,50)))

print("MMD Loss:",mmd(data_1,data_2))

# ## 3. Query Featurize




def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--imbalance', default=False, help="is imbalance?", action='store_true')
    # parser.add_argument("--buffersize", help="buffer size", type=int, default=50)
    # parser.add_argument("--epochs", help="number of epochs (default: 20)", type=int, default=30)
    # parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=1024)
    # args = parser.parse_args()


    ### start writing the results
    # write_results('all', args.buffersize, is_imb, result_per_seed_all)
    # write_results('rs', args.buffersize, is_imb, result_per_seed_rs)
    # write_results('latest', args.buffersize, is_imb, result_per_seed_latest)
    # write_results('cbp', args.buffersize, is_imb, result_per_seed_cbp)
    # write_results('lwp', args.buffersize, is_imb, result_per_seed_lwp)


if __name__ == "__main__":
    main()


