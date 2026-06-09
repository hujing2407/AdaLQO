import numpy as np
import torch
import json
from torch.autograd import Variable
import time
import random
import csv
import copy
from random import shuffle


def get_query_list(x, y, num_train_per_task, num_test_per_task, is_imb=True, num_burnin=4, num_task=3, seed=0):
    file_name = "./dataset/train"
    num_queries_per_file = 100000


    return x_filtered, y_filtered, train_list, test_list


def sle(preds, targets):
    log_preds = np.log(preds)
    log_targets = np.log(targets)

    min_val = np.min(log_targets)
    max_max = np.max(log_targets)

    scaled_preds = [(pred - min_val) / (max_max - min_val) for pred in log_preds]
    scaled_targets = [(pred - min_val) / (max_max - min_val) for pred in log_targets]

    sle_res = []
    for i in range(len(targets)):
        err = np.square(scaled_preds[i] - scaled_targets[i])
        sle_res.append(err)
    return sle_res


def get_results_by_tile(all_performs, tile='mean'):
    ### get the avg errors of given tile at all shifting points

    avg_error_per_point = []
    for res_per_point in all_performs:
        avg_error_per_task = []
        for res_per_task in res_per_point:
            desired_error = res_per_task[tile]
            avg_error_per_task.append(desired_error)
        avg_error_per_point.append(np.mean(avg_error_per_task))
    avg_error = np.mean(avg_error_per_point)

    return avg_error


def write_results(buffer, buffer_size, is_imbalance, result_per_seed):
    avg_mean, avg_median, avg_max = 0, 0, 0
    for seed in result_per_seed:
        avg_mean += result_per_seed[seed]['mean']
        avg_median += result_per_seed[seed]['median']
        avg_max += result_per_seed[seed]['max']

    avg_mean = avg_mean / len(result_per_seed)
    avg_median = avg_median / len(result_per_seed)
    avg_max = avg_max / len(result_per_seed)

    overall_performance = {'mean': avg_mean, 'median': avg_median, 'max': avg_max}

    result_json = {"buffer": buffer, "size": buffer_size, "overall_performance": overall_performance,
                   "result_per_seed": result_per_seed}

    file_name = "./cost_result_imb_{}.txt".format(str(is_imbalance))
    with open(file_name, "a") as f:
        f.write("{}\n".format(json.dumps(result_json)))
        f.flush()
