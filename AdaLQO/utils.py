from typing import Any

import numpy as np
import json
from pathlib import Path
import pandas as pd
import torch
import ast
import re
import matplotlib.pyplot as plt
from pandas import DataFrame, Series


def load_data(dataset_name):
    df = pd.read_csv(
        f"dataset/{dataset_name}/execution_results.csv",
        converters={"latency_list": parse_latency_list}
    )

    def load_plans(plan_path):
        plan_path = Path(str(plan_path).replace("\\", "/"))
        with open(plan_path, "r") as f:
            return json.load(f)

    df["plans"] = df["plan_path"].apply(load_plans)

    df["plans"] = df["plans"].apply(
        lambda plans: [wrap_plan(p) for p in plans if p is not None]
    )

    # df["latency_list"] = df["latency_list"].apply(
    #     lambda xs: [x for x in xs if np.isfinite(x)] if isinstance(xs, list) else xs
    # )

    return df

def parse_latency_list(s):
    if pd.isna(s):
        return []

    s = str(s)

    # 把 inf / -inf / nan 替换成 None，避免 ast.literal_eval 报错
    s = re.sub(r"(?<![\w.])-?inf(?![\w.])", "None", s)
    s = re.sub(r"(?<![\w.])nan(?![\w.])", "None", s, flags=re.IGNORECASE)

    xs = ast.literal_eval(s)
    return xs
    # 读的时候直接跳过 None / inf / nan
    # return [
    #     x for x in xs
    #     if x is not None and np.isfinite(x)
    # ]

def wrap_plan(plan):
    if "Plan" in plan:
        return plan
    return {"Plan": plan}


def split_by_batch_size(df, batch_size=100):
    batches = []

    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batches.append(df.iloc[start:end])

    return batches


def split_dataset(df: DataFrame, batch_size: int, random_state:int):
    phase_batches = []
    if random_state is not None:
        rand = np.random.seed(random_state)
        df_to_split = df.sample(frac=1, random_state=rand).reset_index(drop=True)
    else:
        df_to_split = df

    if "phase" not in df.columns: df_to_split["phase"] = "phase_0"
    for phase_id, phase_df in df_to_split.groupby("phase"):
        phase_batches.append(split_by_batch_size(phase_df, batch_size))

    return phase_batches


def get_training_data(df: DataFrame):
    X = []
    y = []
    for _, row in df.iterrows():
        plans = row["plans"]
        latencies = row["latency_list"]
        for plan, latency in zip(plans, latencies):
            X.append(plan)
            y.append(latency)
    return X, y


def pred_many(model, queries):
    results = []
    for _, query in queries.iterrows():
        result = pred_single_query(model, query)
        results.append(result)
    res = pd.DataFrame(results)
    # res.to_csv(save_path, index=False)
    return res

def pred_single_query(model, query) -> dict:
    if hasattr(query, "_fields"):  # itertuples() 返回的 namedtuple
        query_id = query.query_id
        plans = query.plans
        latencies = query.latency_list
    else:  # pandas Series / dict
        query_id = query["query_id"]
        plans = query["plans"]
        latencies = query["latency_list"]

    preds = model.predict(plans)

    chosen_idx = np.argmin(preds)
    optimal_idx = np.argmin(latencies)

    return {
        "query_id": query_id,
        "chosen_idx": int(chosen_idx),
        "optimal_idx": int(optimal_idx),
        "bao_latency": latencies[chosen_idx],
        "optimal_latency": latencies[optimal_idx],
        "default_latency": latencies[0],
        "regret": latencies[chosen_idx]/ latencies[optimal_idx]
    }


def plot_res(phase_name, pre_results, save_path):
    import matplotlib.pyplot as plt

    default_latency_list = pre_results["default_latency"]
    base_bao_latency_list = pre_results["base_bao_latency"]
    bao_latency_list = pre_results["bao_latency"]
    optimal_latency_list = pre_results["best_latency"]

    # 累积Latency
    default_cum = np.cumsum(default_latency_list)
    base_bao_cum = np.cumsum(base_bao_latency_list)
    bao_cum = np.cumsum(bao_latency_list)
    optimal_cum = np.cumsum(optimal_latency_list)

    # Query编号
    x = np.arange(1, len(default_cum) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(x, default_cum,
             linewidth=2,
             label="PostgreSQL Default")
    plt.plot(x, base_bao_cum,
             linewidth=2,
             label="Bao w/o shifting detector")
    plt.plot(x, bao_cum,
             linewidth=2,
             label="Bao with shifting detector")
    plt.plot(x, optimal_cum,
             linewidth=2,
             label="Optimal")
    plt.xlabel("Number of Queries", fontsize=12)
    plt.ylabel("Cumulative Latency (ms)", fontsize=12)
    plt.title(f"Cumulative Query Latency for {phase_name}", fontsize=14)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{save_path}", dpi=300, bbox_inches="tight")
    plt.show()


def scatter_plot(phase_name, pre_results, mmd_score_p1, save_path):
    regrets = pre_results["regret"]
    x_np = [t.item() for t in mmd_score_p1]
    y_np = list(regrets)

    # 3. Create the scatter plot
    plt.scatter(x_np, y_np)
    plt.xlabel(f"MMD Score of {phase_name}")
    plt.ylabel("Regrets")
    plt.savefig(f"{save_path}", dpi=300, bbox_inches="tight")
    plt.show()


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

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def to_tensor(x, device):
    if torch.is_tensor(x):
        return x.detach().to(device=device, dtype=torch.float32)

    return torch.as_tensor(x, dtype=torch.float32, device=device)

def parse_ks_result(value):
    if pd.isna(value):
        return np.nan, np.nan
    value = str(value)
    stat_match = re.search(
        r"statistic\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        value
    )
    pvalue_match = re.search(
        r"pvalue\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        value
    )
    ks_stat = float(stat_match.group(1)) if stat_match else np.nan
    ks_pvalue = float(pvalue_match.group(1)) if pvalue_match else np.nan
    return ks_stat, ks_pvalue