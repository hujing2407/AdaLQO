import numpy as np
import json
import pandas as pd
import ast
import matplotlib.pyplot as plt


def load_data(dataset_name):
    df = pd.read_csv(f"dataset/{dataset_name}/execution_results.csv")
    df["latency_list"] = df["latency_list"].apply(ast.literal_eval)

    def load_plans(plan_path):
        with open(plan_path, "r") as f:
            return json.load(f)

    df["plans"] = df["plan_path"].apply(load_plans)

    df["plans"] = df["plans"].apply(
        lambda plans: [wrap_plan(p) for p in plans]
    )
    return df


def wrap_plan(plan):
    if "Plan" in plan:
        return plan
    return {"Plan": plan}


def prediction(model, data, save_path):
    results = []
    for _, row in data.iterrows():
        plans = row["plans"]
        latencies = row["latency_list"]
        preds = model.predict(plans)
        chosen_idx = np.argmin(preds)
        best_idx = np.argmin(latencies)
        result = {
            "query_id": row["query_id"],
            "bao_latency":
                latencies[chosen_idx],
            "best_latency":
                latencies[best_idx],
            "default_latency":
                latencies[0],
            "regret":
                latencies[chosen_idx]
                / latencies[best_idx],
        }

        results.append(result)
    res = pd.DataFrame(results)
    res.to_csv(save_path, index=False)
    return res


def plot_res(phase_name, pre_results, save_path):
    import matplotlib.pyplot as plt

    default_latency_list = pre_results["default_latency"]
    bao_latency_list = pre_results["bao_latency"]
    optimal_latency_list = pre_results["best_latency"]

    # 累积Latency
    default_cum = np.cumsum(default_latency_list)
    bao_cum = np.cumsum(bao_latency_list)
    optimal_cum = np.cumsum(optimal_latency_list)

    # Query编号
    x = np.arange(1, len(default_cum) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(x, default_cum,
             linewidth=2,
             label="PostgreSQL Default")
    plt.plot(x, bao_cum,
             linewidth=2,
             label="Bao")
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


def scatter_plot(phase_name, pre_results, mmd_score_p1,save_path):
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
