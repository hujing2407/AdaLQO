import sys
import argparse
from typing import Any

from pandas import DataFrame, Series

from config import Config
from utils import *
from AdaLQO.shift_detector import mmd
from AdaLQO.utils import load_data, plot_res, prediction, scatter_plot, scatter_plot, split_dataset

sys.path.insert(0, 'bao_server')
import bao_server.model as model
from bao_server.model import BaoRegression

# TODO:// Remove the import from ShiftHandler if the retrain strategies no need
# sys.path.append('ShiftHandler')
# from replay_buffer import summarizer

logger = Config.setup_logging()
PHASE_NUM = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="dataset folder name", default='tpc-h_sf10_100-shifting')
    parser.add_argument("--size", help="size of the slide window", type=int, default=20)
    parser.add_argument("--batch", help="batch size (default: 10)", type=int, default=10)
    args = parser.parse_args()

    ds_name = args.dataset
    window_size = args.size
    batch_size = args.batch

    # 1. Load DataSet
    # 1.1 load all datasets - queries, plans, latecies
    df = load_data(ds_name)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    data = split_dataset(df_shuffled, batch_size)

    # data[i] - phase, data[i][j] - batch queries
    p0_df = df[df["phase"] == "phase_0"]
    p1_df = df[df["phase"] == "phase_1"]
    p2_df = df[df["phase"] == "phase_2"]

    # 2. Training BAO Model with the first batch of data
    X, y = train_ds_process(data[0][0])
    reg = train_model(X, y)

    # 4. Queries(Plans) Featurization
    p0_plans = []
    for plans in p0_df["plans"]:
        p0_plans.extend(plans)
    tree_p0 = reg._BaoRegression__tree_transform.transform(p0_plans)
    emb_p0 = reg._BaoRegression__net.get_fixed_features(tree_p0)

    for i in len(data):
        # 3. Prediction and Evaluation
        # pred1 = prediction(reg, p1_df, f"results/p1_pred.csv")
        # plot_res("Phase1", pred1, f"results/p1_pred.png")
        # logger.info("Phase1: Prediction csv and plot saved!")
        # pred2 = prediction(reg, p2_df, f"results/p2_pred.csv")
        # plot_res("Phase2", pred2, f"results/p2_pred.png")
        # logger.info("Phase2: Prediction csv and plot saved!")

        # 5. MMD scores between query/plans and phase0 plans
        mmd_score_p1 = []
        p1_plans = p1_df["plans"]
        for p in p1_plans:
            trees = reg._BaoRegression__tree_transform.transform(p)
            embedding = reg._BaoRegression__net.get_fixed_features(trees)
            mmd_score_p1.append(mmd(embedding, emb_p0))

        scatter_plot("Phase1", pred1, mmd_score_p1, "results/prototype/vs_regrets/mmd_vs_regret_p1.png")
        logger.info("Phase1: mmd vs regret plot saved!")

        mmd_score_p2 = []
        p2_plans = p2_df["plans"]
        for p in p2_plans:
            trees = reg._BaoRegression__tree_transform.transform(p)
            embedding = reg._BaoRegression__net.get_fixed_features(trees)
            mmd_score_p2.append(mmd(embedding, emb_p0))

        scatter_plot("Phase2", pred2, mmd_score_p2, "results/prototype/vs_regrets/mmd_vs_regret_p2.png")
        logger.info("Phase2: mmd vs regret plot saved!")


def train_ds_process(p0_df: Series | DataFrame | Any) -> tuple[list[Any], list[Any]]:
    X = []
    y = []
    for _, row in p0_df.iterrows():
        plans = row["plans"]
        latencies = row["latency_list"]
        for plan, latency in zip(plans, latencies):
            X.append(plan)
            y.append(latency)
    return X, y


def train_model(X: list[Any], y: list[Any]) -> BaoRegression:
    # TODO:// have_cache_data=False, since Buffer Feature missed. Fix it later
    reg = model.BaoRegression(have_cache_data=False, verbose=False)

    try:
        reg.fit_feature_extractor(X, y)
    except Exception as e:
        logger.error("ERROR:", e)

    logger.info("Bao model training......")
    reg.fit_model(X, y, seed=42, ada_size=False)
    torch.save(
        reg._BaoRegression__net.state_dict(),
        "results/model/bao_model.pt"
    )
    logger.info("Bao model trained & saved!")
    return reg


if __name__ == "__main__":
    main()
