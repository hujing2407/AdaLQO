from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from dataclasses import asdict

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist, pdist
from scipy.special import rel_entr
from scipy.stats import ks_2samp, wasserstein_distance
import AdaLQO.utils as utils
import AdaLQO.shift_detector as det

Aggregation = Literal["mean", "median", "sum", "max", "none"]
BinStrategy = Literal["quantile", "uniform"]


@dataclass
class DistributionDistanceResult:
    """
    保存两组样本之间的多个分布距离。

    featurewise_wasserstein:
        每个特征的 Wasserstein distance，shape=(n_features,)

    featurewise_ks:
        每个特征的 KS statistic，shape=(n_features,)

    featurewise_js:
        每个特征的 JS divergence，shape=(n_features,)

    featurewise_kl_x_to_y:
        每个特征的 KL(X || Y)，shape=(n_features,)

    featurewise_kl_y_to_x:
        每个特征的 KL(Y || X)，shape=(n_features,)
    """

    mmd: float
    energy_distance: float

    wasserstein: float
    ks: float
    js_divergence: float
    js_distance: float
    kl_x_to_y: float
    kl_y_to_x: float
    symmetric_kl: float

    featurewise_wasserstein: NDArray[np.float64]
    featurewise_ks: NDArray[np.float64]
    featurewise_js: NDArray[np.float64]
    featurewise_kl_x_to_y: NDArray[np.float64]
    featurewise_kl_y_to_x: NDArray[np.float64]

    def to_dict(self) -> dict[str, float | NDArray[np.float64]]:
        return {
            "mmd": self.mmd,
            "energy_distance": self.energy_distance,
            "wasserstein": self.wasserstein,
            "ks": self.ks,
            "js_divergence": self.js_divergence,
            "js_distance": self.js_distance,
            "kl_x_to_y": self.kl_x_to_y,
            "kl_y_to_x": self.kl_y_to_x,
            "symmetric_kl": self.symmetric_kl,
            "featurewise_wasserstein": self.featurewise_wasserstein,
            "featurewise_ks": self.featurewise_ks,
            "featurewise_js": self.featurewise_js,
            "featurewise_kl_x_to_y": self.featurewise_kl_x_to_y,
            "featurewise_kl_y_to_x": self.featurewise_kl_y_to_x,
        }


def _validate_samples(
    x: ArrayLike,
    y: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    将输入转换为二维浮点数组，并检查数据合法性。

    输入：
        一维数组会转换成 shape=(n_samples, 1)
        二维数组保持 shape=(n_samples, n_features)
    """
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)

    if x_array.ndim == 1:
        x_array = x_array[:, None]

    if y_array.ndim == 1:
        y_array = y_array[:, None]

    if x_array.ndim != 2 or y_array.ndim != 2:
        raise ValueError(
            "x 和 y 必须是一维或二维数组。"
            f"当前 shape 分别为 {x_array.shape} 和 {y_array.shape}。"
        )

    if x_array.shape[0] == 0 or y_array.shape[0] == 0:
        raise ValueError("x 和 y 不能为空。")

    if x_array.shape[1] != y_array.shape[1]:
        raise ValueError(
            "x 和 y 的特征维度必须相同。"
            f"当前分别为 {x_array.shape[1]} 和 {y_array.shape[1]}。"
        )

    if not np.all(np.isfinite(x_array)):
        raise ValueError("x 中包含 NaN 或无穷值。")

    if not np.all(np.isfinite(y_array)):
        raise ValueError("y 中包含 NaN 或无穷值。")

    return x_array, y_array


def _aggregate(
    values: ArrayLike,
    method: Aggregation = "mean",
) -> float | NDArray[np.float64]:
    """
    汇总逐特征距离。
    """
    array = np.asarray(values, dtype=np.float64)

    if method == "none":
        return array

    if method == "mean":
        return float(np.mean(array))

    if method == "median":
        return float(np.median(array))

    if method == "sum":
        return float(np.sum(array))

    if method == "max":
        return float(np.max(array))

    raise ValueError("aggregation 必须是 " "'mean'、'median'、'sum'、'max' 或 'none'。")


def estimate_rbf_bandwidth(
    x: ArrayLike,
    y: ArrayLike,
    max_samples: int = 1000,
    random_state: int | None = 42,
) -> float:
    """
    使用 median heuristic 估计 RBF 核带宽 sigma。

    为避免大样本下计算量过大，最多随机抽取 max_samples 个样本。
    """
    x_array, y_array = _validate_samples(x, y)
    combined = np.vstack([x_array, y_array])

    rng = np.random.default_rng(random_state)

    if combined.shape[0] > max_samples:
        indices = rng.choice(
            combined.shape[0],
            size=max_samples,
            replace=False,
        )
        combined = combined[indices]

    distances = pdist(combined, metric="euclidean")
    positive_distances = distances[distances > 0]

    if positive_distances.size == 0:
        return 1.0

    sigma = float(np.median(positive_distances))

    if not np.isfinite(sigma) or sigma <= 0:
        return 1.0

    return sigma


def mmd_rbf(
    x: ArrayLike,
    y: ArrayLike,
    sigma: float | None = None,
    squared: bool = False,
    unbiased: bool = False,
) -> float:
    """
    使用 RBF 核计算 Maximum Mean Discrepancy。

    核函数：
        k(a, b) = exp(-||a-b||^2 / (2 sigma^2))

    Parameters
    ----------
    sigma:
        RBF 核带宽。
        如果为 None，则使用 median heuristic 自动估计。

    squared:
        True 返回 MMD^2。
        False 返回 sqrt(max(MMD^2, 0))。

    unbiased:
        False 使用 biased/V-statistic 估计。
        True 使用 unbiased/U-statistic 估计。

    注意
    ----
    当其中一组样本很少时，biased 版本通常数值更稳定。
    """
    x_array, y_array = _validate_samples(x, y)

    if sigma is None:
        sigma = estimate_rbf_bandwidth(x_array, y_array)

    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("sigma 必须是正的有限数。")

    gamma = 1.0 / (2.0 * sigma**2)

    xx_squared = cdist(
        x_array,
        x_array,
        metric="sqeuclidean",
    )
    yy_squared = cdist(
        y_array,
        y_array,
        metric="sqeuclidean",
    )
    xy_squared = cdist(
        x_array,
        y_array,
        metric="sqeuclidean",
    )

    kernel_xx = np.exp(-gamma * xx_squared)
    kernel_yy = np.exp(-gamma * yy_squared)
    kernel_xy = np.exp(-gamma * xy_squared)

    n = x_array.shape[0]
    m = y_array.shape[0]

    if unbiased:
        if n < 2 or m < 2:
            raise ValueError("unbiased MMD 要求每组至少有两个样本。")

        xx_term = (kernel_xx.sum() - np.trace(kernel_xx)) / (n * (n - 1))

        yy_term = (kernel_yy.sum() - np.trace(kernel_yy)) / (m * (m - 1))

        xy_term = kernel_xy.mean()

        mmd_squared = xx_term + yy_term - 2.0 * xy_term
    else:
        mmd_squared = kernel_xx.mean() + kernel_yy.mean() - 2.0 * kernel_xy.mean()

    # 无偏估计可能由于有限样本而得到负值。
    if squared:
        return float(mmd_squared)

    return float(np.sqrt(max(mmd_squared, 0.0)))


def energy_distance(
    x: ArrayLike,
    y: ArrayLike,
    squared: bool = False,
    unbiased: bool = False,
) -> float:
    """
    计算多维 Energy Distance。

    D_E^2(P, Q) =
        2 E||X-Y||
        - E||X-X'||
        - E||Y-Y'||

    Parameters
    ----------
    squared:
        True 返回公式中的 Energy statistic。
        False 返回其平方根。

    unbiased:
        False 使用包含对角线项的 V-statistic。
        True 排除组内距离矩阵对角线项。
    """
    x_array, y_array = _validate_samples(x, y)

    xy_distances = cdist(
        x_array,
        y_array,
        metric="euclidean",
    )

    xx_distances = cdist(
        x_array,
        x_array,
        metric="euclidean",
    )

    yy_distances = cdist(
        y_array,
        y_array,
        metric="euclidean",
    )

    cross_term = 2.0 * xy_distances.mean()

    if unbiased:
        n = x_array.shape[0]
        m = y_array.shape[0]

        if n < 2 or m < 2:
            raise ValueError("unbiased Energy Distance 要求每组至少有两个样本。")

        within_x = xx_distances.sum() / (n * (n - 1))
        within_y = yy_distances.sum() / (m * (m - 1))
    else:
        within_x = xx_distances.mean()
        within_y = yy_distances.mean()

    energy_squared = cross_term - within_x - within_y

    if squared:
        return float(energy_squared)

    return float(np.sqrt(max(energy_squared, 0.0)))


def featurewise_wasserstein_distance(
    x: ArrayLike,
    y: ArrayLike,
    aggregation: Aggregation = "mean",
) -> float | NDArray[np.float64]:
    """
    对每个特征分别计算一维 Wasserstein distance。
    """
    x_array, y_array = _validate_samples(x, y)

    values = np.asarray(
        [
            wasserstein_distance(
                x_array[:, feature_index],
                y_array[:, feature_index],
            )
            for feature_index in range(x_array.shape[1])
        ],
        dtype=np.float64,
    )

    return _aggregate(values, aggregation)


def featurewise_ks_statistic(
    x: ArrayLike,
    y: ArrayLike,
    aggregation: Aggregation = "mean",
) -> float | NDArray[np.float64]:
    """
    对每个特征分别计算双样本 KS statistic。

    这里只返回 statistic，不返回 p-value。
    """
    x_array, y_array = _validate_samples(x, y)

    values = np.asarray(
        [
            ks_2samp(
                x_array[:, feature_index],
                y_array[:, feature_index],
                alternative="two-sided",
                method="auto",
            ).statistic
            for feature_index in range(x_array.shape[1])
        ],
        dtype=np.float64,
    )

    return _aggregate(values, aggregation)


def featurewise_ks_test(
    x: ArrayLike,
    y: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    返回每个特征的 KS statistic 和 p-value。

    Returns
    -------
    statistics:
        shape=(n_features,)

    p_values:
        shape=(n_features,)
    """
    x_array, y_array = _validate_samples(x, y)

    statistics = np.empty(x_array.shape[1], dtype=np.float64)
    p_values = np.empty(x_array.shape[1], dtype=np.float64)

    for feature_index in range(x_array.shape[1]):
        result = ks_2samp(
            x_array[:, feature_index],
            y_array[:, feature_index],
            alternative="two-sided",
            method="auto",
        )

        statistics[feature_index] = result.statistic
        p_values[feature_index] = result.pvalue

    return statistics, p_values


def _make_bin_edges(
    x_feature: NDArray[np.float64],
    y_feature: NDArray[np.float64],
    n_bins: int = 3,
    strategy: BinStrategy = "quantile",
    bin_reference: Literal["combined", "x"] = "combined",
) -> NDArray[np.float64]:
    """
    为单个特征创建共享的分箱边界。

    strategy="quantile":
        使用分位数分箱，适合小样本组。

    strategy="uniform":
        在最小值和最大值之间等宽分箱。

    bin_reference="combined":
        使用 x 和 y 的合并数据决定边界。

    bin_reference="x":
        仅使用 x 决定边界，适合将 x 视为参考分布。
    """
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins 必须是大于或等于 1 的整数。")

    if bin_reference == "combined":
        reference = np.concatenate([x_feature, y_feature])
    elif bin_reference == "x":
        reference = x_feature
    else:
        raise ValueError("bin_reference 必须是 'combined' 或 'x'。")

    minimum = float(np.min(reference))
    maximum = float(np.max(reference))

    if minimum == maximum:
        return np.asarray([-np.inf, np.inf], dtype=np.float64)

    if strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(reference, quantiles)

    elif strategy == "uniform":
        edges = np.linspace(minimum, maximum, n_bins + 1)

    else:
        raise ValueError("strategy 必须是 'quantile' 或 'uniform'。")

    # 离散特征或重复值较多时，分位数边界可能重复。
    edges = np.unique(edges)

    if edges.size < 2:
        return np.asarray([-np.inf, np.inf], dtype=np.float64)

    # 使用无穷边界，保证两个数据集的所有值都被统计。
    edges = edges.astype(np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf

    return edges


def _histogram_probabilities(
    x_feature: NDArray[np.float64],
    y_feature: NDArray[np.float64],
    n_bins: int = 3,
    strategy: BinStrategy = "quantile",
    smoothing: float = 0.5,
    bin_reference: Literal["combined", "x"] = "combined",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    使用共享分箱边界，将两组样本转换为离散概率分布。

    smoothing=0.5:
        对每个 bin 增加 0.5 个伪计数。
        这比在概率上直接加极小 epsilon 更适合小样本。
    """
    if smoothing < 0 or not np.isfinite(smoothing):
        raise ValueError("smoothing 必须是非负有限数。")

    edges = _make_bin_edges(
        x_feature=x_feature,
        y_feature=y_feature,
        n_bins=n_bins,
        strategy=strategy,
        bin_reference=bin_reference,
    )

    x_counts, _ = np.histogram(x_feature, bins=edges)
    y_counts, _ = np.histogram(y_feature, bins=edges)

    p = x_counts.astype(np.float64) + smoothing
    q = y_counts.astype(np.float64) + smoothing

    if p.sum() <= 0 or q.sum() <= 0:
        raise ValueError("直方图总计数为零，请检查数据或 smoothing。")

    p /= p.sum()
    q /= q.sum()

    return p, q


def _discrete_kl(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    base: float | None = 2.0,
) -> float:
    """
    计算离散概率分布的 KL(P || Q)。
    """
    value = float(np.sum(rel_entr(p, q)))

    if base is not None:
        if base <= 0 or base == 1 or not np.isfinite(base):
            raise ValueError("对数底数 base 必须为正、有限且不等于 1。")

        value /= np.log(base)

    return value


def featurewise_kl_divergence(
    x: ArrayLike,
    y: ArrayLike,
    n_bins: int = 3,
    strategy: BinStrategy = "quantile",
    smoothing: float = 0.5,
    base: float | None = 2.0,
    aggregation: Aggregation = "mean",
    bin_reference: Literal["combined", "x"] = "combined",
) -> float | NDArray[np.float64]:
    """
    对每个特征计算 KL(X || Y)。

    注意
    ----
    KL 是有方向的：

        KL(X || Y) != KL(Y || X)

    该方法比较的是各个特征的边缘分布，而不是完整的高维联合分布。
    """
    x_array, y_array = _validate_samples(x, y)

    values = np.empty(x_array.shape[1], dtype=np.float64)

    for feature_index in range(x_array.shape[1]):
        p, q = _histogram_probabilities(
            x_feature=x_array[:, feature_index],
            y_feature=y_array[:, feature_index],
            n_bins=n_bins,
            strategy=strategy,
            smoothing=smoothing,
            bin_reference=bin_reference,
        )

        values[feature_index] = _discrete_kl(
            p=p,
            q=q,
            base=base,
        )

    return _aggregate(values, aggregation)


def featurewise_js_divergence(
    x: ArrayLike,
    y: ArrayLike,
    n_bins: int = 3,
    strategy: BinStrategy = "quantile",
    smoothing: float = 0.5,
    base: float | None = 2.0,
    return_distance: bool = False,
    aggregation: Aggregation = "mean",
    bin_reference: Literal["combined", "x"] = "combined",
) -> float | NDArray[np.float64]:
    """
    对每个特征计算 Jensen-Shannon divergence 或 distance。

    JS divergence:
        JS(P, Q) =
            0.5 KL(P || M) + 0.5 KL(Q || M)

        M = 0.5(P + Q)

    当 base=2 时，JS divergence 位于 [0, 1]。

    return_distance=True 时返回：
        sqrt(JS divergence)

    只有平方根形式是严格意义上的距离度量。
    """
    x_array, y_array = _validate_samples(x, y)

    values = np.empty(x_array.shape[1], dtype=np.float64)

    for feature_index in range(x_array.shape[1]):
        p, q = _histogram_probabilities(
            x_feature=x_array[:, feature_index],
            y_feature=y_array[:, feature_index],
            n_bins=n_bins,
            strategy=strategy,
            smoothing=smoothing,
            bin_reference=bin_reference,
        )

        mixture = 0.5 * (p + q)

        js_value = 0.5 * _discrete_kl(
            p=p,
            q=mixture,
            base=base,
        )
        js_value += 0.5 * _discrete_kl(
            p=q,
            q=mixture,
            base=base,
        )

        js_value = max(float(js_value), 0.0)

        if return_distance:
            js_value = float(np.sqrt(js_value))

        values[feature_index] = js_value

    return _aggregate(values, aggregation)


def compute_distribution_distances(
    x: ArrayLike,
    y: ArrayLike,
    *,
    aggregation: Aggregation = "mean",
    n_bins: int = 3,
    bin_strategy: BinStrategy = "quantile",
    bin_reference: Literal["combined", "x"] = "combined",
    smoothing: float = 0.5,
    log_base: float | None = 2.0,
    mmd_sigma: float | None = None,
    mmd_unbiased: bool = False,
    energy_unbiased: bool = False,
) -> DistributionDistanceResult:
    """
    一次性计算全部分布距离。

    针对 shape=(1300, 64) 和 shape=(13, 64) 的推荐默认值：

        n_bins=3
        bin_strategy="quantile"
        smoothing=0.5
        aggregation="mean"
        mmd_unbiased=False
        energy_unbiased=False

    Returns
    -------
    DistributionDistanceResult
    """
    x_array, y_array = _validate_samples(x, y)

    featurewise_wasserstein = np.asarray(
        featurewise_wasserstein_distance(
            x_array,
            y_array,
            aggregation="none",
        ),
        dtype=np.float64,
    )

    featurewise_ks = np.asarray(
        featurewise_ks_statistic(
            x_array,
            y_array,
            aggregation="none",
        ),
        dtype=np.float64,
    )

    featurewise_js = np.asarray(
        featurewise_js_divergence(
            x_array,
            y_array,
            n_bins=n_bins,
            strategy=bin_strategy,
            smoothing=smoothing,
            base=log_base,
            return_distance=False,
            aggregation="none",
            bin_reference=bin_reference,
        ),
        dtype=np.float64,
    )

    featurewise_kl_x_to_y = np.asarray(
        featurewise_kl_divergence(
            x_array,
            y_array,
            n_bins=n_bins,
            strategy=bin_strategy,
            smoothing=smoothing,
            base=log_base,
            aggregation="none",
            bin_reference=bin_reference,
        ),
        dtype=np.float64,
    )

    featurewise_kl_y_to_x = np.asarray(
        featurewise_kl_divergence(
            y_array,
            x_array,
            n_bins=n_bins,
            strategy=bin_strategy,
            smoothing=smoothing,
            base=log_base,
            aggregation="none",
            bin_reference=bin_reference,
        ),
        dtype=np.float64,
    )

    wasserstein_value = _aggregate(
        featurewise_wasserstein,
        aggregation,
    )

    ks_value = _aggregate(
        featurewise_ks,
        aggregation,
    )

    js_value = _aggregate(
        featurewise_js,
        aggregation,
    )

    kl_x_to_y_value = _aggregate(
        featurewise_kl_x_to_y,
        aggregation,
    )

    kl_y_to_x_value = _aggregate(
        featurewise_kl_y_to_x,
        aggregation,
    )

    # 此函数需要返回标量，因此 aggregation 不能是 none。
    scalar_values = {
        "wasserstein": wasserstein_value,
        "ks": ks_value,
        "js": js_value,
        "kl_x_to_y": kl_x_to_y_value,
        "kl_y_to_x": kl_y_to_x_value,
    }

    for name, value in scalar_values.items():
        if isinstance(value, np.ndarray):
            raise ValueError(
                f"compute_distribution_distances 中 aggregation "
                f"不能为 'none'，因为 {name} 需要汇总成标量。"
            )

    js_scalar = float(js_value)
    kl_xy_scalar = float(kl_x_to_y_value)
    kl_yx_scalar = float(kl_y_to_x_value)

    return DistributionDistanceResult(
        mmd=mmd_rbf(
            x_array,
            y_array,
            sigma=mmd_sigma,
            squared=False,
            unbiased=mmd_unbiased,
        ),
        energy_distance=energy_distance(
            x_array,
            y_array,
            squared=False,
            unbiased=energy_unbiased,
        ),
        wasserstein=float(wasserstein_value),
        ks=float(ks_value),
        js_divergence=js_scalar,
        js_distance=float(np.sqrt(max(js_scalar, 0.0))),
        kl_x_to_y=kl_xy_scalar,
        kl_y_to_x=kl_yx_scalar,
        symmetric_kl=0.5 * (kl_xy_scalar + kl_yx_scalar),
        featurewise_wasserstein=featurewise_wasserstein,
        featurewise_ks=featurewise_ks,
        featurewise_js=featurewise_js,
        featurewise_kl_x_to_y=featurewise_kl_x_to_y,
        featurewise_kl_y_to_x=featurewise_kl_y_to_x,
    )

def record_regret(model, cur_embedding, base_embedding):
    device = next(
        model._BaoRegression__net.parameters()
    ).device

    cur_tensor = utils.to_tensor(cur_embedding, device)
    base_tensor = utils.to_tensor(base_embedding, device)
    with torch.no_grad():
        mmd_score = det.mmd(cur_tensor, base_tensor)

    cur_np = utils.to_numpy(cur_embedding)
    base_np = utils.to_numpy(base_embedding)

    ks_result = det.ks_values_pca(cur_np, base_np)
    ws_score = det.ws(cur_np, base_np)

    result = compute_distribution_distances(
        base_np,
        cur_np,
        # 逐特征结果如何汇总
        aggregation="mean",
        # 针对较小样本组只有 13 个样本
        n_bins=3,
        # 分位数分箱比等宽分箱更稳定
        bin_strategy="quantile",
        # 使用两组数据共同确定 bin 边界
        bin_reference="combined",
        # 每个 bin 加 0.5 个伪计数
        smoothing=0.5,
        # KL/JS 使用 log2
        log_base=2.0,
        # None 表示使用 median heuristic
        mmd_sigma=None,
        # 小样本情况下 biased 版本通常更稳定
        mmd_unbiased=False,
        energy_unbiased=False,
    )
    # if hasattr(mmd_score, "detach"):
    #     mmd_value = float(mmd_score.detach().cpu().item())
    # else:
    #     mmd_value = float(mmd_score)

    return {
        "mmd_score": float(mmd_score.detach().cpu()),
        "ks_result": ks_result,
        "ws_score": ws_score,
        **asdict(result),
    }


if __name__ == "__main__":
    # 示例数据
    rng = np.random.default_rng(42)
    x = rng.normal(loc=0.0,scale=1.0,size=(1300, 64),)
    y = rng.normal(loc=0.2,scale=1.1,size=(13, 64),)

    result = compute_distribution_distances(
        x,
        y,
        # 逐特征结果如何汇总
        aggregation="mean",
        # 针对较小样本组只有 13 个样本
        n_bins=3,
        # 分位数分箱比等宽分箱更稳定
        bin_strategy="quantile",
        # 使用两组数据共同确定 bin 边界
        bin_reference="combined",
        # 每个 bin 加 0.5 个伪计数
        smoothing=0.5,
        # KL/JS 使用 log2
        log_base=2.0,
        # None 表示使用 median heuristic
        mmd_sigma=None,
        # 小样本情况下 biased 版本通常更稳定
        mmd_unbiased=False,
        energy_unbiased=False,
    )

    print("===== Aggregated distances =====")
    print(f"MMD:                    {result.mmd:.6f}")
    print(f"Energy Distance:        {result.energy_distance:.6f}")
    print(f"Mean Wasserstein:       {result.wasserstein:.6f}")
    print(f"Mean KS Statistic:      {result.ks:.6f}")
    print(f"Mean JS Divergence:     {result.js_divergence:.6f}")
    print(f"Mean JS Distance:       {result.js_distance:.6f}")
    print(f"Mean KL(X || Y):        {result.kl_x_to_y:.6f}")
    print(f"Mean KL(Y || X):        {result.kl_y_to_x:.6f}")
    print(f"Mean Symmetric KL:      {result.symmetric_kl:.6f}")

    print()
    print("===== Feature-wise result shapes =====")
    print(
        "Wasserstein:",
        result.featurewise_wasserstein.shape,
    )
    print(
        "KS:",
        result.featurewise_ks.shape,
    )
    print(
        "JS:",
        result.featurewise_js.shape,
    )
    print(
        "KL(X || Y):",
        result.featurewise_kl_x_to_y.shape,
    )
