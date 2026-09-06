import torch
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance
from torch.autograd import Variable


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    # n_samples = int(source.size()[0]) + int(target.size()[0])
    # total = torch.cat([source, target], dim=0)
    # total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # L2_distance = ((total0 - total1) ** 2).sum(2)
    total = torch.cat([source, target], dim=0)
    n_total = total.size(0)

    # 只生成 [N, N]，不会生成 [N, N, D]
    L2_distance = torch.cdist(total, total, p=2).pow(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        # bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth = torch.sum(L2_distance.data) / (n_total ** 2 - n_total)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n = int(source.size()[0])
    m = int(target.size()[0])

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)  # K_ss矩阵，Source<->Source
    XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st矩阵，Source<->Target

    YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)  # K_ts矩阵,Target<->Source
    YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)  # K_tt矩阵,Target<->Target

    loss = (XX + XY).sum() + (YX + YY).sum()
    return loss


def ws(embedding_1, embedding_2):
    pca = PCA(n_components=1)
    # Fit on reference data
    pca.fit(embedding_1)
    emb1_1d = pca.transform(embedding_1)
    emb2_1d = pca.transform(embedding_2)

    return wasserstein_distance(emb1_1d[:,0], emb2_1d[:,0])

def wasserstein_random_proj(X, Y, n_proj=10):
    dims = X.shape[1]
    results = []
    for _ in range(n_proj):
        w = np.random.normal(size=dims)
        w = w / np.linalg.norm(w)
        x_proj = X @ w
        y_proj = Y @ w

        d = wasserstein_distance(x_proj, y_proj)
        results.append(d)

    return np.mean(results)

def ks_values_min(embedding_1, embedding_2):
    p_values = []
    for d in range(64):
        stat, p = ks_2samp(embedding_1[:, d], embedding_2[:, d])
        p_values.append(p)

    return (min(p_values))


def ks_values_pca(embedding_1, embedding_2):
    pca = PCA(n_components=1)
    # Fit on reference data
    pca.fit(embedding_1)
    emb1_1d = pca.transform(embedding_1)
    emb2_1d = pca.transform(embedding_2)
    return ks_2samp(emb1_1d[:, 0], emb2_1d[:, 0])

def embedding_plans(model, all_plans):
    """ Embed multiple plans.
    Returns:
        shape: (num_plans, embedding_dim)"""
    trees = model._BaoRegression__tree_transform.transform(all_plans)
    return model._BaoRegression__net.get_fixed_features(trees)

def embedding_single_query(model, query_plans, aggregate=None):
    """
    Embed all candidate plans of a single query.
    Parameters
    ----------
    model:
        Trained BaoRegression model.
    query_plans:
        List of candidate plans for one query.
        For example: 13 plans.
    aggregate:
        None   -> return all 13 plan embeddings
        "mean" -> return one mean query embedding
        "max"  -> return one max-pooled query embedding
    Returns
    -------
    embeddings:
        aggregate=None:
            (13, embedding_dim)
        aggregate="mean"/"max":
            (embedding_dim,)
    """

    embeddings = embedding_plans(model, query_plans)

    # 如果返回的是 torch.Tensor
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()

    if aggregate is None:
        return embeddings

    if aggregate == "mean":
        return np.mean(embeddings, axis=0)

    if aggregate == "max":
        return np.max(embeddings, axis=0)

    raise ValueError("aggregate must be None, 'mean', or 'max'")


def get_plans(queries):
    all_plans = []
    for plans in queries["plans"]:
        all_plans.extend(plans)
    return all_plans

def main():
    # 样本数量可以不同，特征数目必须相同
    # 100和90是样本数量，50是特征数目
    data_1 = torch.tensor(np.random.normal(loc=0, scale=10, size=(100, 50)))
    data_2 = torch.tensor(np.random.normal(loc=10, scale=10, size=(90, 50)))
    print("MMD Loss:", mmd(data_1, data_2))

    data_1 = torch.tensor(np.random.normal(loc=0, scale=10, size=(100, 50)))
    data_2 = torch.tensor(np.random.normal(loc=0, scale=9, size=(80, 50)))

    print("MMD Loss:", mmd(data_1, data_2))


if __name__ == "__main__":
    main()
