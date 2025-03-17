import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from scipy.special import comb

nmi = normalized_mutual_info_score

## juge measure
def rand_index_score(clusters, classes):
    clusters = clusters.astype(int)
    classes = classes.astype(int)
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

def acc(y_true, y_pred, num_cluster):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        num_cluster: number of cluster

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_true = y_true - y_true.min()#将 y_true 映射到 [0, num_cluster)


    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size

    w = np.zeros((num_cluster, num_cluster))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment

    ind = linear_sum_assignment(w.max() - w)
    accuracy = 0.0
    for i in ind[0]:
        accuracy = accuracy + w[i, ind[1][i]]
    return accuracy / y_pred.size
