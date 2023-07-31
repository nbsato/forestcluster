import operator

import numpy
import scipy.cluster.hierarchy
import scipy.spatial.distance
import sklearn.metrics


def cluster(Xt, metric="cosine", method="average", n_jobs=None):
    """Cluster transformed vectors.

    The agglomerative hierarchical clustering is performed.

    Parameters
    ----------
    Xt : scipy.sparse.spmatrix
        Matrix whose row is a transformed vector of a sample.
    metric : {'cosine', 'euclidean', 'jaccard'}, optional
        Distance metric.
    method : str, optional
        Linkage method. Passed to the ``method`` argument of the
        `scipy.cluster.hierarchy.linkage` function.
    n_jobs : int, optional
        Number of parallel jobs.

    Returns
    -------
    Z : numpy.ndarray
        Linkage matrix returned by the `scipy.cluster.hierarchy.linkage`
        function.
    """
    D = _distance_matrix(Xt, metric=metric, n_jobs=n_jobs)
    distances = scipy.spatial.distance.squareform(D, force="tovector")
    Z = scipy.cluster.hierarchy.linkage(distances, method=method)

    return Z


def assign_labels(Z, sort_key="distance", ascending=True):
    """Assign cluster labels to samples.

    When a cluster is divided into two child clusters, then the label is
    inherited by the left child cluster.

    Parameters
    ----------
    Z : numpy.ndarray
        Linkage matrix.
    sort_key : {'distance', 'size'}, optional
        Sort key for swapping child nodes.
    ascending : bool, optional
        If ``True``, child nodes are sorted in ascending order from left to
        right. Otherwise, they are sorted in descending order.

    Returns
    -------
    labels : numpy.ndarray of int
        Cluster labels of samples. The ``(i, j)`` component is the label of the
        ``i``th sample when dividing samples into ``j + 1`` clusters.
    """
    Z = _swap_children(Z, key=sort_key, ascending=ascending)

    n_samples = Z.shape[0] + 1
    n_nodes = n_samples + Z.shape[0]

    labels = numpy.empty((n_samples, n_samples), dtype=int)

    order = [n_nodes - 1]
    for i_node, (i_left, i_right) in zip(
            range(n_nodes - 1, n_samples - 1, -1),
            Z[::-1, [0, 1]].astype(int)):
        k = order.index(i_node)
        order[k] = i_left
        order.append(i_right)
    labels[order, -1] = range(n_samples)

    node_indices = numpy.arange(n_samples)
    for i_node, child_indices in enumerate(
            Z[:, [0, 1]].astype(int), start=n_samples):
        j = n_nodes - 1 - i_node
        is_child = numpy.isin(node_indices, child_indices)
        labels[:, j] = labels[:, j + 1]
        labels[is_child, j] = labels[is_child, j + 1].min()
        node_indices[is_child] = i_node

    return labels


def _distance_matrix(points, metric, n_jobs=None):
    """Compute distance matrix.

    Parameters
    ----------
    points : numpy.ndarray or scipy.sparse.spmatrix
        Points among which distances are measured. Each row is a point.
    metric : {'cosine', 'euclidean', 'jaccard'}
        Distance metric.
    n_jobs : int, optional
        Number of parallel jobs.

    Returns
    -------
    D : numpy.ndarray
        Distance matrix.
    """
    if metric in {"cosine", "euclidean"}:
        return sklearn.metrics.pairwise_distances(
            points, metric=metric, n_jobs=n_jobs)
    elif metric == "jaccard":
        if scipy.sparse.isspmatrix(points):
            points = points.astype(float)
            intersection_sizes = points @ points.T
            counts = points.sum(axis=1)
            union_sizes = counts + counts.T - intersection_sizes

            similarities = numpy.ones(union_sizes.shape)
            numpy.true_divide(
                intersection_sizes.toarray(), union_sizes.A,
                where=(union_sizes != 0), out=similarities)

            return 1. - similarities
        else:
            points = numpy.asarray(points, dtype=bool)

            return sklearn.metrics.pairwise_distances(
                points, metric="jaccard", n_jobs=n_jobs)
    else:
        raise ValueError(
            "`metric` must be 'cosine', 'euclidean', or 'jaccard'")


def _swap_children(Z, key, ascending=True):
    """Swap child nodes to be sorted.

    Parameters
    ----------
    Z : numpy.ndarray
        Linkage matrix.
    key : {'distance', 'size'}
        Sort key.
    ascending : bool, optional
        If ``True``, child nodes are sorted in ascending order from left to
        right. Otherwise, they are sorted in descending order.

    Returns
    -------
    numpy.ndarray
        Linkage matrix for swapped child nodes.
    """
    Z = numpy.array(Z)

    n_samples = Z.shape[0] + 1

    if key == "distance":
        values = Z[:, 2]
        sample_value = 0.
    elif key == "size":
        values = Z[:, 3]
        sample_value = 1
    else:
        raise ValueError("`key` must be 'distance' or 'size'")

    is_swapped = operator.gt if ascending else operator.lt
    for k, child_indices in enumerate(Z[:, [0, 1]].astype(int)):
        child_values = [
            sample_value if i_child < n_samples
            else values[i_child - n_samples]
            for i_child in child_indices]
        if is_swapped(*child_values):
            Z[k, [0, 1]] = Z[k, [1, 0]]

    return Z
