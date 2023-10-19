import functools
import numpy as np
from scipy.sparse import issparse

from sklearn.utils._param_validation import (
    StrOptions,
    validate_params,
)

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y
from sklearn.metrics.pairwise import _VALID_METRICS, pairwise_distances_chunked


def check_number_of_labels(n_labels, n_samples):
    """Check that number of labels are valid.

    Parameters
    ----------
    n_labels : int
        Number of labels.

    n_samples : int
        Number of samples.
    """
    if not 1 < n_labels < n_samples:
        raise ValueError(
            "Number of labels is %d. Valid values are 2 to n_samples - 1 (inclusive)"
            % n_labels
        )


def dunn_reduce(D_chunk, start, labels):
    """Accumulate Dunn Index score - inter- and intraclust distances for vertical chunk of X.

    Parameters
    ----------
    D_chunk : {array-like, sparse matrix} of shape (n_chunk_samples, n_samples)
        Precomputed distances for a chunk. If a sparse matrix is provided,
        only CSR format is accepted.
    start : int
        First index in the chunk.
    labels : array-like of shape (n_samples,)
        Corresponding cluster labels, encoded as {0, ..., n_clusters-1}.
    label_freqs : array-like
        Distribution of cluster labels in ``labels``.
    """ 
    n_chunk_samples = D_chunk.shape[0]

    inter_cluster_distances = np.zeros(
        n_chunk_samples, dtype=D_chunk.dtype
    )
    intra_cluster_distances = np.zeros(
        n_chunk_samples, dtype=D_chunk.dtype
    )
    
    if issparse(D_chunk):
        if D_chunk.format != "csr":
            raise TypeError(
                "Expected CSR matrix. Please pass sparse matrix in CSR format."
            )
        for i in range(n_chunk_samples):
            indptr = D_chunk.indptr
            indices = D_chunk.indices[indptr[i] : indptr[i + 1]]
            sample_dists = D_chunk.data[indptr[i] : indptr[i + 1]]
            sample_labels = np.take(labels, indices)
            intra_cluster_distances[i] += np.max(
                np.where(sample_labels == sample_labels[i],
                         sample_dists,
                         -np.inf)
            )
            inter_cluster_distances[i] += np.min(
                np.where(sample_labels != sample_labels[i],
                         sample_dists,
                         np.inf)
            )
    else:
        for i in range(n_chunk_samples):
            sample_dists = D_chunk[i]
            sample_labels = labels
            intra_cluster_distances[i] += np.max(
                np.where(sample_labels == sample_labels[i],
                         sample_dists,
                         -np.inf)
            )
            inter_cluster_distances[i] += np.min(
                np.where(sample_labels != sample_labels[i],
                         sample_dists,
                         np.inf)
            )
    return intra_cluster_distances, inter_cluster_distances


@validate_params(
        {
            "X": ["array-like", "sparse matrix"],
            "labels": ["array-like"],
            "metric" : [StrOptions(set(_VALID_METRICS) | {"precomputed"}), callable],
        },
        prefer_skip_nested_validation=True,
)
def dunn_index(X, labels, metric = "euclidean", **kwds):
    """Compute the Dunn Index.
    The Intercluster distance is caclulated as the minimal between two data points, one in each cluster.
    The Intracluster distance is caclulated as the maximal between two data points, both in the same cluster.
    The more the value, the better.
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_a, n_samples_a) if metric == \
        "precomputed" or (n_samples_a, n_features) otherwise
        An array of pairwise distances between samples, or a feature array.

    labels : array-like of shape (n_samples,)
        Predicted cluster labels for each sample.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`~sklearn.metrics.pairwise_distances`.
    
    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
    
    Returns
    -------
    score: float
        The resulting Dunn Index.
    """
    X, labels = check_X_y(X, labels, accept_sparse=["csr"])

    # Check for non-zero diagonal entries in precomputed distance matrix
    if metric == "precomputed":
        error_msg = ValueError(
            "The precomputed distance matrix contains non-zero "
            "elements on the diagonal. Use np.fill_diagonal(X, 0)."
        )
        if X.dtype.kind == "f":
            atol = np.finfo(X.dtype).eps * 100
            if np.any(np.abs(X.diagonal()) > atol):
                raise error_msg
        elif np.any(X.diagonal() != 0):  # integral dtype
            raise error_msg
    
    
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples = len(labels)
    check_number_of_labels(len(le.classes_), n_samples)

    kwds["metric"] = metric

    reduce_func = functools.partial(
        dunn_reduce, labels=labels
    )

    results = zip(*pairwise_distances_chunked(X, reduce_func=reduce_func, **kwds))
    intra_clust_dists, inter_clust_dists = results
    intra_clust_dists = np.concatenate(intra_clust_dists)   
    inter_clust_dists = np.concatenate(inter_clust_dists)
    return np.min(inter_clust_dists)/np.max(intra_clust_dists)