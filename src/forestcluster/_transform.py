import sklearn.preprocessing


def transform(estimator, X):
    """Transform feature vectors by an estimator.

    Parameters
    ----------
    estimator : sklearn.BaseEstimator
        Fitted estimator implementing the `apply` method.
    X : array-like or scipy.sparse.spmatrix
        Matrix whose row is a feature vector of a sample.

    Returns
    -------
    Xt : scipy.sparse.spmatrix
        Matrix whose row is a transformed vector of a sample.
    """
    leaves = estimator.apply(X)

    encoder = sklearn.preprocessing.OneHotEncoder(sparse=True)
    Xt = encoder.fit_transform(leaves)

    return Xt
