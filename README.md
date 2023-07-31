Forestcluster
=============

Python package for clustering assisted by the random forest.

Installation
------------

```shellsession
$ python -m pip install git+https://github.com/nbsato/forestcluster.git
```

Usage
-----

Assume that you have a fitted `sklearn.ensemble.RandomForestRegressor` object,
`estimator`, and a set of feature vectors, `X`.

Import the package:
```python console
>>> import forestcluster
```

Transform the feature vectors:
```python console
>>> Xt = forestcluster.transform(estimator, X)
```

Perform clustering:
```python console
>>> Z = forestcluster.cluster(Xt)
```

Assign cluster labels to samples:
```python console
>>> labels = forestcluster.assign_labels(Z)
```

License
-------

[Apache License, Version 2.0](LICENSE.txt)
