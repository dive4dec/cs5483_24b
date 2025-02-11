import os

import matplotlib.pyplot as plt
import numpy as np

def plot_decision_regions(X, Y, clf, target_names=None, ax=None, N=200):
    """
    Plot the decision boundaries of a classifier on a 2D dataset.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input features.
    Y : array-like, shape (n_samples,)
        Target values for X.
    clf : sklearn Classifier
        A classifier trained to predict Y from X.
    target_names : array-like, shape (n_classes,), optional
        Possible target names. If None, inferred from Y. Default is None.
    ax : matplotlib Axes, optional
        The axes on which to plot the boundaries. If None, uses the current axes. Default is None.
    N : int, optional
        Number of points for each dimension to scan for the decision boundaries. Default is 200.

    Returns
    -------
    ax : matplotlib Axes
        The axes for the plot of the decision boundaries.
    """

    def cat2int(target_array):
        return (
            (np.asarray(target_array).reshape(-1, 1) == target_names.reshape(1, -1))
            * np.arange(len(target_names)).reshape(1, -1)
        ).sum(axis=-1)

    if ax is None:
        ax = plt.gca()

    X_, Y_ = np.asarray(X), np.asarray(Y)

    X_min, X_max = X_.min(axis=0), X_.max(axis=0)
    x1, x2 = np.mgrid[X_min[0] : X_max[0] : N * 1j, X_min[1] : X_max[1] : N * 1j]

    if target_names is None:
        target_names = np.unique(Y_)

    yhat = cat2int(clf.predict(np.c_[x1.ravel(), x2.ravel()])).reshape(x1.shape)

    ax.contourf(x1, x2, yhat, alpha=0.4, cmap="Set1")
    scatter = ax.scatter(
        X_[:, 0], X_[:, 1], c=cat2int(Y_), edgecolor="w", s=20, cmap="Set1"
    )
    ax.set_xlim(X_min[0], X_max[0])
    ax.set_ylim(X_min[1], X_max[1])
    ax.add_artist(
        ax.legend(
            scatter.legend_elements()[0],
            target_names,
            loc="upper left",
            title="Classes",
        )
    )
    return ax