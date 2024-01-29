import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog, minimize, brute

from sklearn.linear_model import LinearRegression

from hyperopt import hp, tpe, Trials, fmin, space_eval, STATUS_OK




def get_features(x, power):
    X = x.reshape(-1,1)
    X = np.concatenate([np.power(X, p) for p in range(power + 1)], axis = -1)
    return X


def log_likelihood(X, y, W):
    """
    :param X_data: observable features with (bias term at first coordinate)
                    expected shape: (n_samples, n_features + 1)
    :param y: observable target 
                    expected shape: (n_samples)
    :param W: weight matrix (with bias term at first coordinate):
                    expexted_shape: (n_features + 1, )
    """
    predictions = X@W
    mask = predictions < y
    # print(W, len(mask))
    if not mask.all():
        return -9999
    return np.sum(predictions[mask] - y[mask])



def feasible_point(A, b):
    # finds the center of the largest sphere fitting in the convex hull
    norm_vector = np.linalg.norm(A, axis=1)
    A_ = np.hstack((A, norm_vector[:, None]))
    b_ = b[:, None]
    c = np.zeros((A.shape[1] + 1,))
    c[-1] = -1
    res = linprog(c, A_ub=A_, b_ub=b[:, None], bounds=(None, None))
    return res.x[:-1]

def hs_intersection(A, b):
    interior_point = feasible_point(A, b)
    halfspaces = np.hstack((A, -b[:, None]))
    hs = HalfspaceIntersection(halfspaces, interior_point)
    return hs

def plt_halfspace(a, b, bbox, ax):
    if a[1] == 0:
        ax.axvline(b / a[0])
    else:
        x = np.linspace(bbox[0][0], bbox[0][1], 100)
        ax.plot(x, (b - a[0]*x) / a[1], linewidth=.5)

def add_bbox(A, b, bbox):
    d = A.shape[1]
    tmp = np.zeros(d)
    tmp[0] = 1
    A = np.vstack((A, [(-1)**(i+1) * np.roll(tmp, i//2) for i in range(2*d)] ))
    b = np.hstack((b, [(-1)**(i+1) * x for i, x in enumerate(bbox.flatten())]))
    return A, b

def solve_convex_set(A, b, bbox, ax=None):
    A_, b_ = add_bbox(A, b, bbox)
    interior_point = feasible_point(A_, b_)
    hs = hs_intersection(A_, b_)
    points = hs.intersections
    hull = ConvexHull(points)
    return points[hull.vertices], interior_point, hs

def plot_convex_set(A, b, bbox, ax=None):
    # solve and plot just the convex set (no lines for the inequations)
    points, interior_point, hs = solve_convex_set(A, b, bbox, ax=ax)
    if ax is None:
        _, ax = plt.subplots()
    # ax.fill(points[:, 0], points[:, 1], 'r', alpha=0.2, label='domain')
    return points, interior_point, hs

def plot_inequalities(A, b, bbox, ax=None):
    # solve and plot the convex set,
    # the inequation lines, and
    # the interior point that was used for the halfspace intersections
    points, interior_point, hs = plot_convex_set(A, b, bbox, ax=ax)
    for a_k, b_k in zip(A, b):
        plt_halfspace(a_k, b_k, bbox, ax)
    ax.plot(*interior_point, 'o', color='red')
    ax.set_xlim(bbox[0])
    ax.set_ylim(bbox[1])
    return points, interior_point, hs