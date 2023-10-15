import numpy as np
from typing import SupportsIndex

class KNN(object):
    def __init__(self, n_neighbours: int = 4):
        self.X_train = None
        self.y_train = None
        self.n_neighbours = n_neighbours

    def fit(self, X: np.ndarray, y: SupportsIndex):
        self.X_train, self.y_train = X, y.astype(int)

    
    def predict(self, X: np.ndarray) -> np.ndarray:

    # overall complexity O(NTD) ~ (DN^2) if Train and Fit data of same order
    # N - fit data samples
    # T - train data samples
    # D - dimension of data
    # k - n_neighbours

        labels = np.zeros(X.shape[0])

        #O(NTD)
        distances = np.apply_along_axis(
            np.linalg.norm,
            2,
            X[:, None, :] - self.X_train[None, :, :]
        )
        
        #O(N(T+k)) ~ O(NT)
        for i in range(X.shape[0]):
            indices = np.argpartition(distances[i], self.n_neighbours)      # do not sort is O(NlogN), but find k least values in O(N) 
            k_nearest_labels = self.y_train[indices][:self.n_neighbours] 
            counts = np.bincount(k_nearest_labels)
            labels[i] = np.argmax(counts)
        
        return labels
    
def accuracy(labels_true: np.ndarray, labels_predicted: np.ndarray) -> float:
    return np.sum(labels_true == labels_predicted)/labels_predicted.shape[0]

def TP(labels_true: np.ndarray, labels_predicted: np.ndarray) -> int:
    return np.sum(
        np.logical_and(
            labels_predicted,
            labels_true))

def FN(labels_true: np.ndarray, labels_predicted: np.ndarray) -> int:
    return np.sum(
        np.logical_and(
            np.logical_not(labels_predicted),
            labels_true))

def TN(labels_true: np.ndarray, labels_predicted: np.ndarray) -> int:
    return np.sum(
        np.logical_and(
            np.logical_not(labels_predicted),
            np.logical_not(labels_true)))

def FP(labels_true: np.ndarray, labels_predicted: np.ndarray) -> int:
    return np.sum(
        np.logical_and(labels_predicted,
                       np.logical_not(labels_true)))

def TPR(labels_true: np.ndarray, labels_predicted: np.ndarray) -> float:
    tp = TP(labels_true, labels_predicted)
    fn = FN(labels_true, labels_predicted)
    return tp/(tp + fn)

def FPR(labels_true: np.ndarray, labels_predicted: np.ndarray) -> float:
    tn = TN(labels_true, labels_predicted)
    fp = FP(labels_true, labels_predicted)
    return fp/(fp + tn)

def precision(labels_true: np.ndarray, labels_predicted: np.ndarray) -> float:
    tp = TP(labels_true, labels_predicted)
    fp = FP(labels_true, labels_predicted)
    return tp/(tp + fp)

def recall(labels_true: np.ndarray, labels_predicted: np.ndarray) -> float:
    tp = TP(labels_true, labels_predicted)
    fn = FN(labels_true, labels_predicted)
    return tp/(tp + fn)

def F1(labels_true: np.ndarray, labels_predicted: np.ndarray) -> float:
    R = recall(labels_true, labels_predicted)
    P = precision(labels_true, labels_predicted)
    return 2 * (R * P)/(R + P)