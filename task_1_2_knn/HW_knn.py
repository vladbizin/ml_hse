#HSE ML course 2023
#Bizin Vladislav

import numpy as np
from typing import SupportsIndex
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

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
    # P.S. можно сделать и с помощью KD-дерева и улучшить асимптортику,
    # но мы нормисы

        labels = np.zeros(X.shape[0])

        #O(NTD)
        distances = np.apply_along_axis(
            np.linalg.norm,
            2,
            X[:, None, :] - self.X_train[None, :, :]
        )
        
        #O(N(T+k)) ~ O(NT)
        for i in range(X.shape[0]):
            indices = np.argpartition(distances[i], self.n_neighbours)      # не сортируем за NlogN, а находим k минимальных за N 
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

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(100)

    means0 = [1, -1]
    covs0 = [[7, 3],
             [3, 7]]
    x0, y0 = np.random.multivariate_normal(means0, covs0, 190).T

    means1 = [0, -4]
    covs1 = [[0.1, 0.0],
             [0.0, 25]]
    x1, y1 = np.random.multivariate_normal(means1, covs1, 100).T

    data0 = np.vstack([x0, y0]).T
    labels0 = np.zeros(data0.shape[0])

    data1 = np.vstack([x1, y1]).T
    labels1 = np.ones(data1.shape[0])

    data = np.vstack([data0, data1])
    labels = np.hstack([labels0, labels1])
    total_size = data.shape[0]
    print("Original dataset shapes:", data.shape, labels.shape)

    train_size = int(total_size * 0.7)
    indices = np.random.permutation(total_size)

    X_train, y_train = data[indices][:train_size], labels[indices][:train_size]
    X_test, y_test = data[indices][train_size:], labels[indices][train_size:]
    print("Train/test sets shapes:", X_train.shape, X_test.shape)

    accuracy_scores = np.zeros(5)
    f1_scores = np.zeros(5)
    precision_scores = np.zeros(5)
    recall_scores = np.zeros(5)
    scores = [accuracy_scores, f1_scores, precision_scores, recall_scores]
    metrics = [accuracy, F1, precision, recall]

    for k in range(1, 6):
        
        print('n_neihgbours = %d' % k)
        predictor = KNN(n_neighbours=k)

        predictor.fit(X_train, y_train)
        y_pred = predictor.predict(X_test)

        for score, metric in zip(scores, metrics):
            score[k-1] = metric(y_test, y_pred)
        print("Accuracy: %.4f [ours]" % accuracy(y_test, y_pred))
        assert abs(accuracy_score(y_test, y_pred) - accuracy(y_test, y_pred)) < 1e-5,\
            "Implemented accuracy is not the same as sci-kit learn one!"
        
        print("Precision: %.4f [ours]" % precision(y_test, y_pred))
        assert abs(precision(y_test, y_pred) - precision_score(y_test, y_pred)) < 1e-5,\
            "Implemented precision is not the same as sci-kit learn one!"
        
        print("Recall: %.4f [ours]" % recall(y_test, y_pred))
        assert abs(recall(y_test, y_pred) - recall_score(y_test, y_pred)) < 1e-5,\
            "Implemented recall is not the same as sci-kit learn one!"
        
        print("F1: %.4f [ours]" % F1(y_test, y_pred))
        assert abs(F1(y_test, y_pred) - f1_score(y_test, y_pred)) < 1e-5,\
            "Implemented F1 is not the same as sci-kit learn one!"
        
        assert accuracy_score(y_test, y_pred) > 19. / 29,\
            "Your classifier is worse than the constant !"

        print(classification_report(y_test, y_pred))
    

    predictor_1 = KNN(n_neighbours=1)
    predictor_1.fit(X_train, y_train)
    y_pred_1 = predictor_1.predict(X_test)

    predictor_4 = KNN(n_neighbours=4)
    predictor_4.fit(X_train, y_train)
    y_pred_4 = predictor_4.predict(X_test)



    fig1, ax1 = plt.subplots(1,1)

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('True Tags')

    scatter1_true = ax1.scatter(X_test[:, 0], X_test[:, 1],
                                s=10, c=y_test, cmap='bwr')
    
    scatter1_train = ax1.scatter(X_train[:, 0], X_train[:, 1], marker='x',
                                 s=10, c=y_train, cmap='bwr', alpha=0.5)
    
    legend1 = ax1.legend(handles = scatter1_true.legend_elements()[0] + scatter1_train.legend_elements()[0],
                         labels=['true 0', 'true 1', 'train 0', 'train 1'],
                         title="Classes")
    
    fig1.savefig('true_tags.jpeg', dpi = 350)



    fig2, ax2 = plt.subplots(1,1)

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Nearest Neighbour')

    scatter2_pred = ax2.scatter(X_test[:, 0], X_test[:, 1],
                                s=10, c = y_pred_1, cmap='bwr')
    
    scatter2_train = ax2.scatter(X_train[:, 0], X_train[:, 1],
                                 marker = 'x', s=10, c=y_train, cmap='bwr', alpha=0.5)
    
    legend2 = ax2.legend(handles = scatter2_pred.legend_elements()[0] + scatter2_train.legend_elements()[0],
                         labels=['predict 0', 'predict 1', 'train 0', 'train 1'],
                         title="Classes")
    
    fig2.savefig('1nn.jpeg', dpi = 350)



    fig3, ax3 = plt.subplots(1,1)

    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('4 Nearest Neighbours')

    scatter3_pred = ax3.scatter(X_test[:, 0], X_test[:, 1],
                                s=10, c = y_pred_4, cmap='bwr')
    
    scatter3_train = ax3.scatter(X_train[:, 0], X_train[:, 1],
                                 marker = 'x', s=10, c=y_train, cmap='bwr', alpha=0.5)
    
    legend3 = ax3.legend(handles = scatter3_pred.legend_elements()[0] + scatter3_train.legend_elements()[0],
                         labels=['predict 0', 'predict 1', 'train 0', 'train 1'],
                         title="Classes")
    
    fig3.savefig('4nn.jpeg', dpi = 350)



    fig4, ax4 = plt.subplots(1,1)

    ax4.set_xlabel('N Neighbours')
    ax4.set_ylabel('Metric Value')
    ax4.set_title('Metrics(N)')

    ax4.plot(np.arange(1, 6, 1), accuracy_scores, label = 'accuracy')
    ax4.plot(np.arange(1, 6, 1), f1_scores, label = 'f1')
    ax4.plot(np.arange(1, 6, 1), precision_scores, label = 'precision')
    ax4.plot(np.arange(1, 6, 1), recall_scores, label = 'recall')
    ax4.legend()

    fig4.savefig('scores.jpeg', dpi = 350)