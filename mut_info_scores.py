def mutual_information(y_true, y_pred) -> float:
    """
    Calculates the mutual information score between true labels and predicted labels using sklearn's contingency_matrix.

    Parameters:
    y_true (ndarray): True labels.
    y_pred (ndarray): Predicted labels.

    Returns:
    float: Mutual information score.
    """
    from sklearn.metrics.cluster import contingency_matrix
    from numpy import log
    contingency = contingency_matrix(y_true, y_pred)
    ni = contingency.sum(axis=1)
    nj = contingency.sum(axis=0)
    N = contingency.sum()

    mi = 0.0
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            if contingency[i, j]: mi += (contingency[i, j] / N) * log((N * contingency[i, j]) / (ni[i] * nj[j]))
    return mi

def mutual_dif_information(y_true, predicted_probs) -> float:
    """
    Calculates the mutual differential information between true labels and predicted label probabilities.

    Parameters:
    y_true (ndarray of size K): True labels.
    y_pred (ndarray of size N * K): Predicted probabilities.

    Returns:
    float: Differential Mutual Information score.
    """
    from numpy import log, argmax
    from sklearn.metrics.cluster import contingency_matrix
    
    y_pred = argmax(predicted_probs, axis = 1)
    p_xy = contingency_matrix(y_true, y_pred)/9
    p_y = p_xy.sum(axis=1)
    p_x = predicted_probs.mean(axis = 0) 
    
    dmi = 0.0
    for y_label in range(p_xy.shape[0]):
        for x_label in range(p_xy.shape[1]):
            if p_xy[y_label][x_label]: dmi += p_xy[y_label][x_label] * log(p_xy[y_label][x_label] / (p_x[x_label] * p_y[y_label]))

    return dmi