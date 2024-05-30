import numpy as np

def ranked_probability_score(y_true, y_prob):
    '''
    Compute the ranked probability score

    params:
    ---
    y_true: the true class label
        shape: (n_samples, )
        ordered in 0, 1, 2, ..., K-1
    y_prob: the predicted probability of each class
        shape: (n_samples, n_classes)

    return
    ---
    rps : the ranked probability score
    '''
    n_samples = len(y_true)
    n_classes = y_prob.shape[1]

    # convert the true label into step-increased one-hot encoding
    # y_true[i, j] = P_i(Y <= j)
    y_true_onehot = np.zeros_like(y_prob)
    y_true_onehot[np.arange(n_samples), y_true] = 1.
    y_true_onehot = np.cumsum(y_true_onehot, axis=1)

    # covert the predicted probability into cumulative probability
    # y_prob[i, j] = P_i(Y <= j)
    y_prob = np.cumsum(y_prob, axis=1)

    # mean squared error
    rps = np.sum(np.square(y_true_onehot - y_prob))
    rps = rps / n_samples

    return rps
# =====================================================================================
