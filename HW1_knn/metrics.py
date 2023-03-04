import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    y_pred = y_pred.astype(int)
    y_true = y_true.astype(int)
    tp = np.size(y_pred[(y_true == y_pred) & (y_pred == 1)])
    fp = np.size(y_pred[(y_true != y_pred) & (y_pred == 1)])
    tn = np.size(y_pred[(y_true == y_pred) & (y_pred != 1)])
    fn = np.size(y_pred[(y_true != y_pred) & (y_pred != 1)])

    try:
        precision = tp/(tp+fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = tp/(tp+fn)
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = 2*(precision*recall)/(precision+recall)
    except ZeroDivisionError:
        f1 = 0

    return {'precision': precision, 'recall': recall,
            'f1': f1, 'accuracy': multiclass_accuracy(y_pred, y_true)}


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    accuracy = sum((y_pred.astype(int) == y_true.astype(int)))/len(y_true)
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    
    try:
        r2 = 1 - sum((y_pred - y_true)**2)/(sum((y_true - np.mean(y_true))**2))
    except ZeroDivisionError:
        r2 = 0
        
    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse = sum((y_pred - y_true)**2)/np.size(y_true)
    
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    mae = sum(abs(y_pred - y_true))/np.size(y_true)

    return mae
    