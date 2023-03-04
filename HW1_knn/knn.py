import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    # seting the number of neighbors    
    def __init__(self, k=1):
        self.k = k
    
    # setting train vars and response var
    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        # running the selected algorithm
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        
        # for each element from the test sample we're calculating Manhattan distance to each train sample
        num_train_samples = self.train_X.shape[0]
        num_test_samples = X.shape[0]
        distances = np.zeros((num_test_samples, num_train_samples))
        for test_cell in range(num_test_samples):
            for train_cell in range(num_train_samples):
                distances[test_cell][train_cell] = sum(abs(self.train_X[train_cell] - X[test_cell]))
                
        return distances


    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        
        num_train_samples = self.train_X.shape[0]
        num_test_samples = X.shape[0]
        distances = np.zeros((num_test_samples, num_train_samples))
        for test_cell in range(num_test_samples):
            distances[test_cell] = np.sum(abs(self.train_X - X[test_cell]), axis=1)
        
        return distances


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        
        distances = np.abs(X[:, None] - self.train_X).sum(axis=-1)
        
        return distances


    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """

        n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = []
        # sorting array for each test --> indexes of the K smallest distanses --> 
        # --> check their classes in y_train --> selecting the most frequent --> True if =1 
        for test in range(n_test):
            indexes = np.argsort(distances[test])[:self.k]
            classes = self.train_y[indexes]
            most_freq_class = np.bincount(classes.astype(int)).argmax()
            prediction.append(int(most_freq_class)==1)
        
        return np.array(prediction).astype(int)


    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        n_test = distances.shape[0]
        #prediction = [np.bincount(self.train_y[np.argsort(distances[test])[-self.k : ]].astype(int)).argmax() for test in range(n_test)]
        # one liner didn't work I we can try the same as above:
        prediction = []
        # sorting array for each test --> indexes of the K smallest distanses --> 
        # --> check their classes in y_train --> selecting the most frequent --> True if =1 
        for test in range(n_test):
            indexes = np.argsort(distances[test])[:self.k]
            classes = self.train_y[indexes]
            most_freq_class = np.bincount(classes.astype(int)).argmax()
            prediction.append(str(most_freq_class))
        
        return np.array(prediction)