import numpy as np
from numpy.random import seed


class AdalineSGD(object):
    """An ADAptive LInear NEuron (ADALINE) classifier implementation with stochastic gradien descent

    Parameters:
        learn_rate (float): The learning rate between 0.0 and 1.0
        epochs (int): The number of epochs/iterations over the training set

    Attributes:
        w_ (1d-array): Weights after fitting
        errors_ (list): Number of misclassifications in every epoch.
        Shuffle (bool): Shuffles training data every epoch if True to prevent cycles
        random_state (int): Set randome state for shuffling and initializing weights
    """

    def __init__(self, learn_rate=0.01, epochs=50, shuffle=True, random_state=None):
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        """Fit the training data.

        Args:
            X (array-like shape = [n_samples, n_features]: Training vectors,
                where n_samples is the number of samples and n_features is the
                number of features.
            y (array-like, shape = [n_samples]: Target values

        Returns:
            self (object)
        """

        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.epochs):
            if self.shuffle:
                X, y = self._shuffle(X, y)
                cost = []
                for xi, target in zip(X, y):
                    cost.append(self._update_weights(xi, target))
                avg_cost = sum(cost)/len(y)
                self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing weights.
        """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to zeros"""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.learn_rate * xi.dot(error)
        self.w_[0] += self.learn_rate * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        """Calculate the net input
        
        Args:
            X: Training set
        
        Returns:
            
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
      """Compute linear activation
      """
      return self.net_input(X)

    def predict(self, X):
        """Predicts the class label after unit step

        - Called from the fit function to predict the class label for the weight update
        - Also used to predict class labels for new data
          
        Args:
            X:

        Returns:
              
        """
        return np.where(self.activation(X) >= 0.0, 1, -1)
