import numpy as np


class Perceptron(object):
    """A perceptron classifier implementation

    Could be extended to use more than 2 classes by way of the One-vs-All technique
    
    Parameters:
        learn_rate (float): The learning rate between 0.0 and 1.0
        epochs (int): The number of epochs/iterations over the training set

    Attributes:
        w_ (1d-array): Weights after fitting
        errors_ (list): Number of misclassifications in every epoch.
    """

    def __init__(self, learn_rate=0.01, epochs=10):
        self.learn_rate = learn_rate
        self.epochs = epochs

    def fit(self, X, y):
        """Fit the training data.

        Args:
            X (array-like shape = [n_samples, n_features]: Training vectors,
                where n_samples is the number of samples and n_features is the
                number of features.
            y (array-like, shape = [n_samples]: Target values

        Returns:
            object
        """

        # initialize weights to zero vector R^(m+1), where m = num of dimensions, +1 for threshold
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            # loop over training set and update weights according to learning rule
            for xi, target in zip(X, y):
                update = self.learn_rate * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            # Collect number of misclassifications in each epoch
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate the net input
        
        Calculates the vector dot product (w^T)x
        
        Args:
            X: Training set
        
        Returns:
            
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Predicts the class label after unit step

        - Called from the fit function to predict the class label for the weight update
        - Also used to predict class labels for new data
          
        Args:
            X:

        Returns:
              
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
