import numpy as np


class AdalineGD(object):
    """An ADAptive LInear NEuron (ADALINE) classifier implementation

    Parameters:
        learn_rate (float): The learning rate between 0.0 and 1.0
        epochs (int): The number of epochs/iterations over the training set

    Attributes:
        w_ (1d-array): Weights after fitting
        errors_ (list): Number of misclassifications in every epoch.
    """

    def __init__(self, learn_rate=0.01, epochs=50):
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
            self (object)
        """

        # initialize weights to zero vector R^(m+1), where m = num of dimensions, +1 for threshold
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.epochs):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.learn_rate * X.T.dot(errors)
            self.w_[0] += self.learn_rate * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

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
