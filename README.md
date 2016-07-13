# Perceptron-python

A playground for modeling Perceptron learners in python.

### History
* McCullock and Pitts neuron (1943)
  - first concept of a simplified brain cell
* Rosenblatt **Perceptron** (1957)
  - proposed an algorithm that would automatically learn the optimal weight coefficients
    that are then multiplied by the input features to determine if the neuron fires
* Adaptive Linear Neuron (Adaline) by Widrow and Hoff (1960)
  - illustrates the key concept of defining and minimizing cost functions
  - key difference with Perceptron is that weights are updated based on a linear activation function rather than a unit step
  - quantizer is used to predict class labels
  - weight update is calculated based on all samples in the training set, instead
    of incrementally after each step, => why its called "batch" gradient descent

* Stochastic Gradient Descent
  - fixed learning rate often replaced by adaptive learning rate that decreases over time

* if the two classes can be separated by a linear hyperplane, then the perceptron will converge
  - if not, a max number of epochs must be set if the classes cannot be perfectly separable by
    a linear decision boundary



### References
