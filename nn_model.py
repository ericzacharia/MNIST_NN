import pdb
import time
from tqdm.notebook import tqdm
import numpy as np

DATA_TYPE = np.float32
EPSILON = 1e-12


def xavier(shape, seed=None):
    n_in, n_out = shape
    if seed is not None:
        # set seed to fixed number (e.g. layer idx) for predictable results
        np.random.seed(seed)
    xavarian_matrix = [[np.random.uniform(-np.sqrt(6/(n_in+n_out)),
                                          np.sqrt(6/(n_in+n_out))) for j in range(n_out)] for i in range(n_in)]
    return np.array(xavarian_matrix, dtype=np.float32)


# InputValue: These are input values. They are leaves in the computational graph.
#              Hence we never compute the gradient wrt them.
class InputValue:
    def __init__(self, value=None):
        self.value = DATA_TYPE(value).copy()
        self.grad = None

    def set(self, value):
        self.value = DATA_TYPE(value).copy()


# Parameters: Class for weight and biases, the trainable parameters whose values need to be updated
class Param:
    def __init__(self, value):
        self.value = DATA_TYPE(value).copy()
        self.grad = DATA_TYPE(0)


class Add:  # Add with broadcasting
    '''
  Class name: Add
  Class usage: add two matrices a, b with broadcasting supported by numpy "+" operation.
  Class function:
      forward: calculate a + b with possible broadcasting
      backward: calculate derivative w.r.t to a and b
    '''

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.grad = None if a.grad is None and b.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = self.a.value + self.b.value

    def backward(self):
        if self.a.grad is not None:
            self.a.grad = self.a.grad + self.grad

        if self.b.grad is not None:
            self.b.grad = self.b.grad + self.grad


class Mul:  # Multiply with broadcasting
    '''
    Class Name: Mul
    Class Usage: elementwise multiplication with two matrix
    Class Functions:
    forward: compute the result a*b
    backward: compute the derivative w.r.t a and b
    '''

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.grad = None if a.grad is None and b.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = self.a.value * self.b.value

    def backward(self):
        if self.a.grad is not None:
            self.a.grad = self.a.grad + self.grad * self.b.value

        if self.b.grad is not None:
            self.b.grad = self.b.grad + self.grad * self.a.value


class VDot:  # Matrix multiply (fully-connected layer)
    '''
    Class Name: VDot
    Class Usage: matrix multiplication where a is a vector and b is a matrix
        b is expected to be a parameter and there is a convention that parameters come last.
        Typical usage is a is a feature vector with shape (f_dim, ), b a parameter with shape (f_dim, f_dim2).
    Class Functions:
        forward: compute the vector matrix multplication result
        backward: compute the derivative w.r.t a and b, where derivative of a and b are both matrices
    '''

    def __init__(self, a, b):
        self.a = a  # vector
        self.b = b  # matrix
        self.grad = None if a.grad is None and b.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = []
        for i in range(len(self.b.value[0])):
            summation = 0
            for j in range(len(self.b.value)):
                summation += self.a.value[j] * self.b.value[j][i]
            self.value.append(summation)

    def backward(self):
        if self.a.grad is not None:
            self.a.grad = []
            for i in range(len(self.b.value)):
                summation = 0
                for j in range(len(self.b.value[i])):
                    summation += self.b.value[i][j] * self.grad[j]
                self.a.grad.append(summation)
        self.a.grad = np.array(self.a.grad, dtype=np.float32)

        if self.b.grad is not None:
            self.b.grad = []
            for i in range(len(self.b.value)):
                grad_row = []
                for j in range(len(self.b.value[i])):
                    grad_row.append(self.grad[j] * self.a.value[i])
                self.b.grad.append(grad_row)
        self.b.grad = np.array(self.b.grad, dtype=np.float32)


class Sigmoid:
    '''
    Class Name: Sigmoid
    Class Usage: compute the elementwise sigmoid activation. Input is vector or matrix.
        In case of vector, [a_{0}, a_{1}, ..., a_{n}], output is vector [b_{0}, b_{1}, ..., b_{n}] where b_{i} = 1/(1 + exp(-a_{i}))
    Class Functions:
        forward: compute activation b_{i} for all i.
        backward: compute the derivative w.r.t input vector/matrix a
    '''

    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        if len(self.a.value.shape) == 1:  # if vector
            self.value = [1/(1+np.exp(-self.a.value[i]))
                          for i in range(len(self.a.value))]
        else:  # if matrix
            self.value = []
            for i in range(len(self.a.value)):
                subvalue = []
                for j in range(len(self.a.value[i])):
                    subvalue.append(1/(1+np.exp(-self.a.value[i][j])))
                self.value.append(subvalue)
        self.value = np.array(self.value, dtype=np.float32)

    def backward(self):
        if self.a.grad is not None:
            self.a.grad = []
            for i in range(len(self.a.value)):
                self.a.grad.append(
                    self.grad[i]*self.value[i]*(1-self.value[i]))
        self.a.grad = np.array(self.a.grad, dtype=np.float32)


class RELU:  # optional to compare to sigmoid function
    '''
    Class Name: RELU
    Class Usage: compute the elementwise RELU activation. Input is vector or matrix. In case of vector,
        [a_{0}, a_{1}, ..., a_{n}], output is vector [b_{0}, b_{1}, ..., b_{n}] where b_{i} = max(0, a_{i})
    Class Functions:
        forward: compute activation b_{i} for all i.
        backward: compute the derivative w.r.t input vector/matrix a
    '''

    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = np.maximum(self.a.value, np.zeros_like(self.a.value))

    def backward(self):
        if self.a.grad is not None:
            self.a.grad = np.maximum(self.grad, np.zeros_like(self.a.value))


class SoftMax:
    '''
    Class Name: SoftMax
    Class Usage: compute the softmax activation for each element in the matrix, normalization by each all elements
        in each batch (row). Specifically, input is matrix [a_{00}, a_{01}, ..., a_{0n}, ..., a_{b0}, a_{b1}, ..., a_{bn}],
        output is a matrix [p_{00}, p_{01}, ..., p_{0n},...,p_{b0},,,p_{bn} ] where p_{bi} = exp(a_{bi})/(exp(a_{b0}) + ... + exp(a_{bn}))
    Class Functions:
        forward: compute probability p_{bi} for all b, i.
        backward: compute the derivative w.r.t input matrix a
    '''

    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        summation = 0.0
        for i in range(len(self.a.value)):
            summation += np.exp(self.a.value[i])
        self.value = [np.exp(self.a.value[i]) /
                      summation for i in range(len(self.a.value))]
        self.value = np.array(self.value, dtype=np.float32)

    def backward(self):
        if self.a.grad is not None:
            yhat = self.value
            summations = []
            for i in range(len(yhat)):
                summation = 0
                for j in range(len(yhat)):
                    summation += self.grad[j]*yhat[j]*yhat[i]
                summations.append(summation)
            dytilde = [self.grad[i]*yhat[i]-summations[i]
                       for i in range(len(yhat))]
            self.a.grad = np.array(dytilde, dtype=np.float32)


class Log:  # Elementwise Log
    '''
    Class Name: Log
    Class Usage: compute the elementwise log(a) given a.
    Class Functions:
        forward: compute log(a)
        backward: compute the derivative w.r.t input vector a
    '''

    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = np.array([np.log(self.a.value[i])
                               for i in range(len(self.a.value))], dtype=np.float32)

    def backward(self):
        if self.a.grad is not None:
            self.a.grad = []
            for i in range(len(self.a.value)):
                self.a.grad.append(self.grad[i] / self.a.value[i])
        self.a.grad = np.array(self.a.grad, dtype=np.float32)


class Aref:
    '''
    Class Name: Aref
    Class Usage: get some specific entry in a matrix. a is the matrix with shape (batch_size, N) and idx is vector containing
        the entry index and a is differentiable.
    Class Functions:
        forward: compute a[batch_size, idx]
        backward: compute the derivative w.r.t input matrix a
    '''

    def __init__(self, a, idx):
        self.a = a
        self.idx = idx
        self.grad = None if a.grad is None else DATA_TYPE(0)

    def forward(self):
        xflat = self.a.value.reshape(-1)
        iflat = self.idx.value.reshape(-1)
        outer_dim = len(iflat)
        inner_dim = len(xflat) / outer_dim
        self.pick = np.int32(np.array(range(outer_dim)) * inner_dim + iflat)
        self.value = xflat[self.pick].reshape(self.idx.value.shape)

    def backward(self):
        if self.a.grad is not None:
            grad = np.zeros_like(self.a.value)
            gflat = grad.reshape(-1)
            gflat[self.pick] = self.grad.reshape(-1)
            self.a.grad = self.a.grad + grad


class Accuracy:
    '''
    Class Name: Accuracy
    Class Usage: check the predicted label is correct or not. a is the probability vector where each probability is
                for each class. idx is ground truth label.
    Class Functions:
        forward: find the label that has maximum probability and compare it with the ground truth label.
        backward: None
    '''

    def __init__(self, a, idx):
        self.a = a
        self.idx = idx
        self.grad = None
        self.value = None

    def forward(self):
        self.value = np.mean(
            np.argmax(self.a.value, axis=-1) == self.idx.value)

    def backward(self):
        pass


# Set of allowed/implemented activation functions
ACTIVATIONS = {'relu': RELU,
               'sigmoid': Sigmoid}


class NN:
    def __init__(self, nodes_array, activation):
        # assert nodes_array is a list of positive integers
        assert all(isinstance(item, int) and item > 0 for item in nodes_array)
        # assert activation is supported
        assert activation in ACTIVATIONS.keys()
        self.nodes_array = nodes_array
        self.activation = activation
        self.activation_func = ACTIVATIONS[self.activation]
        self.layer_number = len(nodes_array) - 1
        self.weights = []
        # list of trainable parameters
        self.params = []
        # list of computational graph
        self.components = []
        self.sample_placeholder = InputValue()
        self.label_placeholder = InputValue()
        self.pred_placeholder = None
        self.loss_placeholder = None
        self.accy_placeholder = None

    # helper function for creating a parameter and add it to the list of trainable parameters
    def nn_param(self, value):
        param = Param(value)
        self.params.append(param)
        return param

    # helper function for creating a unary operation object and add it to the computational graph
    def nn_unary_op(self, op, a):
        unary_op = op(a)
        print(
            f"Append <{unary_op.__class__.__name__}> to the computational graph")
        self.components.append(unary_op)
        return unary_op

    # helper function for creating a binary operation object and add it to the computational graph
    def nn_binary_op(self, op, a, b):
        binary_op = op(a, b)
        print(
            f"Append <{binary_op.__class__.__name__}> to the computational graph")
        self.components.append(binary_op)
        return binary_op

    def set_weights(self, weights):
        """
        :param weights: a list of tuples (matrices and vectors)
        :return:
        """
        weights = np.array(weights)
        # assert weights have the right shapes
        if len(weights) != self.layer_number:
            raise ValueError(
                f"You should provide weights for {self.layer_number} layers instead of {len(weights)}")
        for i, item in enumerate(weights):
            weight, bias = item
            if weight.shape != (self.nodes_array[i], self.nodes_array[i + 1]):
                raise ValueError(
                    f"The weight for the layer {i} should have shape ({self.nodes_array[i]}, {self.nodes_array[i + 1]}) instead of {weight.shape}")
            if bias.shape != (self.nodes_array[i + 1],):
                raise ValueError(
                    f"The bias for the layer {i} should have shape ({self.nodes_array[i + 1]}, ) instead of {bias.shape}")
        # reset params to empty list before setting new values
        self.params = []
        # add Param objects to the list of trainable paramters with specified values
        for item in weights:
            weight, bias = item
            weight = self.nn_param(weight)
            bias = self.nn_param(bias)

    def get_weights(self):
        weights = []
        # Extract weight values from the list of Params
        # Every other entry is a matrix of weight paramemters, alternating with bias vectors
        for i in range(0, len(self.params), 2):
            # print('grad', self.params[i].grad)
            weights.append((self.params[i].value, self.params[i+1].value))
        return weights

    def init_weights_with_xavier(self):
        xavier_weights = []
        for i in range(self.layer_number):
            shape = (self.nodes_array[i], self.nodes_array[i+1])
            seed = i
            w = xavier(shape, seed)
            b = np.random.random((self.nodes_array[i+1],)).astype(DATA_TYPE)
            xavier_weights.append((w, b))
        self.set_weights(xavier_weights)

    def build_computational_graph(self):
        if len(self.params) != self.layer_number*2:
            raise ValueError(
                "Trainable Parameters have not been initialized yet. Call init_weights_with_xavier first.")

        # Reset computational graph to empty list
        self.components = []

        prev_output = self.sample_placeholder
        for i in range(self.layer_number):
            prev_output = prev_output

        pred = prev_output
        return pred

    def cross_entropy_loss(self):
        label_prob = self.nn_binary_op(
            Aref, self.pred_placeholder, self.label_placeholder)
        log_prob = self.nn_unary_op(Log, label_prob)
        loss = self.nn_binary_op(Mul, log_prob, InputValue(-1))
        return loss

    def eval(self, X, y):
        if len(self.components) == 0:
            raise ValueError(
                "Computational graph not built yet. Call build_computational_graph first.")
        accuracy = 0.
        objective = 0.
        for k in range(len(y)):
            self.sample_placeholder.set(X[k])
            self.label_placeholder.set(y[k])
            self.forward()
            accuracy += self.accy_placeholder.value
            objective += self.loss_placeholder.value
        accuracy /= len(y)
        objective /= len(y)
        return accuracy, objective

    def fit(self, X, y, alpha, t):
        """
        Use the cross entropy loss.  The stochastic
        gradient descent should go through the examples in order, so
        that your output is deterministic and can be verified.
        :param X: an (m, n)-shaped numpy input matrix
        :param y: an (m,1)-shaped numpy output
        :param alpha: the learning rate
        :param t: the number of iterations
        :return:
        """
        # create sample and input placeholder
        self.pred_placeholder = self.build_computational_graph()
        self.loss_placeholder = self.cross_entropy_loss()
        self.accy_placeholder = self.nn_binary_op(
            Accuracy, self.pred_placeholder, self.label_placeholder)

        train_loss = []
        train_acc = []
        since = time.time()
        for epoch in range(t):
            for i in tqdm(range(X.shape[0])):
                # tqdm adds a progress bar
                for p in self.params:
                    p.grad = DATA_TYPE(0)
                for c in self.components:
                    if c.grad is not None:
                        c.grad = DATA_TYPE(0)
                self.sample_placeholder.set(X[i])
                self.label_placeholder.set(y[i])
            # evaluate on train set
            avg_acc, avg_loss = self.eval(X, y)
            print("Epoch %d: train loss = %.4f, accy = %.4f, [%.3f secs]" % (
                epoch, avg_loss, avg_acc, time.time()-since))
            train_loss.append(avg_loss)
            train_acc.append(avg_acc)
            since = time.time()

    def forward(self):
        for c in self.components:
            c.forward()

    def backward(self, loss):
        loss.grad = np.ones_like(loss.value)
        for c in self.components[::-1]:
            c.backward()

    # Optimization functions
    def sgd_update_parameter(self, lr):
        # update the parameter values in self.params
        for p in range(len(list(self.params.keys()))):
            self.params[0][p].value += lr * self.params[0][p].grad
            self.params[1][p].value += lr * self.params[1][p].grad
