## Setup Functions for NeuralNet Regression

def relu(x):
  # applies relu function by returning whichever is bigger for each element in matrix: x or 0
  # if x is positive, x is greater and x is returned
  # if x is negative, 0 is greater and 0 is returned
  return np.maximum(0, x)

def d_relu(a):
  # derivative of relu
  # derivative is 1 if a > 1, 0 if a < 0
  return np.maximum(a>1,0)

def dZ(W,dZ,d_func,Z):
  # Returns derivative of Z in that layer
  # W parameter is the W of the layer to the right
  # dZ parameter is the derivative of Z in the layer to the right
  # Z parameter is the Z that you're finding the derivative of
  # d_func is the function calculating the derivative of that layer's activation function
  # So if you're trying to find dZ2:
  # W should be W3, dZ should be dZ3, Z should be Z2
  # d_func should be function for the derivative of activation function of layer 2
  return np.dot(W.T, dZ) * d_func(Z)

def dW(dZ,A,m):
  # Returns derivative of W in that layer
  # dZ is derivative of the same layer's Z
  # A is the activations of the previous layer (to the left)
  # m is number of training examples
  return np.dot(dZ, A.T)/m

def db(dZ,m):
  #Returns derivative of b in that layer
  #dZ is derivative of current layer Z, m is number of training examples
  return np.sum(dZ,axis = 1,keepdims = True)/m

#equations for forward propagation
#Z1 = np.dot(W1, X_train) + b1
#A1 = relu(Z1)

