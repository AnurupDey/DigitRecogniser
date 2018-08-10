from test import *

n1 = (28*28)   # number of input nodes
n2 = 25        # number of nodes in layer 2
n3 = 20        # number of nodes in layer 3 
n4 = 10        # number of output nodes

eps12 = np.sqrt(6)/np.sqrt(n1+n2)
eps23 = np.sqrt(6)/np.sqrt(n2+n3)
eps34 = np.sqrt(6)/np.sqrt(n3+n4)
# wieghts from input (layer 1) to layer 2 
w12 = np.random.normal(-eps12,eps12,(n2,n1+1))   
# wieghts from layer 2 to output layer (layer 3)
w23 = np.random.normal(-eps23,eps23,(n3,n2+1))   
# from layer 3 to 4 (output)
w34 = np.random.normal(-eps34,eps34,(n4,n3+1))   
# aggregate all the wieghts

theta = load_weights() 
#theta = [w12,w23,w34]


# X is going to be a n1 x n matrix, where n is the number of samples
X,l  = load("train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz")
theta = grad_descend(X,l,1,500,theta)

X,l = load("t10k-images-idx3-ubyte.gz","t10k-labels-idx1-ubyte.gz")
J,acc,grad = cost(theta,X,l,3)
print("avg cost on test: ", J,"\tacc: ", acc)

print("saving weights")

save_weights(theta)