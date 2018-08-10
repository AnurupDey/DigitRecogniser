# import matplotlib.pyplot as plt
import numpy as np
import gzip
import struct
import array
import matplotlib.pyplot as plt

def sigmoid(X):
    return 1 / (1+np.exp(-X))

def forward_prop(theta,X):

    a = []
    # adding bias nodes to a1
    this_a = np.transpose(X)

    for l in range(0,len(theta)):
        a1 = np.vstack([np.ones(this_a.shape[1]),this_a])     
        a.append(a1)

        z = np.dot(theta[l],a1)                    
        this_a = sigmoid(z)                            
    a.append(this_a)

    return a

def cost(theta,x,y,lambd):
    # here x is the K x m matrix, the inputs 
    # y is the K x m matrix representing the desired outputs
    #   m   -> no. of samples
    #   n   -> no. of input nodes
    #   K   -> no. of classes
    L = len(theta)
    (m,n) = np.asarray(x).shape
    y = np.transpose(y)
    a = forward_prop(theta,x)
    hypothesis = a[L]

    reg = 0
    #regularisation
    for l in range(0,L):
        reg = reg + np.sum((theta[l])[:,1:]**2) # we do not consider the first row, as that is for bias
    reg = reg*lambd*0.5
    
    # cost is the K x m matrix with costs
    cost = np.multiply(y,np.log(hypothesis))+np.multiply((1-y),np.log((1-hypothesis)))
    J = (-(np.sum(cost))+reg)/m

    acc = 0.0
    for i in range(0,m):
        if np.argmax(hypothesis[:,i]) == np.argmax(y[:,i]):
            acc = acc + 1.0
    acc = (acc/m)*100.0

    # back propogation
    del4 = hypothesis - y                               # del4: n4 x m
    del3 = np.dot(np.transpose(theta[2]),del4)*(a[L-1]*(1-a[L-1]))            # del3: n3+1 x m;  each element del3 = (thata2.del4)*g
    del2 = np.dot(np.transpose(theta[1]),del3[1:,:])*(a[L-2]*(1-a[L-2]))      # del2: n2+1 x m;
   
    delta3 = np.dot(del4,np.transpose(a[L-1]))
    delta2 = np.dot(del3[1:,:],np.transpose(a[L-2]))
    delta1 = np.dot(del2[1:,:],np.transpose(a[L-3]))

    theta1_grad = (delta1 + lambd*(np.hstack([np.zeros((theta[0].shape[0],1)),theta[0][:,1:]])))/m 
    theta2_grad = (delta2 + lambd*(np.hstack([np.zeros((theta[1].shape[0],1)),theta[1][:,1:]])))/m
    theta3_grad = (delta3 + lambd*(np.hstack([np.zeros((theta[2].shape[0],1)),theta[2][:,1:]])))/m

    return J,acc,[theta1_grad,theta2_grad,theta3_grad]

def grad_descend(X,l,alpha,its,theta):
    for i in range(0,its):
            J,acc,grads = cost(theta,X,l,3)
            theta[0]    = theta[0] - alpha*grads[0]
            theta[1]    = theta[1] - alpha*grads[1]
            theta[2]    = theta[2] - alpha*grads[2]

            print("avg cost: ",J, "\tacc: ",acc)
    return theta

def load(images_file,labels_file):
    images = gzip.open(images_file,'rb')
    labels = gzip.open(labels_file,'rb')
    print("Loading Data...")
    data_magic_no       = struct.unpack('>I',images.read(4))[0]
    data_num_samples    = struct.unpack('>I',images.read(4))[0]
    data_no_rows        = struct.unpack('>I',images.read(4))[0]
    data_no_cols        = struct.unpack('>I',images.read(4))[0]

    labels_magic_no     = struct.unpack('>I',labels.read(4))[0]
    labels_no_samples   = struct.unpack('>I',labels.read(4))[0]

    l = np.zeros((labels_no_samples,10))
    X = np.zeros((data_num_samples,data_no_rows*data_no_cols))
    for k in range(0 , data_num_samples):
        label = struct.unpack('>B',labels.read(1))[0]
        l[k,label] = 1
        a = [] 
        for j in range(0,28*28):
            X[k,j] = (struct.unpack('>B',images.read(1))[0]/255.0)
    return X,l

def save_weights(theta):
    L = len(theta)
    np.savetxt("trained_wieghts0",np.asarray(theta[0]))
    np.savetxt("trained_wieghts1",np.asarray(theta[1]))
    np.savetxt("trained_wieghts2",np.asarray(theta[2]))

def load_weights():
    theta0 = np.loadtxt("trained_wieghts0")
    theta1 = np.loadtxt("trained_wieghts1")
    theta2 = np.loadtxt("trained_wieghts2")
    return [theta0,theta1,theta2]


