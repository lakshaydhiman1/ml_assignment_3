import autograd.numpy as np
import pandas as pd
from autograd import grad
from sklearn.datasets import load_boston
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sklearn
class NeuralNetwork():
    def __init__(self, n , g, X ,y, type_,n_classes):
        self.n = n
        self.g = g
        self.X = np.array(X)
        self.y = np.array(y)
        self.type_ = type_
        self.n_classes = n_classes
        self.W = []
        self.B = []
        self.W.append(np.ones(n[0]*self.X.shape[1]).reshape(n[0],self.X.shape[1]))
        for i in range(len(n)-1):
            self.W.append(np.ones(n[i+1]*n[i]).reshape(n[i+1],n[i]))
        self.W.append(np.ones(self.n_classes*n[-1]).reshape(self.n_classes,n[-1]))
        for i in range(len(n)):
            self.B.append(np.zeros(n[i]))
        self.B.append(np.zeros(n_classes))
    def activation(self,z_,activation_type):
        if(activation_type == 'relu'):
            return np.maximum(z_,0)
        elif(activation_type == 'sigmoid'):
            return 1/(1+np.exp(-1*z_))
        else:
            return z_

    def forwardpass(self, X):
        a_ = np.array(X)
        # print("A_",a_.shape)
        for i in range(len(self.W)):
            z_ = np.matmul(self.W[i],a_.T)
            for j in range(z_.shape[0]):
                for k  in range(z_.shape[1]):
                    z_[j,k] += self.B[i][j]
            a_ = self.activation(z_,self.g[i])
            a_ = a_.T
        if(self.type_ == 1):
            a_ = np.exp(a_)
            a_ = a_/a_.sum(axis = 0)
        return a_

    def backpass(self,X,y,lr = 0.00001):
        def J(W):
            a_ = np.array(X)
            for i in range(len(W)):
                z_ = np.matmul(W[i],a_.T)
                z_ += self.B[i]
                a_ = self.activation(z_,self.g[i])
                a_ = a_.T
            if(self.type_ == 1):
                a_ = np.exp(a_)
                a_ = a_/a_.sum(axis = 0)
            else:
                # print(np.square(a_- y))
                return np.square(a_- y)

        def J_(B):
            a_ = np.array(X)
            for i in range(len(self.W)):
                z_ = np.matmul(self.W[i],a_.T)
                z_ += B[i]
                a_ = self.activation(z_,self.g[i])
                a_ = a_.T
            if(self.type_ == 1):
                a_ = np.exp(a_)
                a_ = a_/a_.sum(axis = 0)
            else:
                # print(np.square(a_- y))
                return np.square(a_- y)
        dj_dw = grad(J)
        dj_db = grad(J_)
        W_grad = dj_dw(self.W)
        B_grad = dj_db(self.B)
        # print("WGRAD",W_grad)
        # print("BGRAD",B_grad)
        for i in range(len(self.W)):
            self.W[i] -= lr * W_grad[i] 
        for i in range(len(self.B)):
            self.B[i] -= lr * B_grad[i]
        # print("W",self.W)
        # print("B", self.B)
        

data = load_boston()
X = data.data
y = data.target
X = sklearn.preprocessing.normalize(X)
X, y = shuffle(X, y)
kf = KFold(n_splits=3)
kf.get_n_splits(X)
i=1
su = 0
for train_index, test_index in kf.split(X):
    Net = NeuralNetwork([2,5], ['relu','relu', 'relu'], X,y,0,1)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    n_iter = 100
    for i in range(n_iter):
        # print(i)
        for j in range(X_train.shape[0]):
            Net.backpass(X_train[j], y_train[j])
    y_hat = Net.forwardpass(X_test)
    print(y_hat.shape)
    print(y_test.shape)
    temp = 0
    for k in range(X_test.shape[0]):
        temp += np.sqrt(np.square(y_hat[k][0]-y_test[k])) 
    su += temp/(X_test.shape[0])
    print("Root Mean Square Error for a it", temp/(X_test.shape[0]))
print("Overall Mean square error for Network" ,su/3)