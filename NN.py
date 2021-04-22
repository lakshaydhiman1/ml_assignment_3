import autograd.numpy as np
import pandas as pd
from autograd import grad
from sklearn.datasets import load_boston
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

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
        self.W.append(np.random.rand(n[0],self.X.shape[1]))
        for i in range(len(n)-1):
            self.W.append(np.random.rand(n[i+1],n[i]))
        self.W.append(np.random.rand(self.n_classes,n[-1]))
        for i in range(len(n)):
            self.B.append(np.random.rand(n[i]))
        self.B.append(np.random.rand(n_classes))
    def activation(self,z_,activation_type):
        if(activation_type == 'relu'):
            return np.maximum(z_,0)
        elif(activation_type == 'sigmoid'):
            return 1/(1+np.exp(-1*z_))
        else:
            return z_

    def forwardpass(self, X):
        a_ = np.array(X)
        print("A_",a_.shape)
        for i in range(len(self.W)):
            z_ = np.matmul(self.W[i],a_.T)
            for j in range(z_.shape[0]):
                for k  in range(z_.shape[1]):
                    z_[j,k] += self.B[i][j]
            a_ = self.activation(z_,self.g[i])
            a_ = a_.T
            print("A_",a_.shape)
        if(self.type_ == 1):
            a_ = np.exp(a_)
            a_ = a_/a_.sum(axis = 0)
        print("A_",a_)
        return a_

    def backpass(self,X,y,lr = 0.01):
        W_grad = [[] for i in range(len(self.W))]
        B_grad = [[] for i in range(len(self.B))]
        for i in range(len(self.W)):
            def J(W_):
                a_ = np.array(X)
                print("Shape of A",a_.shape)
                for j in range(i):
                    print("J",j)
                    print("Shape of W ",self.W[j].shape)
                    z_ = np.matmul(self.W[j],a_.T)
                    print("Shape of Z",z_.shape)
                    for k in range(z_.shape[0]):
                        for l  in range(z_.shape[1]):
                            z_[k,l] += self.B[j][k]
                    a_ = self.activation(z_,self.g[j])
                    a_ = a_.T
                    print("Shape of A",a_.shape)
                z_ = np.matmul(W_,a_.T)
                for j in range(z_.shape[0]):
                    for k  in range(z_.shape[1]):
                        z_[j,k] += self.B[i][j]
                a_ = self.activation(z_,self.g[i])
                a_ = a_.T
                a_ = activation(z_,g[i])
                for j in range(i+1,len(W)):
                    z_ = np.matmul(self.W[j],a_.T)
                    # for k in range(z_.shape[0]):
                    #     for l  in range(z_.shape[1]):
                    temp = z_ + B[j]
                    a_ = self.activation(z_,self.g[j])
                    a_ = a_.T
                if(self.type_ == 1):
                    a_ = np.exp(a_)
                    a_ = a_/a_.sum(axis = 0)
                    ans = 0
                    for j in range(len(y)):
                        ans += np.log(a_[y[j]])
                    return -1*ans
                else:
                    return np.sum(np.square(a_ - y))
            def J_(B):
                a_ = np.array(X)
                for j in range(i):
                    z_ = np.matmul(self.W[j],a_) + self.B[j]
                    a_ = activation(z_,g[j])
                z_ = np.matmul(W[i],a_) + B
                a_ = activation(z_,g[i])
                for j in range(i+1,len(W)):
                    z_ = np.matmul(self.W[i],a_) + self.B[i]
                    a_ = activation(z_,g[i])
                if(self.type_ == 1):
                    a_ = np.exp(a_)
                    a_ = a_/a_.sum(axis = 0)
                    ans = 0
                    for j in range(len(y)):
                        ans += np.log(a_[y[j]])
                    return -1*ans
                else:
                    return np.sum(np.square(a_ - y))
            dj_dw = grad(J)
            dj_db = grad(J_)
            W_grad[i] = dj_dw(self.W[i])
            B_grad[i] = dj_db(self.B[i])
        for i in range(len(self.W)):
            W[i] -= lr * W_grad[i]
        
        for i in range(len(self.B)):
            B[i] -= lr * W_grad[i]
        

data = load_boston()
X = data.data
y = data.target
X, y = shuffle(X, y)
kf = KFold(n_splits=3)
kf.get_n_splits(X)
print("Shape of X",X.shape)
i=1
su = 0
for train_index, test_index in kf.split(X):
    Net = NeuralNetwork([5,5,5], ['relu','relu', 'relu','relu'], X,y,0,1)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # print(Net.forwardpass(X_test))
    Net.backpass(X_train, y_train)
    print("accuracy for fold:",i,printAccuracy(y_test,y_hat))
    su += printAccuracy(y_test,y_hat)
    i+=1
    # print("theta value",LR.theta)
print("Overall accuracy for logistic" ,su/3)