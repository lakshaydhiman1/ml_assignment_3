import numpy as np
import pandas as pd


class LogisticRegression():
    def __init__(self, n , g, X ,y, type_,n_classes = 1):
        self.n = n
        self.g = g
        self.X = np.array(X)
        self.y = np.array(y)
        self.type_ = type_
        W = []
        B = []
        W.append(np.random.rand(n[0],self.X.shape[0]))
        for i in range(len(n)-1):
            W.append(np.random.rand(n[i+1],n[i]))
        W.append(np.random.rand(self.n_classes,n[-1]))
        for i in range(len(n)):
            B.append(np.random.rand(n[i]))

    def activation(z_,activation_type):
        if(activation_type == 'relu'):
            for i in range(z_.shape[0]):
                z_[i] = max(0,z_[i])
        elif(activation_type == 'sigmoid'):
            for i in range(z_.shape[0]):
                z_[i] = 1/(1+np.exp(-1*z_[i]))
        return z_

    def forwardpass(self, X):
        '''
        Funtion to run the LogisticRegression on a data point
        :param X: pd.DataFrame with rows as samples and columns as features
        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        a_ = np.array(X)
        for i in range(len(W)-1):
            z_ = np.matmul(W[i],a_) + B[i]
            a_ = activation(z_,g[i])

        return a_
    

        
        

