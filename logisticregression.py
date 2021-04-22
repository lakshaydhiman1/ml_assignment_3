import autograd.numpy as np
import numpy as geek
import pandas as pd
from autograd import grad
from matplotlib import cm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import sklearn
import matplotlib.pyplot as plt
import matplotlib
class LogisticRegression():
    def __init__(self, fit_intercept=True, lambda_ = 0, fun_type = 0):

        self.fit_intercept = fit_intercept
        self.theta = None
        self.lambda_ = lambda_
        self.fun_type = fun_type
    def fit_autograd(self, X, y, batch_size, n_iter = 500, lr=0.03, lr_type='constant'):

        self.X = np.array(X)
        # print(self.X)
        self.y = np.array(y)
        self.X = sklearn.preprocessing.normalize(self.X)
        X = self.X
        y = self.y
        # print(self.X)
        if(self.fit_intercept):
            self.X = np.concatenate([np.ones((len(y), 1)), self.X], axis=1)
            X = np.concatenate([np.ones((len(y), 1)), np.array(X)], axis=1)
        if(self.fun_type == 1 or self.fun_type==2):
            self.theta = np.ones(len(self.X[0]))
        else:
            self.theta = np.zeros(len(self.X[0]))
        
        def J(theta):
            X_theta = -1*np.matmul(X_,theta)
            p_ = 1/(1+np.exp(X_theta)) 
            t1 = np.multiply(y_,np.log(p_))
            t2 = np.multiply(1-y_,np.log(1-p_))
            ans = -(1/y_.size)* (np.sum(t1+t2))
            if(self.fun_type == 0):
                return ans
            elif(self.fun_type == 1):
                return ans + self.lambda_ * np.sum(np.absolute(theta))
            else:
                return ans + self.lambda_ * np.sum(np.square(theta))
        
        dj_dtheta = grad(J)

        for iter in range(n_iter):
            i = 0
            batch_base = 0
            while(batch_base<X.shape[0]):
                X_ = np.array(X[batch_base:min(batch_base+batch_size,X.shape[0])])    
                y_ = np.array(y[batch_base:min(batch_base+batch_size,X.shape[0])])    
                lr_ = lr
                if(lr_type != "constant"):
                    lr_ = lr / (iter+1)
                self.theta -= lr_*(dj_dtheta(self.theta))
                # print("theta",self.theta)
                batch_base += batch_size 

    def fit(self, X, y, batch_size, n_iter = 500, lr=0.03, lr_type='constant'):

        self.X = np.array(X)
        self.y = np.array(y)
        self.X = sklearn.preprocessing.normalize(self.X)
        X = self.X
        y = self.y
        if(self.fit_intercept):
            self.X = np.concatenate([np.ones((len(y), 1)), self.X], axis=1)
            X = np.concatenate([np.ones((len(y), 1)), np.array(X)], axis=1)
        if(self.fun_type == 1 or self.fun_type==2):
            self.theta = np.ones(len(self.X[0]))
        else:
            self.theta = np.zeros(len(self.X[0]))
        
        for iter in range(n_iter):
            i = 0
            batch_base = 0
            while(batch_base<X.shape[0]):
                X_ = np.array(X[batch_base:min(batch_base+batch_size,X.shape[0])])    
                y_ = np.array(y[batch_base:min(batch_base+batch_size,X.shape[0])])    
                lr_ = lr
                if(lr_type != "constant"):
                    lr_ = lr / (iter+1)
                X_theta = -1*np.matmul(X_,self.theta)
                p_ = 1/(1+np.exp(X_theta)) 
                t1 = p_ - y_
                gradient = np.dot(X_.T, t1)
                self.theta -= lr_*(gradient)
                batch_base += batch_size 

    def predict(self, X):
        '''
        Funtion to run the LogisticRegression on a data point
        :param X: pd.DataFrame with rows as samples and columns as features
        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        X = np.array(X)
        X = sklearn.preprocessing.normalize(X)
        if(self.fit_intercept):
            X = np.concatenate([np.ones((len(X), 1)), X], axis=1) 
        X_theta = np.matmul(X,self.theta)
        p_ = 1 / (1 + np.exp(-X_theta))
        return pd.Series(p_)

def printAccuracy(y,y_hat):
    for i in range(len(y_hat)):
        if(y_hat[i]<0.5):
            y_hat[i] = 0
        else:
            y_hat[i] = 1

    correct = 0

    for i in range(len(y)):
        if(y_hat[i] == y[i]):
            correct+=1
    return correct*100/len(y)

data = datasets.load_breast_cancer()
X = data.data
y = data.target
X, y = shuffle(X, y)
kf = KFold(n_splits=3)
kf.get_n_splits(X)

print("======================Using Equation==========================")
i=1
su = 0
for train_index, test_index in kf.split(X):
    LR = LogisticRegression(fit_intercept=True)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    LR.fit(pd.DataFrame(X_train), pd.Series(y_train),5,500,0.05) 
    y_hat = LR.predict(X_test)
    print("accuracy for fold:",i,printAccuracy(y_test,y_hat))
    su += printAccuracy(y_test,y_hat)
    i+=1
    # print("theta value",LR.theta)
print("Overall accuracy for logistic" ,su/3)

print("======================Using Autograd==========================")
i=1
su = 0
for train_index, test_index in kf.split(X):
    LR = LogisticRegression(fit_intercept=True)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    LR.fit_autograd(pd.DataFrame(X_train), pd.Series(y_train),5,500,0.05) 
    y_hat = LR.predict(X_test)
    print("accuracy for fold:",i,printAccuracy(y_test,y_hat))
    su += printAccuracy(y_test,y_hat)
    i+=1
    # theta_1 = LR.theta[1]
    # theta_2 = LR.theta[2]
    # X_ = X_test[:,0:2]
    # y_ = y_test[:]
    # label = [0,1,2,3,0,1,2,3]
    # colors = ['red','green']
    # X__ = geek.linspace(0,100,num = 100)
    # y__ = (theta_2/theta_1) * X__
    # fig = plt.figure(figsize=(8,8))
    # plt.scatter(X_[:,0], X_[:,1], c = y_, cmap=matplotlib.colors.ListedColormap(colors))
    # plt.plot(X__,y__)
    # cb = plt.colorbar()
    # loc = np.arange(0,max(label),max(label)/float(len(colors)))
    # cb.set_ticks(loc)
    # cb.set_ticklabels(colors)
    # plt.show()

    # print("theta value",LR.theta)
print("Overall accuracy for logistic" ,su/3)


print("======================"+"Printing for L1 normalized"+"==========================")
lambdas = [0.0001,0.001,0.1,1,5,10,50,100,500,1000]
accuracy = []
thetas = []
for temp in lambdas:
    i=1
    su = 0
    for train_index, test_index in kf.split(X):
        LR = LogisticRegression(fit_intercept=True,lambda_ = temp, fun_type = 1)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        LR.fit_autograd(pd.DataFrame(X_train), pd.Series(y_train),5,100,0.005) 
        y_hat = LR.predict(X_test)
        su += printAccuracy(y_test,y_hat)
        i+=1
        thetas.append(LR.theta)
    accuracy.append(su/3)
print("lambdas", lambdas)
# print("Thetas",thetas)
print("Accuracies for L1 Norm",accuracy)
print("====================== Printing for L2 normalized ==========================")

accuracy_ = []
for temp in lambdas:
    i=1
    su = 0
    for train_index, test_index in kf.split(X):
        LR = LogisticRegression(fit_intercept=True,lambda_ = temp, fun_type = 2)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        LR.fit_autograd(pd.DataFrame(X_train), pd.Series(y_train),5,100,0.001) 
        y_hat = LR.predict(X_test)
        su += printAccuracy(y_test,y_hat)
        i+=1
    accuracy_.append(su/3)

print("Accuracies for L2 Norm",accuracy_)
