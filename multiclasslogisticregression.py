import autograd.numpy as np
import pandas as pd
from autograd import grad
from matplotlib import cm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import sklearn
class LogisticRegression():
    def __init__(self, fit_intercept=True, k=2):

        self.fit_intercept = fit_intercept
        self.theta = None
        self.k = k
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
        if(self.k==2):
            self.theta = np.zeros(len(self.X[0]))
        else:
            self.theta = np.zeros(len(self.X[0])*self.k).reshape(len(self.X[0]),self.k)
        def J(theta):
            X_theta = np.matmul(X_,theta)
            p_ = np.exp(X_theta) 
            ans = 0
            for i in range(len(y_)):
                
            return ans
        
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

    def predict(self, X):
        '''
        Funtion to run the LogisticRegression on a data point
        :param X: pd.DataFrame with rows as samples and columns as features
        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        X = np.array(X)
        if(self.fit_intercept):
            X_ = np.concatenate([np.ones((len(X), 1)), X], axis=1) 
        X_theta = np.matmul(X_,self.theta)
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

data = datasets.load_digits()
X = data.data
y = data.target
X, y = shuffle(X, y)
kf = KFold(n_splits=4)
kf.get_n_splits(X)

i=1
su = 0
for train_index, test_index in kf.split(X):
    LR = LogisticRegression(fit_intercept=True, k=10)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    LR.fit_autograd(pd.DataFrame(X_train), pd.Series(y_train),5,500,0.05) 
    
    # y_hat = LR.predict(X_test)
    # print("accuracy for fold:",i,printAccuracy(y_test,y_hat))
    # su += printAccuracy(y_test,y_hat)
    # i+=1
print("Overall accuracy for logistic" ,su/3)
