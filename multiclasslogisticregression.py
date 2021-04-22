import autograd.numpy as np
import pandas as pd
from autograd import grad
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import sklearn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class LogisticRegression():
    def __init__(self, fit_intercept=True, k=2):

        self.fit_intercept = fit_intercept
        self.theta = None
        self.k = k
    def fit_autograd(self, X, y, batch_size, n_iter = 100, lr=0.03, lr_type='constant'):

        self.X = np.array(X)
        self.y = np.array(y)
        self.X = sklearn.preprocessing.normalize(self.X)
        X = self.X
        y = self.y
        self.theta = np.ones(len(self.X[0])*self.k).reshape(len(self.X[0]),self.k)

        def J(theta):
            X_theta = np.matmul(X_,theta)
            temp = np.exp(X_theta)
            temp_ = temp.sum(axis=1)
            temp = temp/temp_[:, np.newaxis]
            ans = 0
            for i in range(len(y_)):
                ans += np.log(temp[i,y_[i]])
            return -1*ans
        
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
                batch_base += batch_size         

    def fit(self, X, y, batch_size, n_iter = 100, lr=0.03, lr_type='constant'):

        self.X = np.array(X)
        self.y = np.array(y)
        self.X = sklearn.preprocessing.normalize(self.X)
        X = self.X
        y = self.y
        self.theta = np.ones(len(self.X[0])*self.k).reshape(len(self.X[0]),self.k)

        #     X_theta = np.matmul(X_,theta)
        #     temp = np.exp(X_theta)
        #     temp_ = temp.sum(axis=1)
        #     temp = temp/temp_[:, np.newaxis]
        #     ans = 0
        #     for i in range(len(y_)):
        #         ans += np.log(temp[i,y_[i]])
        #     return -1*ans
        
        # dj_dtheta = grad(J)

        for iter in range(n_iter):
            i = 0
            batch_base = 0
            while(batch_base<X.shape[0]):
                X_ = np.array(X[batch_base:min(batch_base+batch_size,X.shape[0])])    
                y_ = np.array(y[batch_base:min(batch_base+batch_size,X.shape[0])])    
                lr_ = lr
                if(lr_type != "constant"):
                    lr_ = lr / (iter+1)
                grad_theta = np.ones(len(self.X[0])*self.k).reshape(len(self.X[0]),self.k)
                X_theta = np.matmul(X_,theta)
                temp = np.exp(X_theta)
                temp_ = temp.sum(axis=1)
                temp = temp/temp_[:, np.newaxis]
                
                for i in range(len(y_)):
                    ans += np.log(temp[i,y_[i]])
                return -1*ans
                self.theta -= lr_*(dj_dtheta(self.theta))
                batch_base += batch_size         

    def predict(self, X):
        X = np.array(X)
        X = sklearn.preprocessing.normalize(X)
        X_theta = np.matmul(X,self.theta)
        temp = np.exp(X_theta)
        temp_ = temp.sum(axis=1)
        temp = temp/temp_[:, np.newaxis]
        return temp

def printAccuracy(y,y_hat):
    y_ = []
    for i in range(len(y_hat)):
        t=0
        for j in range(len(y_hat[0])):
            if(y_hat[i,j]>y_hat[i,t]):
                t = j
        y_.append(t)

    correct = 0
    for i in range(len(y)):
        if(y_[i] == y[i]):
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
LR = LogisticRegression(fit_intercept=True, k=10)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    LR.fit_autograd(pd.DataFrame(X_train), pd.Series(y_train),1,100,0.05) 
    y_hat = LR.predict(X_test)
    print("accuracy for fold:",i,printAccuracy(y_test,y_hat))
    su += printAccuracy(y_test,y_hat)
    i+=1
print("Overall accuracy for logistic" ,su/4)
print(X.shape)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, pd.Series(data = y)], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [i for i in range(10)]
colors = ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf[0] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
# print(principalComponents)
# print(y)
plt.show()