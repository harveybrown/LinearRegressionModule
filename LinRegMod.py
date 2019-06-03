#This Module will have 5 classes
#For Each class we will have methods/functions that will perform certain operations


# Gradient Descent

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
    
class Gradient_Descent:
    
    def __init__(self, N, epochs, eta):
        self.N = N
        self.epochs= epochs
        self.eta = eta
        print('N={} epochs={} eta={}')
            
    def grad_desc(N,epochs, eta):
        X = np.hstack((np.ones((N,1)),(np.random.rand(N,2)-0.5)*10))
        b = np.array([[1],[2],[3]])
        Y = X@b+np.random.randn(N,1)*5
        
    def plot(X,Y):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:,1],X[:,2],Y,s=4,c='k')
        fig.show()
        # normalization of data
        X[:,[1,2]] = (X[:,[1,2]]-X[:,[1,2]].min(axis=0))/(X[:,[1,2]].max(axis=0)-X[:,[1,2]].min(axis=0))
        print(X)
        # random weight initialization
        w = np.random.randn(3,1)
        ##### gradient descent
        # create empty variable to store errors in
        Errors = np.empty(epochs)
        
        # for loop until epochs
        for i in range(epochs):
            # Check error of the model now
            E = (Y-X@w).T@(Y-X@w)
            #print('Iteration = {}, Error = {}'.format(i+1,E))
            # store errors 
            Errors[i] = E
            # update model weights by stepping opposite to gradient
            grad = X.T@(X@w-Y)
            w -= eta*grad
            plt.plot(Errors)
            plt.xlabel('Iteration')
            plt.ylabel('Error')
            fig = plt.figure(figsize=(7,7))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X[:,1],X[:,2],Y,s=1,c='k')
            ax.scatter(X[:,1],X[:,2],X@w,s=4,c='r')
            fig.show()
    
########################################################       
#########################################################     
# Closed Form Solution for Regression

class OLSSolution:
    
    def weights(N):
        X = np.hstack((np.ones((N,1)), np.random.rand(N,2)*1000))
        b = np.array([[50.], [2.2], [3.7]])
        Y = X@b + np.random.randn(N,1)*N
        print('Dimensions of X={}, b={}, Y={}.format(X.shape, b.shape, Y.shape')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(X[:,1],X[:,2],Y,s=4,c='k')
        w = np.linalg.inv(X.T@X)@X.T@Y
        print('w={} \n\n whereas \n\n b={}'.format(w.flatten(), b.flatten()))
        
 

# Calculation of R Squared

class RSquared
    def Rsquared(N):
        X = np.hstack((np.ones((N,1)),np.random.rand(N,1)))
        b = np.array([[2],[3]])
        Y = X@b+np.random.randn(N,1)*0.1
        Y[210:215] = Y[210:215]+np.random.rand(5,1)*10
        w = np.linalg.inv(X.T@X)@X.T@Y
        plt.scatter(X[:,[1]],Y,s=2)
        plt.plot(X[:,[1]],X@w,'-r')
        plt.scatter(X[:,[1]],Y)
        Rsq = 1- ((Y-X@w).T@(Y-X@w))/((Y-Y.mean()).T@(Y-Y.mean()))        
        print('Rsq={}'.format(Rsq[0][0]))
        
    def L2(N):
        X = np.hstack((np.ones((N,1)),np.random.rand(N,1)))
        b = np.array([[2],[3]])
        Y = X@b+np.random.randn(N,1)*0.1
        Y[210:215] = Y[210:215]+np.random.rand(5,1)*10
        lamb = np.array([[5],[2]])
        wreg = np.linalg.inv(X.T@X+np.eye(b.shape[0])@lamb)@X.T@Y
        plt.scatter(X[:,[1]],Y,s=2)
        plt.plot(X[:,[1]],X@wreg,'-r')
        RsqReg = 1- ((Y-X@wreg).T@(Y-X@wreg))/((Y-Y.mean()).T@(Y-Y.mean()))
        print('Regularized Rsq={}'.format(RsqReg[0][0]))

# Calculation of the Sum of Squared Error

class SumSqdError
    
    def Sum_SquaredError(N):
        X = np.hstack((np.ones((N,1)),(np.random.rand(N,2)-0.5)*10))
        b = np.array([[50.], [2.2], [3.7]])
        Y = X@b + np.random.randn(N,1)*N
        SSE=(Y-X@b).T@(Y-X@b)
        print('The Total Sum of Squares Error is SSE={}')


# One-Hot Encoder

class OneHotEncoder:
    
    def Encode(col):
        cats = list(set(col.flatten()))
        out =col ==cats
        return out