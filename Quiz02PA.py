# IDS 575
# University of Illinois at Chicago
# Spring 2021
# quiz #02
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages (not specified in
# the assignment) then you need prior approval from course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================

import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

class MyLinearRegression:  
    theta = None

    def fit(self, X, y, option, alpha, epoch):
        X = np.concatenate((np.array(X), np.ones((X.shape[0], 1), dtype=np.float64)), axis=1)
        y = np.array(y)       
        if option.lower() in ['bgd', 'gd']:
          # Run batch gradient descent.
          self.theta = self.batchGradientDescent(X, y, alpha, epoch)      
        elif option.lower() in ['sgd']:
          # Run stochastic gradient descent.
          self.theta = self.stocGradientDescent(X, y, alpha, epoch)
        else:
          # Run solving the normal equation.      
          self.theta = self.normalEquation(X, y)

    def predict(self, X):
        X = np.concatenate((np.array(X), np.ones((X.shape[0], 1), dtype=np.float64)), axis=1)
        if isinstance(self.theta, np.ndarray):
          # TO-DO: ############################################# 
            y_pred = np.dot(X,self.theta)
          ######################################################
            return y_pred
        return None

    def batchGradientDescent(self, X, y, alpha=0.00001, epoch=100000):
        (m, n) = X.shape     
        theta = np.zeros((n, 1), dtype=np.float64)
        temp = np.zeros((n, 1), dtype=np.float64)
        initial = mean_squared_error(y, X.dot(theta))
        for iter in range(epoch):
            if (iter % 1000) == 0:
                print('- currently at %d epoch...' % iter)    
            for j in range(n):
            # TO-DO: ############################################# 
                sum1=[0.0]
                sum1 = sum([((np.dot(X[i], theta)-y[i])* X[i][j]) for i in range(m)])
                theta[j] = theta[j] - ((alpha*2.0/m) * sum1)
            ######################################################
        return theta

    def stocGradientDescent(self, X, y, alpha=0.000001, epoch=10000):
        (m, n) = X.shape
        theta = np.zeros((n, 1), dtype=np.float64)
        for iter in range(epoch):
            if (iter % 100) == 0:
                print('- currently at %d epoch...' % iter)
            for i in range(m):
                for j in range(n):
                    # TO-DO: ############################################# 
                    theta[j] = theta[j] - alpha*((X[i].dot(theta)) - y[i])*X[i][j]
                      ######################################################    
        return theta

    def normalEquation(self, X, y):
    # TO-DO: ############################################# 
        theta = np.linalg.inv((X.T.dot(X))).dot(X.T).dot(y)
        ######################################################
        return theta

    @staticmethod
    def toyLinearRegression(df, feature_name, target_name, option, alpha, epoch):
        # This function performs a simple linear regression.
        # With a single feature (given by feature_name)
        # With a rescaling (for stability of test)
        x = rescaleVector(df[feature_name])
        y = rescaleVector(df[target_name])
        x_train = x.values.reshape(-1, 1)
        y_train = y.values.reshape(-1, 1)

        # Perform linear regression.    
        lr = MyLinearRegression()
        lr.fit(x_train, y_train, option, alpha, epoch)
        y_train_pred = lr.predict(x_train)

        # Return training error and (x_train, y_train, y_train_pred)
        return mean_squared_error(y_train, y_train_pred), (x_train, y_train, y_train_pred)



def getDataFrame(dataset):
    featureColumns = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    targetColumn = pd.DataFrame(dataset.target, columns=['Target'])
    return featureColumns.join(targetColumn)

def rescaleVector(x):
    min = x.min()
    max = x.max()
    return pd.Series([(element - min)/(max - min) for element in x])

def splitTrainTest(df, size):
    X, y = df.drop('Target', axis=1), df.Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, test_size=X.shape[0] - size, random_state=0)
    return (X_train, y_train), (X_test, y_test)

def toyLinearRegression(df, feature_name, target_name):
    # This function performs a simple linear regression.
    # With a single feature (given by feature_name)
    # With a rescaling (for stability of test)
    x = rescaleVector(df[feature_name])
    y = rescaleVector(df[target_name])
    x_train = x.values.reshape(-1, 1)
    y_train = y.values.reshape(-1, 1)

    # Perform linear regression.
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_train_pred = lr.predict(x_train)
  
    # Return training error and (x_train, y_train, y_train_pred)
    return mean_squared_error(y_train, y_train_pred), (x_train, y_train, y_train_pred)


def testYourCode(df, feature_name, target_name, option, alpha, epoch):
    trainError0, (x_train0, y_train0, y_train_pred0) = toyLinearRegression(df, feature_name, target_name)
    trainError1, (x_train1, y_train1, y_train_pred1) = MyLinearRegression.toyLinearRegression(df, feature_name, target_name, option, alpha, epoch)
    return trainError0, trainError1


# Use the main function to test your code when running it from a terminal
# output should be a list of floats
def main():
	HousingDataset = load_boston()
	DataFrame = getDataFrame(HousingDataset)
	Df = DataFrame[DataFrame.Target < 22.5328 + 2*9.1971]
	TrainError0, TrainError1 = testYourCode(Df, 'DIS', 'Target', option='sgd', alpha=0.001, epoch=500)
	print("Scikit's training error = %.6f / My training error = %.6f --> Difference = %.4f" % (TrainError0, TrainError1, np.abs(TrainError0 - TrainError1)))
	TrainError0, TrainError1 = testYourCode(Df, 'RM', 'Target', option='bgd', alpha=0.1, epoch=5000)
	print("Scikit's training error = %.6f / My training error = %.6f --> Difference = %.4f" % (TrainError0, TrainError1, np.abs(TrainError0 - TrainError1)))
    


#########################################################################################################
#Open question Q5:Play with different parameters alpha, epoch. 
#                 Describe your understanding about BGD vs SGD.
###########################################################################################################
# Please write your answer below as comment : 
# 
#In SGD, we are use the cost gradient of 1 example at each iteration, whereas in BGD we use the sum of the cost gradient of the given samples.
#
#
# In SGD (keeping aplha = 0.001 and epoch = 500, as the reference point), keeping the epoch constant , increase in aplha causes an increase in the training error. However, when epoch increases keeping the alpha same, there is no change in the training error. While decreasing the epoch upto 200 causes no change in the error, but if epoch is decreased to 100, 50 etc there is a slight increase in the error. 
#
#
#
#
#
#In BGD (keeping aplha = 0.1 and epoch 5000 as ref point), keeping the aplha constant , an increase or decrease causes no change in the error. However, when alpha decreases (eg 0.01 or 0.001), keeping epoch constant the error increases slightly (0.0012 or 0.0076 in this case). When alpha increases to a value like 0.5 or 0.8, keeping epoch constant, there is no change in the error. 
##########################################################################################################

################ Do not make any changes below this line ################
if __name__ == '__main__':
	main()