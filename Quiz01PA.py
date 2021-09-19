# IDS 575
# University of Illinois at Chicago
# Spring 2021
# quiz #01
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
from sklearn.datasets import load_breast_cancer


#Load the data
CancerDataset = load_breast_cancer()

# covert the dataset to a DataFrame
def getDataFrame(dataset):
  numData = dataset['target'].shape[0]
  newDataset = np.concatenate((dataset['data'], dataset['target'].reshape(numData, -1)), axis=1)
  newNames = np.append(dataset['feature_names'], ['target'])
  return pd.DataFrame(newDataset, columns=newNames)

DataFrame = getDataFrame(CancerDataset)

# Data split
from sklearn.model_selection import train_test_split
def splitData(df, size):
  X, y = df[df.columns[:-1]], df.target
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, test_size=X.shape[0] - size, random_state=0)
  return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = splitData(DataFrame, 400)
assert X_train.shape == (400, 30)
assert y_train.shape == (400, )

# Training
from sklearn.neighbors import KNeighborsClassifier
def trainKnn(X, y, k=1):
  model = KNeighborsClassifier(n_neighbors=k)
  model.fit(X, y)
  pred = model.predict(X)
  accuracy = sum(pred == y) / len(X)    
  return model, accuracy

# Test
def testKnn(model, X, y):
  pred = model.predict(X)
  accuracy = sum(pred == y) / len(X)
  return accuracy 

# Students' implementation
from collections import Counter
class MyKNeighborsClassifier:
  X_train = None
  y_train = None

  def __init__(self, n_neighbors):
    self.k = n_neighbors

  @staticmethod
  def distance(src, dst):
    ######################################################
    # TO-DO: Return the Euclidean distance. 
    
    return 
    ######################################################

  def fit(self, X, y):
    # Convert training data to numpy array.
    # There is nothing to do more for kNN as it avoids explicit generalization.
    self.X_train = np.array(X)
    self.y_train = np.array(y)    
    
  ## Predict the label for just one example.
  def predict_one(self, x):
    # Measure the distance to each of training data.
    # Then sort by increasing order of distances.
    distances = []
    for (i, x_train) in enumerate(self.X_train):      
      distances.append([i, self.distance(x, x_train)])      
    distances.sort(key=lambda element: element[1])

    ########################################################################
    # TO-DO: Extract the indexes of the examples in the k-Nearest Neighbors.    
    
    
    ########################################################################
    
    # Extract k target values corresponding to the example indexes in kNN.    
    targets = [self.y_train[i] for i in kNN]
    
    # Return the majority-voted target value.
    return Counter(targets).most_common(1)[0][0]
  
  ## Predict the labels for every example.
  def predict(self, X):    
    predictions = []
    for (i, x) in enumerate(np.array(X)):
      predictions.append(self.predict_one(x))
    return np.asarray(predictions)

# students' Train Knn
def myTrainKnn(X, y, k=1):
  model = MyKNeighborsClassifier(n_neighbors=k)
  model.fit(X, y)
  pred = model.predict(X)
  accuracy = sum(pred == y) / len(X)    
  return model, accuracy

def myTestKnn(model, X, y):
  pred = model.predict(X)
  accuracy = sum(pred == y) / len(X)
  return accuracy

# Use the main function to test your code when running it from a terminal
def main():
	# test the built-in KNeighborsClassifier
	for k in range(1, 20):
		Model_k, Acc_train = trainKnn(X_train, y_train, k)
		Acc_test = testKnn(Model_k, X_test, y_test)
		print('%d-NN --> training accuracy = %.4f  /  test accuracy = %.4f' % (k, Acc_train, Acc_test))
	
  # test the KNN implemented by yourself
	for k in range(1, 20):
		Model_k, Acc_train = myTrainKnn(X_train, y_train, k)
		Acc_test = myTestKnn(Model_k, X_test, y_test)
		print('%d-NN --> training accuracy = %.4f  /  test accuracy = %.4f' % (k, Acc_train, Acc_test))



################ Do not make any changes below this line ################
if __name__ == '__main__':
	main()