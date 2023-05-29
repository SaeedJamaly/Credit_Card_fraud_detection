import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.mixture
import sklearn.model_selection
import sklearn.metrics
import sklearn.utils
import sklearn.preprocessing
import sklearn.svm
import sklearn.naive_bayes
import sklearn.linear_model

def printResults(model):
    train_acc = sklearn.metrics.accuracy_score(model.predict(X_train), y_train)
    test_acc = sklearn.metrics.accuracy_score(model.predict(X_test), y_test) 

    train_prec = sklearn.metrics.precision_score(model.predict(X_train), y_train)
    test_prec = sklearn.metrics.precision_score(model.predict(X_test), y_test) 

    train_recall = sklearn.metrics.recall_score(model.predict(X_train), y_train)
    test_recall = sklearn.metrics.recall_score(model.predict(X_test), y_test) 

    print("Training Accuracy with {} estimator: {} %".format(type(model).__name__, np.round(train_acc, 2)*100))
    print("Testing Accuracy with {} estimator: {} %".format(type(model).__name__, np.round(test_acc, 2)*100))

    print("Training Precision with {} estimator: {} %".format(type(model).__name__, np.round(train_prec, 2)*100))
    print("Testing Precision with {} estimator: {} %".format(type(model).__name__, np.round(test_prec, 2)*100))

    print("Training Recall with {} estimator: {} %".format(type(model).__name__, np.round(train_recall, 2)*100))
    print("Testing Recall with {} estimator: {} % \n".format(type(model).__name__, np.round(test_recall, 2)*100))

# load data 
path = './creditcard.csv'
data = pd.read_csv(path)
downsample = True # choose whether to use the whole dataset or a portion of the class 0 samples
downsample_multiplier = 1 # variable controlling how many class 0 samples we choose relative to class 1. 10 gives best result

# downsample class=0
if downsample:
    y_class1 = data[data["Class"]==1]
    y_class0 = data[data["Class"]==0]

    X_downsample = sklearn.utils.resample(y_class0, n_samples=downsample_multiplier*len(y_class1), random_state=0)
    data_downsample = pd.concat([X_downsample, y_class1])
    data = data_downsample.sample(frac=1) #shuffle rows

# Scale data to Standard mean=0, variance=1, then split into train and testing 80 - 20 respectively
X = data.iloc[:, 1: data.shape[1] - 1]
y = data.iloc[:, -1]
X = sklearn.preprocessing.StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.20, random_state=0)

# fit to GMM
# values obtained through testing @ Iymen Abdella
gmm = sklearn.mixture.GaussianMixture(n_components=2, random_state=0, verbose=1, tol=1e-05, init_params='k-means++')
gmm.fit(X_train, y_train)

# fit to SVM
# values obtained from cv @ Sepehr Seifpour
svc = sklearn.svm.SVC(C = 10, gamma = 0.01, random_state=0)
svc.fit(X_train, y_train)

# fit to Naive Bayes
# values obtained from cv @ Saeed JamaliFashi
gNB_clf = sklearn.naive_bayes.GaussianNB(var_smoothing=1.0)
gNB_clf.fit(X_train, y_train)

# fit to Logistic Regression
# values obtained from cv @ Maryam Hatami
lr = sklearn.linear_model.LogisticRegression(random_state = 0)
lr.fit(X_train, y_train)

# output train and test results
printResults(gmm)
printResults(svc)
printResults(gNB_clf)
printResults(lr)