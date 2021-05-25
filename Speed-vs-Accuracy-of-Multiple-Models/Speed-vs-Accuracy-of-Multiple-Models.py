import numpy as np
import pandas as pd 


# from matplotlib.colors import ListedColormap
from sklearn.metrics import pairwise_distances
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from scipy.spatial.distance import pdist
import os
import time



########## LOADING DATASET ##########
import scipy.io as sio
data = sio.loadmat("../data/mnist_10digits.mat")
xtrain = data["xtrain"]
xtest = data["xtest"]
ytrain = data["ytrain"].reshape(-1,)
ytest = data["ytest"].reshape(-1,)

# down sampling and ntest to save time since constrained by local computer resources
ntest = ytest.shape[0]
ndownsample = 5000

xtrain = xtrain/255
xtest  = xtest/255


def logisitc_regression(xtrain,ytrain,xtest,ytest,show = False):

    """
    Logistic Regression Model 
    """
    
    # timing 
    time0 = time.time()
    
    # train the model
    clf = LogisticRegression(random_state=0,max_iter=200, solver='liblinear',multi_class='auto').fit(xtrain, ytrain)

    # performance evaluation
    y_pred_lr = clf.predict(xtest)
    acc_lr = sum(y_pred_lr==ytest)/ntest
    
    
    # time to train and test
    time1 = time.time()
    
    # getting confusion matrix
    conf_nn = confusion_matrix(ytest, y_pred_lr)
    
    print("\n")
    print("####### Logistic Regression Results #######\n")
    print('running time: ', round((time1-time0), 2), 'seconds\n')
    print("Confusion Matrix\n")
    print(conf_nn)
    print("\n")
    
    # visual of confusion matrix
    plt.imshow(conf_nn, cmap="hot")
    plt.colorbar()
    plt.title('Logistic Regression') 
    if show:
        plt.show()
        print("\n")
    
    # save image
    cwd = os.getcwd()
    output_path = os.path.join(cwd,'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # saves image to output folder
    fname = str('Logistic_Regression') + ".png"
    pout = os.path.join(output_path,fname)
    plt.savefig(pout)

    
    print("Additional Results\n")
    print(classification_report(ytest,y_pred_lr))
    print("\n")
    print('Accuracy:',acc_lr)
    print("\n")

    return round((time1-time0), 2) , acc_lr
    

def neural_networks(xtrain,ytrain,xtest,ytest,show = False):

    """
    Neural Network
    """
    
    # timing 
    time0 = time.time()
    
    # training model
    mlp = MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=500, random_state=0).fit(xtrain,ytrain)

    # performance evaluation
    y_pred_nn = mlp.predict(xtest)
    acc_nn = sum(y_pred_nn==ytest)/ntest
    
    # time to train and test
    time1 = time.time()
    
    # getting confusion matrix
    conf_nn = confusion_matrix(ytest, y_pred_nn)
    
    print("\n")
    print("####### Neural Network Results #######\n")
    print('running time: ', round((time1-time0), 2), 'seconds\n')
    print("Confusion Matrix\n")
    print(conf_nn)
    print("\n")
    
    # visual of confusion matrix
    plt.imshow(conf_nn, cmap="hot")
    plt.colorbar()
    plt.title('Neural Network') 
    if show:
        plt.show()
        print("\n")

    # save image
    cwd = os.getcwd()
    output_path = os.path.join(cwd,'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # saves image to output folder
    fname = str('Neural_Network') + ".png"
    pout = os.path.join(output_path,fname)
    plt.savefig(pout)
    
    print("Additional Results\n")
    print(classification_report(ytest,y_pred_nn))
    print("\n")

    return round((time1-time0), 2) , acc_nn

    


def knn(xtrain,ytrain,xtest,ytest,show = False):
    
    """
    KNN model
    """
    
    # timing 
    time0 = time.time()

    
    # training model
    clf = KNeighborsClassifier(n_neighbors = 3).fit(xtrain[0:ndownsample], ytrain[0:ndownsample])

    # performance evaluation
    y_pred_knn = clf.predict(xtest)
    acc_knn = sum(y_pred_knn==ytest)/ntest

    # time to train and test
    time1 = time.time()
    
    
    # getting confusion matrix
    conf_knn = confusion_matrix(ytest, y_pred_knn)
    
    print("\n")
    print("####### KNN Results #######\n")
    print('running time: ', round((time1-time0), 2), 'seconds\n')
    print("Confusion Matrix\n")
    print(conf_knn)
    print("\n")
    
    # visual of confusion matrix
    plt.imshow(conf_knn, cmap="hot")
    plt.colorbar()
    plt.title('KNN')
    if show: 
        plt.show()
        print("\n")
    
    # save image
    cwd = os.getcwd()
    output_path = os.path.join(cwd,'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # saves image to output folder
    fname = str('KNN') + ".png"
    pout = os.path.join(output_path,fname)
    plt.savefig(pout)
    
    print("Additional Results\n")
    print(classification_report(ytest,y_pred_knn))
    print("\n")

    return round((time1-time0), 2) , acc_knn
    


def kernel_svm(xtrain,ytrain,xtest,ytest,show = False):
    
    """
    Kernel SVM model
    """
    
    # timing 
    time0 = time.time()

    
    # getting pair_wise_distance for kernel 
    first = pdist(xtrain[0:ndownsample], 'sqeuclidean')
    median = np.median(first)
    sigma = (median/2)**(1/2)
  
    # training model
    sv = SVC(kernel='rbf', random_state=1,gamma = 'scale', C=sigma).fit(xtrain[0:ndownsample],ytrain[0:ndownsample])

    # performance evaluation
    y_pred_sv_ker = sv.predict(xtest)
    acc_sv_ker = sum(y_pred_sv_ker==ytest)/ntest

    # time to train and test
    time1 = time.time()
    
    
    # getting confusion matrix
    conf_sv_ker = confusion_matrix(ytest, y_pred_sv_ker)
    
    print("\n")
    print("####### Kernel SVM Results #######\n")
    print('running time: ', round((time1-time0), 2), 'seconds\n')
    print("Confusion Matrix\n")
    print(conf_sv_ker)
    print("\n")
    
    # visual of confusion matrix
    plt.imshow(conf_sv_ker, cmap="hot")
    plt.colorbar()
    plt.title('Kernel SVM') 
    if show: 
        plt.show()
        print("\n")
    
    # save image
    cwd = os.getcwd()
    output_path = os.path.join(cwd,'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # saves image to output folder
    fname = str('kernel_svm') + ".png"
    pout = os.path.join(output_path,fname)
    plt.savefig(pout)
    
    print("Additional Results\n")
    print(classification_report(ytest,y_pred_sv_ker))
    print("\n")

    return round((time1-time0), 2) , acc_sv_ker



def linear_svm(xtrain,ytrain,xtest,ytest,show = False):
    
    """
    Linear SVM model
    """
    
    # timing 
    time0 = time.time()
  
    # training model
    sv = SVC(kernel='linear', random_state=0).fit(xtrain[0:ndownsample],ytrain[0:ndownsample])

    # performance evaluation
    y_pred_sv = sv.predict(xtest)
    acc_sv = sum(y_pred_sv==ytest)/ntest

    # time to train and test
    time1 = time.time()
    
    
    # getting confusion matrix
    conf_sv = confusion_matrix(ytest, y_pred_sv)
    
    print("\n")
    print("####### Linear SVM Results #######\n")
    print('running time: ', round((time1-time0), 2), 'seconds\n')
    print("Confusion Matrix\n")
    print(y_pred_sv)
    print("\n")
    
    # visual of confusion matrix
    plt.imshow(conf_sv, cmap="hot")
    plt.colorbar()
    plt.title('Linear SVM') 
    if show: 
        plt.show()
        print("\n")
    
    # save image
    cwd = os.getcwd()
    output_path = os.path.join(cwd,'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # saves image to output folder
    fname = str('linear_svm') + ".png"
    pout = os.path.join(output_path,fname)
    plt.savefig(pout)
    print("Additional Results\n")
    print(classification_report(ytest,y_pred_sv))
    print("\n")


    return round((time1-time0), 2) , acc_sv
    


if __name__ == "__main__":
    
    time_lr, acc_lr = logisitc_regression(xtrain,ytrain,xtest,ytest)
    time_nn, acc_nn = neural_networks(xtrain,ytrain,xtest,ytest)
    time_knn, acc_knn = knn(xtrain,ytrain,xtest,ytest)
    time_sv_ker, acc_sv_ker = kernel_svm(xtrain,ytrain,xtest,ytest)
    time_sv, acc_sv = linear_svm(xtrain,ytrain,xtest,ytest)


    # compare results of models
    times = [time_lr,time_nn,time_knn,time_sv_ker,time_sv]
    results = [acc_lr,acc_nn,acc_knn,acc_sv_ker,acc_sv]
    models = ['Linear Regression','Neural Network','KNN','Kernel SVM','Linear SVM']
    df = pd.DataFrame(list(zip(models, times, results)),columns =['Models','Time', 'Accuracy'])

    df.index = list(df["Models"])
    df = df.drop(columns=['Models'])


    df[['Time']].plot.bar()
    plt.show()
    df[['Accuracy']].plot.bar(ylim = (.90,1))
    plt.show()

    print(df)

