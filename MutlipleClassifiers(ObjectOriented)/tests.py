import platform
import random
import pandas as pd


def dataTest(Data):
    datatest = Data()
    data = '../data/pima-indians-diabetes.csv'
    try:
        x_data,y_data = datatest.dataAllocation(data)
        print("dataAllocation Function Executed")
    except:
        print("Data not imported correctly")
    try:
        x_train, x_test, y_train, y_test = datatest.trainSets(x_data,y_data)
        print("trainSets Function Executed")
    except:
        print("Data not imported correctly")

def linearTest(Data,LinearRegression):
    dataset = Data()
    linear = LinearRegression()
    data = '../data/pima-indians-diabetes.csv'
    x_data,y_data = dataset.dataAllocation(data)
    x_train, x_test, y_train, y_test = dataset.trainSets(x_data,y_data)
    try:
        y_predict_train, y_predict_test = linear.linearClassifer(x_train,x_test, y_train)
        print("linearClassifer Function Executed")
    except:
        print("Failed to execute linearClassifer()")
    try:
        print("Linear Regression Train Accuracy: ", linear.lgTrainAccuracy(y_train,y_predict_train))
    except:
        print("Failed to execute lgTrainAccuracy()")
    try:
        print("Linear Regression Test Accuracy: ", linear.lgTestAccuracy(y_test,y_predict_test))
    except:
        print("Failed to execute lgTestAccuracy()")

def RandomForestTest(Data,RFClassifier):
    dataset = Data()
    rf = RFClassifier()
    data = '../data/pima-indians-diabetes.csv'
    x_data,y_data = dataset.dataAllocation(data)
    x_train, x_test, y_train, y_test = dataset.trainSets(x_data,y_data)
    try:
        rf_clf,y_predict_train, y_predict_test = rf.randomForestClassifier(x_train,x_test, y_train)
        print("randomForestClassifier Function Executed")
    except:
        print("Failed to execute randomForestClassifier()")
    try: 
       print("Random Forest Train Accuracy: ",rf.rfTrainAccuracy(y_train,y_predict_train))
    except:
        print("Failed to execute rfTrainAccuracy()")
    try: 
       print("Random Forest Test Accuracy: ",rf.rfTestAccuracy(y_test,y_predict_test))
    except:
        print("Failed to execute rfTrainAccuracy()")
    try: 
       print("Random Forest Feature Importance: ",rf.rfFeatureImportance(rf_clf))
    except:
        print("Failed to execute rfFeatureImportance()")
    try: 
        print("Random Forest Sorted Feature Importance: ",rf.sortedRFFeatureImportanceIndicies(rf_clf))
    except:
        print("Failed to execute sortedRFFeatureImportanceIndicies()")
    try: 
        gscv_rfc, gscv_rfc_fit = rf.hyperParameterTuning(rf_clf,x_train,y_train)
        print("HyperParameterTuning Function Executed")
    except:
        print("Failed to execute hyperParameterTuning()")
    try: 
        print("Random Forest Best Parameters: ",rf.bestParams(gscv_rfc))
    except:
        print("Failed to execute bestParams()")
    try: 
        print("Random Forest Best Score: ",rf.bestScore(gscv_rfc))
    except:
        print("Failed to execute bestScore()")

def SupportVectorMachineTest(Data,SupportVectorMachine):
    dataset = Data()
    svm = SupportVectorMachine()
    data = '../data/pima-indians-diabetes.csv'
    x_data,y_data = dataset.dataAllocation(data)
    x_train, x_test, y_train, y_test = dataset.trainSets(x_data,y_data)
    try: 
        scaled_x_train, scaled_x_test = svm.dataPreProcess(x_train,x_test)
        print("dataPreProcess Function Executed")
    except:
        print("Failed to execute dataPreProcess()")
    try: 
        y_predict_train,y_predict_test = svm.SVCClassifer(scaled_x_train,scaled_x_test, y_train)
        print("SVCClassifer Function Executed")
    except:
        print("Failed to execute SVCClassifer()")
    try: 
        print("Support Vector Machine Trian Accuracy: ",svm.SVCTrainAccuracy(y_train,y_predict_train))
    except:
        print("Failed to execute SVCTrainAccuracy()")
    try: 
        print("Support Vector Machine Test Accuracy: ",svm.SVCTestAccuracy(y_test,y_predict_test))
    except:
        print("Failed to execute SVCTestAccuracy()")
    try: 
        svm_cv, best_score = svm.SVMBestScore(scaled_x_train, y_train)
        print("Support Vector Machine Best Score: ", best_score)
    except:
        print("Failed to execute SVMBestScore()")
    try: 
        y_predict_train,y_predict_test = svm.SVCClassiferParam(svm_cv,scaled_x_train,scaled_x_test,y_train)
        print("SVCClassiferParam Function Executed")
    except:
        print("Failed to execute SVCClassiferParam()")
    try: 
        print("Support Vector Machine Trian Accuracy: ",svm.svcTrainAccuracy(y_train,y_predict_train))
    except:
        print("Failed to execute SVCTrainAccuracy()")
    try: 
        print("Support Vector Machine Test Accuracy: ",svm.svcTestAccuracy(y_test,y_predict_test))
    except:
        print("Failed to execute SVCTestAccuracy()")
    try: 
        print("Support Vector Machine Rank Test Score: ",svm.SVMRankTestScore(svm_cv))
    except:
        print("Failed to execute SVCTestAccuracy()")
    try: 
        print("Support Vector Machine Mean Test Score: ",svm.SVMMeanTestScore(svm_cv))
    except:
        print("Failed to execute SVCTestAccuracy()")


def PCATest(Data,PCAClassifer):
    dataset = Data()
    pc = PCAClassifer()
    data = '../data/pima-indians-diabetes.csv'
    x_data,y_data = dataset.dataAllocation(data)
    #x_train, x_test, y_train, y_test = dataset.trainSets(x_data,y_data)
    try: 
        pca = pc.pcaClassifer(x_data)
        print("pcaClassifer Function Executed")
    except:
        print("Failed to execute pcaClassifer()")
    try: 
        print("PCA Explained Variance Ratio: ",pc.pcaExplainedVarianceRatio(pca))
    except:
        print("Failed to execute pcaExplainedVarianceRatio()")
    try: 
        print("PCA Singular Values: ",pc.pcaSingularValues(pca))
    except:
        print("Failed to execute pcaSingularValues()")