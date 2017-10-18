# Import python modules
import numpy as np
import kaggle
import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


class classficationHw2(object):
 
    def __init__(self):
      pass
  
    # Read in train and test data
    def read_image_data(self):
        print('Reading image data ...')
        temp = np.load('../../Data/data_train.npz')
        train_x = temp['data_train']
        temp = np.load('../../Data/labels_train.npz')
        train_y = temp['labels_train']
        temp = np.load('../../Data/data_test.npz')
        test_x = temp['data_test']
            
        print (" image data shape trainX, trainY, testX: ", train_x.shape, train_y.shape, test_x.shape)
        return (train_x, train_y, test_x)


   

    
    #decision tree train model use cv
    def executeTrainDT(self, data, kfold, depthLst, fileTestOutputDT):
        trainX = data[0]
        trainY = data[1]
        testX = data[2]
        
        tree_para = {'criterion':['gini'],'max_depth':depthLst}
        clf = GridSearchCV(DecisionTreeClassifier(), tree_para, cv=kfold, n_jobs=12)
        clf.fit(trainX, trainY)
        meanTestAccuracy = clf.cv_results_['mean_test_score']
        
        bestPara = clf.best_estimator_
        print ("DT cvResult : ",  bestPara.max_depth,  1.0 - meanTestAccuracy)
        
        kwargs = {'criterion':'gini', 'max_depth': bestPara.max_depth}
        predY = self.trainTestWholeData(trainX, trainY, testX, DecisionTreeClassifier, kwargs)
        #print ("predY DT: ", predY)
        #output to file
        if fileTestOutputDT != "":
            kaggle.kaggleize(predY, fileTestOutputDT)
  
        
        return (min(1.0 - meanTestAccuracy),  kfold, bestPara.max_depth)
    
    

   #decision tree train model use cv
    def executeTrainKNN(self, data, kfold, knnLst, fileTestOutputDT):
        trainX = data[0]        #[0:1000, : ]                #smaller first for debugging
        trainY = data[1]        #[0:1000]
        testX = data[2]
        
        knn_para = {'n_neighbors': knnLst}
        clf = GridSearchCV(KNeighborsClassifier(), knn_para, cv=kfold, n_jobs=12)
        clf.fit(trainX, trainY)
        meanTestAccuracy = clf.cv_results_['mean_test_score']
        
        bestPara = clf.best_estimator_
        print ("KNN cvResult : ",  bestPara.n_neighbors,  1.0 - meanTestAccuracy)
        
        kwargs = {'n_neighbors': bestPara.n_neighbors}
        predY = self.trainTestWholeData(trainX, trainY, testX, KNeighborsClassifier, kwargs)
        #print ("predY DT: ", predY)
        #output to file
        if fileTestOutputDT != "":
            kaggle.kaggleize(predY, fileTestOutputDT)
  
        
        #return (min(1.0 - meanTestAccuracy),  kfold, bestPara.n_neighbors)
    
    
     #logistic regression classifier to train model use cv
    def executeTrainLinearReg(self, data, kfold, alphaLst, fileTestOutputDT):
        trainX = data[0]
        trainY = data[1]
        testX = data[2]
        
        logReg_para = {'loss': ['hinge', 'log'], 'alpha': alphaLst}
        clf = GridSearchCV(linear_model.SGDClassifier(), logReg_para, cv=kfold, n_jobs=12)
        clf.fit(trainX, trainY)
        meanTestAccuracy = clf.cv_results_['mean_test_score']
        
        bestPara = clf.best_estimator_
        print ("logistic Regreesion cvResult : ", bestPara.loss,  bestPara.alpha,  1.0 - meanTestAccuracy)
        
        kwargs = {'loss':  'hinge', 'alpha': bestPara.alpha}
        predY = self.trainTestWholeData(trainX, trainY, testX, linear_model.SGDClassifier, kwargs)
        #print ("predY DT: ", predY)
        #output to file
        if fileTestOutputDT != "":
            kaggle.kaggleize(predY, fileTestOutputDT + 'hinge')
  
        kwargs = {'loss':  'log', 'alpha': bestPara.alpha}
        predY = self.trainTestWholeData(trainX, trainY, testX, linear_model.SGDClassifier, kwargs)
        #print ("predY DT: ", predY)
        #output to file
        if fileTestOutputDT != "":
            kaggle.kaggleize(predY, fileTestOutputDT + 'log')
            
            
        return (min(1.0 - meanTestAccuracy),  kfold, bestPara.alpha)
    
    # use whole train data to do train and then test
    def trainTestWholeData(self, trainX, trainY, testX, modelFunc, kwargs):
        model =  modelFunc(**kwargs)
        model.fit(trainX, trainY)
            
        #print ("parameter: ", neigh.get_params(deep=True))
        predY = model.predict(testX)
        
        return predY
    
    
    #use diefferent model to classify images
    def predictDifferentModels(self):

        dataImage = self.read_image_data()
        
        print (" -----Begin decision tree classification CV--------")
        depthLst = [3, 6, 9, 12, 14]              #range(1, 20) try different alpha from test
        kfold = 5
        fileTestOutputDT  = "../Predictions/best_DT.csv"
        timeBegin = time.time()
        self.executeTrainDT(dataImage, kfold, depthLst, fileTestOutputDT)
        timeEnd = time.time()
        print ("time spent on DT: ", timeEnd - timeBegin)


        print (" -----Begin knn classification CV--------")
        knnLst = [3, 5, 7, 9, 11]              #range(1, 20) try different alpha from test
        kfold = 5
        fileTestOutputKNN  = "../Predictions/best_KNN.csv"
        
        timeBegin = time.time()
        self.executeTrainKNN(dataImage, kfold, knnLst, fileTestOutputKNN)
        timeEnd = time.time()
        print ("time spent on KNN: ", timeEnd - timeBegin)

        print (" -----Begin LR classification CV--------")
        alphaLst = [1e-6, 1e-4, 1e-2, 1, 10]               #range(1, 20) try different alpha from test
        kfold = 5
        fileTestOutputLR  = "../Predictions/best_LR.csv"
        
        timeBegin = time.time()
        self.executeTrainLinearReg(dataImage, kfold, alphaLst, fileTestOutputLR)
        timeEnd = time.time()
        print ("time spent on linear regression classifer: ", timeEnd - timeBegin)
        

def main():
    
    classifyHwObj = classficationHw2()
    
    #for assigment querstion 1,2,3
    classifyHwObj.predictDifferentModels()
    
    
if __name__== "__main__":
  main()


