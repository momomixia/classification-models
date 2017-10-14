# Import python modules
import numpy as np
import kaggle
import time

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

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




    #decision tree train model
    def executeTrainDT(self, data, kfold, depthLst, fileTestOutputDT):
        trainX = data[0]
        trainY = data[1]
        testX = data[2]
        
        tree_para = {'criterion':['gini'],'max_depth':depthLst}
        clf = GridSearchCV(DecisionTreeClassifier(), tree_para, cv=kfold)
        clf.fit(trainX, trainY)
        meanTestAccuracy = clf.cv_results_['mean_test_score']
        
        bestPara = clf.best_estimator_
        print ("cvResult : ",  bestPara.max_depth)
        

        '''
        i = 0
        smallestError = 2^32              #out of sample error
        bestDepth = depthLst[0]
        
        for depth in depthLst:
            
            args = ("mse", "best", depth)         #mae takes long long time?   # {"criterion": "mae", "splitter": "best", "max_depth": depth} 
            averageMAE = self.modelSelectionCV(trainX, trainY, kfold, DecisionTreeRegressor, *args)

            print ("averageMAE cv MAE error DT: ", averageMAE)
            if averageMAE < smallestMAE:
                smallestError = averageMAE
                bestDepth = depth
        '''
            

    def predictDifferentModels(self):

        dataImage = self.read_image_data()
        
        print (" -----Begin decision tree classification CV--------")
        depthLst = [3, 6, 9, 12, 14]              #range(1, 20) try different alpha from test
        kfold = 5
        fileTestOutputDT  = "../Predictions/best_DT.csv"
        timeBegin = time.time()
        self.executeTrainDT(dataImage, kfold, depthLst, fileTestOutputDT)
        timeEnd = time.time()
        print ("time spent: ", timeEnd - timeBegin)


def main():
    
    classifyHwObj = classficationHw2()
    
    #for assigment querstion 1
    classifyHwObj.predictDifferentModels()
    
    
if __name__== "__main__":
  main()


    
'''
############################################################################
train_x, train_y, test_x = read_image_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

# Create dummy test output values to compute accuracy
test_y = np.ones(test_x.shape[0])
predicted_y = np.random.randint(0, 4, test_x.shape[0])
print('DUMMY Accuracy=%0.4f' % accuracy_score(test_y, predicted_y, normalize=True))

# Output file location
file_name = '../Predictions/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)
'''

