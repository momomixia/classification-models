import autograd.numpy as np
import autograd
from autograd.util import flatten
from plotting import plotNN
import time
from sklearn.model_selection import train_test_split
import kaggle


# Function to compute classification accuracy
def mean_zero_one_loss(weights, x, y_integers, unflatten):
	(W, b, V, c) = unflatten(weights)
	out = feedForward(W, b, V, c, x)
	pred = np.argmax(out, axis=1)
	return(np.mean(pred != y_integers))

# Feed forward output i.e. L = -O[y] + log(sum(exp(O[j])))
def feedForward(W, b, V, c, train_x):
    hid = np.tanh(np.dot(train_x, W) + b)
    out = np.dot(hid, V) + c
    return out

    
# Logistic Loss function
def logistic_loss_batch(weights, x, y, unflatten):
    # regularization penalty
    lambda_pen = 10

    # unflatten weights into W, b, V and c respectively 
    (W, b, V, c) = unflatten(weights)

    # Predict output for the entire train data
    out  = feedForward(W, b, V, c, x)
    pred = np.argmax(out, axis=1)

	    # True labels
    true = np.argmax(y, axis=1)
    # Mean accuracy
    class_err = np.mean(pred != true)

    # Computing logistic loss with l2 penalization
    logistic_loss = np.sum(-np.sum(out * y, axis=1) + np.log(np.sum(np.exp(out),axis=1))) + lambda_pen * np.sum(weights**2)
    #logistic_loss = np.sum(np.sum(np.log(1+np.exp(-1*out*y)), axis=1)) + lambda_pen * np.sum(weights**2)

    # returning loss. Note that outputs can only be returned in the below format
    return (logistic_loss, [autograd.util.getval(logistic_loss), autograd.util.getval(class_err)])


#read image data
def read_image_data():
    # Loading the dataset
    print('Reading image data ...')
    temp = np.load('../../Data/data_train.npz')
    train_x = temp['data_train']
    temp = np.load('../../Data/labels_train.npz')
    train_y_integers = temp['labels_train']
    temp = np.load('../../Data/data_test.npz')
    test_x = temp['data_test']
    return (train_x, train_y_integers, test_x)


#one time of forward propagtion for optimization
def trainNN(epsilon, momentum, train_x, train_y, train_y_integers, weights, unflatten, smooth_grad):
    
    # Batch compute the gradients (partial derivatives of the loss function w.r.t to all NN parameters)
    grad_fun = autograd.grad_and_aux(logistic_loss_batch)
    
    # Compute gradients (partial derivatives) using autograd toolbox
    weight_gradients, returned_values = grad_fun(weights, train_x, train_y, unflatten)
    #print('logistic loss: ', returned_values[0], 'Train error =', returned_values[1])
    
    # Update weight vector
    smooth_grad = (1 - momentum) * smooth_grad + momentum * weight_gradients
    weights = weights - epsilon * smooth_grad
    
    #print('Train accuracy =', 1-mean_zero_one_loss(weights, train_x, train_y_integers, unflatten))
    meanZeroOneLoss = mean_zero_one_loss(weights, train_x, train_y_integers, unflatten)
    
    grad_fun= logistic_loss_batch(weights, train_x, train_y, unflatten)
    meanLogisticloss = grad_fun[0]/train_x.shape[0]
    return smooth_grad, weights, meanLogisticloss, meanZeroOneLoss
    

#for train 1000 epochs with different NN models;    part 4 1 plot
def nnOneLayerTrainEntry():
    data = read_image_data()
    train_x = data[0]
    train_y_integers = data[1]
    test_x = data[2]

    # Make inputs approximately zero mean (improves optimization backprob algorithm in NN)
    train_x -= .5
    test_x  -= .5
    
    # Number of output dimensions
    dims_out = 4
    # Number of hidden units
    dims_hid_list = [5, 40, 70]       #5
    # Learning rate
    epsilon = 0.0001
    # Momentum of gradients update
    momentum = 0.1
    # Number of epochs
    nEpochs = 1000            #10
    # Number of train examples
    nTrainSamples = train_x.shape[0]
    # Number of input dimensions
    dims_in = train_x.shape[1]
    
    # Convert integer labels to one-hot vectors
    # i.e. convert label 2 to 0, 0, 1, 0
    train_y = np.zeros((nTrainSamples, dims_out))
    train_y[np.arange(nTrainSamples), train_y_integers] = 1
    
    print ("trainy shape: ", train_y.shape)
    
    assert momentum <= 1
    assert epsilon <= 1
    
    xnEpochsLst = range(1, nEpochs+1, 1)
    yLossLst = []
    for dims_hid in dims_hid_list:
        trainStart = time.time()*1000

        # Initializing weights
        W = np.random.randn(dims_in, dims_hid)
        b = np.random.randn(dims_hid)
        V = np.random.randn(dims_hid, dims_out)
        c = np.random.randn(dims_out)
        smooth_grad = 0
        # Compress all weights into one weight vector using autograd's flatten
        all_weights = (W, b, V, c)
        weights, unflatten = flatten(all_weights)
        yLossInns = []
        for epo in xnEpochsLst: #range(0, nEpochs):
            smooth_grad, weights, meanLogisticloss, meanZeroOneLoss = trainNN(epsilon, momentum, train_x, train_y, train_y_integers, weights, unflatten, smooth_grad)
            yLossInns.append(meanLogisticloss)
        yLossLst.append(yLossInns)
        #print ("YLossLsttttttt: ", yLossLst)
        print ( "NN time for different M: ", dims_hid, time.time()*1000 - trainStart)
    labels = [ "M = " + str(dims_hid) for dims_hid in dims_hid_list]
    #print('Train yLossInns =', xnEpochsLst, yLossLst)
    plotNN(xnEpochsLst, yLossLst, labels)
    
    
#test stage use beest model from trained model   
def testDataOutputFile(weights, test_x, unflatten, fileTestOutput):  
    (W, b, V, c) = unflatten(weights)
    out = feedForward(W, b, V, c, test_x)
    predY= np.argmax(out, axis=1)
    #output to file
    if fileTestOutput != "":
        kaggle.kaggleize(predY, fileTestOutput)    
    
#stratify once only to test validation set
def stratifyDataTrainTestNN():
    data = read_image_data()
    train_x = data[0]
    train_y_integers = data[1]
    test_x = data[2]
    
    # Make inputs approximately zero mean (improves optimization backprob algorithm in NN)
    train_x -= .5
    test_x  -= .5
    
    dims_out = 4
   
    xsplitTrain, xsplitTest, ysplitTrain_integer, ysplitTest_integer = train_test_split(train_x, train_y_integers, test_size=0.2, random_state=0, stratify=train_y_integers)
       
    dims_in = xsplitTrain.shape[1]
    nTrainSamples = xsplitTrain.shape[0]
    ysplitTrain = np.zeros((nTrainSamples, dims_out))
    ysplitTrain[np.arange(nTrainSamples), ysplitTrain_integer] = 1

    # Learning rate
    epsilon = 0.0001
    # Momentum of gradients update
    momentum = 0.1
    dims_hid_list = [5, 40, 70]
    nEpochs = 1000
    xnEpochsLst = range(1, nEpochs+1, 1)

    smallestValidationError = 2**32
    bestParas = []
    best_dims_hid = 0
    for dims_hid in dims_hid_list:

        # Initializing weights
        W = np.random.randn(dims_in, dims_hid)
        b = np.random.randn(dims_hid)
        V = np.random.randn(dims_hid, dims_out)
        c = np.random.randn(dims_out)
        smooth_grad = 0
        # Compress all weights into one weight vector using autograd's flatten
        all_weights = (W, b, V, c)
        weights, unflatten = flatten(all_weights)
        meanZeroOneLoss = 0

        for epo in xnEpochsLst: #range(0, nEpochs):
            smooth_grad, weights, meanLogisticloss, meanZeroOneLoss = trainNN(epsilon, momentum, xsplitTrain, ysplitTrain, ysplitTrain_integer, weights, unflatten, smooth_grad)
        
        
        #get validation data set zero-one-loss-error
        zeroOnelossEach = mean_zero_one_loss(weights, xsplitTest, ysplitTest_integer, unflatten)
        print ("zeroOnelossEach: ", zeroOnelossEach)
        if  zeroOnelossEach < smallestValidationError:
            smallestValidationError = zeroOnelossEach
            bestParas = [weights, unflatten, smooth_grad]
            best_dims_hid = dims_hid
    
    print ("smallestValidationError: ", smallestValidationError, "M = ",  best_dims_hid)
    #train whole data 
    nTrainSamples = train_x.shape[0]
    train_y = np.zeros((nTrainSamples, dims_out))
    train_y[np.arange(nTrainSamples), train_y_integers] = 1
    
    weights = bestParas[0]
    unflatten = bestParas[1]
    smooth_grad = bestParas[2]

    smooth_grad, weights, meanLogisticloss, meanZeroOneLoss = trainNN(epsilon, momentum, train_x, train_y, train_y_integers, weights, unflatten, smooth_grad)

    fileTestOutputNN = "../Predictions/best_NN2.csv"
    
    testDataOutputFile(weights, test_x, unflatten, fileTestOutputNN)



#main entry function
if __name__== "__main__":
    print ("begin train whole data and test plot and time ... for b. (1) (2)")
    nnOneLayerTrainEntry()               # for plot
    print ("begin stratified data traing and test for b. (3)")
    stratifyDataTrainTestNN()