import numpy as np
import glob
import os
from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt
from scipy import misc

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D,UpSampling2D, Dense, Flatten, Dropout
from keras.datasets import cifar10
from keras.utils import to_categorical
#from keras.applications.nasnet import NASNetMobile

from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

#model
def get_images(path, split=0.2, test_only=False):
    X_train = []
    X_test = []
    for directory in os.listdir(path):#tqdm(os.listdir(path),total=len(os.listdir(path)), unit=" labels"):
        if directory[0] != ".":
            total_len = len(os.listdir(os.path.join(path,directory)))
            #get X_test
            test_len = int(total_len * split)
            test_dir = glob.glob(os.path.join(path,directory,"*.png"))[-test_len:]
            for image_path in tqdm(test_dir,total=test_len, unit=" images"):
                image = misc.imread(image_path)
                X_test.append(image)
            if not test_only: #get X_train
                train_len = int(total_len * (1-split))
                train_dir = glob.glob(os.path.join(path,directory,"*.png"))[:train_len]
                for image_path in tqdm(train_dir,total=train_len, unit=" images"):
                    image = misc.imread(image_path)
                    X_train.append(image)
    X_test_red = np.array(X_test)[:,:,:,2].reshape(-1,217, 334, 1) #delete unnecessary color channels
    X_test_norm = X_test_red / 255. #get pixel values in range 0 to 1 -> Only values of 0 or 1 should appear
    if test_only:
        return X_test_norm.astype(np.float32) #float 32 saves 50% memory, 8 or 16 bit are not well supported though
    else:
        X_train_red = np.array(X_train)[:,:,:,2].reshape(-1,217, 334, 1) #delete unnecessary color channels
        X_train_norm = X_train_red / 255. #get pixel values in range 0 to 1 -> Only values of 0 or 1 should appear
        return X_train_norm.astype(np.float32), X_test_norm.astype(np.float32) #float 32 saves 50% memory, 8 or 16 bit are not well supported though

def get_labels(path,split=0.2, test_only=False):
    Y_train = []
    Y_test = []
    labelMap = {}
    labelNum = 0
    labelDir = os.listdir(path)
    for label in tqdm(labelDir, total=len(labelDir), unit=" labels"):
        if label[0] != ".":
            total_len = len(os.listdir(os.path.join(path,label)))
            train_len = int(total_len * (1-split))
            test_len = int(total_len * split)
            labelMap[labelNum] = label
            if not test_only:
                for image_path in glob.glob(os.path.join(path,label,"*.png"))[:train_len]:
                    Y_train.append(labelNum)
            for image_path in glob.glob(os.path.join(path,label,"*.png"))[-test_len:]:
                Y_test.append(labelNum)
            labelNum += 1
    if test_only:
        return to_categorical(np.array(Y_test)), labelMap
    else:
        return to_categorical(np.array(Y_train)), to_categorical(np.array(Y_test)), labelMap

def save_model(model,path):
        if os.path.exists(path):
            path = path[:-4]+str(int(path[-4])+1)+".h5"
            save_model(model,path)
        else:
            model.save(path)
            print("model saved")
            
#fake charts
def getFakePrices(originalPrices, n=100, sensitivity = 10):
    fakePrices = []
    sensitivity = sensitivity *1e-3
    for _ in range(n):
        fake = []
        differ = 0
        for price in originalPrices:
            differ = differ + np.random.randn()*sensitivity
            fakePrice = price + differ
            fake.append(fakePrice)
        fakePrices.append(fake)
    return fakePrices

def getExtraEntries(prices, n=10):
    newPrices = []
    for i in range(len(prices)-1):
        yStart = prices[i]
        yEnd = prices[i+1]
        yDiff = (yEnd - yStart) / n
        fillers = []
        price = yStart
        for _ in range(n):
            price += yDiff
            fillers.append(price)
        newPrices.extend(fillers)
    return newPrices

def generatePlots(patternPrices, patternName,directory,xEntries = 100, numberOfFakes = 100, sensitivity = 10):
    if not os.path.isdir(directory):
        print("create directory: "+directory)
        os.makedirs(directory)
    path = directory+"/"+patternName+"/"
    #create directory if it does not exist already
    if not os.path.isdir(path):
        print("create directory: "+path)
        os.makedirs(path)
    
    #plot original
    print("Original Plot")
    plt.plot(patternPrices)
    plt.xlim(0,len(patternPrices))
    plt.show()

    #get fakes
    fakePrices = getFakePrices(getExtraEntries(patternPrices,xEntries),numberOfFakes,sensitivity)
    #plot and save fakes
    print("First fake Plots")
    numberOfLastPlot = len(os.listdir(path))
    for i,fakePrice in tqdm(enumerate(fakePrices),total=numberOfFakes, unit=" plots"):        
        #save image without axis
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        plt.axis('off')
        plt.plot(fakePrice)
        plt.xlim(0,len(fakePrices[0]))
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(path+str(numberOfLastPlot+i)+".png", bbox_inches=extent)
        if i < 1:
            plt.show()
        plt.close()
    return True