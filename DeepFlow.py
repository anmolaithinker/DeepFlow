from google.colab import files  
import zipfile

# Importing Modules
import numpy as np
import pandas as pd
import io
import os
import re
from keras.preprocessing import sequence
from keras.models import Sequential
import numpy as np
from keras.layers import Flatten,Masking
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation
from keras.layers.wrappers import TimeDistributed 
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from sklearn.cross_validation import train_test_split
from keras.layers import Merge ,  Bidirectional , Dense
from keras.backend import tf
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB




class GoogleColabHelper():

  def __init__(self):
    print 'Initializing Google Helper : '
    print 'Helper Functions That you can use : '
    print '---------------------------------------------------------------------'
    print 'To Upload Files from local system to colab : - UploadFilesFromLocal()'
    print '---------------------------------------------------------------------'
    print 'To Download Files                          : - downloadFile(path)'
    print '---------------------------------------------------------------------'
    print 'Getting Data From URL                      : - getDataFromUrl(url)'
    print '---------------------------------------------------------------------'
    print 'Unzip                                      : - Unzip(path , directory)'
    print '---------------------------------------------------------------------'
    print 'Download and Install Glove                 : - DownAndInstallGlove(directory)'
    print '---------------------------------------------------------------------'


  ## Uploading files from Local  
  def UploadFilesFromLocal(self):
    global files
    uploaded = files.upload()
    for fn in uploaded.keys():
      print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
    
  ## Downloading the file  
  def downloadFile(self,path):
    global files
    files.download(path)
  
  ## Getting data from URL
  def getDataFromUrl(self , url):
    print "Getting Data From Url"
    os.system('wget ' + str(url))
    print "Done !!"
    os.system('ls')
  
  ## Unzipping the file
  def Unzip(self , path , directory):
    zip_ref = zipfile.ZipFile(path, 'r')
    zip_ref.extractall(directory)
    zip_ref.close()  
    print('Done!!')
    os.system('ls')
    
  ## Download and Install Glove  
  def DownAndInstallGlove(self , directory):
    curr = !ls
    print("Current Directory : \n" + str(curr))
    print("---------------------------Downloading : ")
    !wget = 'http://nlp.stanford.edu/data/glove.6B.zip'
    print("---------------------------Extracting : ")
    zip_ref = zipfile.ZipFile('./glove.6B.zip', 'r')
    zip_ref.extractall(directory)
    zip_ref.close()
    print "Done !!!"
    os.system('ls')
    
    
    
#################################### Neural Network ####################################################    
    
class NN:
    def __init__(self):
      print 'Model Initialized :)' 
      self.model = Sequential()

    #input_dim: This is the size of the vocabulary in the text data. For example, if your data is integer encoded to values between 0-10, then the size of the vocabulary would be 11 words.
    #output_dim: This is the size of the vector space in which words will be embedded. It defines the size of the output vectors from this layer for each word. For example, it could be 32 or 100 or even larger. Test different values for your problem.
    #input_length: This is the length of input sequences, as you would define for any input layer of a Keras model. For example, if all of your input documents are comprised of 1000 words, this would be 1000.
    def EmbeddingLayer(self,input_dim,emb_dim,input_len = None):
        print('Input Dimension : ' + str(input_dim))
        print('Embedding Dimension : ' + str(emb_dim))
        print('Input Length : ' + str(input_len))
        self.model.add(Embedding(input_dim,emb_dim,input_length=input_len))
        print('Embedd Success')
        
    def Conv1D(self,n_filters,kernel,padding = 'valid'):
      self.model.add(Convolution1D(n_filters,kernel,padding = padding))
        
    # Flattening
    def Flatten(self):
      self.model.add(Flatten())
        
    # Input Layer
    def InputLayer(self,input_dim,output,activation):
      self.model.add(Dense(output, input_dim=input_dim,activation=activation,kernel_initializer='normal'))
      
      
    # Hidden Layer    
    def HiddenLayer(self,output,activation):
        self.model.add(Dense(output,activation=activation,kernel_initializer='normal'))
                    
    # Output Layer                   
    def Output(self,output,activation):
        self.model.add(Dense(output,activation=activation,kernel_initializer='normal'))
    
    def Compile(self,loss,opt,metr):
        self.model.compile(loss = loss , optimizer = opt , metrics = [metr]) 
        print("Successfully Compiled")
                        
    def modelSummary(self):
        print("Model Summary \n")
        print(self.model.summary())   
        
    def FitTrain(self , x_train , y_train , batch_size , epochs):
        self.model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs)
        
    def Predictions(self,x_test):
        return self.model.predict(x_test)
    
    def LSTM(self,output,return_sequences = False , drop_out = None):
        self.model.add(LSTM(output ,return_sequences = return_sequences , dropout = drop_out))
        
    def Tokenizer(self , max_features ,data,split = ' '):
        self.tokenizer = Tokenizer(num_words=max_features, split=split)
        self.tokenizer.fit_on_texts(data)
    
    def PadSequences(self,data):
        X = self.tokenizer.texts_to_sequences(data)
        #print ("Pad Seq. X : " + str(X))
        X = pad_sequences(X)
        return X    
      

########################## Machine Learning ###########################################      
      
def is_categorical(array_like):
  return array_like.dtype.name == 'object'  

class MachineLearning():
  
  def __init__(self,dataset):
    self.dataset = dataset
    print ('Functions : ')
    print '----------------------------------------------------'
    print 'For taking the updated dataset        : getDataSet()'
    print '----------------------------------------------------'
    print 'For Checking Null Percentage          : checkCounts()'
    print '----------------------------------------------------'
    print 'For checking Null Values              : checkNull()'
    print '----------------------------------------------------'
    print 'For Getting Out Unique Values         : uniqueValues(column)'
    print '----------------------------------------------------'
    print 'For drawing bargraph distribution     : drawDistribution(column)'
    print '----------------------------------------------------'
    print 'Label Encoding                        : LabelEncoder(X)'
    print '----------------------------------------------------'
    print 'Filling Null Values without Imputer   : FillNA(cols , filling_data)'
    print '----------------------------------------------------'
    print 'Filling Null Values with Imputer      : FillNAImputer(strategy , X)'
    print '----------------------------------------------------'
    print 'Train and Test Splitting              : traintestSplit(X , y)'
    print '----------------------------------------------------'
    print 'XGBoost                               : XGBoost(X_train , y_train)'
    print '----------------------------------------------------'
    print 'SVM                                   : SVM(X_train , y_train)'
    print '----------------------------------------------------'
    print 'GaussianNB                            : GaussianNB(X_train,y_train)'
    print '----------------------------------------------------'
    print 'CheckAccuracy                         : CheckAccuracy(X_test , y_test)'
    print '----------------------------------------------------'
    print 'ConfusianMatrix                       : ConfusianMatrix(y_test , y_pred)'
    
    
  
  def getDataSet(self):
    return self.dataset
  
  def checkCounts(self): 
    for i in self.dataset.columns:
      if(is_categorical(self.dataset[i])):   
        print 'Info of Column : ' + str(i)
        counts_without_null = self.dataset[i].value_counts().sum()
        null_counts = self.dataset[i].isnull().sum()
        print 'Counts Without Null : '
        print self.dataset[i].value_counts()
        print 'Null Counts  : '
        print null_counts
        print 'Null Percentage : ' + str(float((float(null_counts)/float(null_counts + counts_without_null)) * 100))
        print '------------------------------------------------------------'
      
  
  
  def checkNull(self):
    print self.dataset.isnull().sum()
    
    
  def uniqueValues(self , cols):
    print 'Unique Values of Column ' + str(cols) + ' :'
    print set(self.dataset[cols].values)
    print '-------------------------------------------'
    
    
  def drawDistribution(self , cols):
    print 'Drawing Distribution of Column ' + str(cols) + ' :'
    print  self.dataset[cols].value_counts().plot.bar()
    print '---------------------------------------------'
    
  def LabelEncoder(self , X):
    print 'Label Encoding Starts :'
    le = preprocessing.LabelEncoder()
    tranformed_x = le.fit_transform(X)
    print 'Classes : '
    print le.classes_
    print '---------------------------------------------'
    return tranformed_x
   
  
  def FillNA(self , cols , filling_data):
    self.dataset[cols] = self.dataset[cols].fillna(filling_data)
    print 'Done !!'
  
  def FillNAImputer(self , strategy , X):
    imputer = preprocessing.Imputer(strategy = strategy)
    X = imputer.fit_transform(X)
    print 'Done !'
    return X   
  
  
  def traintestSplit(self,X,y,test_size = 0.2 ,rs = 0):
    print 'Test Size : ' + str(test_size) + ' Random State : ' + str(rs)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    print 'Done!!'
    return (X_train , X_test , y_train , y_test)
  
  
  def XGBoost(self , X_train , y_train , max_depth = 3 , n_estimators = 100):
    classifier = XGBClassifier(max_depth = max_depth , n_estimators = n_estimators)
    classifier.fit(X_train,y_train)
    self.classifier = classifier
    print 'Done!!'
    return classifier
  
  def SVM(self,X_train,y_train,kernel,rs):
    self.classifier = SVC(kernel = kernel ,random_state = rs)
    self.classifier.fit(X_train,y_train)
    print 'Done !!'
    return self.classifier        
        
  def GaussianNB(self,X_train,y_train):
    self.classifier = GaussianNB()
    self.classifier.fit(X_train,y_train)
    print 'Done !!'
    return self.classifier
  
  def ConfusionMatrix(self,y_test,y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print ("Confusion Matrix : \n" + str(cm))
    total = cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]
    accurate = cm[0][0] + cm[1][1]
    print ("Accuracy : " + str(int(float(accurate/float(total))*100)))  
  
  
  def CheckAccuracy(self , X_test , y_test):
    y_pred = self.classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print cm
    print 'Accuracy : '
    correct = cm[0][0] + cm[1][1]
    total = cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0]
    print str(float((float(correct) / float(total)) * 100))
    print '--------------------------------------------------' 
    return y_pred
  
  
  def NeuralNetwork(self , X_Train , y_train ,X_Test,y_test, batch_size ,  epochs):
    n = NN()
    n.InputLayer(15 , 10 , 'relu')
    n.HiddenLayer(5 , 'relu')
    n.Output(1 , 'sigmoid')
    n.Compile('binary_crossentropy' , 'adam' , 'accuracy')
    n.modelSummary()
    print 'Batch Size : ' + str(batch_size) + ' Epochs : ' + str(epochs)
    n.FitTrain(X_train,y_train,batch_size,epochs)
    y_pred = n.Predictions(X_Test) > 0.5
    self.ConfusionMatrix(y_test , y_pred)
    return (n,y_pred)      
