
from google.colab import drive
drive.mount('/content/drive')


!ls "/content/drive/My Drive/ml_dataset"

import io
import sklearn.svm as sk
import urllib.request
from PIL import Image
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt 
from scipy import stats

import torch
import torchvision
from torchvision import transforms
import torch.utils.data as utils
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD


# urllib.request.urlretrieve("https://i.ytimg.com/vi/JxCqMLJbRnE/mqdefault.jpg","image.jpg")

file = pd.read_csv("drive/My Drive/ml_dataset/data.csv")

file = file.loc[:,['thumbnail','duration','dislikeCount','commentCount','likeCount','channel_ViewCount','channel_subscriberCount','channel_videoCount','No_of_tags','viewCount']]

file = file.sample(frac=1)

X = file.drop(['viewCount'], axis =1)
X = file.drop(['thumbnail'], axis = 1)
Y = file[file.columns[-1]]
X_urls = file[file.columns[0]]
X = np.array(X)
X_urls = np.array(X_urls)
Y = np.array(Y)

X = X[:10000,:]
X_urls = X_urls[:10000]
Y = Y[:10000]

# meanY = np.mean(Y)
# stdY = np.std(Y)

# delRows = []

# Y_temp = (Y-meanY)/stdY
# for i in range(len(Y)):
# 	if(abs(Y_temp[i])>3):
# 		delRows.append(i)

# Y = np.delete(Y,delRows,axis=0)
# X = np.delete(X,delRows,axis=0)


X = np.array(X, dtype=np.float64)
Y = np.array(Y, dtype=np.float64)

mean = np.mean(X,axis=0)
std = np.std(X,axis=0)
print(std)
meanLab = np.mean(Y,axis=0)
stdLab = np.std(Y,axis=0)

# print(mini)
# print(maxi)
# print(miniLab)
# print(maxiLab)
# X = X - mini
# X = X / (maxi-mini)

# Y = Y - miniLab
# Y = Y / (maxiLab-miniLab)
print(stdLab)
print(mean)
X = (X-mean)/std
Y = (Y-meanLab)/stdLab

# indices = np.where(X[:,6]<=4)
# X_new = X[indices]
# Y_new = Y[indices]

# indices = np.where(Y_new<=40)
# X_new = X_new[indices]
# Y_new = Y_new[indices]

# data = X
# data = np.vstack((data,Y))
delRows = []

# zscore = np.abs(stats.zscore(df))

for i in range(len(X)):
	for j in range(len(X[0])):
		if(abs(X[i,j])>4):
			delRows.append(i)
			break

X_new = np.delete(X,delRows,axis=0)
X_urls = np.delete(X_urls, delRows, axis = 0)
Y_new = np.delete(Y,delRows,axis=0)

print(X_new.shape)
print(Y_new.shape)


delRows = []

for i in range(len(Y_new)):
	if(abs(Y_new[i])>10):
		delRows.append(i)

X_new = np.delete(X_new,delRows,axis=0)
X_urls = np.delete(X_urls, delRows, axis = 0)
Y_new = np.delete(Y_new,delRows,axis=0)


index=0

trainData = []
print(X_urls)
for url in X_urls:
	try:
		urllib.request.urlretrieve(url,"image.jpg")
		im = np.asarray(Image.open("image.jpg").convert('L'))
		if (im.shape[0] == 180 and im.shape[1] == 320):
			print(index)
			print(im.shape)
			trainData.append(im)
			os.remove("image.jpg")
			index+=1
		else:
			X_new = np.delete(X_new,index,0)
			X_urls = np.delete(X_urls, index, 0)
			Y_new = np.delete(Y_new,index,0)
	except:
		X_new = np.delete(X_new,index,0)
		X_urls = np.delete(X_urls, index, 0)
		Y_new = np.delete(Y_new,index,0)

print(Y_new)
Y = np.copy(Y_new)
labels = np.copy(Y)
Y = (Y-np.min(Y))/(np.max(Y)-np.min(Y))
for i in range(len(Y)):
	# print(Y[i])
	if (Y[i] == 1.0):
		Y[i] = 0.9
	Y[i] = int(Y[i]*10)
	# if (Y[i] == 10):
	# 	Y[i] = 9
print(Y)
print(np.unique(Y))
trainLabels = []
for i in Y:
	trainLabels.append(np.array(i))
# print(im.shape)
# print(im)
# trainData = np.asarray(trainData)
# print(trainData.shape)
# print(Y.shape)
# X = X.ravel()
# Y = Y.ravel()
# trainData = np.concatenate((X,Y),axis=1)

print(Y)
tensor_x = torch.stack([torch.Tensor(np.array(i)) for i in trainData]) # transform to torch tensors
tensor_y = torch.stack([torch.Tensor(np.array(i)) for i in trainLabels])
tensor_x = torch.unsqueeze(tensor_x, dim = 1)
my_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
features = utils.DataLoader(my_dataset, batch_size=128)

# features = DataLoader(trainData, batch_size = 128)
# labels = DataLoader(Y, batch_size=128)
# print(len(features))
# print(len(labels))
# pix_val = list(im.getdata())
# print(len(pix_val))



# inp = np.delete(X, 0, 1)
# classifier = sk.SVR(kernel='rbf')

# classifier.fit(inp,labels)
# print(classifier.score(inp,labels))

# for i in range(0,10):
#   print(len(np.where(Y==i)[0]))
class ConvNet(Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.layer1 = Sequential(Conv2d(1,16,kernel_size=5,padding=2),ReLU(),
                                  MaxPool2d(2),Dropout(p=0.25))
        
        self.layer2 = Sequential(Conv2d(16,32,kernel_size=5,padding=2),ReLU(),
                                 MaxPool2d(2),Dropout(p=0.25))
        
        self.layer3 = Linear(45*80*32,100,Dropout(p=0.25))
        self.layer4 = Linear(100,10)
        
    def forward(self,X):
        global svm_features
        #print(1)
        outputs = self.layer1(X)
        #print(2)
        outputs = self.layer2(outputs)
        #print(3)
        outputs = outputs.reshape(outputs.shape[0],-1)
        #print(4)
        outputs = self.layer3(outputs)
        #print(5)
        outputs = self.layer4(outputs)
        #print(6)
        svm_features = outputs
        #print(7)
        return outputs
        
Network = ConvNet()
# device = torch.device("cuda")
# Network.to(device)
weight_optimization = Adam(Network.parameters(),lr=0.001)
criterion = CrossEntropyLoss()

print(Network)

total_step = len(features)
epochs = 10
trainingLoss = []
trainingAccuracy = []
testLoss = []
predictedLabels_train = []
trueLabels_train = []
SVM_features = []
outputs = None
print(len(features))

for epoch in range(epochs):
    total_loss = 0
    print(epoch)
    for i,(img,label) in enumerate(features):
      # print(type(label))
      print(i)
      img = Variable(img.float())
      label = Variable(label)
      label = label.long()
      outputs = Network(img)
      loss = criterion(outputs,label)
      total_loss += loss.data
      #trainingLoss.append(loss)
      
      weight_optimization.zero_grad()
      loss.backward()
      weight_optimization.step()

      if (i + 1) % (len(features)) == 0:
          print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, i + 1, total_step, total_loss))
          trainingLoss.append(total_loss)
          total_loss = 0


      # for s in svm_features:
      #   s = list(s)
      #   feature = []
      #   for s1 in s:
      #     feature.append(float(s1))
      #   SVM_features.append(np.array(feature).reshape(1,10))
      # print(np.array(SVM_features).shape)


    # SVM_features = np.array(SVM_features)


    # for j,(im,lab) in enumerate(featuresTest):
    #   im = Variable(im.float())
    #   lab = Variable(lab)
    #   outputs = Network(im)
    #   loss = criterion(outputs,lab)
    #   total_loss += loss.data
    #   #trainingLoss.append(loss)

    #   if (j + 1) % (len(featuresTest)) == 0:
    #       print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, i + 1, total_step, total_loss))
    #       testLoss.append(total_loss)
    #       total_loss = 0

data = "final_data"
path = F"/content/drive/My Drive/{data}" 
x = "final_x"
lab = "final_label"

# torch.save(X_new,F"/content/drive/My Drive/{x}")
# torch.save(labels,F"/content/drive/My Drive/{lab}")
# torch.save(features, path)

features = torch.load(F"/content/drive/My Drive/{data}")
X_new = torch.load(F"/content/drive/My Drive/{x}")
labels = torch.load(F"/content/drive/My Drive/{lab}")

import matplotlib.pyplot as plt
import numpy as np
trainingLoss = [33.5139,27.3183,22.7148,20.7330,18.3330,17.4438,17.2675,15.8269,13.9874]
epochs = np.arange(9)
plt.plot(epochs,trainingLoss)
plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")
plt.show()

# features = f2
predictedLabels_train = []
trueLabels_train = []
for i,(img,label) in enumerate(features):
  # print(type(label))
  print(i)
  img = Variable(img.float())
  label = Variable(label)
  label = label.long()
  outputs = Network(img)

  print(outputs.size())
  a,predictions = torch.max(outputs.data,1)
  print(predictions)
  for j in predictions.flatten():
    predictedLabels_train.append(j)
  for j in label.flatten():
    trueLabels_train.append(j)

from sklearn.model_selection import GridSearchCV 
import sklearn.svm as sk
# defining parameter range 
param_grid = { 'kernel': ['rbf','poly','sigmoid','linear']}  
  
classifier = GridSearchCV(sk.SVR(), param_grid) 
  
# fitting the model for grid search 
# grid.fit(X_train, y_train)

import sklearn.svm as sk
import sklearn
# import numpy as np
# file = pd.read_csv("drive/My Drive/ml_dataset/data.csv")

# file = file.loc[:,['duration','dislikeCount','commentCount','likeCount','channel_ViewCount','channel_subscriberCount','channel_videoCount','No_of_tags','viewCount']]

# file = file.sample(frac=1)

# X = file.drop(['viewCount'], axis =1)
# Y = file[file.columns[-1]]
# X = np.array(X)
# Y = np.array(Y)

# X = X[:10000,:]
# Y = Y[:10000]

# Y = np.delete(Y,delRows,axis=0)
# X = np.delete(X,delRows,axis=0)

#inp = np.delete(X, 0, 1)
thumbnail_predictions = np.asarray(predictedLabels_train)
#print(inp.shape)
#print(thumbnail_predictions.shape)
thumbnail_predictions = np.reshape(thumbnail_predictions,(X_new.shape[0],1))
F = np.concatenate((X_new,thumbnail_predictions),axis=1)
print(F.shape)
classifier = sk.SVR(kernel='rbf')

classifier.fit(F,labels)
print(classifier.score(F,labels))

from sklearn.ensemble import RandomForestRegressor

# param_grid = { 'max_depth': [3,4,5,6,7,8]}  
  
# regr = GridSearchCV(RandomForestRegressor(), param_grid) 
regr = RandomForestRegressor(max_depth = 6)
regr.fit(F,labels)

print("Random Forest, max-depth=5:")
print()
print("Accuracy on the Training Data:" + " " + str(regr.score(F,labels)))
# print(regr.best_estimator_)

print(classifier.best_estimator_)

import io
import sklearn.svm as sk
import urllib.request
from PIL import Image
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt 
from scipy import stats

import torch
import torchvision
from torchvision import transforms
import torch.utils.data as utils
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD


# urllib.request.urlretrieve("https://i.ytimg.com/vi/JxCqMLJbRnE/mqdefault.jpg","image.jpg")

file = pd.read_csv("drive/My Drive/ml_dataset/data.csv")

file = file.loc[:,['thumbnail','duration','dislikeCount','commentCount','likeCount','channel_ViewCount','channel_subscriberCount','channel_videoCount','No_of_tags','viewCount']]

file = file.sample(frac=1)

X = file.drop(['viewCount'], axis =1)
X = file.drop(['thumbnail'], axis = 1)
Y = file[file.columns[-1]]
X_urls = file[file.columns[0]]
X = np.array(X)
X_urls = np.array(X_urls)
Y = np.array(Y)

X = X[90000:94000,:]
X_urls = X_urls[90000:94000]
Y = Y[90000:94000]

# meanY = np.mean(Y)
# stdY = np.std(Y)

# delRows = []

# Y_temp = (Y-meanY)/stdY
# for i in range(len(Y)):
# 	if(abs(Y_temp[i])>3):
# 		delRows.append(i)

# Y = np.delete(Y,delRows,axis=0)
# X = np.delete(X,delRows,axis=0)


X = np.array(X, dtype=np.float64)
Y = np.array(Y, dtype=np.float64)

mean = np.mean(X,axis=0)
std = np.std(X,axis=0)
print(std)
meanLab = np.mean(Y,axis=0)
stdLab = np.std(Y,axis=0)

# print(mini)
# print(maxi)
# print(miniLab)
# print(maxiLab)
# X = X - mini
# X = X / (maxi-mini)

# Y = Y - miniLab
# Y = Y / (maxiLab-miniLab)
print(stdLab)
print(mean)
X = (X-mean)/std
Y = (Y-meanLab)/stdLab

# indices = np.where(X[:,6]<=4)
# X_new = X[indices]
# Y_new = Y[indices]

# indices = np.where(Y_new<=40)
# X_new = X_new[indices]
# Y_new = Y_new[indices]

# data = X
# data = np.vstack((data,Y))
delRows = []

# zscore = np.abs(stats.zscore(df))

for i in range(len(X)):
	for j in range(len(X[0])):
		if(abs(X[i,j])>4):
			delRows.append(i)
			break

X_new = np.delete(X,delRows,axis=0)
X_urls = np.delete(X_urls, delRows, axis = 0)
Y_new = np.delete(Y,delRows,axis=0)

print(X_new.shape)
print(Y_new.shape)


delRows = []

for i in range(len(Y_new)):
	if(abs(Y_new[i])>10):
		delRows.append(i)

X_new = np.delete(X_new,delRows,axis=0)
X_urls = np.delete(X_urls, delRows, axis = 0)
Y_new = np.delete(Y_new,delRows,axis=0)


index=0

trainData = []
print(X_urls)
for url in X_urls:
	try:
		urllib.request.urlretrieve(url,"image.jpg")
		im = np.asarray(Image.open("image.jpg").convert('L'))
		print(index)
		print(im.shape)
		trainData.append(im)
		os.remove("image.jpg")
		index+=1
	except:
		X_new = np.delete(X_new,index,0)
		X_urls = np.delete(X_urls, index, 0)
		Y_new = np.delete(Y_new,index,0)

print(Y_new)
Y = np.copy(Y_new)
labels = np.copy(Y)
Y = (Y-np.min(Y))/(np.max(Y)-np.min(Y))
for i in range(len(Y)):
	# print(Y[i])
	if (Y[i] == 1.0):
		Y[i] = 0.9
	Y[i] = int(Y[i]*10)
	# if (Y[i] == 10):
	# 	Y[i] = 9
print(Y)
print(np.unique(Y))
trainLabels = []
for i in Y:
	trainLabels.append(np.array(i))
# print(im.shape)
# print(im)
# trainData = np.asarray(trainData)
# print(trainData.shape)
# print(Y.shape)
# X = X.ravel()
# Y = Y.ravel()
# trainData = np.concatenate((X,Y),axis=1)

print(Y)
tensor_x = torch.stack([torch.Tensor(np.array(i)) for i in trainData]) # transform to torch tensors
tensor_y = torch.stack([torch.Tensor(np.array(i)) for i in trainLabels])
tensor_x = torch.unsqueeze(tensor_x, dim = 1)
my_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
features = utils.DataLoader(my_dataset, batch_size=128)

# features = DataLoader(trainData, batch_size = 128)
# labels = DataLoader(Y, batch_size=128)
# print(len(features))
# print(len(labels))
# pix_val = list(im.getdata())
# print(len(pix_val))



# inp = np.delete(X, 0, 1)
# classifier = sk.SVR(kernel='rbf')

# classifier.fit(inp,labels)
# print(classifier.score(inp,labels))

predictedLabels_test = []
trueLabels_test = []

for i,(img,label) in enumerate(features):
  # print(type(label))
  print(i)
  img = Variable(img.float())
  label = Variable(label)
  label = label.long()
  outputs = Network(img)

  print(outputs.size())
  a,predictions = torch.max(outputs.data,1)
  print(predictions)
  for j in predictions.flatten():
    predictedLabels_test.append(j)
  for j in label.flatten():
    trueLabels_test.append(j)

import sklearn.svm as sk
# file = pd.read_csv("drive/My Drive/ml_dataset/data.csv")

# file = file.loc[:,['duration','dislikeCount','commentCount','likeCount','channel_ViewCount','channel_subscriberCount','channel_videoCount','No_of_tags','viewCount']]

# file = file.sample(frac=1)

# X = file.drop(['viewCount'], axis =1)
# Y = file[file.columns[-1]]
# X = np.array(X)
# Y = np.array(Y)

# X = X[:10000,:]
# Y = Y[:10000]

# Y = np.delete(Y,delRows,axis=0)
# X = np.delete(X,delRows,axis=0)

#inp = np.delete(X, 0, 1)
thumbnail_predictions = np.asarray(predictedLabels_test)
#print(inp.shape)
#print(thumbnail_predictions.shape)

thumbnail_predictions = np.reshape(thumbnail_predictions,(X_new.shape[0],1))
F = np.concatenate((X_new,thumbnail_predictions),axis=1)
print(F.shape)

print(classifier.score(F,labels))

results = classifier.predict(F)*stdLab + meanLab
print(results.astype(int))
ans = labels*stdLab + meanLab

print(ans.astype(int))

print(regr.score(F,labels))

saved_model = "final_network"
path = F"/content/drive/My Drive/{saved_model}" 
# torch.save(Network.state_dict(), path)

Network.load_state_dict(torch.load(path))