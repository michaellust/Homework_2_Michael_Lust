#!/usr/bin/env python
# coding: utf-8

# In[49]:


#Michael Lust
#801094861
#Real Time AI (4106)
#March 1, 2022


# In[50]:


from torchvision import models, datasets, transforms
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd 
import matplotlib.pyplot as plt  


# In[51]:


#Problem 1 Building Fully connected neural network for housing dataset


# In[52]:


#Problem 1 part a Using 1 hidden layer with 8 nodes


# In[53]:


dataset = pd.DataFrame(pd.read_csv('Housing.csv'))
dataset.head()


# In[54]:


m = len(dataset)
m


# In[55]:


dataset.shape


# In[56]:


#This is taking an assumption that we are focusing on these explanatory variables from homework 1 and not all of them
#-> that can be found in the housing.csv dataset.
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price'] 
Newtrain = dataset[num_vars] 
Newtrain.head() 


# In[57]:


Newtrain.values[:, 0]


# In[58]:


#Copying dataset to fix SettingWithCopyWarning: 
Newtrain_copy = Newtrain.copy()

#Apply normalization to dataset to 
for column in Newtrain.columns:
    Newtrain_copy[column] = Newtrain_copy[column]  / Newtrain_copy[column].abs().max()

Newtrain_copy.head()


# In[59]:


#Splitting the Data into Training and Testing Sets
from sklearn.model_selection import train_test_split

#The split is 80%-20% for training and testing respectively. 
np.random.seed(0)
Newtrain,Newtest = train_test_split(Newtrain_copy, train_size = 0.8, test_size = 0.2, random_state = 1)

#Displaying training shape to verify
Newtrain.shape


# In[60]:


#Displaying test shape to verify
Newtest.shape


# In[61]:


#Training set
y_Newtrain = Newtrain.pop('price')
x_Newtrain = Newtrain
#Validation set
y_Newtest = Newtest.pop('price')
x_Newtest = Newtest


# In[62]:


#Labeling for model 

#Training
x_T = torch.tensor(x_Newtrain.values).float()
x_V = torch.tensor(x_Newtest.values).float()

#Validation
y_T = torch.tensor(y_Newtrain.values).float().unsqueeze(-1)
y_V = torch.tensor(y_Newtest.values).float().unsqueeze(-1)

#Verifying test split 
x_T.shape


# In[63]:


#Modified training loop to return Epoch and Cost Values to graph
def training_loop(n_epochs, optimizer, model, loss_fn, x_T, x_V, y_T, y_V):

    #Main training loop for both training set and validation set
    for epoch in range(1, n_epochs + 1):
        t_p_T = model(x_T)
        loss_T = loss_fn(t_p_T, y_T)
        
        t_p_V = model(x_V)
        
        loss_V = loss_fn(t_p_V, y_V)

        #Passing optimizer and loss function
        optimizer.zero_grad()
        loss_T.backward()
        optimizer.step()

    #Printing out Epochs and Loss
    if epoch == 1 or epoch % 10 == 0:
        print(f"Epoch {epoch}, Training Loss is {loss_T.item():.4f},"
              f" Validation Loss is {loss_V.item():.4f}")

    #return optimizer, loss_arr_T, loss_arr_V, n_epochs_arr


# In[64]:


#Now creating hidden layer for model
#Starting of with 1 hidden layer or linear module. 8 hidden features or nodes was chosen arbitrarily.
#Naming each module in Sequential using OrderedDict
from collections import OrderedDict

seq_model = nn.Sequential(OrderedDict([
            ('hidden_linear', nn.Linear(len(num_vars)-1,8)), #Hidden Layer 1
            ('hidden_activation', nn.Tanh()),
            ('output_linear', nn.Linear(8,1)) #Outer Layer
]))
seq_model


# In[65]:


#Collecting weights and biases using model.parameters()
[param.shape for param in seq_model.parameters() ]


# In[66]:


#Showing tensors from the optimizer
for name, param in seq_model.named_parameters():
    print(name, param.shape)


# In[67]:


len(num_vars) #Checking length of num_vars to correctly establish linear model


# In[68]:


#Explanatory names for submodule
for name, param in seq_model.named_parameters():
    print(name, param.shape)


# In[69]:


#Accessing particular parameters using submodules as atributes
seq_model.output_linear.bias


# In[70]:


#Now testing neural network with a learing rate at 0.001 with hidden features and 1 layer
optimizer = optim.SGD(seq_model.parameters(), lr = 1e-2)

training_loop(
    n_epochs = 200,
    optimizer = optimizer,
    model = seq_model,
    loss_fn = nn.MSELoss(), #This replaces the loss function from earlier
    x_T = x_T,
    x_V = x_V,
    y_T = y_T,
    y_V = y_V,
    )

print('output', seq_model(x_V))
print('answer', y_V)
print('hidden', seq_model.hidden_linear.weight.grad)


# In[71]:


#Problem 1 Part b Expanding network with 2 more additional layers


# In[72]:


seq_model_2 = nn.Sequential(OrderedDict([
            ('hidden_linear', nn.Linear(len(num_vars)-1,8)), #Hidden Layer 1
            ('hidden_activation', nn.Tanh()),
            ('hidden_linear', nn.Linear(8,5)), #Hidden Layer 2
            ('hidden_activation', nn.Tanh()),
            ('hidden_linear', nn.Linear(5,2)), #Hidden Layer 3 
            ('hidden_activation', nn.Tanh()),
            ('output_linear', nn.Linear(2,1)) #Outer Layer
]))
seq_model_2


# In[73]:


#Now testing neural network with a learing rate at 0.01 with hidden features and 1 layer
optimizer = optim.SGD(seq_model_2.parameters(), lr = 1e-2)

training_loop(
    n_epochs = 200,
    optimizer = optimizer,
    model = seq_model_2,
    loss_fn = nn.MSELoss(), #This replaces the loss function from earlier
    x_T = x_T,
    x_V = x_V,
    y_T = y_T,
    y_V = y_V,
    )

print('output', seq_model_2(x_V))
print('answer', y_V)
print('hidden', seq_model_2.hidden_linear.weight.grad)


# In[74]:


#Problem 2 Creating Neural Network for all 10 features with CIFAR-10


# In[75]:


#Problem 2 Part a Using 1 hidden layer with size of 512


# In[76]:


#Choosing the 10 classes to base model out off (using the read me file to pick classes)
class_names = ['dolphin', 'seal', 'otter', 'shark', 'ray', 'flatfish', 'beaver', 'aquarium fish', 'trout', 'whale']


# In[77]:


#Dowloading CIFAR-10
from torchvision import datasets
data_path = 'CIFAR'
cifar10 = datasets.CIFAR10(data_path, train = True, download = True) #Gathering training data
cifar10_val = datasets.CIFAR10(data_path, train = False, download = True) #Gathering validation data


# In[78]:


#Checking length of CIFAR 10
len(cifar10)


# In[79]:


img, label = cifar10[99]


# In[80]:


img, label, class_names[label]


# In[81]:


#Converting PIL images to PyTorch using transform function
from torchvision import transforms

#Seeing available objects
#dir(transforms) 


# In[82]:


#Turning PIL images to tensors
to_tensor = transforms.ToTensor()
img_t = to_tensor(img)
img_t.shape


# In[83]:


#Passing transfomr to CIFAR10
tensor_cifar10 = datasets.CIFAR10(data_path, train = True, download = False, transform = transforms.ToTensor())

img_t, _ = tensor_cifar10[99]
type(img_t)


# In[84]:


img_t.shape, img_t.dtype


# In[85]:


#Verifying Image output and changing CxHxW to HxWxC for matplotlib
plt.imshow(img_t.permute(1,2,0))
plt.show()


# In[86]:


#Normalizing tensor as shown in lecture 9
transformed_cifar10 = datasets.CIFAR10(data_path, train = True, download = False, transform = transforms.Compose([
                                       transforms.ToTensor(), 
                                       transforms.Normalize((0.4915, 0.4823, 0.4468),(0.2470, 0.2435, 0.2616))
                                        ]))
img_t, _ = tensor_cifar10[99]

plt.imshow(img_t.permute(1,2,0))
plt.show()


# In[87]:


#Creating neural network for testing and validation


# In[88]:


#Normalizing PIL to tensor for training and validation
cifar10_T = datasets.CIFAR10(data_path, train = True, download = False, transform = transforms.Compose([
                                       transforms.ToTensor(), 
                                       transforms.Normalize((0.4915, 0.4823, 0.4468),(0.2470, 0.2435, 0.2616))
                                        ]))
cifar10_V = datasets.CIFAR10(data_path, train = False, download = False, transform = transforms.Compose([
                                       transforms.ToTensor(), 
                                       transforms.Normalize((0.4915, 0.4823, 0.4468),(0.2470, 0.2435, 0.2616))
                                        ]))


# In[89]:


#Creating subclass to include the labels in class_names
label_map = {0: 0, 2: 1}

cifar10_train = [(img, label_map[label]) for img, label in cifar10_T if label in [0,2]]
cifar10_test = [(img, label_map[label]) for img, label in cifar10_V if label in [0,2]]


# In[90]:


#Setting GPU to run nn
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)


# In[91]:


#Creating sequential model using train loader and steps in problem 1
train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size = 64, shuffle = True)

model = nn.Sequential(
            nn.Linear(3072, 512), #Hidden Layer 1
            nn.Tanh(),
            nn.Linear(512, 2), #Outer Layer
            nn.LogSoftmax(dim = 1))

#Running on GPU
model.to(device)

learning_rate = 1e-2
optimizer = optim.SGD(model.parameters(), lr = learning_rate)

loss_fn = nn.NLLLoss()
n_epochs = 200

for epoch in range(n_epochs + 1):
    for imgs, labels in train_loader:
        
        imgs, labels = imgs.to(device), labels.to(device) #Used for GPU
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    #Printing out Epochs and Loss
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))  


# In[92]:


#Measurning accuracy of the model on the training set
train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=64, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in train_loader:
        
        imgs, labels = imgs.to(device), labels.to(device) #Used for GPU
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        _, predicted = torch.max(outputs, dim = 1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
        
print("Training Accuracy: %f" % (correct / total))


# In[93]:


#Measurning accuracy of the model on the validation set
val_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=64, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        
        imgs, labels = imgs.to(device), labels.to(device) #Used for GPU
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        _, predicted = torch.max(outputs, dim = 1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
        
print("Validation Accuracy: %f" % (correct / total))


# In[94]:


#Problem 2 part b Increasing network to two more additional layers


# In[95]:


#Increasing model complexity using intermediate layers to improve model performance
#Dropping nn.LogSoftMax
#The combination of nn.LogSoftmax and nn.NLLLoss is equivalent to using nn.CrossEntropyLoss.
model2 = nn.Sequential(
            nn.Linear(3072, 1024), #Hidden Layer 1
            nn.Tanh(),
            nn.Linear(1024, 512), #Hidden Layer 2
            nn.Tanh(),
            nn.Linear(512, 128), #Hidden Layer 3
            nn.Tanh(),
            nn.Linear(128, 2)) #Outer Layer

#Running on GPU
model2.to(device)

#Repeating steps for accuracy for training and validation
learning_rate = 1e-2
optimizer = optim.SGD(model2.parameters(), lr = learning_rate)

loss_fn = nn.CrossEntropyLoss()
n_epochs = 200

for epoch in range(n_epochs + 1):
    for imgs, labels in train_loader:
        
        imgs, labels = imgs.to(device), labels.to(device) #Used for GPU
        batch_size = imgs.shape[0]
        outputs = model2(imgs.view(batch_size, -1))
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    #Printing out Epochs and Loss
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))  


# In[96]:


#Measurning accuracy of the model on the training set for model2
train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=64, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in train_loader:
        
        imgs, labels = imgs.to(device), labels.to(device) #Used for GPU
        batch_size = imgs.shape[0]
        outputs = model2(imgs.view(batch_size, -1))
        _, predicted = torch.max(outputs, dim = 1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
        
print("Training Accuracy: %f" % (correct / total))


# In[97]:


#Measurning accuracy of the model on the validation set for model2
val_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=64, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        
        imgs, labels = imgs.to(device), labels.to(device) #Used for GPU
        batch_size = imgs.shape[0]
        outputs = model2(imgs.view(batch_size, -1))
        _, predicted = torch.max(outputs, dim = 1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
        
print("Training Accuracy: %f" % (correct / total))

