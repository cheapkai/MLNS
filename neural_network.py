#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[43]:


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import scipy.io
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from sklearn.preprocessing import normalize
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler


# In[44]:


class NDStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X


# In[39]:


# Get input feature dataset
features = scipy.io.loadmat('/home/mehthab/feature_vector3.mat')



# Turn feature dataset into seperate arrays
AEVs = np.transpose(np.array(features['AEVs']), (2, 0, 1))
Atomic_Num = np.array(features['Atomic_Num'], dtype=np.long)
Target = np.array(features['labels'][0])
'''

AEVs = np.random.rand(7165, 23, 520)
Atomic_Num = np.random.randint(6, size=(7165, 23))
Target = np.random.rand(7165)

'''
AEVs = np.round(AEVs,4)
#AEVs = normalize(AEVs)
print("Shapes:")
print(np.shape(AEVs))
print(np.shape(Atomic_Num))
print(np.shape(Target))
print(AEVs[50][2][41:80])
list_atoms = []
for row in Atomic_Num:
    for elem in row:
        list_atoms.append(elem)
print(np.unique(np.array(list_atoms)))

# Seperate into training and testing samples
#train_atoms =    
#train_samples = 
#train_labels = 

#test_atoms = 
#test_samples = 
#test_labels = 


# In[45]:


Ar= AEVs
scaler = NDStandardScaler()
data = scaler.fit_transform(Ar)
print(np.shape(data))


# In[52]:


print(data[50][3][280:320])


# In[53]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(520, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = (self.fc4(x))
        #print(x)
        return x #need activation function on x or loss function


# In[63]:


def doNN():
    # Declare Nets
    NNP_H = Net()
    NNP_C = Net()
    NNP_N = Net()
    NNP_O = Net()
    NNP_S = Net()
    
    #torch.nn.init.normal_(NNP_H.weight, mean=0, std=1)
    #torch.nn.init.normal_(NNP_C.weight, mean=0, std=1)
    #torch.nn.init.normal_(NNP_N.weight, mean=0, std=1)
    #torch.nn.init.normal_(NNP_O.weight, mean=0, std=1)
    #torch.nn.init.normal_(NNP_S.weight, mean=0, std=1)
    
    # put in a list for ease of access
    nets = [NNP_H, NNP_C, NNP_N, NNP_O, NNP_S]

        
    # corresponding optimizers
    optimizers = []
    #criterions = []
    for net in nets:
        optimizers.append( optim.Adam(net.parameters(), lr=0.001,betas=(0.9,0.999) ,eps=1e-08))
        #criterions.append(nn.MSELoss())
    
    criterion = nn.SmoothL1Loss()
    
    # wonder if we need seperate criterions
    #criterion = nn.MSELoss()
    
    ##########################################################################################
    # training
    epochs = 20
    molecules = 300
    
    for epoch in range(epochs) :
        print("epoch: ", epoch)
        outp = []
            
        for molecule in range(molecules):
        
            # initialize optimizer
            for optimizer in optimizers:
                optimizer.zero_grad()

        
            out_f = torch.zeros(1,1)
            
            # use all relevant aev's on the relevant nets
            for atom in range(23):
                #print(Atomic_Num[molecule][atom], end=" "),
                if(Atomic_Num[molecule][atom]==0):
                    continue
                
                aev = torch.from_numpy(AEVs[molecule][atom])
                out = nets[Atomic_Num[molecule][atom]-1](aev.float())
                #print("out", out)
                out_f = out_f + out
                
                '''
                #extra
                targett = Target[molecule]
                targett = torch.from_numpy(np.array(targett)
                targett = targett.view(1,-1)
                loss = criterion(out, targett.float())
                loss.backward()
                optimizers[Atomic_Num[molecule][atom]-1].step()
                '''
            
        
            ''''''
            #print("outf", out_f)
            
            # setting the parameters for the entire net to be zero
            for net in nets:
                net.zero_grad()

            
            
            targett = Target[molecule]
            targett = torch.from_numpy(np.array(targett))
            targett = targett.float()
            targett = targett.view(1,-1)
            ''''''
            # get list of used atoms i.e NNP's
            used_atoms = []
            for atom in range(23):
                if(Atomic_Num[molecule][atom]!=0):
                    used_atoms.append(Atomic_Num[molecule][atom])
            used_atoms = np.unique(np.array(used_atoms))
            
            # if no nets were used, just move on to the next molecule
            if(len(used_atoms)==0) :
                print("scream")
                break
            
            ''''''
            # use loss function
            loss = criterion(out_f,targett)

            # backpropagate
            loss.backward()
            
            # step only the used optimizers            
            for atom in used_atoms:
                optimizers[atom-1].step()    
            ''''''
    
    
    ###################################################################################
    losses = []
    for molecule in range(301,500) :

        molecule_out = torch.zeros(1,1)
        for atom in range(23):
            #print(Atomic_Num[molecule][atom], end=" ")
            if(Atomic_Num[molecule][atom]==0):
                continue
            aev = torch.from_numpy((AEVs[molecule][atom]))
            net_out = nets[Atomic_Num[molecule][atom]-1](aev.float())
            #print(net_out)
            molecule_out = molecule_out + net_out
        #print("")

        targett = Target[molecule]
        targett = torch.from_numpy(np.array(targett))
        targett = targett.float()
        targett = targett.view(1, -1)
        #losses.append(targett - molecule_out)
        loss = criterion(molecule_out, targett)
        if(molecule%10==0):
            print("molecule")
            print("Target: ", targett)
            print("Our_vl: ", molecule_out)
            print("Loss  : ", loss)
            #print(targett - molecule_out)
    '''
    train(nets, optimizers, criterion)
    testing(nets)
    '''
            
    
doNN()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


'''
for molecule in range(10):
    for atom in range(23):
        aev = []
        for elem in AEVs[molecule][atom]:
            if(elem>0.0000000001):
                aev.append(elem)
        print(aev)
'''


# In[ ]:




