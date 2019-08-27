#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import random


# In[ ]:


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


# In[ ]:


# Get input feature dataset
features = scipy.io.loadmat('./feature_vector.mat')

rans = np.arange(7165)
# Turn feature dataset into seperate arrays
AEVs = np.transpose(np.array(features['AEVs']), (2, 0, 1))
Atomic_Num = np.array(features['Atomic_Num'], dtype=np.long)
Target = np.array(features['labels'][0])

AEVs = np.round(AEVs,4)

print("Shapes:")
print(np.shape(AEVs))
print(np.shape(Atomic_Num))
print(np.shape(Target))
print(AEVs[50][2][41:80])


# In[ ]:


Ar= AEVs
scaler = NDStandardScaler()
data = scaler.fit_transform(Ar)
AEVs = data
print(np.shape(data))


# In[ ]:


print(data[50][3][280:320])


# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(520, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128,64)
        self.fc5 = nn.Linear(64, 1)
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = (self.fc5(x))
        #print(x)
        return x #need activation function on x or loss function


# In[ ]:


def doNN():
    # Declare Nets
    NNP_H = Net()
    NNP_C = Net()
    NNP_N = Net()
    NNP_O = Net()
    NNP_S = Net()
    
    # put in a list for ease of access
    nets = [NNP_H, NNP_C, NNP_N, NNP_O, NNP_S]
    
    Los = []
    
    # corresponding optimizers
    optimizers = []
    #criterions = []
    for net in nets:
        optimizers.append( optim.Adam(net.parameters(), lr=0.0099,betas=(0.5,0.59) ,eps=1e-08))
    
    criterion = nn.L1Loss()
        
    ##########################################################################################
    # training
    epochs = 20
    molecules = 5000
    batch_size = 20
    rep = int(molecules/batch_size)
    
    for epoch in range(epochs) :
        print("epoch: ", epoch)
        
        for mols in range(rep) :
            
            outp = torch.zeros(1,batch_size)
            toget = torch.zeros(1,batch_size)
            ck = int(0)
            
            start = mols*batch_size
            
            for mole in range(start,start+batch_size):
                molecule = int(rans[int(mole)])
                molecule = mole

                out_f = torch.zeros(1,1)

                # use all relevant aev's on the relevant nets
                for atom in range(23):
                    if(Atomic_Num[molecule][atom]==0):
                        continue

                    aev = torch.from_numpy(AEVs[molecule][atom])
                    out = nets[Atomic_Num[molecule][atom]-1](aev.float())
                    out_f = out_f + out

                outp[0][ck] = out_f
                
                targett = Target[molecule]
                targett = torch.from_numpy(np.array(targett))
                targett = targett.float()
                toget[0][ck] = targett
                ck = int(ck+1)
                
                # Get a list of only the nets that were used. I.e list of atoms in the molecule
                used_atoms = []
                for atom in range(23):
                    if(Atomic_Num[molecule][atom]!=0):
                        used_atoms.append(Atomic_Num[molecule][atom])
                used_atoms = np.unique(np.array(used_atoms))
            
                # if no nets were used, just move on to the next molecule
                if(len(used_atoms)==0) :
                    print("scream")
                    break
                
            # setting the parameters for the entire net to be zero
            for net in nets:
                net.zero_grad()
            
            for optimizer in optimizers:
                    optimizer.zero_grad()

            toget = toget.view(1,-1)
            
            # use loss function
            loss = criterion(outp,toget)

            # backpropagate
            loss.backward()
            
            # step only the used optimizers            
            for atom in range(5) :
                optimizers[atom].step()
    
    
    ###################################################################################
    # Testing
    losses = []
    answ = 0
    for mole in range(5501,7100) :
        molecule = int(rans[int(mole)])
        molecule = mole
        

        molecule_out = torch.zeros(1,1)
        for atom in range(23):
            if(Atomic_Num[molecule][atom]==0):
                continue
            aev = torch.from_numpy((AEVs[molecule][atom]))
            net_out = nets[Atomic_Num[molecule][atom]-1](aev.float())
            molecule_out = molecule_out + net_out
        
        targett = Target[molecule]
        targett = torch.from_numpy(np.array(targett))
        targett = targett.float()
        targett = targett.view(1, -1)
        
        loss = criterion(molecule_out, targett)
        if ((targett-15) <= molecule_out <= (targett+15)) :
            answ = answ + 1
        
    print(answ)
    
doNN()

