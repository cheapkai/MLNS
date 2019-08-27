#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Libraries
import scipy.io
import numpy as np
import sys
import math
from math import exp
from math import cos
from math import sqrt
from math import *
from sklearn.preprocessing import normalize


# In[ ]:


# Utility Functions
def switch_demo(argument):
    switcher = {
        0: 0,
        1: 1,
        6: 2,
        7: 3,
        8: 4,
        
    }
    return switcher.get(argument, 5)

def dot(Vi, Vj) :
    return abs(Vi[0]*Vj[0] + Vi[1]*Vj[1] + Vi[2]*Vj[2])

def norm(Vi) :
    return sqrt(dot(Vi, Vi))


# In[ ]:


# Functions that help calculate radial and angular components of AEVs

# Radial cutoff function
def cutoffradial(Rij) :
    if (Rij >8.1) :
        return 0.0

    return (0.5*cos((3.14*Rij)/8.1) + 0.5)	

# Angular cutoff function
def cutoffangular(Rij) :
    if (Rij > 8.1) :
        return 0.0

    return (0.5*cos((3.14*Rij)/8.1) + 0.5)

# Radial component for each pair of atoms to be added up into the AEV
def radial(Ri,Rj,Rs,eta) :
    Rij = math.sqrt((Ri[0]-Rj[0])*(Ri[0]-Rj[0]) + (Ri[1]-Rj[1])*(Ri[1]-Rj[1]) + (Ri[2]-Rj[2])*(Ri[2]-Rj[2]))
    
    cc = cutoffradial(Rij)

    exx = exp((-1)*(Rij-3.9)*(Rij-3.9))

    radf = exp((-1)*(Rij-3.9)*(Rij-3.9))*cc

    return (radf)

# Angular component for each triplet of atoms to be added up into the AEV
def angular(Ri,Rj,Rk,Rs,eta,theta,zeta) :
    Ri = np.array(Ri)
    Rj = np.array(Rj)
    Rk = np.array(Rk)
    Vj = Rj - Ri
    Vk = Rk - Ri

    cosan = dot(Vj, Vk) / (norm(Vj) * norm(Vk))
    cosan = round(cosan,5)
    try :
        acos(cosan)

    except :
        print(cosan)
        sys.exit()

    angle = acos(cosan)

    result = pow(2, 1-3)*pow( 1+cos(angle-theta),zeta )
    result = result * exp(-1*eta*pow( (norm(Vj)+norm(Vk))/2 - Rs , 2) )

    result = result * cutoffangular(norm(Vj)) * cutoffangular(norm(Vk))
    result = round(result,5)
    return result


# In[ ]:


# Loading up the QM7 Dataset
qm7 = scipy.io.loadmat('./qm7.mat')

X = np.array(qm7['X']) #X is Coulomb Matrix
R = np.array(qm7['R']) #R is cartesian coordinates of each atom in the molecule
T = np.array(qm7['T']) #T is Atomization Energies - labels
P = np.array(qm7['P']) #P is splits for cross-validation
Z = np.array(qm7['Z']) #Z is atomic charge of each atom in the molecule


# In[ ]:


# Setting up the parameters. (R[], eta, zeta, theta, )
Rs = [0.50,1.17,1.83,2.50,3.17,3.83,4.50,5.17]
eta = 4.00
zeta = 8.00
the = [0.00,1.57,3.14,4.71]


# In[ ]:


# Setting up indices for sub-AEVs
a1 = [0,8,16,24,32]
a2 = np.zeros([5,5])

a2[0][0] = 40
a2[0][1] = 72
a2[0][2] = 104
a2[0][3] = 136
a2[0][4] = 168
a2[1][1] = 200
a2[1][2] = 232
a2[1][3] = 264
a2[1][4] = 296
a2[2][2] = 328
a2[2][3] = 360
a2[2][4] = 392
a2[3][3] = 424
a2[3][4] = 456
a2[4][4] = 488

a2[0][0] = 40
a2[1][0] = 72
a2[2][0] = 104
a2[3][0] = 136
a2[4][0] = 168
a2[1][1] = 200
a2[2][1] = 232
a2[3][1] = 264
a2[4][1] = 296
a2[2][2] = 328
a2[3][2] = 360
a2[4][2] = 392
a2[3][3] = 424
a2[4][3] = 456
a2[4][4] = 488


# In[ ]:


# Making RC
rc = np.zeros([7165,23])

for i in range(7165) :
    for j in range(23) :
        u = int(Z[i][j])
        v = switch_demo(u)
        rc[i][j] = v
        rc[i][j] = int(rc[i][j])


# In[ ]:


# Making the AEVs
Ar = np.zeros([23,520,7165])

for k in range(7165) :
    for i in range(23) :

        if(int(rc[k][i])==0) :
            continue

        for j in range(23) :
            if (i==j) :
                continue

            if (int(rc[k][j]) == 0) :
                continue

            start = a1[int(rc[k][j])-1]

            for y in range(8) :
                uu = radial(R[k][i],R[k][j],Rs[y],eta)
                Ar[i][start + y][k] = Ar[i][start + y][k] + uu
                
            for y2 in  range((j+1),23):
                if (y2==i) :
                    continue
                if (y2==j) :
                    continue
                if (int(rc[k][y2]) == 0) :
                    continue

                start = a2[int(rc[k][j]-1)][int(rc[k][y2]-1)]

                for y3 in range(8) :
                    for y4 in range(4) :
                        uvv = angular(R[k][i],R[k][j],R[k][y2],Rs[y3],eta,the[y4],zeta)
                        Ar[i][int(start + int((y3+1)*(y4+1) - 1))][k] = Ar[i][int(start + int((y3+1)*(y4+1) - 1))][k] + uvv
                        
print(np.shape(Ar))

Ar = np.transpose(Ar)

print(np.shape(Ar))
Ar =Ar*1000

for k in range(7165) :
    aa = Ar[k]
    aa = np.transpose(aa)
    aa[0:23] = normalize(aa[0:23])
    aa = np.transpose(aa)
    Ar[k] = aa

Ar = Ar*1000
Ar = np.round(Ar,decimals = 3)
aa = Ar[70]
aa = np.transpose(aa)

Ar = np.transpose(Ar)
print(np.shape(Ar))
print(np.shape(rc))
print(np.shape(T))


# In[ ]:


# Saving the extracted features
scipy.io.savemat('feature_vector.mat', {'AEVs' : Ar , 'Atomic_Num' : rc , 'labels' : T},do_compression = True)


# In[ ]:





# In[ ]:





# In[ ]:




