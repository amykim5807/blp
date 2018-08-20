import scipy.io as sio
import numpy as np
import os
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 14:37:47 2018

@author: L1YAK01
"""

##C
os.chdir('S:/GOLD_Interns/AmyKim2018/blp_shared')

#Importing helper functions
import helper as hlp

data = sio.loadmat('BLP_data.mat')
random = sio.loadmat('random.mat')

#Setting random seed
np.random.seed(1)

#Initializing binary variables
debug = 0
solveforprices = 1
usequadrature = 0

#Number of Monte Carlo Simulations
MS=200

#Number of Simulations
NS=50

BS=2000
Nmarkets=100;
Nproducts=2;

#Total number of products
N=Nmarkets*Nproducts

#Taking the smaller value of NS and Nmarkets
m=min(NS,Nmarkets);

#Creating covariate matrix as numpy array
if debug:
    covariates = np.hstack([data['hpwt'],data['space']])
else:
    covariates = np.hstack([data['hpwt'],data['air'],data['mpd'],data['space']])

#Getting number of covariates
ncovariates = covariates.shape[1]

#Adding row of ones
Xdata = np.hstack([np.ones((len(covariates),1)),covariates])

Ndata = len(Xdata)
dimX = Xdata.shape[1]
alpha = 0.05
t=140#0;%0.5*N*(2*dimX+1);
B=560#00;%4*t;

covx = np.cov(covariates.T)
meanx = covariates.mean(axis=0)

varp = np.var(data['price'])
meanprice = data['price'].T.mean(axis=1) [0]

#Initializing matrix of zeros for sum_other and sum_rival
sum_other = np.zeros(Xdata.shape)
sum_rival = np.zeros(Xdata.shape)

#Filling sum matrices with sum of characteristics from other and rival products
for i in range(Ndata):
    other_ind = [(data['firmid']==data['firmid'][i]) & (data['cdid']==data['cdid'][i]) & (data['id']!=data['id'][i])][0] #Products in the same market and same firm
    rival_ind = [(data['firmid']!=data['firmid'][i]) & (data['cdid']==data['cdid'][i])][0] #Products in the same market but different firm
    total_ind = [(data['cdid']==data['cdid'][i])][0] #All products in the same market -->> Necessary?
    sum_other[i,:] = sum(Xdata[np.where(other_ind)[0],:])
    sum_rival[i,:] = sum(Xdata[np.where(rival_ind)[0],:])

#Creating Instr. Var. matrix
IV = np.hstack([Xdata, sum_other, sum_rival])

#Original Code
P = np.matmul(np.linalg.solve(np.matmul(IV.T,IV).T,IV.T).T,IV.T)

#Step by step
p1 = np.matmul(IV.T,IV)
p2 = np.linalg.solve(p1.T,IV.T)
p3 = np.matmul(p2.T,IV.T) #p3 is the same thing as P

q2 = np.linalg.lstsq(p1.T,IV.T) #Using least squares --> same matrix as p2

p4 = np.matmul(np.matmul(IV,np.linalg.inv(p1))) #Multiplying inverse --> same matrix as p3

max_val = np.max(np.abs(IV))
IV_temp = IV/max_val
p5 = np.matmul(np.linalg.solve(np.matmul(IV_temp.T,IV_temp).T,IV_temp.T).T,IV_temp.T) #Normalizing IV matrix to 1 --> same matrix as p3

#Importing P matrix directly from MATLAB
temp = sio.loadmat('temp.mat')
P_comp = temp['P']

print(P-P_comp)





