# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 16:08:26 2018

@author: L1YAK01
"""

## IMPORTING PACKAGES ##
import scipy.io as sio
import scipy.stats as st
import numpy as np
import scipy as sp
import os
import time

######################################################################
########SETTING DIRECTORIES AND RETRIEVING DATA/FUNCTIONS#############
######################################################################
#os.chdir('S:/GOLD_Interns/NicoleTrachman2018/blp')
#import simulatemarketsharesCtranslation as simms

#os.chdir('S:/GOLD_Interns/AmyKim2018/blp_shared') #If working locally
#os.chdir('/mnt/lan-shared/GOLD_Interns/AmyKim2018/blp_shared') #If working on cluster

#Importing helper functions
import helper as hlp


def initialization(data, debug, usequadrature, NS, BS, Nmarkets, Nproducts):
    #Total number of products
    N=Nmarkets*Nproducts
    
    #Taking the smaller value of NS and Nmarkets
    m=min(NS,Nmarkets)
    
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

    
    covx = np.cov(covariates.T)
    meanx = covariates.mean(axis=0)
    
    varp = np.var(data['price'], ddof=1)
    meanprice = data['price'].T.mean(axis=1)[0]
    
    ######################################################################
    ####################### KEY MATRIX CREATION ##########################
    ######################################################################
    
    #Initializing matrix of zeros for sum_other and sum_rival
    sum_other = np.zeros(Xdata.shape)
    sum_rival = np.zeros(Xdata.shape)
    
    #Filling sum matrices with sum of characteristics from other and rival products
    
    ##GENERAL VERSION
    #for i in range(Ndata):
    #    other_ind = [(data['firmid']==data['firmid'][i]) & (data['cdid']==data['cdid'][i]) & (data['id']!=data['id'][i])][0] #Products in the same market and same firm
    #    rival_ind = [(data['firmid']!=data['firmid'][i]) & (data['cdid']==data['cdid'][i])][0] #Products in the same market but different firm
    #    total_ind = [(data['cdid']==data['cdid'][i])][0] #All products in the same market -->> Necessary?
    #
    #    sum_other[i,:] = sum(Xdata[np.where(other_ind)[0],:])
    #    sum_rival[i,:] = sum(Xdata[np.where(rival_ind)[0],:])
    
    ##MATCHING MATLAB
    for i in range(Ndata):
        other_ind = [(data['firmid']==data['firmid'][i]) & (data['cdid']==data['cdid'][i]) & (data['id']!=data['id'][i])][0] #Products in the same market and same firm
        rival_ind = [(data['firmid']!=data['firmid'][i]) & (data['cdid']==data['cdid'][i])][0] #Products in the same market but different firm
        total_ind = [(data['cdid']==data['cdid'][i])][0] #All products in the same market -->> Necessary?
        
        if len(other_ind[other_ind==True])==1:
            sum_other[i,:] = np.sum(Xdata[np.where(other_ind)[0],:])
        
        else:
            sum_other[i,:] = sum(Xdata[np.where(other_ind)[0],:])
        sum_rival[i,:] = sum(Xdata[np.where(rival_ind)[0],:])
    
    #Creating Instr. Var. matrix
    IV = np.hstack([Xdata, sum_other, sum_rival])
    covIVnotX = np.cov(IV[:,dimX:len(IV)].T, ddof=1)
    varIVnotX = np.var(IV[:,dimX:len(IV)], axis = 0, ddof=1)
    meanIVnotX = IV[:,dimX:len(IV)].mean(axis=0)
    dimIVnotX = len(meanIVnotX)
    
    if debug:
        theta2true = [2.009,1.586,1.51]
        Sigmatrue = np.diag(theta2true)
        betatrue = [-7.304,2.185,2.604,-0.2]
        gammatrue = [0.726,0.313,1.499]
        
    else:
        theta2true = [2.009,1.586,1.215,0.67,1.51]
        Sigmatrue = np.diag(theta2true);
        betatrue = [-7.304,2.185,0.579,-0.049,2.604,-0.2]
        gammatrue = [0.726,0.313,0.290,0.293,1.499]
        
    thetatrue = np.hstack([betatrue,theta2true])
        
    if usequadrature:
        [J,vdraws10,weights] = hlp.GH_Quadrature(10,dimX,np.identity(dimX))
        weights=weights.T
        musimtrue=np.matmul(np.matmul(Xdata,Sigmatrue),vdraws10.T)
    else:
        vdraws = np.random.multivariate_normal([0]*dimX,np.identity(dimX),NS)
        musimtrue=np.matmul(np.matmul(Xdata,Sigmatrue),vdraws.T)
        weights= np.tile(1/NS,(1,NS))
    
    ######################################################################
    ################## CREATING DELTAHAT AND XIHAT #######################
    ######################################################################
        
    #deltahat = simms.simulatemarketshares(data['share'],data['outshr'],musimtrue,np.shape(musimtrue)[1],data['cdindex'],weights,1e-4)
    deltahat = np.log(data['share']/data['outshr'])  ##PLACEHOLDER FOR COMPUTEDELTAFROMSIMULATIONCCODE
    C = np.hstack([Xdata,data['price']])
    P = np.matmul(np.linalg.solve(np.matmul(IV.T,IV).T,IV.T).T,IV.T)
    betahat = np.linalg.solve(np.matmul(np.matmul(C.T,P),C),np.matmul(np.matmul(C.T,P),deltahat))
    xihat = (deltahat - np.matmul(C,betahat))
    varxi = np.var(xihat,axis=0,ddof=1)
    
    #Getting covariance between xihat and price
    covmatpricexihat = np.cov(xihat,data['price'],rowvar=False)
    covpricexihat = covmatpricexihat[0,1]
    
    return [dimX, meanx, covx, varxi, varp, covpricexihat, meanIVnotX, covIVnotX, gammatrue, Sigmatrue, betatrue, Xdata, thetatrue]


def datageneration(solveforprices, usequadrature, NS, Nmarkets, Nproducts, dimX, meanx, covx, varxi, varp, covpricexihat, meanIVnotX, covIVnotX, gammatrue, Sigmatrue, betatrue, Xdata):
    
    N = Nmarkets * Nproducts
    X= np.zeros((N, dimX))
    xi= np.zeros((N,1))
    
    for j in range(Nmarkets):
        xcharacteristics= np.random.multivariate_normal(meanx,covx,[Nproducts])
        X[(j*Nproducts):((j+1)*Nproducts), 0:5] = np.hstack([np.ones((Nproducts,1)), xcharacteristics])
        ximarket= np.random.normal(0, 0.5*varxi, Nproducts)
        xi[((j-1)*Nproducts):(j*Nproducts), :] = ximarket.shape
    
    v1= np.random.normal(0,abs(covpricexihat-varxi), Nmarkets*Nproducts)
    v2= np.random.normal(0,varxi, Nmarkets*Nproducts)
    v3= np.random.normal(0, abs(varp-varxi), Nmarkets*Nproducts)
    v4= np.random.multivariate_normal(meanIVnotX, covIVnotX,[N])
    
    v3ones= np.ones((v3.size,10))
    addstep= np.add(v3.size, v4[:]) #unsure of accuracy of this calculation 
    IV= np.hstack((X, addstep)) 
    
    
    etaconst=0.001
    eta=etaconst*(v1+v3)
    
    mc=X@gammatrue+eta
    
    #want to check that this produces the same value as MATLAB version-- may need to use sparse.kron or different complex conjugate transpose
    cdid= np.kron(range(1,Nmarkets+1),np.ones((1,Nproducts))).T
    cdindex = [i for i in range(Nproducts,N+1,Nproducts)]
    ######################################################################
    ###################### SOLVING FOR PRICES ############################
    ######################################################################
    
    start_time = time.time()
    #solving for prices
    if solveforprices:
        [J,vdraws10,weights] = hlp.GH_Quadrature(10,dimX,np.identity(dimX))
        weights=weights.T
        musim = np.matmul(np.matmul(X,Sigmatrue),vdraws10.T)
        print('Solving for prices using Bertrand-Nash')
        price = np.array([1.0]*N)
        profits = np.array([1]*Nmarkets)
        for j in range(Nmarkets):
            print('Market #:',j+1,'/',Nmarkets)
            cdindexformarket = cdindex[j]
            p0 = np.random.random(2)
            lb = np.array([0.01]*Nproducts)
            ub = np.array([100]*Nproducts)
            while p0[0] < lb[0] or p0[1] < lb[0]:
                p0 = np.random.random(2)
            priceformarket = sp.optimize.least_squares(lambda price: hlp.equationtosolveforprice(price, X[(cdindexformarket-Nproducts):cdindexformarket,:], betatrue, musim[(cdindexformarket-Nproducts):cdindexformarket,:],
                                                                                     np.shape(musim)[1],Nproducts,1, mc[(cdindexformarket-Nproducts):cdindexformarket],weights),p0,bounds=(lb,ub),verbose=1)
            print(priceformarket['x'])
            price[(cdindexformarket-Nproducts):cdindexformarket] = priceformarket['x']
    else:
        e = st.truncnorm.rvs(0,1,size=(N,1))
        price = 0.5*abs(2+0.5*xi+e+np.sum(X,axis=1))
    
    price_time = time.time()-start_time
    print("Elapsed time:", str(price_time))
    ##NOTE: ElAPSED TIME = 1700s = approx. 28 min
    ######################################################################
    ################### SIMULATING MARKET SHARES #########################
    ######################################################################
    
    deltatrue = (np.matmul(np.c_[np.array(X),np.array(price)],np.array(betatrue).T) + np.reshape(xi,(-1,1)).T).T
    if usequadrature:
        [J,vdraws10,weights] = hlp.GH_Quadrature(10,dimX,np.identity(dimX))
        weights=weights.T
        musim = np.matmul(np.matmul(X,Sigmatrue),vdraws10.T)
    else:
        vdraws = np.random.multivariate_normal([0]*dimX,np.identity(dimX),NS)
        musim = np.matmul(np.matmul(Xdata,Sigmatrue),vdraws.T)
        weights = np.tile(1/NS,(1,NS))
    
    individualshares, outsideshares = hlp.simulateMarketShares(deltatrue, musim, np.shape(musim)[1],cdindex)
    simshare = np.sum(np.tile(weights,(N,1))*individualshares,axis=1)
    simoutshare = np.sum(np.tile(weights,(N,1))*individualshares,axis=1)
    
    deltahat = np.log(simshare/simoutshare) ##PLACEHOLDER FOR COMPUTEDELTAFROMSIMULATIONCCODE
    C = np.hstack([X,price[:,None]])
    P = np.linalg.solve((IV.T @ IV).T,IV.T).T @ IV.T
    dimIV = np.shape(IV)[1]
    
    beta0 = np.linalg.lstsq((C.T @ P @ C),(C.T @ P @ deltahat),rcond=None)[0]
    theta0 = np.concatenate([beta0,np.diag(Sigmatrue)])
    
    return [C, X, xi, weights, price, simshare, simoutshare, cdid, cdindex, IV, vdraws, vdraws10, theta0, beta0]
