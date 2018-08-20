# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 16:09:06 2018

@author: L1YAK01
"""
######################################################################
################## DATA & HELPER FUNCTION IMPORTS ####################
######################################################################
os.chdir('S:/GOLD_Interns/AmyKim2018/blp_shared') #If working locally
#os.chdir('/mnt/lan-shared/GOLD_Interns/AmyKim2018/blp_shared') #If working on cluster
import data_generation_final as datagen
import helper as hlp

## IMPORTING PACKAGES ##
import scipy.io as sio
import scipy.stats as st
import numpy as np
import scipy as sp
import os
import time
import matplotlib.mlab as matlab

#Loading BLP data
data = sio.loadmat('BLP_data.mat')

######################################################################
##################### VARIABLE INITIALIZATION ########################
######################################################################
#Setting random seed
np.random.seed(0)

#Initializing binary variables
debug = 0
solveforprices = 1
usequadrature = 0

#Number of Monte Carlo Simulations
MS=20

#Number of Simulations
NS=50

BS=2000
#Nmarkets=2 #For testings
Nmarkets=100 #For point estimates
Nproducts=2

alpha = 0.05
t=140
B=560
m = min(NS,Nmarkets)

[dimX, meanx, covx, varxi, varp, covpricexihat, meanIVnotX, covIVnotX, gammatrue, Sigmatrue, betatrue, Xdata, thetatrue] = datagen.initialization(data, debug, 
    usequadrature, NS, BS, Nmarkets, Nproducts)

#Initializing indicator lists
[coverageindictatorscorrect,coverageindictatorswrong, coverageindictatorswrong2, coverageindictatorsdeltacorrect,coverageindicatorsbootstrap,
     standarderrorscorrect,standarderrorswrong,standarderrorswrong2,betahats, coverageindictatorscorrectmedian, coverageindictatorswrongmedian,
     coverageindictatorswrong2median,coverageindictatorsdeltacorrectmedian,coverageindicatorsbootstrapmedian,standarderrorscorrectmedian,
     standarderrorswrongmedian,standarderrorswrong2median,betahatsmedian] = [np.zeros((dimX+1,MS))]*18

[posteriormeanspre,posteriormedianspre,posteriorquantilesalpha2pre,posteriorquantilesoneminusalpha2pre,coverageindicatorspre,coverageindicatorssymmetricpre,
     criticalvaluessymmetricpre,posteriormeanspost,posteriormedianspost,posteriorquantilesalpha2post,posteriorquantilesoneminusalpha2post,coverageindicatorspost,
     coverageindicatorssymmetricpost,criticalvaluessymmetricpost,acceptanceratiostokeep,sigmastokeep] = [np.zeros((2*dimX+1,MS))]*16


######################################################################
############ FOR LOOP FOR DATA GENERATION & SIMULATION ###############
######################################################################
##ADD VDRAWS10 TO FUNCTION RETURN
#for s in range(MS):
[C, X, xi, weights, price, simshare, simoutshare, cdid, cdindex, IV, vdraws, vdraws10, theta0, beta0] = datagen.datageneration(solveforprices, usequadrature, NS, Nmarkets, Nproducts, dimX, meanx, covx, varxi, varp, covpricexihat, meanIVnotX, covIVnotX, gammatrue, Sigmatrue, betatrue, Xdata)

#initializing function constant variables
N = Nmarkets * Nproducts
tolerance = 0.001
nogradient = 0

thetaspost = hlp.HMCMC(lambda theta: hlp.computeGMMobjective(theta, simshare, simoutshare, cdindex, weights, price, X, IV, vdraws, Nproducts, N, tolerance, nogradient), theta0, B)

posteriormeanpost = np.mean(thetaspost[t-1:,:], axis=0)
posteriormedianpost = np.median(thetaspost[t-1:,:], axis=0)
thetasdemedianed = np.abs(thetaspost[t-1:,:] - (np.ones((B-t+1,1)) @ posteriormedianpost[:,None].T))
criticalvaluesymmetricpost = matlab.prctile(thetasdemedianed, 100*(1-alpha)) 

posteriorquantilealpha2post = matlab.prctile(thetaspost[t-1:,:],100*alpha/2)
posteriorquantileoneminusalpha2post = matlab.prctile(thetaspost[t-1:,:],100*(1-alpha/2))

### STANDARD ERRORS ###
betahat=np.zeros((1, dimX+1))
for e in range (dimX+1):
    betanew=posteriormeanpost[e]
    betahat[0,e]= betanew 
betahat=betahat.conj().transpose()

theta2hat=np.zeros((1,dimX))
for c in range (dimX):
    thetanew= posteriormeanpost[(dimX+1)+c].size
    theta2hat[0,c]=thetanew

if usequadrature:
    muhat=X@np.diag(theta2hat)@vdraws10.T
else: 
    muhat= X@np.diag(theta2hat)@vdraws.T

delta = np.log(simshare/simoutshare)
if any(np.isnan(delta)):
    delta = np.log(simshare/simoutshare)

#Computing standard errors
[sebetahatcorrect, sebetahatwrong, sebetahatwrong2, Ghat, W] = hlp.computeStandardErrorsforBetahat(delta, betahat, cdindex, cdid, muhat, IV, dimX, C, weights, np.shape(muhat)[1], N, Nmarkets)

#bootstrap
bootdist = hlp.bootstrap(IV, delta, betahat, BS, NS, Nmarkets, Nproducts, cdindex, C, simshare, simoutshare, usequadrature, vdraws10, vdraws, X, Sigmatrue, theta2hat, Ghat, W, m)

bootpercentiles = matlab.prctile(bootdist, [100*alpha/2, 100*(1-alpha/2)])
bootCI = np.vstack([betahat - (m**(-0.5))*bootpercentiles[1:2].T, betahat-(m**(-0.5))*bootpercentiles[1:2].T]).T

CIwrong = np.vstack([betahat-1.96*sebetahatwrong, betahat+1.96*sebetahatwrong]).T
CIwrong2 = np.vstack([betahat-1.96*sebetahatwrong2, betahat+1.96*sebetahatwrong2]).T
CIcorrect = np.vstack([betahat-1.96*sebetahatcorrect, betahat+1.96*sebetahatcorrect]).T

covwrong = (all(betatrue >= CIwrong[:,0]) and all(betatrue <= CIwrong[:,1]))
covwrong2 = (betatrue >= CIwrong2[:,0] and betatrue <= CIwrong2[:,1])
covcorrect = (betatrue >= CIcorrect[:,0] and betatrue <= CIcorrect[:,1])
covboot = (betatrue >= bootCI[:,0] and betatrue <= bootCI[:,1])

betahatmedian = posteriormedianpost[0:(dimX+1)]
theta2hatmedian = posteriormeanpost[(dimX+1):(2*dimX+2)]

if usequadrature:
    muhatmedian = X @ np.diag(theta2hatmedian) @ vdraws10.T
else:
    muhatmedian = X @ np.diag(theta2hatmedian) @ vdraws.T
    
deltamedian = np.log(simshare/simoutshare)

if any(np.isnan(deltamedian)):
    deltamedian = np.log(simshare/simoutshare)

#Computing standard errors
[sebetahatcorrectmedian,sebetahatwrongmedian,sebetahatwrong2median,Ghat,W]= hlp.computeStandardErrorsforBetahat(deltamedian,betahatmedian,cdindex,cdid,muhatmedian,IV,dimX,C,weights,np.shape(muhatmedian)[1],N,Nmarkets)
#
##bootstrap
#bootdist = hlp.bootstrap(IV, deltamedian, betahatmedian, BS, NS, Nmarkets, Nproducts, cdindex, C, simshare, simoutshare, usequadrature, vdraws10, vdraws, X, Sigmatrue, theta2hatmedian, Ghat, W, m)
#
#bootpercentiles = matlab.prctile(bootdist, [100*alpha/2, 100*(1-alpha/2)])
#bootCI = np.hstack([betahatmedian - (m**(-0.5))@bootpercentiles[1:2,:].T, betahatmedian-(m**(-0.5))@bootpercentiles[1:2,:].T])
#
#CIwrongmedian = np.hstack([betahatmedian-1.96*sebetahatwrongmedian, betahatmedian+1.96*sebetahatwrongmedian])
#CIwrong2median = np.hstack([betahatmedian-1.96*sebetahatwrong2median, betahatmedian+1.96*sebetahatwrong2median])
#CIcorrectmedian = np.hstack([betahatmedian-1.96*sebetahatcorrectmedian, betahatmedian+1.96*sebetahatcorrectmedian])
#
#covwrongmedian = (betatrue >= CIwrongmedian[:,0] and betatrue <= CIwrongmedian[:,1])
#covwrong2median = (betatrue >= CIwrong2median[:,0] and betatrue <= CIwrong2median[:,1])
#covcorrectmedian = (betatrue >= CIcorrectmedian[:,0] and betatrue <= CIcorrectmedian[:,1])
#covbootmedian = (betatrue >= bootCI[:,0] and betatrue <= bootCI[:,1])
#
#s = 0
#criticalvaluessymmetricpost[:,s]=criticalvaluesymmetricpost
#coverageindicatorssymmetricpost[:,s]=(thetatrue.T >= posteriormedianpost-criticalvaluesymmetricpost) and (thetatrue.T <= posteriormedianpost+criticalvaluesymmetricpost)
#posteriormeanspost[:,s]=posteriormeanpost
#posteriormedianspost[:,s]=posteriormedianpost
#posteriorquantilesalpha2post[:,s]=posteriorquantilealpha2post
#posteriorquantilesoneminusalpha2post[:,s]=posteriorquantileoneminusalpha2post
#
#coverageindicatorspost[:,s]=(thetatrue.T>=posteriorquantilealpha2post) and (thetatrue.T <= posteriorquantileoneminusalpha2post)
#
#standarderrorscorrect[:,s]=sebetahatcorrect
#standarderrorswrong[:,s]=sebetahatwrong
#standarderrorswrong2[:,s]=sebetahatwrong2
#
#standarderrorscorrectmedian[:,s]=sebetahatcorrectmedian
#standarderrorswrongmedian[:,s]=sebetahatwrongmedian
#standarderrorswrong2median[:,s]=sebetahatwrong2median
#
#betahats[:,s]=betahat
#betahatsmedian[:,s]=betahatmedian
#
#coverageindictatorswrong2[:,s]=covwrong2
#coverageindictatorswrong[:,s]=covwrong
#coverageindictatorscorrect[:,s]=covcorrect
#coverageindicatorsbootstrap[:,s]=covboot
#
#coverageindictatorswrong2median[:,s]=covwrong2median
#coverageindictatorswrongmedian[:,s]=covwrongmedian
#coverageindictatorscorrectmedian[:,s]=covcorrectmedian
#coverageindicatorsbootstrapmedian[:,s]=covbootmedian
#
###END OF FOR LOOP
#
#coveragefreqpre= np.mean(coverageindicatorspre, axis=1)
#coveragefreqpost= np.mean(coverageindicatorspost, axis=1)
#
#coveragefreqsymmetricpre=np.mean(coverageindicatorssymmetricpre, axis=1)
#coveragefreqsymmetricpost=np.mean(coverageindicatorssymmetricpost, axis=1)
#
#covfreqwrong=np.mean(coverageindictatorswrong, axis=1)
#covfreqwrong2=np.mean(coverageindictatorswrong2, axis=1)
#covfreqcorrect=np.mean(coverageindictatorscorrect, axis=1)
#covfreqboot=np.mean(coverageindicatorsbootstrap, axis=1)
#
#covfreqwrongmedian=np.mean(coverageindictatorswrongmedian, axis=1)
#covfreqwrong2median=np.mean(coverageindictatorswrong2median, axis=1)
#covfreqcorrectmedian=np.mean(coverageindictatorscorrectmedian, axis=1)
#covfreqbootmedian=np.mean(coverageindicatorsbootstrapmedian, axis=1)
#
#standarderrorscorrectmin=standarderrorscorrect.min(axis=1)
#standarderrorscorrectmax=standarderrorscorrect.max(axis=1)
#standarderrorscorrectmean=np.mean(standarderrorscorrect, axis=1)
#standarderrorswrongmin=standarderrorswrong.min(axis=1)
#standarderrorswrongmax=standarderrorswrong.max(axis=1)
#standarderrorswrongmean=np.mean(standarderrorswrong, axis=1)
#standarderrorswrong2min=standarderrorswrong2.min(axis=1)
#standarderrorswrong2max=standarderrorswrong2.max(axis=1)
#standarderrorswrong2mean=np.mean(standarderrorswrong2, axis=1)
#
#
#standarderrorscorrectminmedian=standarderrorscorrectmedian.min(axis=1)
#standarderrorscorrectmaxmedian=standarderrorscorrectmedian.max(axis=1)
#standarderrorscorrectmeanmedian=np.mean(standarderrorscorrectmedian, axis=1)
#standarderrorswrongminmedian=standarderrorswrongmedian.min(axis=1)
#standarderrorswrongmaxmedian=standarderrorswrongmedian.max(axis=1)
#standarderrorswrongmeanmedian=np.mean(standarderrorswrongmedian, axis=1)
#standarderrorswrong2minmedian=standarderrorswrong2median.min(axis=1)
#standarderrorswrong2maxmedian=standarderrorswrong2median.max(axis=1)
#standarderrorswrong2meanmedian=np.mean(standarderrorswrong2median, axis=1)
#
#betahatmin=betahats.min(axis=1)
#betahatmax=betahats.max(axis=1)
#betahatmean=np.mean(betahats, axis=1)
#
#betahatmedianmin=betahatsmedian.min(axis=1)
#betahatmedianmax=betahatsmedian.max(axis=1)
#betahatmedianmean=np.mean(betahatsmedian, axis=1)
#
#
#
#
#
#
#
#
#
#
