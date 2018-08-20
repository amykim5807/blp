# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 15:42:59 2018

@author: L1YAK01
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:10:26 2018

@author: L1YAK01
"""

import numpy as np
import scipy.io as sio

def GH_Quadrature(Qn, N, vcv):
#    if Qn == 1:                 # Number of nodes in each dimension, Qn <=10
#        eps = [0]             # Set of integration nodes
#        weight = [sqrt(pi)]   # Set of integration weights      
#    elif Qn == 2:      
#        eps = [0.7071067811865475, -0.7071067811865475]
#        weight = [0.8862269254527580,  0.8862269254527580]
#    elif Qn == 3:
#        eps = [1.224744871391589, 0, -1.224744871391589]
#        weight = [0.2954089751509193,1.181635900603677,0.2954089751509193]
#    elif Qn == 4:
#        eps = [1.650680123885785, 0.5246476232752903,-0.5246476232752903,-1.650680123885785]
#        weight = [0.08131283544724518,0.8049140900055128, 0.8049140900055128, 0.08131283544724518]
#    elif Qn == 5:
#        eps = [2.020182870456086,0.9585724646138185,0,-0.9585724646138185,-2.020182870456086]
#        weight = [0.01995324205904591,0.3936193231522412,0.9453087204829419,0.3936193231522412,0.01995324205904591]
#    elif Qn == 6:
#        eps = [2.350604973674492,1.335849074013697,0.4360774119276165,-0.4360774119276165,-1.335849074013697,-2.350604973674492]
#        weight = [0.004530009905508846,0.1570673203228566,0.7246295952243925,0.7246295952243925,0.1570673203228566,0.004530009905508846]
#    elif Qn == 7:
#        eps = [2.651961356835233,1.673551628767471,0.8162878828589647,0,-0.8162878828589647,-1.673551628767471,-2.651961356835233]
#        weight = [0.0009717812450995192, 0.05451558281912703,0.4256072526101278,0.8102646175568073,0.4256072526101278,0.05451558281912703,0.0009717812450995192]
#    elif Qn == 8:
#        eps = [2.930637420257244,1.981656756695843,1.157193712446780,0.3811869902073221,-0.3811869902073221,-1.157193712446780,-1.981656756695843,-2.930637420257244]
#        weight = [0.0001996040722113676,0.01707798300741348,0.2078023258148919,0.6611470125582413,0.6611470125582413,0.2078023258148919,0.01707798300741348,0.0001996040722113676] 
#    elif Qn == 9:
#        eps = [3.190993201781528,2.266580584531843,1.468553289216668,0.7235510187528376,0,-0.7235510187528376,-1.468553289216668,-2.266580584531843,-3.190993201781528]
#        weight = [0.00003960697726326438,0.004943624275536947,0.08847452739437657,0.4326515590025558,0.7202352156060510,0.4326515590025558,0.08847452739437657,0.004943624275536947,0.00003960697726326438] 
#    else:
#        Qn =10 # The default option
#        eps = [3.436159118837738,2.532731674232790,1.756683649299882,1.036610829789514,0.3429013272237046,-0.3429013272237046,-1.036610829789514,-1.756683649299882,-2.532731674232790,-3.436159118837738]
#        weight = [7.640432855232621e-06,0.001343645746781233,0.03387439445548106,0.2401386110823147,0.6108626337353258,0.6108626337353258,0.2401386110823147,0.03387439445548106,0.001343645746781233,7.640432855232621e-06]
#  
    
    eps, weight = np.polynomial.hermite.hermgauss(Qn)
    # 2. N-dimensional integration nodes and weights for N uncorrelated normally 
    # distributed random variables with zero mean and unit variance
    # ------------------------------------------------------------------------ 
    n_nodes = Qn**N        # Total number of integration nodes (in N dimensions)

    z1 = np.zeros((n_nodes,N)) # A supplementary matrix for integration nodes 
                           # n_nodes-by-N 
    w1 = [1]*n_nodes # A supplementary matrix for integration weights 
                           # n_nodes-by-1
    
    for i in range(N):            
       z1i = []           # A column for variable i to be filled in with nodes 
       w1i = []           # A column for variable i to be filled in with weights 
       for j in range(Qn**(N-i-1)):
           for u in range(Qn):
               z1i += [eps[u]]*(Qn**(i))
               
               w1i += [weight[u]]*(Qn**(i))
        
       z1[:,i] = z1i      # z1 has its i-th column equal to z1i 
       w1 = np.multiply(w1,w1i)       # w1 is a product of weights w1i
    
    z = np.sqrt(2)*z1       # Integration nodes n_nodes-by-N for example, 
                           # for N = 2 and Qn=2, z = [1 1 -1 1 1 -1 -1 -1]
    
    w = w1/np.sqrt(np.pi)**N     # Integration weights see condition (B.6) in the 
                           # Supplement to JMM (2011) n_nodes-by-1
    # 3. N-dimensional integration nodes and weights for N correlated normally 
    # distributed random variables with zero mean and variance-covariance matrix, 
    # vcv 
    # -----------------------------------------------------------------------                      
    sqrt_vcv = np.linalg.cholesky(vcv).T            # Cholesky decomposition of the variance-
                                     # covariance matrix
                                     
    epsi_nodes = np.matmul(z,sqrt_vcv)         # Integration nodes; see condition (B.6)  
                                     # in the Supplement to JMM (2011); 
                                     # n_nodes-by-N                                
    
    return [n_nodes,epsi_nodes,w]

def simulateMarketShares(delta, mu, NS, cdindex):
    N = np.shape(delta)[0]
    individualshares = np.zeros((N,NS))
    outsideshares = np.zeros((N,NS))
    
    #Initializing starting index
    startindex = 0
    
    if type(cdindex)==int:
        cdindex = [cdindex]
        
    for q in range(len(cdindex)):
        endindex = cdindex[q]
        deltat = delta[startindex:endindex]
        mut = mu[startindex:endindex]
        J = endindex - startindex
        if len(np.shape(deltat))==1:
            deltat=deltat[:,None]
        deltatmatrix = np.matmul(deltat,[[1]*J])
        deltatdiffs = deltatmatrix-deltatmatrix.T
        sumdiffs = np.ones((J,NS))
        
        for r in range(NS):
            murtmatrix = np.matmul(mut[:,r][:,None],[[1]*J])
            murtdiffs = murtmatrix-murtmatrix.T
            sumdiffs[:,r] = np.sum(np.exp(deltatdiffs+murtdiffs),axis=0).T

        marketdenom = np.exp(-(np.tile(deltat,(1,NS)) + mut)) + sumdiffs
        shares = 1/marketdenom
        individualshares[startindex:endindex,:] = shares
        outsideshares[startindex:endindex,:] = np.tile((1-np.sum(shares,0)),(J,1))
        startindex = endindex
    return individualshares, outsideshares

    
def equationtosolveforprice(price, X, betatrue, mu, NS, cdindex, cdid, mc, weights):    
    individshares,outshare=simulateMarketShares(np.matmul(np.c_[X,price],betatrue),mu,NS,cdindex)
    marketshares = np.sum(np.tile(weights,(len(price),1))*individshares,axis=1)
    output = (price - mc.T[0]) + 1/(betatrue[-1]*(1-marketshares))
    return output


# Hamiltonian Metropolis Hastings Sampler
#L: objective and gradient function
#theta0: initial theta value (mx1)
#B: number of draws
#Assumes transition kernel q(x|y)=f(|x-y|) where f is Normal(0,1).
#returns Bxm matrix of theta draws.
def HMCMC(L, theta0, B):
    
    #Initialization of constants
    P = len(theta0)
    numacceptances = 0
    thetaprevious = theta0
    thetaspost = np.ones((B,P))
    C = 50
    numleaps = 10
    epsilon = 0.1
    uppthres = 0.8
    lowthres = 0.6
    cfactor = 1
    Kprevious = np.random.normal(0,1,(P,))
    
    if P==11:
        lower=[-8,1,0,-1,1,-1,1,1,1,0,1]
        upper=[-6,3,1,0,3, 0,3,3,3,2,3]
        covtheta = sio.loadmat('covthetafull.mat')
        
    else:
        lower=[-8,1,1,-1,1,1,1]
        upper=[-6,3,3, 0,3,3,3]
        covtheta = sio.loadmat('covtheta.mat')
    
    for j in range(B):
        [objold,gradold] = L(thetaprevious)
        K = Kprevious + epsilon*gradold/2
        xitheta = thetaprevious
        
        for i in range(numleaps):
            xitheta = xitheta + epsilon * K
            for p in range(P):
                if xitheta[p] < lower[p]:
                    xitheta[p] = lower[p] + (lower[p] - xitheta[p])
                    K[p] = -K[p]
                
                if xitheta[p] > upper[p]:
                    xitheta[p] = upper[p] - (xitheta[p] - upper[p])
                    K[p] = -K[p]
            
            if i != numleaps:
                [objnew,gradnew] = L(xitheta)
                K = K + epsilon*gradnew
        
        [objnew, gradnew] = L(xitheta)
        K = K + epsilon*gradnew/2
        K = -K
        randunif = np.random.uniform(0,1)
        if randunif < (np.exp(-objold + objnew + Kprevious.T @ covtheta['covtheta'] @ Kprevious/2 - K.T @ covtheta['covtheta'] @ K/2)):##FIX
            thetanew = xitheta
            numacceptances = numacceptances + 1
        else:
            thetanew = thetaprevious
        thetaprevious = thetanew
        thetaspost[j,:] = thetanew.T
        
        if j%C == 0:
            acceptanceratio = numacceptances/C
            if acceptanceratio > uppthres:
                epsilon = epsilon * (1 + cfactor * (lowthres - acceptanceratio) / lowthres)
            
            if acceptanceratio < lowthres:
                epsilon = epsilon / (1 + cfactor * (lowthres - acceptanceratio) / lowthres)
            
            numacceptances = 0
        
    return thetaspost
    
    
def computeGMMobjective(theta, simshare, simoutshare, cdindex, weights, price, X, IV, vdraws, Nproducts, N, tolerance, nogradient):
    dimX = np.shape(X)[1]
    dimIV = np.shape(IV)[1]
    Sigma = np.diagflat(theta[(dimX + 1):(2*dimX + 1)])
    musim = X @ Sigma @ vdraws.T
    delta = np.log(simshare/simoutshare) ##PLACEHOLDER FOR COMPUTEDELTAFROMSIMULATIONCCODE
    
    beta = theta[0:(dimX + 1)]
    Nmarkets = N//Nproducts
    NS = len(vdraws)
    
    if any(np.isnan(delta)):
        print('delta is nan')
        L = -float('inf')
        gradL = np.zeros((2*dimX+1,1))
    
    else:
        C = np.hstack([X, price[:,None]])
        resid = delta - C @ beta
        startindex = 0
        residIVproductsums = np.ones((dimIV, Nmarkets))
        
        for q in range(len(cdindex)):
            endindex = cdindex[q]
            residformarket = resid[:,None][startindex:endindex,:]
            ivformarket = IV[startindex:endindex, :]
            residIVproductsumformarket = np.sum(np.matmul(residformarket, np.ones((1,dimIV))) * ivformarket, axis = 0)
            residIVproductsums[:, q] = residIVproductsumformarket
            startindex = endindex
        
        Omega = (1/(N*Nproducts)) * residIVproductsums @ residIVproductsums.T
        Omegainverse = np.linalg.pinv(Omega) #pseudo-inverse of Omega matrix
        L = - min(Nmarkets,NS) / (N**2)*(delta - C @ beta).T @ IV @ Omegainverse @ IV.T @ (delta - C @ beta)
        
        if nogradient:
            gradL = np.zeros((2 * dimX + 1, 1))
    
        else:
            [individualshares, outsideshares] = simulateMarketShares(delta,musim,NS,cdindex)
            
            ddelta2 = np.zeros((N, dimX))
            Gtheta2 = np.zeros((N, dimX))
            Gdeltasuminverse = np.zeros((N,N))
            startindex = 0
            
            for q in range(len(cdindex)):
                endindex = cdindex[q]
                sharesformarket = individualshares[startindex:endindex, :]
                xformarket = X[startindex:endindex, :]
                sxvsum = np.zeros((Nproducts, dimX))
                ssxvsum = np.zeros((Nproducts, dimX))
                
                for s in range(dimX):
                    foo = sharesformarket * (xformarket[:,s:s+1] @ np.ones((1,NS))) * (np.ones((Nproducts, 1)) @ vdraws[:,s:s+1].T)
                    sxvsum[:,s:s+1] = foo @ weights.T
                    foobar = sharesformarket*(np.ones((Nproducts,1)) @ np.sum(sharesformarket*(xformarket[:,s:s+1] @ np.ones((1,NS)))*(np.ones((Nproducts,1)) @ vdraws[:,s:s+1].T), axis = 0)[None,:])
                    ssxvsum[:,s:s+1] = foobar @ weights.T
                
                Gtheta2[startindex:endindex, :] = sxvsum - ssxvsum
                tempmat = np.tile(weights,(Nproducts,1))*sharesformarket
                gsum = np.sum(tempmat, axis=1)
                gcrosssum = - (tempmat @ sharesformarket.T)
                Gdelta = gcrosssum
                Gdelta[np.identity(2)==1] = gsum + np.diag(gcrosssum)
                Gdeltasuminverse[startindex:endindex, startindex:endindex] = np.linalg.pinv(Gdelta)
                startindex = endindex
            
            ddelta2 = Gdeltasuminverse @ Gtheta2
            Gamma = np.hstack([-IV.T @ C, IV.T @ ddelta2])
            gradL = -2 * min(Nmarkets,NS)/(N**2)*Gamma.T @ Omegainverse @ IV.T @ (delta - C @ beta)
    
    if np.isnan(L):
        print('L is nan')
        L = -float('inf')
        gradL = np.zeros((2*dimX+1,1))
    
    return [L, gradL]
    
    
    
def computeStandardErrorsforBetahat(delta,betahat,cdindex,cdid,musim,IV,dimX,C,weights,NS,N,Nmarkets):
    P = np.linalg.solve((IV.T @ IV).T,IV.T).T @ IV.T
    dimIV = np.shape(IV)[1]
    
    resid = delta - C @ betahat
    resid2 = resid**2
    
    varbeta = (np.sum(resid2, axis=0)/(N-(dimX + 1))*np.identity(dimX + 1)) @ np.linalg.inv(C.T @ P @ C)
    sebetahatwrong = np.diag(varbeta)**2
    
    #Getting estimated individual shares
    [individualshares, outsideshares] = simulateMarketShares(delta, musim, NS, cdindex)
    
    #Cumulative sum of Gdelta matrices
    #Assign off diagonal elements (in the same market) to -sum(s_i*s_j)
    Gdeltasum = -(np.tile(weights, (N,1))*individualshares) @ individualshares.T
    
    #Sum of s_i
    gsum = np.sum(np.tile(weights, (N,1))*individualshares, axis = 1)
    
    #Matrix of 1s and 0s to index markets
    startindex = 0
    marketindex = np.zeros((N,N))
    Gdeltasuminverse = np.zeros((N,N))
    
    for q in range(len(cdindex)):
        endindex = cdindex[q]
        Gdeltasum[startindex:endindex,(endindex+1):] = 0
        Gdeltasum[(endindex + 1):, startindex:endindex] = 0
        Gdeltasumblock = Gdeltasum[startindex:endindex, startindex:endindex]
        
        Gdeltasumblock[np.identity(endindex-startindex) == 1] = gsum[startindex:endindex] + np.diag(Gdeltasumblock)
        Gdeltasuminverse[startindex:endindex, startindex:endindex] = np.linalg.pinv(Gdeltasumblock)
        marketindex[startindex:endindex, startindex:endindex] = 1
        startindex = endindex
    
    Ghat = -(1/N) * IV.T @ C
    
    #Construct hs for all NS
    hs = -(1/N) * IV.T @ Gdeltasuminverse @ (individualshares - gsum[:,None] @ np.ones((1,NS)))
    Sigmahelements = np.asarray([np.outer(hs[:,j], hs[:,j]) for j in range(NS)])
    weights3dIV = np.asarray([weights[0,r]*np.ones((dimIV,dimIV)) for r in range(NS)])
    Sigmah = np.sum(weights3dIV*Sigmahelements, axis=0)
    
    
    startindex = 0
    residIVproductsums = np.ones((dimIV, Nmarkets))
    
    for q in range(len(cdindex)):
        endindex = cdindex[q]
        residformarket = resid[startindex:endindex]
        ivformarket = IV[startindex:endindex, :]
        residIVproductsumformarket = np.sum((residformarket[:,None] @ np.ones((1, dimIV))) * ivformarket, axis=0)
        residIVproductsums[:,q] = residIVproductsumformarket
        startindex = endindex
    
    Nproducts = N/Nmarkets
    Omega= (1/(N*Nproducts)) * (residIVproductsums @ residIVproductsums.T)
    
    k = NS/Nmarkets
    Sigmav = min(1,k)*Omega + min(1,(1/k))*Sigmah
    W = N*np.identity(dimIV)/(IV.T @ IV)
    m = min(Nmarkets, NS)
    
    varbetahatwrong2 = (1/N)*np.linalg.pinv(Ghat.T @ W @ Ghat) @ (Ghat.T @ W @ Omega @ W @ Ghat) @ np.linalg.pinv(Ghat.T @ W @ Ghat)
    sebetahatwrong2 = np.diag(varbetahatwrong2)**0.5
    
    varbetahatcorrect = (1/m)*np.linalg.pinv(Ghat.T @ W @ Ghat) @ (Ghat.T @ W @ Omega @ W @ Ghat) @ np.linalg.pinv(Ghat.T @ W @ Ghat)
    sebetahatcorrect = np.diag(varbetahatcorrect)**0.5
    
    return [sebetahatcorrect,sebetahatwrong,sebetahatwrong2,Ghat,W]
    
def bootstrap(IV, delta, betahat, BS, NS, Nmarkets, Nproducts, cdindex, C, simshare, simoutshare, usequadrature, vdraws10, vdraws, X, Sigmatrue, theta2hat, Ghat, W, m):

    gammahat = IV.T @ (delta-C@betahat)
    bootdist = np.zeros((BS, len(betahat)))

    for bs in range(BS):
        indices = np.random.choice(Nmarkets, size=Nmarkets, replace=True)
        cdindexstar = np.asarray([cdindex[i] for i in indices])
        IVstar = IV
        Cstar = C
        simsharestar = simshare
        simoutsharestar = simoutshare
        
        for h in range(Nmarkets):
            startindex = cdindexstar[h] - Nproducts
            endindex = cdindexstar[h]
            IVstar[h*Nproducts:(h+1)*Nproducts,:] = IV[startindex:endindex, :]
            Cstar[h*Nproducts:(h+1)*Nproducts,:] = C[startindex:endindex, :]
        
        if usequadrature:
            vindices = np.random.choice(len(vdraws10), size=len(vdraws10), replace=True)
            vdrawsstar = np.asarray([vdraws10[i,:] for i in vindices])
            musimstar = X @ Sigmatrue @ vdrawsstar.T
            muhatstar = X @ np.diag(theta2hat) @ vdrawsstar.T
        
        else:
            vindices = np.random.choice(NS, size=NS, replace=True)
            vdrawsstar = np.asarray([vdraws[i,:] for i in vindices])
            musimstar = X @ Sigmatrue @ vdrawsstar.T
            muhatstar = X @ np.diag(theta2hat) @ vdrawsstar.T
        
        deltastar = np.log(simsharestar/simoutsharestar) ##COMPUTEDELtAFROMSIMULATIONCODE REPLACE
        if any(np.isnan(deltastar)):
            deltastar = np.log(simsharestar/simoutsharestar) ##COMPUTEDELtAFROMSIMULATIONCODE REPLACE
        
        gammahatstar = IVstar.T @ (deltastar - Cstar@betahat)
        bootdist[bs:bs+1,:] = -np.linalg.lstsq(Ghat.T @ W @ Ghat, m**0.5 * Ghat.T @ W @ (gammahatstar - gammahat), rcond=None)[0].T
    
    return bootdist

    
    
    