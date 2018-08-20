# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:10:26 2018

@author: L1YAK01
"""

import numpy as np

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


    
    
    
    
    
    
    














                           