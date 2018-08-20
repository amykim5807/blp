#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mex.h"

void simulateMarketShares(double *delta, int N, int Nmarkets,int NS,double mu[][NS], double *cdindex, double *weights, double *share, double *marketshare, double *error)
{
	int q,r,c,s,endindex,J,startindex;
	startindex = 0;
	double sumshares[NS];
	/* int Nproducts=N/Nmarkets; */
	for(q=0;q<Nmarkets;q++){
		endindex = (int) cdindex[q] -1;
		J=endindex-startindex+1;
		double deltatdiffs[J][J];
		for(r=0;r<J;r++){
			for(c=0;c<J;c++){
				deltatdiffs[r][c]= delta[startindex+r] - delta[startindex+c];
			}
		}

		double sumdiffs[J][NS];
		for(s=0;s<NS;s++){
			double murtdiffs[J][J];
			for(r=0;r<J;r++){
				for(c=0;c<J;c++){
					murtdiffs[r][c] = mu[startindex+r][s]-mu[startindex+c][s];
				}
			}
			for(c=0;c<J;c++){
				sumdiffs[c][s]=0;
				for(r=0;r<J;r++){
					sumdiffs[c][s]+=exp(deltatdiffs[r][c] + murtdiffs[r][c]);
				}
			} 		
			
		}

		
		double shares[J][NS];
		for(r=0;r<J;r++){
			marketshare[startindex + r] = 0;
			for(s=0;s<NS;s++){
				shares[r][s]=1/(exp(-(delta[startindex+r]+mu[startindex+r][s]))+sumdiffs[r][s]);
				/*individualshares[NS*(startindex+r)+s]=shares[r][s];*/
				marketshare[startindex + r] += weights[s]*shares[r][s];
			}
			
		}		
		for(s=0;s<NS;s++){
			sumshares[s]=0;
			for(r=0;r<J;r++){
				sumshares[s] += shares[r][s];
			}
		} 
		/*for(r=0;r<J;r++){
			for(s=0;s<NS;s++){
				outsideshares[NS*(startindex+r)+s]=1-sumshares[s];
			}
		}*/
		
		startindex=endindex+1;
		
	}
	
	double gap;
	
	for(r=0;r<N;r++){
		gap = log(share[r])-log(marketshare[r]);
		error[0] += fabs(gap/delta[r]);
		delta[r] = delta[r] + gap;
		
		
	}
	return;
}
/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
/* variable declarations here */
	int N,Nmarkets,NS;
	double delta, mu[][NS], cdindex, weights, share, marketshare, error;
	
	
	simulateMarketShares(delta, N,Nmarkets,NS,mu, cdindex, weights, share, marketshare, error);
/* code here */
}