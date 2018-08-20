/*
 * simulateMarketShares.c - routine in BLP estimation
 *
 * Calculate choice probabilities for each individual for each product in each market. 
 * Choice probabilities are a mixed logit integral. Use simulation to approximate the integral
 *
 * Input 
 *  product-market fixed effect delta (1 x N row vector) N = Nproducts x Nmarkets 
 *     delta(1) to delta(J_1) is the fixed effects for products 1 to J_1 in market 1
 *     delta(J_1+1) to delta(J_1+J_2) is the fixed effects for products 1 to J_2 in market J_2, so on
 *
 *  consumer specific utility mu (1 x (N x NS) row vector, NS is the number of simulation draws) 
 *    mu(1) to mu(NS) is the draws for product 1 in market 1, mu(NS+1) to mu(2*NS) is that for product 2 in market 1, so on
 *
 *  last index in delta associated with each market cdindex (1 x Nmarkets, row vector)
 *    cdindex(1) is the J_1, cdindex(2) is J_1+J_2, so one
 *  
 *  weights on each simulation point (1xNS, row vector) 
 * 
 * Outputs are 
 *  1. individualshares, 1x (NxNS) row vector 
 *     element 1 to element NS is the predicted shares of 1 in market 1 from simulation draw 1, 2,..., NS
 *  2. outside, 1x (NxNS) row vector 
 *     element 1 is the predicted share of the outside good in market 1 for simulation draw 1
 *     element 2 is the predicted share of the outside good in market 1 for simulation draw 2
 *     
 *  3. marketshare, 1xN row vector
 *     element 1 to element J_1 is the predicted market share of product 1 to J_1 in market 1 from averaging over simulation draws
 *
 * The calling syntax is:
 *
 * [individualshares,outsideshares] = simulateMarketShares(double *delta,int,double mu[][NS],int N,int Nmarkets,double   
 *  individualshares[][NS],double outsideshares[][NS])
 *
 * This is a MEX file for MATLAB.
*/ 

#include "mex.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Input Arguments */

#define	D_IN	prhs[0]
#define	M_IN	prhs[1]
#define C_IN	prhs[2]

/* Output Arguments */

#define	I_OUT	plhs[0]
#define	O_OUT	plhs[1]

/****************************************************************************/
/* Function Declarations                                                    */
/****************************************************************************/
/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

 double *delta, *mu, *cdindex, *weights,*share;                 /* input array of doubles */
 size_t N, Nmarkets, NS, NNS;			/* size of input matrices */
 double *marketshare,*error;    /* output matrices */
 

/* verifies five input */

if(nrhs != 5) {
    mexErrMsgIdAndTxt("MyToolbox:simulateMarketShares:nrhs",
                      "Three inputs required.");
}

/* verifies two output */
if(nlhs != 2) {
    mexErrMsgIdAndTxt("MyToolbox:simulateMarketShares:nlhs",
                      "Two outputs required.");
}

/* make sure the first input argument is an array */
if( !mxIsDouble(prhs[0]) || 
     mxIsComplex(prhs[0])) {
    mexErrMsgIdAndTxt("MyToolbox:simulateMarketShares:notDouble",
        "First input matrix must be type double.");
}

/* make sure the second input argument is an array */
if( !mxIsDouble(prhs[1]) || 
     mxIsComplex(prhs[1])) {
    mexErrMsgIdAndTxt("MyToolbox:simulateMarketShares:notDouble",
        "Second input matrix must be type double.");
}

/* make sure the third input argument is a scalar 
if( !mxIsDouble(prhs[2]) || 
     mxIsComplex(prhs[2]) ||
     mxGetNumberOfElements(prhs[2]) != 1 ) {
    mexErrMsgIdAndTxt("MyToolbox:simulateMarketShares:notScalar",
                      "Third input must be a scalar.");
}
*/

/* make sure the third input argument is an array of integers 
if( !mxIsDouble(prhs[2])  ||
     mxIsComplex(prhs[2]) ) {
    mexErrMsgIdAndTxt("MyToolbox:simulateMarketShares:notInteger",
                      "Third input must be type double.");
}*/

/* make sure the fourth input argument is an array of integers 
if( !mxIsInteger(prhs[3]) || 
     mxIsDouble(prhs[3])  ||
     mxIsComplex(prhs[3]) ) {
    mexErrMsgIdAndTxt("MyToolbox:simulateMarketShares:notInteger",
                      "Fifth input must be type integer.");
}*/

    /* get the value of the scalar input  
    NS = mxGetScalar(prhs[2]);*/

    /* create a pointer to the real data in the first input matrix  */
    delta = mxGetPr(prhs[0]);

    /* create a pointer to the real data in the second input matrix  */
    mu = mxGetPr(prhs[1]);	
  /*  printf("mu[1] is %d\n", *mu[1]);*/

    /* get dimensions of the input matrix */
    NNS = mxGetN(prhs[1]);   /* get the number of columns */
    N  = mxGetN(prhs[0]);   /* get the numbre of rows */
    NS = NNS/N;
    
    /* create a pointer to the real data in the third input matrix  */
    cdindex = mxGetPr(prhs[2]);
    weights = mxGetPr(prhs[3]);
    share = mxGetPr(prhs[4]);
    /*printf("cdindex[1] is %d\n", *cdindex[1]);*/
    
     /* get dimensions of the third input matrix */
    Nmarkets = mxGetN(prhs[2]);

    /* create a pointer to the real data in the fourth input matrix  
    cdid = mxGetPr(prhs[3]);*/

    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix(1,(mwSize) N,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
    /* get a pointer to the real data in the output matrix */
      marketshare  = mxGetPr(plhs[0]);
      error = mxGetPr(plhs[1]);
    /* call the computational routine */
    /* TODO: Uncomment in the final version. */
    
   simulateMarketSharesCCode(delta,mu,cdindex,weights, share, marketshare,error,(mwSize)N,(mwSize)Nmarkets,(mwSize)NS);

   return;

}

void simulateMarketSharesCCode(double *delta, double *mu, double *cdindex, double *weights, double *share, double *marketshare, double *error, mwSize N, mwSize Nmarkets, mwSize NS)
{
	mwSize q,r,c,s,endindex,J,startindex;
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
					murtdiffs[r][c] = mu[NS*(startindex+r)+s] - mu[NS*(startindex+c)+s];
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
				shares[r][s]=1/(exp(-(delta[startindex+r] + mu[(startindex+r)*NS+s])) + sumdiffs[r][s]);
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
/*void release_matrix_pointers(double **a)
{
    mxFree(a+1);
} */

