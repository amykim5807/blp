% Hamiltonian Metropolis Hastings Sampler
%L: objective and gradient function
%theta0: initial theta value (mx1)
%B: number of draws
%Assumes transition kernel q(x|y)=f(|x-y|) where f is Normal(0,1).
%returns Bxm matrix of theta draws.
function [thetaspost]=HMCMC(L,theta0,B)

P=length(theta0);
numacceptances=0;
thetaprevious=theta0;
thetaspost=ones(B,P);
C=50;
numleaps=10;
epsilon=0.1;
uppthres=0.8;
lowthres=0.6;
cfactor=1;
Kprevious=normrnd(0,1,P,1);
if(P==11)
    lower=[-8,1,0,-1,1,-1,1,1,1,0,1]';
    upper=[-6,3,1,0,3, 0,3,3,3,2,3]';
    load covthetafull.mat
else
    lower=[-8,1,1,-1,1,1,1]';
    upper=[-6,3,3, 0,3,3,3]';
    load covtheta.mat
end
%covtheta=eye(P);
for j=1:B
	   [objold,gradold]=L(thetaprevious);
	   K=Kprevious+epsilon*gradold/2;
	   xitheta=thetaprevious;
	   for i=1:numleaps
			xitheta=xitheta+epsilon*K;
			for p=1:P
				if (xitheta(p)<lower(p))
					xitheta(p) = lower(p) + (lower(p) - xitheta(p));
					K(p) = -K(p);
				end
				if (xitheta(p)>upper(p))
					xitheta(p) = upper(p) - (xitheta(p) - upper(p));
					K(p) = -K(p);
				end
			end
			if(i~=numleaps)
				[objnew,gradnew]=L(xitheta);
				K=K+epsilon*gradnew;
			end
	   end
       [objnew,gradnew]=L(xitheta);
	   K=K+epsilon*gradnew/2;
	   K=-K;
       randunif=unifrnd(0,1);
       if(randunif<exp(-objold+objnew+Kprevious'*covtheta*Kprevious./2-K'*covtheta*K./2))
           thetanew=xitheta;
           numacceptances=numacceptances+1;
       else
           thetanew=thetaprevious;
       end
       thetaprevious=thetanew;
	   thetaspost(j,:)=thetanew';
   
  
		if(rem(j,C)==0)
           acceptanceratio = numacceptances/C;
           if (acceptanceratio > uppthres)
              epsilon=epsilon*(1 + cfactor*(acceptanceratio - uppthres) / (1 - uppthres));
           end
           if (acceptanceratio < lowthres)
              epsilon=epsilon/(1 + cfactor*(lowthres - acceptanceratio) / lowthres);
           end
           numacceptances=0;
       
		end
end
