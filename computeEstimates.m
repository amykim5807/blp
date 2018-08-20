function [betahat,theta2hat,deltahat]=computeEstimates(deltatrue,betatrue,theta2true,...
                         cdindex,cdid,mu,musim,simshare,IV,dimIV,dimX,X,C,dimC,...
                         vdraws,weights,NS,N,Nmarkets)
[setheta0correct,setheta0wrong]=computeStandardErrors(deltatrue,betatrue,theta2true,...
                                                         cdindex,cdid,mu,musim,IV,dimIV,dimX,X,C,...
                                                         vdraws,weights,10,NS,N,Nmarkets);
                                                     
betadraw=mvnrnd(zeros(dimC-1,1),diag(setheta0correct(1:dimC-1).^2))'; 
adjustmentbeta=min(max(betadraw,-abs(betatrue(1:dimC-1))),abs(betatrue(1:dimC-1)));
beta0=[betatrue(1:dimC-1)+adjustmentbeta;betatrue(end)];

theta2draw=mvnrnd(zeros(dimC-1,1),diag(setheta0correct((dimC+1):end).^2))';
adjustmenttheta2=min(max(theta2draw,-abs(theta2true(1:dimC-1))),abs(theta2true(1:dimC-1)));
theta20=theta2true+adjustmenttheta2;
delta0=deltatrue;

% beta0=betatrue;
% theta20=theta2true;
% delta0=deltatrue;

theta0=[beta0;theta20;delta0];

% options=optimset('Algorithm','interior-point',...
%                  'GradObj','on',...
%                  'ScaleProblem','obj-and-constr');
% 
% thetaopt = fmincon(@(thetaanddelta) BLPobjAll(thetaanddelta,IV,C),thetaanddelta0,[],[],[],[],[],[],...
%     @(thetaanddelta)BLPConAll(thetaanddelta,simshare,musim,cdindex,cdid,weights,dimC,vdraws,X),options);

% thetatrue=[betatrue;theta2true;deltatrue];
% thetaLB=thetatrue-0.5*abs(thetatrue);
% thetaUB=thetatrue+0.5*abs(thetatrue);
options=optimset('Algorithm','levenberg-marquardt');
thetaopt=lsqnonlin(@(theta) BLPMoments(theta,IV,C,simshare,musim,cdindex,cdid,weights),theta0,[],[],options);

betahat=thetaopt(1:dimC);
theta2hat=thetaopt((dimC+1):(2*dimC-1));

deltahat=thetaopt((2*dimC):end);

end