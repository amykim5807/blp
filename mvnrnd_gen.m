% Set the seed to recall same generator
seed=1;

%Random number generator stream
RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));

NS = 50;
dimX = 5;

%Random number generation
vdraws=mvnrnd(zeros(NS,dimX),eye(dimX));
save('random')

% xcharacteristics=mvnrnd(ones(Nproducts,1)*meanx,covx);
% ximarket=normrnd(0,0.5*varxi)+normrnd(0,0.5*varxi,Nproducts,1);
% 
% v4=mvnrnd(ones(N,1)*meanIVnotX,covIVnotX);
