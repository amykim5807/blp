% Set the seed to recall same generator
seed=1;

%Random number generator stream
RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));

close all;

debug=0;
solveforprices=1;
usequadrature=0;

load  BLP_data.mat
% Number of Monte Carlo Simulations
MS=200

%Number of Simulations
NS=50

BS=2000
Nmarkets=10;
Nproducts=2;

%Total number of products
N=Nmarkets*Nproducts

%Taking the smaller value of NS and Nmarkets
m=min(NS,Nmarkets);

if(debug)
    covariates=[hpwt,space];
else
    covariates=[hpwt,air,mpd,space];
end
%Size of row of ncovariates (number of covariates)
ncovariates=size(covariates,2);

%Adding a column of ones to the front of the covariates matrix
Xdata=[ones(size(covariates,1),1) covariates];

%ncovariates + 1
dimX=size(Xdata,2);

%Number of rows
Ndata=size(Xdata,1);
alpha=0.05;

%Burn in stage
t=140%0;%0.5*N*(2*dimX+1);

%total number of draws
B=560%00;%4*t;

%Get covariance matrix of covariates in data.
covx=cov(covariates);

%Get mean of each covariate
meanx=mean(covariates);

%Get variance of price in data
varp=var(price);
meanprice=mean(price);

%Get variance of instruments
%Instruments are exogenous characteristics, sum of characteristics across
%own firm products, and sum of characteristics across rival firm products

%Other = other products in the same market, from the same firm
sum_other=[]; %This is the sum of each characteristics from Other Products for a given product

%Rival = products in the same market, from a different firm
sum_rival=[]; %This is the sum of each characteristic from Rival Products for a given product

for i=1:size(id,1)
        other_ind=(firmid==firmid(i)  & cdid==cdid(i) & id~=id(i));   % id contains unique identifier of each product x market
        % cdid contains flag for each market and firmid marks the firm
        % producing the product
        rival_ind=(firmid~=firmid(i)  & cdid==cdid(i));
        total_ind=(cdid==cdid(i));
        sum_other(i,:)=sum(Xdata(other_ind==1,:));
        sum_rival(i,:)=sum(Xdata(rival_ind==1,:));
end

IV=[Xdata,sum_other,sum_rival];
covIVnotX=cov(IV(:,(dimX+1):end));
varIVnotX=var(IV(:,(dimX+1):end));
meanIVnotX=mean(IV(:,(dimX+1):end));
dimIVnotX=size(meanIVnotX,2);


%Get variance of xi=deltahat-x*beta from simulation
%assume some true values for Sigma
if(debug)
    theta2true=[2.009,1.586,1.51]';
    Sigmatrue=diag(theta2true);
    betatrue=[-7.304,2.185,2.604,-0.2]';
    gammatrue=[0.726,0.313,1.499]';
else
    theta2true=[2.009,1.586,1.215,0.67,1.51]';
    Sigmatrue=diag(theta2true);
    betatrue=[-7.304,2.185,0.579,-0.049,2.604,-0.2]';
    gammatrue=[0.726,0.313,0.290,0.293,1.499]';
end
thetatrue=[betatrue;theta2true];

if(usequadrature)
[J,vdraws10,weights] = GH_Quadrature(10,dimX,eye(dimX));
weights=weights'; %need weights to be row vector (1xNS)
musimtrue=Xdata*Sigmatrue*vdraws10';
else
vdraws1=mvnrnd(zeros(NS,dimX),eye(dimX));
musimtrue=Xdata*Sigmatrue*vdraws1';
weights=repmat(1/NS,1,NS);
end
deltahat=computeDeltaFromSimulationCCode(share,outshr,musimtrue,size(musimtrue,2),cdindex,weights,1e-4);

C=[Xdata,price];
P1=IV/(IV'*IV)*IV';
betahat=(C'*P1*C)\(C'*P1*deltahat);
xihat=(deltahat-C*betahat);
varxi=var(xihat);

%Get covariance between xihat and price
covmatpricexihat=cov(xihat,price);
covpricexihat=covmatpricexihat(1,2);

coverageindictatorscorrect=zeros(dimX+1,MS);
coverageindictatorswrong=zeros(dimX+1,MS);
coverageindictatorswrong2=zeros(dimX+1,MS);
coverageindictatorsdeltacorrect=zeros(dimX+1,MS);
coverageindicatorsbootstrap=zeros(dimX+1,MS);

standarderrorscorrect=zeros(dimX+1,MS);

standarderrorswrong=zeros(dimX+1,MS);
standarderrorswrong2=zeros(dimX+1,MS);
betahats=zeros(dimX+1,MS);

coverageindictatorscorrectmedian=zeros(dimX+1,MS);
coverageindictatorswrongmedian=zeros(dimX+1,MS);
coverageindictatorswrong2median=zeros(dimX+1,MS);
coverageindictatorsdeltacorrectmedian=zeros(dimX+1,MS);
coverageindicatorsbootstrapmedian=zeros(dimX+1,MS);

standarderrorscorrectmedian=zeros(dimX+1,MS);

standarderrorswrongmedian=zeros(dimX+1,MS);
standarderrorswrong2median=zeros(dimX+1,MS);
betahatsmedian=zeros(dimX+1,MS);

posteriormeanspre=zeros(2*dimX+1,MS);
posteriormedianspre=zeros(2*dimX+1,MS);
posteriorquantilesalpha2pre=zeros(2*dimX+1,MS);
posteriorquantilesoneminusalpha2pre=zeros(2*dimX+1,MS);
coverageindicatorspre=zeros(2*dimX+1,MS);
coverageindicatorssymmetricpre=zeros(2*dimX+1,MS);
criticalvaluessymmetricpre=zeros(2*dimX+1,MS);

posteriormeanspost=zeros(2*dimX+1,MS);
posteriormedianspost=zeros(2*dimX+1,MS);
posteriorquantilesalpha2post=zeros(2*dimX+1,MS);
posteriorquantilesoneminusalpha2post=zeros(2*dimX+1,MS);
coverageindicatorspost=zeros(2*dimX+1,MS);
coverageindicatorssymmetricpost=zeros(2*dimX+1,MS);
criticalvaluessymmetricpost=zeros(2*dimX+1,MS);

acceptanceratiostokeep=zeros(2*dimX+1,MS);
sigmastokeep=zeros(2*dimX+1,MS);


%IN A FOR LOOP x200
tic;
X=zeros(N,dimX);
xi=zeros(N,1);
%Generate own data
for j=1:Nmarkets
    xcharacteristics=mvnrnd(ones(Nproducts,1)*meanx,covx);
    X(((j-1)*Nproducts+1):(j*Nproducts),:)=[ones(Nproducts,1) xcharacteristics];
    ximarket=normrnd(0,0.5*varxi)+normrnd(0,0.5*varxi,Nproducts,1);
    xi(((j-1)*Nproducts+1):(j*Nproducts),:)=ximarket;
end

v1=normrnd(0,abs(covpricexihat-varxi),Nmarkets*Nproducts,1);
v2=normrnd(0,varxi,Nmarkets*Nproducts,1);
v3=normrnd(0,abs(varp-varxi),Nmarkets*Nproducts,1);
v4=mvnrnd(ones(N,1)*meanIVnotX,covIVnotX);

IV=[X,v3*ones(1,size(v4,2))+v4];
etaconst=0.001;
eta=etaconst*(v1+v3);

mc=X*gammatrue+eta;

  
cdid=kron((1:Nmarkets)',ones(Nproducts,1));
cdindex=(Nproducts:Nproducts:N)';

if(solveforprices)
    [J,vdraws10,weights] = GH_Quadrature(10,dimX,eye(dimX));
    weights=weights';
    musim=X*Sigmatrue*vdraws10';

    %Solve for prices assuming Bertrand Nash competition
    disp('solving for prices using Bertrand-Nash');
    price=ones(N,1);
    profits=ones(Nmarkets,1);
    for j=1:Nmarkets
        cdindexformarket=cdindex(j);
        opts = optimset('Algorithm','active-set','Display','off');
        p0=rand([Nproducts,1]);
        lb=0.01*ones(Nproducts,1);
        ub=1e2*ones(Nproducts,1);
        options=optimset('Algorithm','trust-region-reflective','Display','off');
        priceformarket=lsqnonlin(@(price) equationtosolveforprice(price,X((cdindexformarket-Nproducts+1):cdindexformarket,:),...
            betatrue,musim((cdindexformarket-Nproducts+1):cdindexformarket,:),size(musim,2),Nproducts,1,...
            mc((cdindexformarket-Nproducts+1):cdindexformarket,:),weights),...
            p0,lb,ub,options);
        disp(priceformarket)
        price((cdindexformarket-Nproducts+1):cdindexformarket,:)=priceformarket;
    end

else
  trunc=truncate(makedist('Normal'),0,1);
  e=random(trunc,N,1);
  price=0.5*abs(2+0.5*xi+e+sum(X,2));
end

deltatrue=[X,price]*betatrue+xi;

 %mu is a NxNS matrix
if(usequadrature)
[J,vdraws10,weights] = GH_Quadrature(10,dimX,eye(dimX));%vdraws10 (10^dimX)xdimX
weights=weights';
musim=X*Sigmatrue*vdraws10';
else
vdraws2=mvnrnd(zeros(NS,dimX),eye(dimX));
musim=X*Sigmatrue*vdraws2';
weights=repmat(1/NS,1,NS);
end
 
[individualshares,outsideshares]=simulateMarketShares(deltatrue,musim,size(musim,2),cdindex);
simshare = sum(repmat(weights,N,1).*individualshares,2); %market share of each product in each market
simoutshare = sum(repmat(weights,N,1).*outsideshares,2);  %market share of outside good.
delta=computeDeltaFromSimulationCCode(simshare,simoutshare,musim,size(musim,2),cdindex,weights,1e-4);
C=[X,price];
P2=IV/(IV'*IV)*IV';
dimIV=size(IV,2);
beta0=(C'*P2*C)\(C'*P2*delta);
theta0=[beta0;diag(Sigmatrue)];