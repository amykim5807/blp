%Monte Carlo Simulations for BLP
%parpool('local',16);
% Set the seed
seed=1;
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
Nmarkets=100;
Nproducts=2;
N=Nmarkets*Nproducts
m=min(NS,Nmarkets);

if(debug)
    covariates=[hpwt,space];
else
    covariates=[hpwt,air,mpd,space];
end
ncovariates=size(covariates,2);
Xdata=[ones(size(covariates,1),1) covariates];
dimX=size(Xdata,2);
Ndata=size(Xdata,1);
alpha=0.05;
%Burn in stage
t=140%0;%0.5*N*(2*dimX+1);
%total number of draws
B=560%00;%4*t;
%Get covariance matrix of covariates in data.
covx=cov(covariates);
meanx=mean(covariates);

%Get variance of price in data
varp=var(price);
meanprice=mean(price);

%Get variance of instruments
%Instruments are exogenous characteristics, sum of characteristics across
%own firm products, and sum of characteristics across rival firm products
sum_other=[];
sum_rival=[];
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
vdraws=mvnrnd(zeros(NS,dimX),eye(dimX));
musimtrue=Xdata*Sigmatrue*vdraws';
weights=repmat(1/NS,1,NS);
end
deltahat=computeDeltaFromSimulationCCode(share,outshr,musimtrue,size(musimtrue,2),cdindex,weights,1e-4);

C=[Xdata,price];
P=IV/(IV'*IV)*IV';
betahat=(C'*P*C)\(C'*P*deltahat);
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

for s=1:MS
tic;
s
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
vdraws=mvnrnd(zeros(NS,dimX),eye(dimX));
musim=X*Sigmatrue*vdraws';
weights=repmat(1/NS,1,NS);
end
 
[individualshares,outsideshares]=simulateMarketShares(deltatrue,musim,size(musim,2),cdindex);
simshare = sum(repmat(weights,N,1).*individualshares,2); %market share of each product in each market
simoutshare = sum(repmat(weights,N,1).*outsideshares,2);  %market share of outside good.
delta=computeDeltaFromSimulationCCode(simshare,simoutshare,musim,size(musim,2),cdindex,weights,1e-4);
C=[X,price];
P=IV/(IV'*IV)*IV';
dimIV=size(IV,2);
beta0=(C'*P*C)\(C'*P*delta);
theta0=[beta0;diag(Sigmatrue)];

disp('running Hmcmc')
thetaspost= HMCMC(@(theta) computeGMMobjective(theta,simshare,simoutshare,cdindex, weights,price, X, IV,vdraws, Nproducts, N,0.001,0),theta0,B);
disp('done with Hmcmc')

% posteriormeanpre=mean(thetaspre(t:end,:))
% posteriormedianpre=median(thetaspre(t:end,:));
% thetasdemedianed=abs(thetaspre(t:end,:)-ones(B-t+1,1)*posteriormedianpre);
% criticalvaluesymmetricpre=prctile(thetasdemedianed,100*(1-alpha));

% posteriorquantilealpha2pre=prctile(thetaspre(t:end,:),100*alpha/2);
% posteriorquantileoneminusalpha2pre=prctile(thetaspre(t:end,:),100*(1-alpha/2));

posteriormeanpost=mean(thetaspost(t:end,:))
posteriormedianpost=median(thetaspost(t:end,:))

thetasdemedianed=abs(thetaspost(t:end,:)-ones(B-t+1,1)*posteriormedianpost);
criticalvaluesymmetricpost=prctile(thetasdemedianed,100*(1-alpha));

posteriorquantilealpha2post=prctile(thetaspost(t:end,:),100*alpha/2);
posteriorquantileoneminusalpha2post=prctile(thetaspost(t:end,:),100*(1-alpha/2));

%GMM standard errors

betahat=posteriormeanpost(1:(dimX+1))';
theta2hat=posteriormeanpost((dimX+2):(2*dimX+1));
if(usequadrature)
muhat=X*diag(theta2hat)*vdraws10';
else
muhat=X*diag(theta2hat)*vdraws';
end
delta=computeDeltaFromSimulationCCode(simshare,simoutshare,muhat,size(muhat,2),cdindex,weights,1e-4);
if(find(isnan(delta)))
    delta=computeDeltaFromSimulationCCode(simshare,simoutshare,musim,size(musim,2),cdindex,weights,1e-4);
end
% delta=computeDeltaFromSimulation(simshare,simoutshare,muhat,size(muhat,2),N,cdindex,weights,1e-3);
[sebetahatcorrect,sebetahatwrong,sebetahatwrong2,Ghat,W]=...
    computeStandardErrorsforBetahat(delta,betahat,...
    cdindex,cdid,muhat,IV,dimX,C,weights,size(muhat,2),N,Nmarkets);
%bootstrap
gammahat=IV'*(delta-C*betahat);
bootdist=zeros(BS,length(betahat));
for bs=1:BS
    indices=randsample(Nmarkets,Nmarkets,true); %resample at  the market level
    cdindexstar=cdindex(indices); %indices of products corresponding to resampled markets
    IVstar=IV;
    Cstar=C;
    simsharestar=simshare;
    simoutsharestar=simoutshare;
    for h=1:Nmarkets
        startindex=cdindexstar(h)-Nproducts+1;
        endindex=cdindexstar(h);
        IVstar((h-1)*Nproducts+1:h*Nproducts,:)=IV(startindex:endindex,:);
        Cstar((h-1)*Nproducts+1:h*Nproducts,:)=C(startindex:endindex,:);
       % simsharestar((h-1)*Nproducts+1:h*Nproducts,:)=simshare(startindex:endindex,:);
       % simoutsharestar((h-1)*Nproducts+1:h*Nproducts,:)=simoutshare(startindex:endindex,:);
    end
    if(usequadrature)
        vindices=randsample(size(vdraws10,1),size(vdraws10,1),true);
        vdrawsstar=vdraws10(vindices,:);
        musimstar=X*Sigmatrue*vdrawsstar';
        muhatstar=X*diag(theta2hat)*vdrawsstar';

    else
        vindices=randsample(NS,NS,true);
        vdrawsstar=vdraws(vindices,:);
        musimstar=X*Sigmatrue*vdrawsstar';
        muhatstar=X*diag(theta2hat)*vdrawsstar';

    end

    deltastar=computeDeltaFromSimulationCCode(simsharestar,simoutsharestar,muhatstar,size(muhatstar,2),cdindex,weights,1e-4);
    if(find(isnan(deltastar)))
        deltastar=computeDeltaFromSimulationCCode(simsharestar,simoutsharestar,musimstar,size(musimstar,2),cdindex,weights,1e-4);
    end
    gammahatstar=IVstar'*(deltastar-Cstar*betahat);
    bootdist(bs,:)=-(Ghat'*W*Ghat)\(Ghat'*W*sqrt(m)*(gammahatstar-gammahat));
end

bootpercentiles=prctile(bootdist,[100*alpha/2,100*(1-alpha/2)]);
bootCI=[betahat-(m^(-1/2))*bootpercentiles(2,:)',betahat-(m^(-1/2))*bootpercentiles(1,:)'];

CIwrong=[betahat-1.96*sebetahatwrong,betahat+1.96*sebetahatwrong];
CIwrong2=[betahat-1.96*sebetahatwrong2,betahat+1.96*sebetahatwrong2];
CIcorrect=[betahat-1.96*sebetahatcorrect,betahat+1.96*sebetahatcorrect];

covwrong2=(betatrue>=CIwrong2(:,1) & betatrue<=CIwrong2(:,2));
covwrong=(betatrue>=CIwrong(:,1) & betatrue<=CIwrong(:,2));
covcorrect=(betatrue>=CIcorrect(:,1) & betatrue<=CIcorrect(:,2));
covboot=(betatrue>=bootCI(:,1) & betatrue<=bootCI(:,2));

betahatmedian=posteriormedianpost(1:(dimX+1))';
theta2hatmedian=posteriormeanpost((dimX+2):(2*dimX+1));
if(usequadrature)
muhatmedian=X*diag(theta2hatmedian)*vdraws10';
else
muhatmedian=X*diag(theta2hatmedian)*vdraws';
end
deltamedian=computeDeltaFromSimulationCCode(simshare,simoutshare,muhatmedian,size(muhatmedian,2),cdindex,weights,1e-4);
if(find(isnan(deltamedian)))
    deltamedian=computeDeltaFromSimulationCCode(simshare,simoutshare,musim,size(musim,2),cdindex,weights,1e-4);
end
% deltamedian=computeDeltaFromSimulation(simshare,simoutshare,muhatmedian,size(muhat,2),N,cdindex,weights,1e-3);
[sebetahatcorrectmedian,sebetahatwrongmedian,sebetahatwrong2median,Ghat,W]=...
    computeStandardErrorsforBetahat(deltamedian,betahatmedian,...
    cdindex,cdid,muhatmedian,IV,dimX,C,weights,size(muhatmedian,2),N,Nmarkets);
%bootstrap
gammahat=IV'*(deltamedian-C*betahatmedian);
bootdist=zeros(BS,length(betahatmedian));
for bs=1:BS
    indices=randsample(Nmarkets,Nmarkets,true);
    cdindexstar=cdindex(indices);
    IVstar=IV;
    Cstar=C;
    simsharestar=simshare;
    simoutsharestar=simoutshare;
    for h=1:Nmarkets
        startindex=cdindexstar(h)-Nproducts+1;
        endindex=cdindexstar(h);
        IVstar((h-1)*Nproducts+1:h*Nproducts,:)=IV(startindex:endindex,:);
        Cstar((h-1)*Nproducts+1:h*Nproducts,:)=C(startindex:endindex,:);
        %simsharestar((h-1)*Nproducts+1:h*Nproducts,:)=simshare(startindex:endindex,:);
        %simoutsharestar((h-1)*Nproducts+1:h*Nproducts,:)=simoutshare(startindex:endindex,:);
    end
    if(usequadrature)
        vindices=randsample(size(vdraws10,1),size(vdraws10,1),true);
        vdrawsstar=vdraws10(vindices,:);
        musimstar=X*Sigmatrue*vdrawsstar';
        muhatmedianstar=X*diag(theta2hatmedian)*vdrawsstar';

    else
        vindices=randsample(NS,NS,true);
        vdrawsstar=vdraws(vindices,:);
        musimstar=X*Sigmatrue*vdrawsstar';
        muhatmedianstar=X*diag(theta2hatmedian)*vdrawsstar';

    end

    deltastar=computeDeltaFromSimulationCCode(simsharestar,simoutsharestar,muhatmedianstar,size(muhatmedianstar,2),cdindex,weights,1e-4);
    if(find(isnan(deltastar)))
        deltastar=computeDeltaFromSimulationCCode(simsharestar,simoutsharestar,musimstar,size(musimstar,2),cdindex,weights,1e-4);
    end
    gammahatstar=IVstar'*(deltastar-Cstar*betahatmedian);
    bootdist(bs,:)=-(Ghat'*W*Ghat)\(Ghat'*W*sqrt(m)*(gammahatstar-gammahat));
end

bootpercentiles=prctile(bootdist,[100*alpha/2,100*(1-alpha/2)]);
bootCI=[betahatmedian-(m^(-1/2))*bootpercentiles(2,:)',betahatmedian-(m^(-1/2))*bootpercentiles(1,:)'];

CIwrongmedian=[betahatmedian-1.96*sebetahatwrongmedian,betahatmedian+1.96*sebetahatwrongmedian];
CIwrong2median=[betahatmedian-1.96*sebetahatwrong2median,betahatmedian+1.96*sebetahatwrong2median];
CIcorrectmedian=[betahatmedian-1.96*sebetahatcorrectmedian,betahatmedian+1.96*sebetahatcorrectmedian];

covwrong2median=(betatrue>=CIwrong2median(:,1) & betatrue<=CIwrong2median(:,2));
covwrongmedian=(betatrue>=CIwrongmedian(:,1) & betatrue<=CIwrongmedian(:,2));
covcorrectmedian=(betatrue>=CIcorrectmedian(:,1) & betatrue<=CIcorrectmedian(:,2));
covbootmedian=(betatrue>=bootCI(:,1) & betatrue<=bootCI(:,2));


% criticalvaluessymmetricpre(:,s)=criticalvaluesymmetricpre;
% coverageindicatorssymmetricpre(:,s)=(thetatrue'>=posteriormedianpre-criticalvaluesymmetricpre) ...
                                    % & (thetatrue'<=posteriormedianpre+criticalvaluesymmetricpre);
% posteriormeanspre(:,s)=posteriormeanpre;
% posteriormedianspre(:,s)=posteriormedianpre;
% posteriorquantilesalpha2pre(:,s)=posteriorquantilealpha2pre;
% posteriorquantilesoneminusalpha2pre(:,s)=posteriorquantileoneminusalpha2pre;

% coverageindicatorspre(:,s)=(thetatrue'>=posteriorquantilealpha2pre) & (thetatrue'<=posteriorquantileoneminusalpha2pre);

criticalvaluessymmetricpost(:,s)=criticalvaluesymmetricpost;
coverageindicatorssymmetricpost(:,s)=(thetatrue'>=posteriormedianpost-criticalvaluesymmetricpost) ...
                                    & (thetatrue'<=posteriormedianpost+criticalvaluesymmetricpost);
posteriormeanspost(:,s)=posteriormeanpost;
posteriormedianspost(:,s)=posteriormedianpost;
posteriorquantilesalpha2post(:,s)=posteriorquantilealpha2post;
posteriorquantilesoneminusalpha2post(:,s)=posteriorquantileoneminusalpha2post;

coverageindicatorspost(:,s)=(thetatrue'>=posteriorquantilealpha2post) & (thetatrue'<=posteriorquantileoneminusalpha2post);

standarderrorscorrect(:,s)=sebetahatcorrect;
standarderrorswrong(:,s)=sebetahatwrong;
standarderrorswrong2(:,s)=sebetahatwrong2;

standarderrorscorrectmedian(:,s)=sebetahatcorrectmedian;
standarderrorswrongmedian(:,s)=sebetahatwrongmedian;
standarderrorswrong2median(:,s)=sebetahatwrong2median;

betahats(:,s)=betahat;
betahatsmedian(:,s)=betahatmedian;

coverageindictatorswrong2(:,s)=covwrong2;
coverageindictatorswrong(:,s)=covwrong;
coverageindictatorscorrect(:,s)=covcorrect;
coverageindicatorsbootstrap(:,s)=covboot;

coverageindictatorswrong2median(:,s)=covwrong2median;
coverageindictatorswrongmedian(:,s)=covwrongmedian;
coverageindictatorscorrectmedian(:,s)=covcorrectmedian;
coverageindicatorsbootstrapmedian(:,s)=covbootmedian;
toc;
save(sprintf('HMCMCtest_N%i_MS%i_NS%i_t%i_B%i_BS%i_solveprices%i_usequadratureforshares_eta%1.3g.mat',[N,MS,NS,t,B,BS,solveforprices,etaconst]));
end
coveragefreqpre=mean(coverageindicatorspre,2)
coveragefreqpost=mean(coverageindicatorspost,2)

coveragefreqsymmetricpre=mean(coverageindicatorssymmetricpre,2)
coveragefreqsymmetricpost=mean(coverageindicatorssymmetricpost,2)

covfreqwrong=mean(coverageindictatorswrong,2)
covfreqwrong2=mean(coverageindictatorswrong2,2)
covfreqcorrect=mean(coverageindictatorscorrect,2)
covfreqboot=mean(coverageindicatorsbootstrap,2)

covfreqwrongmedian=mean(coverageindictatorswrongmedian,2)
covfreqwrong2median=mean(coverageindictatorswrong2median,2)
covfreqcorrectmedian=mean(coverageindictatorscorrectmedian,2)
covfreqbootmedian=mean(coverageindicatorsbootstrapmedian,2)

standarderrorscorrectmin=min(standarderrorscorrect,[],2)
standarderrorscorrectmax=max(standarderrorscorrect,[],2)
standarderrorscorrectmean=mean(standarderrorscorrect,2)
standarderrorswrongmin=min(standarderrorswrong,[],2)
standarderrorswrongmax=max(standarderrorswrong,[],2)
standarderrorswrongmean=mean(standarderrorswrong,2)
standarderrorswrong2min=min(standarderrorswrong2,[],2)
standarderrorswrong2max=max(standarderrorswrong2,[],2)
standarderrorswrong2mean=mean(standarderrorswrong2,2)


standarderrorscorrectminmedian=min(standarderrorscorrectmedian,[],2)
standarderrorscorrectmaxmedian=max(standarderrorscorrectmedian,[],2)
standarderrorscorrectmeanmedian=mean(standarderrorscorrectmedian,2)
standarderrorswrongminmedian=min(standarderrorswrongmedian,[],2)
standarderrorswrongmaxmedian=max(standarderrorswrongmedian,[],2)
standarderrorswrongmeanmedian=mean(standarderrorswrongmedian,2)
standarderrorswrong2minmedian=min(standarderrorswrong2median,[],2)
standarderrorswrong2maxmedian=max(standarderrorswrong2median,[],2)
standarderrorswrong2meanmedian=mean(standarderrorswrong2median,2)

betahatmin=min(betahats,[],2)
betahatmax=max(betahats,[],2)
betahatmean=mean(betahats,2)

betahatmedianmin=min(betahatsmedian,[],2)
betahatmedianmax=max(betahatsmedian,[],2)
betahatmedianmean=mean(betahatsmedian,2)

save(sprintf('HMCMCtest_N%i_MS%i_NS%i_t%i_B%i_BS%i_solveprices%i_usequadratureforshares_eta%1.3g.mat',[N,MS,NS,t,B,BS,solveforprices,etaconst]));
