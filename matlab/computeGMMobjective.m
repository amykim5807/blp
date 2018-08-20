function [L,gradL]=computeGMMobjective(theta,simshare,simoutshare,cdindex, weights,...
                               price, X, IV,vdraws, Nproducts, N,tolerance,nogradient)
dimX=size(X,2);
dimIV=size(IV,2);
Sigma=diag(theta((dimX+2):(2*dimX+1)));
musim=X*Sigma*vdraws';
delta=computeDeltaFromSimulationCCode(simshare,simoutshare,musim,size(vdraws,1),cdindex,weights,tolerance);
%delta=computeDeltaFromSimulation(simshare,simoutshare,musim,size(vdraws,1),N,cdindex,weights,tolerance);
beta=theta(1:(dimX+1));
Nmarkets=N/Nproducts;
NS=size(vdraws,1);
if(find(isnan(delta)))
	disp('delta is nan');
    L=-Inf;
	gradL=zeros(2*dimX+1,1);
else
    C=[X,price];
	resid=(delta-C*beta);
	startindex=1;
	residIVproductsums=ones(dimIV,Nmarkets);
	for q=1:length(cdindex)
		endindex=cdindex(q);
		residformarket=resid(startindex:endindex,:);
		ivformarket=IV(startindex:endindex,:);
		residIVproductsumformarket=sum((residformarket*ones(1,dimIV)).*ivformarket); %1xdimIV
		residIVproductsums(:,q)=residIVproductsumformarket';
		startindex=endindex+1;
	end
	Omega=(1/(N*Nproducts))*(residIVproductsums*residIVproductsums');
	Omegainverse=eye(dimIV)/Omega; %pinv(Omega)
	L=-min(Nmarkets,NS)/(N^2)*(delta-C*beta)'*IV*Omegainverse*IV'*(delta-C*beta);
	if(nogradient==1)
		gradL=zeros(2*dimX+1,1);
	else
		[individualshares,outsideshares]=simulateMarketShares(delta,musim,NS,cdindex); %NxNS
		
		ddelta2=zeros(N,dimX);
		Gtheta2=zeros(N,dimX);
		Gdeltasuminverse=zeros(N,N);
		startindex=1;
		for q=1:length(cdindex)
			endindex=cdindex(q);
			sharesformarket=individualshares(startindex:endindex,:); %NproductsxNS
			xformarket=X(startindex:endindex,:); %Nproductsx(dimX)
			sxvsum=zeros(Nproducts,dimX);
			ssxvsum=zeros(Nproducts,dimX);
			for s=1:dimX
				foo=sharesformarket.*((xformarket(:,s)*ones(1,NS)).*(ones(Nproducts,1)*vdraws(:,s)')); %NproductsxNS
				sxvsum(:,s)=foo*weights';
				foobar=(sharesformarket.*(ones(Nproducts,1)*sum(sharesformarket.*(xformarket(:,s)*ones(1,NS)).*(ones(Nproducts,1)*vdraws(:,s)'))));
				ssxvsum(:,s)=foobar*weights';
			end
			Gtheta2(startindex:endindex,:)=sxvsum-ssxvsum;
			gsum=sum(repmat(weights,Nproducts,1).*sharesformarket,2);
			gcrosssum=-(repmat(weights,Nproducts,1).*sharesformarket)*sharesformarket';
			Gdelta=gcrosssum;
			Gdelta(logical(eye(Nproducts)))=gsum+diag(gcrosssum);
			Gdeltasuminverse(startindex:endindex,startindex:endindex)=pinv(Gdelta);
			startindex=endindex+1;
		end
		ddelta2=Gdeltasuminverse*Gtheta2;
		Gamma=[-IV'*C,IV'*ddelta2]; %dimIV*(2*dimX+1)
		gradL=-2*min(Nmarkets,NS)/(N^2)*Gamma'*Omegainverse*IV'*(delta-C*beta);
	end
end
if(isnan(L))
	disp('L is nan');
	L=-Inf;
	gradL=zeros(2*dimX+1,1);
end
end
