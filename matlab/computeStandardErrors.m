function [sethetahatcorrect,sethetahatwrong,sebetahatcorrect]=...
    computeStandardErrors(deltahat,betahat,theta2hat,...
    cdindex,cdid,mu,musim,IV,dimIV,dimX,X,C,vdraws,weights,NS,NSactual,N,Nmarkets)

Sigmahat=diag(theta2hat);
%Get estimated individual shares
if(NS<=10)
    [individualshares,outsideshares]=simulateMarketShares(deltahat,musim(Sigmahat),10,cdindex,cdid); %NxNS
else
    [individualshares,outsideshares]=simulateMarketShares(deltahat,mu(Sigmahat),NS,cdindex,cdid); %NxNS
end
%Cumulative sum of Gdelta matrices
%Assign off diagonal elements (IN THE SAME MARKET) to -sum(s_i*s_j)
if(NS<=10)
    Gdeltasum=-(repmat(weights,N,1).*individualshares)*individualshares';
else
    Gdeltasum=-((1/sqrt(NS))*individualshares)*((1/sqrt(NS))*individualshares');
end

%Sum of s_i
if(NS<=10)
    gsum=sum(repmat(weights,N,1).*individualshares,2);
else
    gsum=(1/NS)*sum(individualshares,2);
end

%Matrix of 1s and 0s to index markets
startindex=1;
marketindex=zeros(N,N);
Gdeltasuminverse=zeros(N,N);
for q=1:length(cdindex)
    endindex=cdindex(q);
    Gdeltasum(startindex:endindex,(endindex+1):end)=0;
    Gdeltasum((endindex+1):end,startindex:endindex)=0;
    Gdeltasumblock=Gdeltasum(startindex:endindex,startindex:endindex);
    %Assign diagonal elements to sum(s_i(1-s_i))=sum(s_i)-sum(s_i^2)
    Gdeltasumblock(logical(eye(endindex-startindex+1)))=gsum(startindex:endindex)+diag(Gdeltasumblock);
    Gdeltasuminverse(startindex:endindex,startindex:endindex)=pinv(Gdeltasumblock);
    marketindex(startindex:endindex,startindex:endindex)=1;
    startindex=endindex+1;
end



% badindicesinf=find(isinf(Gdeltasum)==1);
% badindicesnan=find(isnan(Gdeltasum)==1);
% if(~isempty(badindicesinf))
%     Gdeltasum(badindicesinf)=1;
% end
% if(~isempty(badindicesnan))
%     Gdeltasum(badindicesnan)=1;
% end
% 
% 

Ghat1=-(1/N)*IV'*C; 

Gtheta2=zeros(N,dimX,NS);
weights3d=zeros(N,dimX,NS);
for r=1:NS
%    Gtheta2(:,:,r)=(individualshares(:,r)*ones(1,dimX)).*...
%        ((X.*(ones(N,1)*vdraws(r,:)))-marketindex*...
%        ((individualshares(:,r)*ones(1,dimX)).*(X.*(ones(N,1)*vdraws(r,:)))));
 
   Gtheta2(:,:,r)=(individualshares(:,r)*ones(1,dimX)).*...
       ((X*(vdraws(r,:)'*ones(1,dimX)))-...
       (marketindex*individualshares(:,r)*ones(1,dimX)).*(X*(vdraws(r,:)'*ones(1,dimX))));
   if(NS<=10)
       weights3d(:,:,r)=weights(r)*ones(N,dimX);
   end
end

if(NS<=10)
    Gtheta2sum=sum(weights3d.*Gtheta2,3);
else
    Gtheta2sum=(1/NS)*sum(Gtheta2,3);
end
ddeltadtheta2=Gdeltasuminverse*Gtheta2sum;
Ghat2=(1/N)*IV'*ddeltadtheta2;

Ghat=[Ghat1,Ghat2];

%Construct hs for all NS
hs=-(1/N)*IV'*Gdeltasuminverse*(individualshares-gsum*ones(1,NS));

Sigmahelements=arrayfun(@(j) (hs(:,j))*(hs(:,j))', 1:NS ,'UniformOutput',0);
catA=cat(3,Sigmahelements{:}); %15x15x10
if(NS<=10)
    weights3dIVelements=arrayfun(@(r) weights(r)*ones(dimIV,dimIV), 1:NS,'UniformOutput',0);
    weights3dIV=cat(3,weights3dIVelements{:});
    Sigmah=sum(weights3dIV.*catA,3);
else
    Sigmah=(1/NS)*sum(catA,3); %for summation
end

%Omega is for the moment conditions
resid=(deltahat-C*betahat);

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
Nproducts=N/Nmarkets;
Omega=(1/(N*Nproducts))*(residIVproductsums*residIVproductsums');

k=NSactual/Nmarkets;
Sigmav=min(1,k)*Omega+min(1,(1/k))*Sigmah;
W=N*eye(dimIV)/(IV'*IV);
m=min(Nmarkets,NSactual);


varthetahatwrong=(1/N)*pinv(Ghat'*W*Ghat)*(Ghat'*W*Omega*W*Ghat)*pinv(Ghat'*W*Ghat);
% varthetahatwrong=(1/m)*pinv(Ghat'*W*Ghat);
% P=IV/(IV'*IV)*IV';
% resid2=resid.^2;
% varbetahatwrong=sum(resid2)/(N-(dimX+1))*eye(dimX+1)/(C'*P*C);

sethetahatwrong=sqrt(diag(varthetahatwrong));

varthetahatcorrect=(1/m)*pinv(Ghat'*W*Ghat)*(Ghat'*W*Sigmav*W*Ghat)*pinv(Ghat'*W*Ghat);
sethetahatcorrect=sqrt(diag(varthetahatcorrect));

varbetahatcorrect=(1/m)*pinv(Ghat1'*W*Ghat1)*(Ghat1'*W*Sigmav*W*Ghat1)*pinv(Ghat1'*W*Ghat1);
sebetahatcorrect=sqrt(diag(varbetahatcorrect));
end