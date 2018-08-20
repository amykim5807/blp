
function [sebetahatcorrect,sebetahatwrong,sebetahatwrong2,Ghat,W]=...
    computeStandardErrorsforBetahat(delta,betahat,...
    cdindex,cdid,musim,IV,dimX,C,weights,NS,N,Nmarkets)

P=IV/(IV'*IV)*IV';
dimIV=size(IV,2);

resid=(delta-C*betahat);

resid2=resid.^2;


varbeta=sum(resid2)/(N-(dimX+1))*eye(dimX+1)/(C'*P*C);
sebetahatwrong=sqrt(diag(varbeta));


%Get estimated individual shares

[individualshares,outsideshares]=simulateMarketShares(delta,musim,NS,cdindex); %NxNS

%Cumulative sum of Gdelta matrices
%Assign off diagonal elements (IN THE SAME MARKET) to -sum(s_i*s_j)
Gdeltasum=-(repmat(weights,N,1).*individualshares)*individualshares';

%Sum of s_i
gsum=sum(repmat(weights,N,1).*individualshares,2);

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



Ghat=-(1/N)*IV'*C; 


%Construct hs for all NS
hs=-(1/N)*IV'*Gdeltasuminverse*(individualshares-gsum*ones(1,NS));

Sigmahelements=arrayfun(@(j) (hs(:,j))*(hs(:,j))', 1:NS ,'UniformOutput',0);
catA=cat(3,Sigmahelements{:}); %15x15x10
weights3dIVelements=arrayfun(@(r) weights(r)*ones(dimIV,dimIV), 1:NS,'UniformOutput',0);
weights3dIV=cat(3,weights3dIVelements{:});
Sigmah=sum(weights3dIV.*catA,3);
% else
    % Sigmah=(1/NS)*sum(catA,3); %for summation
% end
%Omega is for the moment conditions

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

k=NS/Nmarkets;
Sigmav=min(1,k)*Omega+min(1,(1/k))*Sigmah;
W=N*eye(dimIV)/(IV'*IV);
m=min(Nmarkets,NS);


varbetahatwrong2=(1/N)*pinv(Ghat'*W*Ghat)*(Ghat'*W*Omega*W*Ghat)*pinv(Ghat'*W*Ghat);
sebetahatwrong2=sqrt(diag(varbetahatwrong2));

varbetahatcorrect=(1/m)*pinv(Ghat'*W*Ghat)*(Ghat'*W*Sigmav*W*Ghat)*pinv(Ghat'*W*Ghat);
sebetahatcorrect=sqrt(diag(varbetahatcorrect));


end
