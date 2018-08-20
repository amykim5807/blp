function [individualshares,outsideshares]=simulateMarketShares(delta,mu,NS,cdindex)
N=size(delta,1);
individualshares=zeros(N,NS);
outsideshares=zeros(N,NS); 
% numer = exp( repmat(delta,1,NS) + mu );
% if(~isempty(find(isinf(numer), 1)))
%     disp('exponents are exploding');
%     return;
% end
startindex=1;
for q=1:length(cdindex)
    endindex=cdindex(q);
    deltat=delta(startindex:endindex,:);
    mut=mu(startindex:endindex,:);
    J=endindex-startindex+1;
    deltatmatrix=deltat*ones(1,J);
    deltatdiffs=deltatmatrix-deltatmatrix';
    sumdiffs=ones(J,NS);
    %disp(deltatdiffs)
    for r=1:NS
        %disp(mut(:,r))
        %disp(ones(1,J))
        murtmatrix=mut(:,r)*ones(1,J);
        murtdiffs=murtmatrix-murtmatrix';
        sumdiffs(:,r)=sum(exp(deltatdiffs+murtdiffs))';
    end
    
    marketdenom=exp(-(repmat(deltat,1,NS) + mut ))+sumdiffs;
    shares=1./marketdenom;
    individualshares(startindex:endindex,:)=shares;
    
    outsideshares(startindex:endindex,:)=repmat(1-sum(shares,1),J,1);
    startindex=endindex+1;
end
end
