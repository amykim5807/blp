function delta=computeDeltaFromSimulationCCode(share,outshare,mu,NS,cdindex,weights,tolerance)

relativemarketshares=share./outshare;
delta=log(relativemarketshares);
error=1;
maxiters=1e6;
iter=0;        
delta = delta';
mu = mu';
mu = mu(:);
mu = mu';
cdindex = cdindex';
share = share';


if NS > 10;
   weights = ones(1,NS)/NS;
end;
while (error > tolerance && iter<maxiters)
    iter=iter+1;
    [marketshares,error]=simulateMarketSharesCCode(delta,mu,cdindex,weights,share); %NxNS
    %c code re writes delta.
%     if(sum(marketshares)==0)
%         delta=NaN;
%         return;
%     end
    % deltaprevious=delta;
    % delta=deltaprevious+(log(share)-log(marketshares));
    % error=norm((delta-deltaprevious)./deltaprevious,1);
	% delta = delta';
end

delta = delta';
end