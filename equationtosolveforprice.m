function output=equationtosolveforprice(price,X,betatrue,mu,NS,cdindex,cdid,mc,weights)
[individshares,outshare]=simulateMarketShares([X,price]*betatrue,mu,NS,cdindex);
marketshares=sum(repmat(weights,size(price,1),1).*individshares,2);
output=price-mc+1./(betatrue(end)*(1-marketshares));
end