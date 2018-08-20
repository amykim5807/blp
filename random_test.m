seed=1;
RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));

randn(zeros(1,5),eye(5))

random_nums = rand(1,5);

seed=1;
RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));

random_mvnrnd = normrnd(0,0.5*varxi,Nproducts,1);