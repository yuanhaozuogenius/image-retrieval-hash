clear 
close all
% addpath('.\utilizes');
% seed = 0;
% rng('default');
% rng(seed);

nbits_set = [16,32,64,128];
opt.top_K=5000;  %5000 or 50
dataname ='Caltech256_1024';%Caltech256_1024£¬place205£¬nuswide21,cifar10
opt.alpha = 100;


exp_data = load_data(dataname);
testgnd = exp_data.testgnd;
traingnd = exp_data.traingnd;
X = exp_data.traindata;


n_anchors = 1500;
anchor = X(randsample( size(exp_data.traindata,1), n_anchors),:);
Dis = EuDist2(X, anchor, 0);
sigma = mean(min(Dis,[],2).^0.5);
Phi_testdata = exp(-sqdist_sdh(exp_data.testdata,anchor)/(2*sigma*sigma));
Phi_traindata = exp(-sqdist_sdh(exp_data.traindata,anchor)/(2*sigma*sigma));
X = [Phi_traindata ; Phi_testdata];

%normZeroMean
data_our.indexTrain=1:size(exp_data.traindata,1);
data_our.indexTest=size(exp_data.traindata,1)+1:size(exp_data.traindata,1) + size(exp_data.testdata,1);
data_our.X = normZeroMean(X);
data_our.X = normEqualVariance(X);
data_our.label = double([traingnd;testgnd]);
yt=exp_data.testgnd;
if isvector(yt)
    Yt = sparse(1:length(yt), double(yt), 1);
    Yt = full(Yt)';
else
    Yt = yt';
end
Ytr=exp_data.traingnd;
if isvector(Ytr)
    Y = sparse(1:length(Ytr), double(Ytr), 1);
    Y = full(Y)';
else
    Y = Ytr';
end
total_res=[];


for ii = 1:length(nbits_set)        
        nbits = nbits_set(ii);        
        [MAP,precision,recall,Precision_top,NDCG]=train_DCHL(data_our, opt, nbits,Y,Yt);        
        fprintf('Bits: %d, MAP: %.4f...   \n', nbits, MAP);           
end






