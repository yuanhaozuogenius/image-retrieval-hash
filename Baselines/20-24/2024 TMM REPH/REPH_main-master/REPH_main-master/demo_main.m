clear all
close all

addpath(genpath(fullfile('utils/')));
addpath(genpath(fullfile('dataset/')));
seed = 0;
rng('default');
rng(seed);
opt.seed = seed;



dataname ='cifar10';
candidate_alpha = 1e-4;
candidate_beta = 1e3;


%dataname ='mnist'; 
% candidate_alpha = 1e-4;
% candidate_beta = 1e3;



% dataname ='Caltech256_1024';
% candidate_alpha = 1e-3;
% candidate_beta = 1e4;


% dataname ='place205';
% candidate_alpha = 1e-3;
% candidate_beta = 1e3;


exp_data = load_data(dataname);
testgnd = exp_data.testgnd;
traingnd = exp_data.traingnd;
X = exp_data.traindata;

% anchors construct
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


total_res=[];
top_K=50;
nbits_set =[8,16,32,64,128,256,512,1024];


for ii=1:length(nbits_set)  
            paras.alpha = candidate_alpha;
            paras.beta = candidate_beta;
            nbits=nbits_set(ii);
            [MAP,precision,recall,Precision_top,NDCG,time] =train_EPH(exp_data, nbits, paras.alpha,paras.beta,top_K);
            fprintf('bits: %d...   \n', nbits );
            fprintf('MAP result of REPH: %d...   \n', MAP );
            fprintf('NDCG result of REPH: %d...   \n', NDCG );
            fprintf('Precision_top result of REPH: %d...   \n', Precision_top );
            total_res = [total_res; MAP,Precision_top,NDCG,paras.alpha, paras.beta, nbits,time];
end




