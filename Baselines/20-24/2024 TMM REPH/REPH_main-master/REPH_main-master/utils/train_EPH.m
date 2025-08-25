function [MAP,precision,recall,NDCG,Precision_top,time] = train_EPH(exp_data, nbits , alpha,beta,top_K)
seed = 0;
rng('default');
rng(seed);
fprintf('Preprocessing...\n');
X = exp_data.traindata;
yt=exp_data.testgnd;
if isvector(yt)
    Yt = sparse(1:length(yt), double(yt), 1); 
    Yt = full(Yt)';
else
    Yt = yt';
end

n_anchors = 1500; 
anchor = X(randperm(size(exp_data.traingnd,1), n_anchors),:);
Dis = EuDist2(X,anchor,0);
sigma = mean(min(Dis,[],2).^0.5);
Phi_testdata = exp(-sqdist_sdh(exp_data.testdata,anchor)/(2*sigma*sigma));
Phi_traindata = exp(-sqdist_sdh(exp_data.traindata,anchor)/(2*sigma*sigma));
X=[Phi_traindata ; Phi_testdata];

data_our.indexTrain=1:size(exp_data.traindata,1);
data_our.indexTest=size(exp_data.traindata,1)+1:size(exp_data.traindata,1) + size(exp_data.testdata,1);
data_our.X=normZeroMean(X);
data_our.X=normEqualVariance(X);
data_our.label=exp_data.traingnd;

% Training
fprintf('Training...\n');
[U_logical_trn,U_logical_tst,Y,time]= EPH(data_our,nbits, alpha,beta);

% Evaluation
fprintf('\nEvaluating...\n');
B_compact_trn = compactbit(U_logical_trn);
B_compact_tst = compactbit(U_logical_tst);
DHamm = hammingDist(B_compact_tst, B_compact_trn);
[~, orderH] = sort(DHamm, 2);

MAP = cal_map(Y',Yt',orderH');
[precision, recall] = precision_recall(orderH', Y',Yt');
% save('pre_rec_place_EPSH.mat','precision','recall');
NDCG=ndcg2_k(orderH,Y',Yt',100);
Precision_top_tmp = precision_at_k(orderH', Y',Yt',top_K);
Precision_top = mean(Precision_top_tmp);
end

