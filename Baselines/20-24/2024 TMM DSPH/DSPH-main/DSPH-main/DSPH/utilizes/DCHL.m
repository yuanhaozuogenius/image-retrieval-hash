function [B_train , B_test,Y] = DCHL (data_our, opt, nbits,Y)
seed = 0;
rng('default');
rng(seed);

gamma_SPLs=0.2;
gamma_SPLb=0.2;
[~,dFea] = size(data_our.X); %Ntrain
X  = data_our.X(data_our.indexTrain, :); X=X'; %train,
[~,Ntrain] = size(X);
X2 = data_our.X(data_our.indexTest, :); X2 = X2';%test

%----------------------------initialize-----------------------------
B=randn(nbits,Ntrain);
B(B>= 0) = 1; B(B< 0) = -1;
W=randn(nbits,dFea);
LOSS = zeros(size(B));
L_SPLs = zeros(size(LOSS,2),1);
R_SPLs = ones(size(LOSS,2),1);
L_SPLb = zeros(size(LOSS,1),1);
R_SPLb = ones(size(LOSS,1),1);
%---------------------------------training---------------------------

for iter=1:opt.Iter_num
    fprintf('iter: %d \n', iter);  
    %calculate r*V
    SPLsXT=bsxfun(@times,R_SPLs,X');
    
    %solve W  
    W = B*SPLsXT/(X*SPLsXT);
    %solve P
    [Ptmp2,~,SR] = svd(Y*B','econ');
    P = SR*Ptmp2';
    %solve B   
    AW=bsxfun(@times,R_SPLb,W);
    XC=bsxfun(@times,R_SPLs,X');XC=XC';
    B = sgn(AW*XC+opt.alpha*P*Y);
    %solve rs
    LOSS_s = zeros(size(B));
    LOSS_s = LOSS_s+bsxfun(@times,R_SPLb,W*X-B);
    %     LOSS = LOSS + W*X - B;
    for ind = 1:size(LOSS_s,2)
        L_SPLs(ind) = norm(LOSS_s(:,ind),'fro');
    end
    L_SPLs = L_SPLs';
    L_SPLs = mapminmax(L_SPLs, 0, 1);%1
    L_SPLs= L_SPLs';
       
    for ind = 1:size(L_SPLs,1)
        me = (1+exp(-1*gamma_SPLs));
        de = (1+exp(L_SPLs(ind)-gamma_SPLs));
        R_SPLs(ind) = me/de;
    end
    gamma_SPLs =  1.1 * gamma_SPLs;
      
    %solve rb
    LOSS_b = zeros(size(B));
    LOSS_b = LOSS_b+bsxfun(@times,W*X-B,R_SPLb);
    %     LOSS = LOSS + W*X - B;
    for ind = 1:size(LOSS_b,1)
        L_SPLb(ind) = norm(LOSS_b(:,ind),'fro');
    end
    L_SPLb = L_SPLb';
    L_SPLb = mapminmax(L_SPLb, 0, 1);
    L_SPLb= L_SPLb';
    
    
    for ind = 1:size(L_SPLb,1)
        me = (1+exp(-1*gamma_SPLb));
        de = (1+exp(L_SPLb(ind)-gamma_SPLb));
        R_SPLb(ind) = me/de;
    end
    gamma_SPLb =  1.1 * gamma_SPLb;
   
end


B_train=B'>0;
B_test=(W*X2)'>0;


end