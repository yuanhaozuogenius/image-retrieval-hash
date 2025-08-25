function [B_train , B_test,Y,time ] = EPH (data_our, nbits,alpha,beta)
seed = 0;
rng('default');
rng(seed);

Iter_num = 5;
N1 = length(data_our.indexTrain)'; %Ntrain
X = data_our.X(data_our.indexTrain, :)'; %train
X2 = data_our.X(data_our.indexTest, :)'; %test
y=data_our.label(data_our.indexTrain, :)';

if isvector(y)
    Y = sparse(1:length(y), double(y), 1);
    Y = full(Y)';
else
    Y = y';
end


%---------------initialize---------------------------
[dv,~] = size(X);
B=randn(nbits,N1);
B=B>0;
Q=randn(nbits,dv);
[P1,~,SR] = svd(Q*X*X','econ');
P = (P1*SR')';
[R1,~,SR] = svd(Q*X*B','econ');
R = (R1*SR')';
% W=(beta*B*Y')/(beta*Y*Y'+lambda*eye(size(Y,1)));

tic
%-----------------------------------------------------training---------------------------------
for iter=1:Iter_num
    fprintf('%d...',iter);
    
    L_old(iter) = norm(X-P*Q*X,'fro');
    %--------------solve Q
    Q=(R'*B*X'+alpha*P'*X*X')/((1+alpha)*X*X');
    %--------------solve P
    [P1,~,SP] = svd(Q*X*X','econ');
    P = (P1*SP')';
    %--------------solve R
    [R1,~,SR] = svd(Q*X*B','econ');
    R = (R1*SR')'; 
    %--------------solve W
    [W1,~,SW] = svd(Y*B','econ');
    W= (W1*SW')';
    %--------------solve B
    B =sgn(R*Q*X+beta*W*Y);
    
    


end
  

time=toc
B_train=B'>0;

%----------------------Out-of-Sample
B_test=(R*Q*X2)'>0;

end



