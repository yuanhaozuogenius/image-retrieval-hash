 function [MAP,precision,recall,Precision_top,NDCG] =train_DCHL(data_our, opt, nbits,Y,Yt)
    opt.Iter_num=5;    
%----------------------Hash codes-------------------------
    [B_trn, B_tst,~] = DCHL(data_our,opt, nbits,Y);   
    B1 = compactbit(B_trn);
    B2 = compactbit(B_tst);  
%----------------------Evaluation-------------------------
    DHamm = hammingDist(B2, B1);
    [~, orderH] = sort(DHamm, 2);
%     MAP = calcMAP(orderH, exp_data.WTT);   
    MAP=cal_map( Y',Yt', orderH');
    [precision, recall] = precision_recall(orderH', Y',Yt');
    NDCG=ndcg2_k(orderH,Y',Yt',100);
    Precision_top_tmp = precision_at_k(orderH', Y',Yt',opt.top_K);
    Precision_top = mean(Precision_top_tmp);
end
