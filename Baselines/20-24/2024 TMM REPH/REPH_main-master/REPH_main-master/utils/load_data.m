function exp_data = load_data(dataname)
switch dataname
     case 'cifar10'
        load dataset/cifar10.mat     
        exp_data.traingnd=train_label_cifar;
        exp_data.testgnd=test_label_cifar;
        cateTrainTest = bsxfun(@eq, train_label_cifar, test_label_cifar');%similarity
        exp_data.WTT=cateTrainTest';
        exp_data.traindata = double(train_data_cifar');
        exp_data.testdata = double(test_data_cifar');
            
    case 'mnist'
        load dataset/mnist.mat         
        traingnd=traingnd+ones(length(traingnd),1);
        testgnd=testgnd+ones(length(testgnd),1);
        exp_data.traingnd=traingnd;
        exp_data.testgnd=testgnd;
        cateTrainTest = bsxfun(@eq, traingnd, testgnd');
        exp_data.WTT=cateTrainTest';
        exp_data.traindata = double(traindata);
        exp_data.testdata = double(testdata);
        
    case 'Caltech256_1024'
        load dataset/Caltech256_1024.mat 
        exp_data.traingnd=train_label_caltech256;
        exp_data.testgnd=test_label_caltech256;
        cateTrainTest = bsxfun(@eq, train_label_caltech256, test_label_caltech256');%similarity
        exp_data.WTT=cateTrainTest';
        exp_data.traindata = double(train_data_caltech256);
        exp_data.testdata = double(test_data_caltech256); 
        
    case 'place205'
        load dataset/place205.mat 
        exp_data.traingnd=train_label_place205;
        exp_data.testgnd=test_label_place205;
        cateTrainTest = bsxfun(@eq, train_label_place205, test_label_place205');%similarity
        exp_data.WTT=cateTrainTest';
        exp_data.traindata = double(train_data_place205);
        exp_data.testdata = double(test_data_place205);
   
    case 'nuswide21'
        load dataset/nuswide21.mat
 
        exp_data.traingnd=LTrain;
        exp_data.testgnd=LTest;
        exp_data.traindata = double(XTrain);
        exp_data.testdata = double(XTest);
       
       
end
end

