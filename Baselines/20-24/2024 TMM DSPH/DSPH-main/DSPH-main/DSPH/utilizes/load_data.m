function exp_data = load_data(dataname)
switch dataname
    case 'cifar10'
        load cifar10.mat
        exp_data.traingnd=train_label_cifar;
        exp_data.testgnd=test_label_cifar;
        exp_data.traindata = double(train_data_cifar');
        exp_data.testdata = double(test_data_cifar');
    case 'Caltech256_1024'
        load Caltech256_1024.mat
        exp_data.traingnd=train_label_caltech256;
        exp_data.testgnd=test_label_caltech256;
        exp_data.traindata = double(train_data_caltech256);
        exp_data.testdata = double(test_data_caltech256);
    case 'place205'
        load place205.mat
        exp_data.traingnd=train_label_place205;
        exp_data.testgnd=test_label_place205;
        exp_data.traindata = double(train_data_place205);
        exp_data.testdata = double(test_data_place205);
    case 'nuswide21'
        load nuswide21.mat
        exp_data.traingnd=LTrain;
        exp_data.testgnd=LTest;
        exp_data.traindata = double(XTrain);
        exp_data.testdata = double(XTest);
end
end

