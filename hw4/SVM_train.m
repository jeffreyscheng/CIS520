function [pred_labels] = SVM_train(test_data, kerneltype)
    % INPUT : 
    % test_data   - m X n matrix, where m is the number of test points and n is number of features
    % kerneltype  - one of strings 'poly', 'rbf'
    %               corresponding to polynomial, and RBF kernels
    %               respectively.
    
    % OUTPUT
    % returns a m X 1 vector predicted labels for each of the test points. The labels should be +1/-1 doubles

    
    % Default code below. Fill in your code on all the relevant positions

    m = size(test_data , 1);
    n = size(test_data, 2);

    %load train_data

    datadir = 'Breast-Cancer/';

    load(strcat(datadir,'train.mat'));


    %load cross-validation data

    %your code
    
    fold_test = '/cv-test.mat';
    fold_train = '/cv-train.mat';
    cv_fold = 'CrossValidation/Fold';



    % Do cross-validation
    % For all c
    % For all kernel parameters
    % Calculate the average cross-validation error for the 5-folds
    
    C = [1, 10, 10^2, 10^3, 10^4, 10^5];
    q = [1, 2, 3, 4, 5];
    sig = [0.01, 1, 10, 10^2, 10^3];
    i = 1;
    
    bestC = zeros(1,5);
    iter = 0;
    
    % Loop through possible q values
    while i < 6
        j = 1;
        
        error_for_c = zeros(1,6);
        % Loop through possible C values
        while j < 7
            fold = 1;
            
            errors = zeros(1,5);
            % Loop through all Folds
            while fold < 6
                iter = iter + 1
                test_data = importdata(strcat(datadir, cv_fold, num2str(fold), fold_test));
                train_data = importdata(strcat(datadir, cv_fold, num2str(fold), fold_train));
                
                train_labels = train_data(:,10);
                train_features = train_data(:,1:9);
               
                % train on q(i), C(j)
                if strcmp(kerneltype, 'poly')
                    model = svmtrain(train_labels, train_features, sprintf('-t 1 -d %f -g 1 -r 1 -c %f -q', q(i), C(j))); 
                else
                    model = svmtrain(train_labels, train_features, sprintf('-t 2 -g %f -c %f -q', sig(i), C(j))); 
                end

                test_labels = test_data(:,10);
                test_features = test_data(:,1:9);

                % Use this to build vector for test_data
                w = model.SVs' * model.sv_coef;
                b = -model.rho;

                if model.Label(1) == -1
                  w = -w;
                  b = -b;
                end

                pred = sign(test_features * w + b);

                errors(fold) = classification_error(pred, test_labels);
                
                fold = fold + 1;
            end
            error_for_c(j) = mean(errors);
            j = j +1;
            
        end
        [val, idx] = min(error_for_c);
        bestC(i) = C(idx);
        i = i + 1;
    end
    
    

    %your code


    %Train SVM on training data for the best parameters
    train_data = importdata(strcat(datadir, 'train.mat'));

    train_labels = train_data(:,10);
    train_features = train_data(:,1:9);
    
    i = 1;
    while i < 6
        % train on q(i) with bestC(i)
        if strcmp(kerneltype, 'poly')
            model = svmtrain(train_labels, train_features, sprintf('-t 1 -d %f -g 1 -r 1 -c %f -q', q(i), bestC(i))); 
        else
            model = svmtrain(train_labels, train_features, sprintf('-t 2 -g %f -c %f -q', sig(i), bestC(i))); 
        end
        
    end

    % Do prediction on the test data
    % pred_labels = your prediction on the test data
    % your code

    test_features = test_data(:,1:9);


    pred_labels = svmpredict(ones(683,1), test_features, model);

end
