kerneltype = 'poly';

synDir = 'Synthetic/';
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
            test_data = importdata(strcat(synDir, cv_fold, num2str(fold), fold_test));
            train_data = importdata(strcat(synDir, cv_fold, num2str(fold), fold_train));

            train_labels = train_data(:,3);
            train_features = train_data(:,1:2);

            % train on q(i), C(j)
            if strcmp(kerneltype, 'poly')
                model = svmtrain(train_labels, train_features, sprintf('-t 1 -d %f -g 1 -r 1 -c %f -q', q(i), C(j))); 
            else
                model = svmtrain(train_labels, train_features, sprintf('-t 2 -g %f -c %f -q', sig(i), C(j))); 
            end

            test_labels = test_data(:,3);
            test_features = test_data(:,1:2);

            pred = svmpredict(test_labels, test_features, model);

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
test_data = importdata(strcat(synDir, 'test.mat'));
train_data = importdata(strcat(synDir, 'train.mat'));

train_labels = train_data(:,3);
train_features = train_data(:,1:2);

testerrors = zeros(1,5);
trainerrors = zeros(1,5);

i = 1;
while i < 6
    iter = iter + 1
    % train on q(i) with bestC(i)
    if strcmp(kerneltype, 'poly')
        model = svmtrain(train_labels, train_features, sprintf('-t 1 -d %f -g 1 -r 1 -c %f -q', q(i), bestC(i))); 
    else
        model = svmtrain(train_labels, train_features, sprintf('-t 2 -g %f -c %f -q', sig(i), bestC(i))); 
    end
    
    %Plot the decision boundary for this q
%     if strcmp(kerneltype, 'poly')
%         decision_boundary_SVM(train_features, train_labels, model, 100, sprintf('q_%d', q(i)));
%     else
%         decision_boundary_SVM(train_features, train_labels, model, 100, sprintf('sigma_%d', sig(i)));
%     end
    % pred_labels = your prediction on the test data
    

    test_features = test_data(:,1:2);

    % Use this to build vector for test_data
    w = model.SVs' * model.sv_coef;
    b = -model.rho;

    if model.Label(1) == -1
      w = -w;
      b = -b;
    end
    
    test_labels = test_data(:,3);

    pred_test_labels = svmpredict(test_labels, test_features, model);
    pred_train_labels = svmpredict(train_labels, train_features, model);
    
    
    trainerrors(i) = classification_error(pred_train_labels, train_labels);
    testerrors(i) = classification_error(pred_test_labels, test_labels);
    i = i + 1;
    
    

end

hold on
if strcmp(kerneltype, 'poly')
    plot(q, testerrors)
    plot(q, trainerrors)
    title('Testing and Training errors as a function of q')
    legend('Testing error', 'Training error')
else
    plot(log(sig), testerrors)
    plot(log(sig), trainerrors)
    title('Testing and Training errors as a function of log(sigma)')
    legend('Testing error', 'Training error')
end
hold off
% Do prediction on the test data

