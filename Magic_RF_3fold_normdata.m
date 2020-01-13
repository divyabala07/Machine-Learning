%% Random Forests for normalised data before ADASYN is performed on the dataset
% to calculate the amount of time the model takes to run
tic

%% set seed
rng(1);

%% load the testing dataset
%load('norm_origin_testing.mat')

%% run treebagger on single values and see if it works on our sample

%initiate a range of values for hyperparameters
NumTree = [1, 10 , 20 , 30, 40 , 50 , 60 , 70 , 80 , 90 , 100];
MinLeaf = [1:10];
NumPredictors = [1:10];

%initiate lists to store values from the grid search
Parameters = [];
oob_Errors = [] ;  
Final = table;
Accuracy = [] ; 
ErrorAllTable = [] ;
class_error = [];
class_TP = [];
class_FP = [];
class_TN = [];
class_FN = [];
TPR = [];
TNR = [];
PPV = [];
NPV = [];
F1 = [];
AUC = [];

%define the variable types of the table
varTypes = {'double','double','double','double','double','double',...
    'double','double','double','double','double'};
%define size of the table
RF_summary_norm = table('Size',[0 11],'VariableTypes',varTypes);
%add headers for the table
RF_headers = {'Parameters' 'oob_Errors','Accuracy','class_error',...
                'TrainTime','TPR','TNR',...
                'PPV','NPV','F1','AUC'};
RF_summary_norm.Properties.VariableNames = RF_headers;

%create a new table with the same headers
RF_folds_summary_norm = RF_summary_norm;

%initiate a list
RF_summary_table_norm = [];

%initiate lists to store performance metrics
oob_Errors_mean_mdl = [];
Accuracy_mean_mdl = [];
class_error_mean_mdl = [];
TPR_mean_mdl = [];
TNR_mean_mdl = [];
PPV_mean_mdl = [];
NPV_mean_mdl = [];
F1_mean_mdl = [];
AUC_mean_mdl = [];
TrainTime_mean_mdl =[];
%generate list of class labels to use for confusion matrix 
labels_training = train_tab{:,11}; 

%initiate the grid search
disp("START")

for i = 1:length(NumTree)
    
    for j = 1:length(MinLeaf)
    
        for p = 1:length(NumPredictors)
            %initiate lists to store values for each fold
            Parameters = [];
            oob_Errors = [] ;  
            Final = table;
            Accuracy = [] ; 
            ErrorAllTable = [] ;
            class_error_mdl = [];
            class_TP_mdl = [];
            class_FP_mdl = [];
            class_TN_mdl = [];
            class_FN_mdl = [];
            TPR_mdl = [];
            TNR_mdl = [];
            PPV_mdl = [];
            NPV_mdl = [];
            F1_mdl = [];
            AUC_mdl = [];
            TrainTime =[];
            
            varTypes = {'double','double','double','double','double','double',...
                        'double','double','double','double','double'};
            RF_summary_norm = table('Size',[0 11],'VariableTypes',varTypes);
            RF_headers = {'Parameters' 'oob_Errors','Accuracy','class_error',...
                'TrainTime','TPR','TNR',...
                'PPV','NPV','F1','AUC'};
            RF_summary_norm.Properties.VariableNames = RF_headers;
            tic
            %initiate k fold cross validation
            for a = 1:k
                train_k = cell2mat(train_set(a));
                test_k = cell2mat(test_set(a));
                %to calculate the time taken for each run initiate tic
                tstart = tic
                %call treebagger model for random forests with respective
                %hyperparameters
                Mdl = TreeBagger(NumTree(i), train_k(:,1:10), train_k(:,11), 'OOBPrediction', 'on','Method','Classification', ...
                      'MinLeafSize', MinLeaf(j),'NumPredictorsToSample', NumPredictors(p)); 
                %assign runtime of each fold
                runtime = toc(tstart)
                TrainTime = [TrainTime; runtime];
                Parameters = [Parameters; NumTree(i), MinLeaf(j), NumPredictors(p)]; %input the parameters tested into an array
        
        %generate the error (or the misclassification probability) for each
        %model for out-of-bag observations in the training data 
        %using ensemble to calculate an average for all the trees in that
        %model
                model_error = oobError(Mdl, 'Mode', 'Ensemble') ;         
                Error_all = oobError(Mdl); %generate the errors for each of the trees in the model for plotting
      
        %input the oobError values into an array
                oob_Errors = [oob_Errors; model_error];
        
                %ErrorAllTable = [ErrorAllTable ; Error_all];
        
       %use the trained model to predict classes on the out of bag
       %observations stored in the model
                %[predicted_labels,posterior_mdl]= str2double(oobPredict(Mdl), test_k(:,1:10));
                [predicted_labels_new, posterior_mdl] = predict(Mdl, test_k(:,1:10));
                predicted_labels_new = str2double(predicted_labels_new)
                Sum = (sum(predicted_labels_new==test_k(:,11)))/length(predicted_labels_new);
                class_error = (1- Sum);
       %generate the confusion matrix 
                %CM_model= confusionmat(labels_training,predicted_labels);
                CM_model= confusionmat((test_k(:,11)),predicted_labels_new);
       %calculate the accuracy of the model using the confusion matrix 
                Accuracy_model = 100*sum(diag(CM_model))./sum(CM_model(:));
       
       %store the model accuracy values in an array
                Accuracy = [Accuracy; Accuracy_model] ; 
       % Calculate the metrics
                class_TN = CM_model(1,1); % true negative
                class_FN = CM_model(2,1); % false negative
                class_TP = CM_model(2,2); % true positive
                class_FP = CM_model(1,2); % false positive
                TPR = class_TP/(class_TP + class_FN); %true positive rate
                TNR = class_TN/(class_TN + class_FP); %true negative rate
                PPV = class_TP/(class_TP + class_FP); %positive prediction rate
                NPV = class_TN/(class_TN + class_FN); % negative prediction rate
                F1 = 2*(TPR*PPV)/(TPR+PPV); %F1 score
                
       % AUC
                scores = max(posterior_mdl');
                [e, f, T, AUC] = perfcurve(test_k(:,11), scores', 1);
       %join the parameters test and the model error and accuracy in a row
       %in an array 
                class_error_mdl = [class_error_mdl; class_error];
                class_TN_mdl = [class_TN_mdl; class_TN];
                class_FN_mdl = [class_FN_mdl; class_FN];
                class_TP_mdl = [class_TP_mdl; class_TP];
                class_FP_mdl = [class_FP_mdl; class_FP]; 
                TPR_mdl = [TPR_mdl; TPR];
                TNR_mdl = [TNR_mdl; TNR];
                PPV_mdl = [PPV_mdl; PPV];
                NPV_mdl = [NPV_mdl; NPV];
                F1_mdl = [F1_mdl; F1];
                AUC_mdl = [AUC_mdl; AUC];
                Final = {Parameters oob_Errors Accuracy class_error_mdl TrainTime...
                    TPR_mdl TNR_mdl PPV_mdl NPV_mdl F1_mdl AUC_mdl}      
            end 
            toc
            %calculate the error of all the performance metrics
            oob_Errors_mean = mean(oob_Errors);
            Accuracy_mean = mean(Accuracy);
            class_error_mean = mean(class_error);
            TPR_mean = mean(TPR_mdl);
            TNR_mean = mean(TNR_mdl);
            PPV_mean = mean(PPV_mdl);
            NPV_mean = mean(NPV_mdl);
            F1_mean = mean(F1_mdl);
            AUC_mean = mean(AUC_mdl);
            TrainTime_mean = mean(TrainTime);
            RF_summary_norm = [Parameters(1,:) oob_Errors_mean Accuracy_mean class_error_mean...
                            TrainTime_mean TPR_mean TNR_mean PPV_mean NPV_mean F1_mean AUC_mean]
            RF_summary_table_norm = [RF_summary_table_norm; RF_summary_norm];
            RF_folds_summary_norm = [RF_folds_summary_norm; Final];
       end
   end 
end

%% convert array to table
RF_summary_final_norm = array2table(RF_summary_table_norm,'VariableNames',{'NumTree', 'NumLeaves' ,'NumPredictors',...
                                  'oobError', 'Accuracy', 'class_error',...
                                  'TrainTime','TPR','TNR','PPV','NPV', 'F1', 'AUC'});

%% save the tables
save ('RF_summary_table_norm.mat');
save ('RF_folds_summary_norm.mat');
save ('RF_summary_final_norm.mat');

%% First convert to mat(as in array) from cell
%v2 = cell2mat (Final.Final1);
%v2(:,4) = cell2mat(Final.Final2);
%v2(:,5)= cell2mat (Final.Final3);

%% transform the array to a table
%transform the array to a table
%Final_table = array2table(v2)
%assign column names to table
%Final_table.Properties.VariableNames = {'NumTree', 'NumLeaves' ,...
 %                                 'NumPredictors', 'Error', 'Accuracy', 'TPR','TNR','PPV',...
  %                                'NPV', 'F1'} ; 

%% Automatically pick up the highest accuracy and its respective number of trees,
% minimum leaf and number of predictors to fit into the best model
highestAccuracy = max(RF_summary_final_norm{:,5})
best_model_norm = RF_summary_final_norm(RF_summary_final_norm.Accuracy == highestAccuracy, :)
best_NumTree_norm = best_model_norm{:,1} ; 
best_MinLeaf_norm = best_model_norm{:,2};
best_NumPredictors_norm = best_model_norm{:,3} ;

%% Accuracy calculation using class error
best_accuracy_new = 1 - best_model_norm{1,6}

%% Add it to the best model table
best_model_norm.accuracy_final = [best_accuracy_new]

%% %% CREATING BEST MODEL FROM OPTIMISED PARAMETERS
net_RF_3fold_norm = TreeBagger(best_NumTree_norm,test_k(:,1:10), test_k(:,11),'OOBPrediction', 'on','Method','Classification', ...
            'MinLeafSize', best_MinLeaf_norm,'NumPredictorsToSample', best_NumPredictors_norm);
%saving the best model
save('net_RF_3fold_norm.mat');

toc;

%% save the best model table 
save ('best_model_norm.mat');
