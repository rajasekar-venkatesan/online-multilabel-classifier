%%%%% ONLINE MULTI-LABEL CLASSIFIER %%%%%
%
% This code performs Online Multi-label Classification
% The input to this code is the dataset file
% The output of this code is performance metrics of multi-label classifier
% The parameters to be set are:
%     Number of hidden layer neurons,
%     Activation function,
%     Number of samples in initial block of data and
%     Number of samples per every iteration of online learning.
% Randomize/Shuffle the dataset before running the code
% More details are provided inline
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% Clearing the MATLAB Workspace
clear all
close all
clc

%%% Initializing the Parameters
NumberofHiddenNeurons = 200;     %Number of Hidden Layer Neurons
ActivationFunction = 'sig';      %Activation Function (Refer activation_fn.m)
N0 = 200;                        %Number of Samples in Initial Block
Block = 10;                      %Number of Samples for each iteration

%%% Loading and Processing Dataset
%
%     TrainIP - Train Data Input (Features)
%     TrainOP - Train Data Output (Labels)
%     TestIP - Test Data Inpput (Features)
%     TestOP - Test Data Output (Labels)
%     TrainIP, TrainOP, TestIP, TestOP are of the following format
%     Each row representing one sample of data
%     Columns in TrainIP, TestIP corresponds to Features
%     Columns in TrainOP, TestOP corresponds to Labels
%     Each elements of TrainOP and TestOP should be +1/-1, with +1 denoting
%       that the sample belong to that label and -1 denoting that the
%       sample does not belong to that label.

load yeast_train_test.mat
TrainIP = train_data;            
TrainOP = train';                
TestIP = test_data;              
TestOP = test';                  

NumberofTrainingData=size(TrainIP,1);
NumberofTestingData=size(TestIP,1);
NumberofInputNeurons=size(TrainIP,2);
NumberofOutputNeurons=size(TrainOP,2);

NumberofTrials=5;        %Number of times the experiment should be repeated

clear train test train_data test_data t

%%% Calculate weights & biases

Training_Set_Metrics = zeros(NumberofTrials,5);
Testing_Set_Metrics = zeros(NumberofTrials,5);
for i=1:NumberofTrials          %Repeat the Experiment NumberofTrials times
    
    %%% Initialization
    P0=TrainIP(1:N0,:);
    T0=TrainOP(1:N0,:);
    IW = rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
    BiasofHiddenNeurons=rand(1,NumberofHiddenNeurons);
    
    %%% Initial Block Processing
    % Training the ELM: train_multilabel (refer train_multilabel.m)
    
    [M,beta] = train_multilabel(P0,IW',BiasofHiddenNeurons,ActivationFunction,T0,'Initial');
    clear P0 T0
    
    %%% Sequential Learning
    
    nTrainingData = NumberofTrainingData;
    for n = N0 : Block : nTrainingData
        clear tempH
        if (n+Block-1) > nTrainingData
            Pn = TrainIP(n:nTrainingData,:);    
            Tn = TrainOP(n:nTrainingData,:);
            Block = size(Pn,1);              % correct the block size
        else
            Pn = TrainIP(n:(n+Block-1),:);    
            Tn = TrainOP(n:(n+Block-1),:);
        end
        
        [M,beta] = train_multilabel(Pn,IW',BiasofHiddenNeurons,ActivationFunction,Tn,'Sequential',M,beta);
    end
    clear Pn Tn M n nTrainingData
    
%%% Prediction and Evaluation
%    Prediction: predict_multilabel (refer predict_multilabel.m)
%    Evaluation: evaluation_multilabel (refer evaluation_multilabel.m)

%
%    Evaluation of Individual Metrics:   
%       hamm_loss = hamming_loss(Actual, Predicted);  %Refer hamming_loss.m
%       accuracy = accuracy(Actual, Predicted);       %Refer accuracy.m
%       precision = precision(Actual, Predicted);     %Refer precision.m
%       recall = recall(Actual, Predicted);           %Refer recall.m
%       F1 = fmeasure(Actual, Predicted);             %Refer fmeasure.m


%%% Prediction and Evaluation on Training Data (Optional)
    
    Predicted_TrainOP = predict_multilabel(TrainIP,IW',BiasofHiddenNeurons,ActivationFunction,beta);
    Predicted = Predicted_TrainOP>0;
    Actual = TrainOP>0;
    [HL,ACC,PRSN,RCLL,F1] = evaluation_multilabel(Actual, Predicted);
    Training_Set_Metrics(i,:) = [HL,ACC,PRSN,RCLL,F1];
    
    clear HL ACC PRSN RCLL F1
    clear Predicted Actual Predicted_TrainOP  %comment this line to hold the actual output and predicted output in workspace
%%% Prediction and Evaluation on Testing Data

    Predicted_TestOP = predict_multilabel(TestIP,IW',BiasofHiddenNeurons,ActivationFunction,beta);
    Predicted = Predicted_TestOP>0;
    Actual = TestOP>0;
    [HL,ACC,PRSN,RCLL,F1] = evaluation_multilabel(Actual, Predicted);
    Testing_Set_Metrics(i,:) = [HL,ACC,PRSN,RCLL,F1];

    clear HL ACC PRSN RCLL F1
        
end

clear N0 Block
clear Predicted Actual TrainIP TrainOP TestIP TestOP Predicted_TestOP %comment this line to hold the actual output and predicted output in workspace
%display(Training_Set_Metrics)        %Uncomment if required

Evaluation_Metrics = Testing_Set_Metrics;
Evaluation_Metrics_Mean = mean(Testing_Set_Metrics);
Evaluation_Metrics_Std_Dev = std(Testing_Set_Metrics);
display('Hamming Loss, Accuracy, Precision, Recall, F-measure')
display(Evaluation_Metrics)



figure
subplot(2,3,1)
boxplot(Evaluation_Metrics(:,1)),title('Hamming Loss')
subplot(2,3,2)
boxplot(Evaluation_Metrics(:,2)),title('Accuracy')
subplot(2,3,3)
boxplot(Evaluation_Metrics(:,3)),title('Precision')
subplot(2,3,4)
boxplot(Evaluation_Metrics(:,4)),title('Recall')
subplot(2,3,5)
boxplot(Evaluation_Metrics(:,5)),title('F1-measure')