function [HL, ACC, PRSN, RCLL, F1] = evaluation_multilabel(Actual, Predicted)
% This function evaluates the multi-label classifier performance metrics
% The metrics are: hamming loss, accuracy, precision, recall, F1-measure
%    Inputs: Actual Output, Predicted Output 
%    Outputs: Hamming Loss, Accuracy, Precision, Recall, F1-measure

HL = hamming_loss(Actual, Predicted);
ACC = accuracy(Actual, Predicted);
PRSN = precision(Actual, Predicted);
RCLL = recall(Actual, Predicted);
F1 = 2*((PRSN*RCLL)/(PRSN+RCLL));

end