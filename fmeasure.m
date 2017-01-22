function F1 = fmeasure(Actual, Predicted)
% This function evaluates the F1-measure of the Online Multi-label Classifier
% Inputs: Actual Output, Predicted Output
% Output: F1-measure value

PRSN = precision(Actual, Predicted);
RCLL = recall(Actual, Predicted);
F1 = 2*((PRSN*RCLL)/(PRSN+RCLL));

end