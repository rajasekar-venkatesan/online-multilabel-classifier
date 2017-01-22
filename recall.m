function RCLL = recall(Actual, Predicted)
% This function evaluates the recall of the Online Multi-label Classifier
% Inputs: Actual Output, Predicted Output
% Output: Recall value

Num_samples = size(Actual,1);
err_count = 0;

for i = 1:length(Actual)
    err_and = and(Actual(i,:),Predicted(i,:));
    err_count = err_count + (sum(err_and)/sum(Actual(i,:)));
end

RCLL = err_count / Num_samples;
end