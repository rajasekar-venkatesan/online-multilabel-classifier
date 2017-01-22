function PRSN = precision(Actual, Predicted)
% This function evaluates the precision of the Online Multi-label Classifier
% Inputs: Actual Output, Predicted Output
% Output: Precision value

Num_samples = size(Actual,1);
err_count = 0;

for i = 1:length(Actual)
    err_and = and(Actual(i,:),Predicted(i,:));
    if sum(err_and) ~= 0
        err_count = err_count + (sum(err_and)/sum(Predicted(i,:)));
    end
end

PRSN = err_count / Num_samples;
end