function HL = hamming_loss(Actual, Predicted)
% This function evaluates the hamming loss of the Online Multi-label Classifier
% Inputs: Actual Output, Predicted Output
% Output: Hamming Loss
% Lower the hamming loss (Closer to zero) means better classifier

Num_labels = size(Actual,2);
Num_samples = size(Actual,1);

err_count = zeros(1,Num_labels);

    for i = 1:Num_samples
        for j = 1:Num_labels
            if Actual(i,j) ~= Predicted(i,j)
                err_count(1,j) = err_count(1,j)+1;
            end
        end
    end
    sum_err = sum(err_count);
    HL = sum_err/(Num_labels*Num_samples);
end