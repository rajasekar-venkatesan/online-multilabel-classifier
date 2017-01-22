function Y = predict_multilabel(Data,IPWeights,Bias,ActivationFunction,OPWeights)
% This function predicts the output of multi-label classifier using the
% trained ELM parameters
% Inputs: Samples, Input Weight Matrix, Bias Matrix, Activation Function, 
%         Output Weight Matrix
% Output: Raw Multi-label Predicted Output

    tempH=Data*IPWeights;
    ind=ones(1,size(Data,1));
    BiasMatrix=Bias(ind,:);
    tempH=tempH+BiasMatrix;
    
    Htrain = activation_fn(tempH,ActivationFunction);
    Y = Htrain * OPWeights;
    
end