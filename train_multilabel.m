function [M,beta] = train_multilabel(IP,IPWeights,Bias,ActivationFunction,OP,TrainingType,varargin)
% This function trains the ELM multi-label classifier using the
% given parameters and input data
% Inputs: Samples, Input Weight Matrix, Bias Matrix, Activation Function, 
%         Output Weight Matrix, Training Type, M(for sequential training
%         type only), beta(for sequential training type only)
% Output: Raw Multi-label Predicted Output
%
% Two different training types are present:
% 1) Training Type - Initial / init
% This training type is used for the training of initial block of data
% 2) Training Type - Sequential / seq
% This training type is used for the sequential training data

numvarargs = length(varargin);
if numvarargs == 2
    M = varargin{1};
    beta = varargin{2};
else if numvarargs ~= 0
        error('Wrong number of variable arguements to the funtion train_multilabel')
    end
end

tempH=IP*IPWeights;
ind=ones(1,size(IP,1));
BiasMatrix=Bias(ind,:);
tempH=tempH+BiasMatrix;

H = activation_fn(tempH,ActivationFunction);

switch lower(TrainingType)
    case {'initial','init'}
        M = pinv(H' * H);
        beta = pinv(H) * OP;
    case {'sequential','seq'}
        M = M - M * H' * (eye(size(IP,1)) + H * M * H')^(-1) * H * M;
        beta = beta + M * H' * (OP - H * beta);
    otherwise
        error('Invalid Training Type')
end

end