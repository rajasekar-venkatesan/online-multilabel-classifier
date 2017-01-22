function opH = activation_fn(ipH,ActivationFunction)
% This functions applies the selected activation function over input H and 
% generates output H
% The activation function can be any of the following:
% Sigmoidal - 'sig' or 'sigmoid'
% Sinusoidal - 'sin' or 'sine'
% Hard Limit - 'hardlim'
% Triangular Basis - 'tribas'
% Radial Basis - 'radbas'
% If further parameters are to be added for tribas/radbas include in the
% appropriate case statement.

 switch lower(ActivationFunction)
     case {'sig','sigmoid'}             %Sigmoidal Activation Function
         opH = 1 ./ (1 + exp(-ipH));
     case {'sin','sine'}                %Sine Activation Function
         opH = sin(ipH);        
     case {'hardlim'}                   %Hard Limit Activation Function
         opH = hardlim(ipH);        
     case {'tribas'}                    %Triangual Basis Activation Function
         opH = tribas(ipH);        
     case {'radbas'}                    %Radial Basis Activation Function
         opH = radbas(ipH);        
         
         %%%More activation functions can be added here        
 end
 
end