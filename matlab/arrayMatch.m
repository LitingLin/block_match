function Result = arrayMatch(A, B, MeasureMethod)
% MeasureMethod can be
%  'mse', 'cc'
%  'mse' for default

if nargin < 2
    error('Too few input parameters');
end

if nargin == 2
    MeasureMethod = 'mse';
end

Result = arrayMatchMex(A, B, MeasureMethod);

end
