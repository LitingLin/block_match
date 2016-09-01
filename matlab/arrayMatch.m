function Result = arrayMatch(A, B, MeasureMethod)
if nargin < 2
    error('Too few input parameter');
end

if nargin == 2
    MeasureMethod = 'mse';
end

Result = arrayMatchMex(A, B, MeasureMethod);

end
