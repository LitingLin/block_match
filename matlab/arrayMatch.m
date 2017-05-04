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

%% Data type
% Data type of result, can be
%  'same': the same as input
%  'double', 'single',
%  'logical', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64'
ResultDataType = 'same';
% Data type in computation, can be
%  'same': the same as input
%  'double', 'single'
IntermediateDataType = 'single';

Result = arrayMatchMex(A, B, MeasureMethod, ...    
    'ResultDataType', ResultDataType, ...
    'IntermediateDataType', IntermediateDataType, ...);
