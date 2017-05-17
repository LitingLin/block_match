function Result = arrayMatch(A, B, MeasureMethod, opt)
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
% Data type of index, can be
%  'auto': the minimal type fit for the result
%  'double', 'single',
%  'logical', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64'
IndexDataType = 'auto';

%% Results Post-Processing
% Sort result, can be
%  logical
Sort = true;
% After sorting the result, retain specified number of blocks, can be
%  scalar
%  'all': retain all blocks
Retain = 'all';

if nargin == 4
    if isfield(Options, 'ResultDataType')
        ResultDataType = Options.ResultDataType;
    end
    if isfield(Options, 'IntermediateDataType')
        IntermediateDataType = Options.IntermediateDataType;
    end
    if isfield(Options, 'IndexDataType')
        IndexDataType = Options.IndexDataType;
    end
    if isfield(Options, 'Sort')
        Sort = Options.Sort;
    end
    if isfield(Options, 'Retain')
        Retain = Options.Retain;
    end
end

Result = arrayMatchMex(A, B, MeasureMethod, ...    
    'ResultDataType', ResultDataType, ...
    'IntermediateDataType', IntermediateDataType, ...
    'IndexDataType', IndexDataType, ...
    'Sort', Sort, ...
    'Retain', Retain);
