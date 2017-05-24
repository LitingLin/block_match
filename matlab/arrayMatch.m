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
% Threshold of result, can be
%  scalar,
%  'no'
Threshold = 'no';
% Sort result, can be
%  logical
Sort = true;
% After sorting the result, retain specified number of blocks, can be
%  scalar
%  'all': retain all blocks
Retain = 'all';

%% Performance Tunning
% Number of worker threads, can be
%  scalar
%  'auto': depends on environment
NumberOfThreads = 'auto';
% Index of GPU, can be
%  scalar
IndexOfDevice = 0;

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
    if isfield(Options, 'Threshold')
        Threshold = Options.Threshold;
    end
    if isfield(Options, 'Sort')
        Sort = Options.Sort;
    end
    if isfield(Options, 'Retain')
        Retain = Options.Retain;
    end
    if isfield(Options, 'NumberOfThreads')
        NumberOfThreads = Options.NumberOfThreads;
    end
    if isfield(Options, 'IndexOfDevice')
        IndexOfDevice = Options.IndexOfDevice;
    end
end

Result = arrayMatchMex(A, B, MeasureMethod, ...    
    'ResultDataType', ResultDataType, ...
    'IntermediateDataType', IntermediateDataType, ...
    'IndexDataType', IndexDataType, ...
    'Threshold', Threshold, ...
    'Sort', Sort, ...
    'Retain', Retain, ...
    'NumberOfThreads', NumberOfThreads, ...
    'IndexOfDevice', IndexOfDevice);
