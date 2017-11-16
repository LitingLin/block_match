function [Result, Index, APadded, BPadded] = blockMatch(A, B, BlockSize, Options)
%% Parameter Description
% Input:
%  A:
%   matrix A, can be
%    MxN image,
%    MxNxC multi-channel image
%  B:
%   matrix B, the size must be the same as A, can be
%    MxN image, 
%    MxNxC multi-channel image
%  BlockSize:
%   Size of block, can be
%    scalar, 1x2 matrix
%  Options:
%   Options, can be
%    struct, members are shown below
%
% Output:
%  Result:
%   MxO, O: the number of patches from A, M: the number of matching result
%  Index:
%   Mx2xO stores the index of B in Result, 
%    Index(:,1,o) for first-dimension(row), Index(:,2,o) for second-dimension(column)

% Demo:
%  blockSize = [3,3];
%  opt.SearchBlock = [3,3];
%  [res,ind] = blockMatch(A, B, blockSize, opt);

%% Check number of input parameter
if nargin < 3
    error('Too few input arguments');
end

%% Perform Full Search
SearchWindow = 'full';

%% Perform Local Search
% Define search window
% Size of search region, can be
%  scalar: window size = [2xscalar+1,2xscalar+1],
%  1x2 matrix: window size = [2xmat(1)+1,2xmat(2)+1],
%  2x2 matrix: window size = [mat(1,1)+1+mat(2,1),mat(1,2)+1+mat(2,2)]
% SearchWindow = 6;
% SearchWindow = [5,5];
% SearchWindow = [3,7;3,7];

%% Measure Method
% Measure method can be
%  'mse': Mean Sequare Error
%  'cc': Pearson Correlation Coefficient
MeasureMethod = 'mse';

%% Stride
% Stride size of matrix A, can be
%  scalar, 1x2 matrix
StrideA = [1,1];
% Stride size of matrix B, can be
%  scalar, 1x2 matrix
StrideB = [1,1];

%% Stride in block
% Stride size of blocks get from matrix A, can be
%  scalar, 1x2 matrix
StrideBlockA = [1,1];
% Stride size of blocks get from matrix B, can be
%  scalar, 1x2 matrix
StrideBlockB = [1,1];

%% Border
% Blocks near the borders
%  can be
%  'normal'
%  'includeLastBlock': make use of the pixels near the borders of A when AStride > 1
BorderA = 'normal';

%% Padding Method
% Padding size of matrix A, can be
%  scalar, 1x2 matrix, 2x2 matrix,
%  'same', 'full': same behavior in conv2 (shape parameter)
PaddingA = 0;
% Padding method of matrix A, can be
%  'zero', 'circular', 'replicate', 'symmetric'
PaddingMethodA = 'symmetric';
% Padding size of matrix B, can be
%  scalar, 1x2 matrix, 2x2 matrix,
%  'same', 'full': same behavior in conv2 (shape parameter)
PaddingB = 0;
% Padding method of matrix B, can be
%  'zero', 'circular', 'replicate', 'symmetric'
PaddingMethodB = 'symmetric';

%% Results Post-Processing
% Threshold of result, can be
%  scalar,
%  'no'
Threshold = 'no';
% Sort result, can be
%  logical
Sort = false;
% After sorting the result, retain specified number of blocks, can be
%  scalar
%  'all': retain all blocks
Retain = 'all';

%% Data type
% Data type of result, can be
%  'same': the same as input
%  'double', 'single'
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

%% Performance Tunning
% Number of worker threads, can be
%  scalar
%  'auto': depends on environment
NumberOfThreads = 'auto';
% Index of GPU, can be
%  scalar
IndexOfDevice = 0;

%% Parse option parameter
if nargin == 4
    if isfield(Options, 'SearchWindow')
        SearchWindow = Options.SearchWindow;
    end
    if isfield(Options, 'StrideA')
        StrideA = Options.StrideA;
    end
    if isfield(Options, 'StrideB')
        StrideB = Options.StrideB;
    end
    if isfield(Options, 'MeasureMethod')
        MeasureMethod = Options.MeasureMethod;
    end
    if isfield(Options, 'PaddingA')
        PaddingA = Options.PaddingA;
    end
    if isfield(Options, 'PaddingMethodA')
        PaddingMethodA = Options.PaddingMethodA;
    end
    if isfield(Options, 'PaddingB')
        PaddingB = Options.PaddingB;
    end
    if isfield(Options, 'PaddingMethodB')
        PaddingMethodB = Options.PaddingMethodB;
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
    if isfield(Options, 'ResultDataType')
        ResultDataType = Options.ResultDataType;
    end
    if isfield(Options, 'IntermediateDataType')
        IntermediateDataType = Options.IntermediateDataType;
    end
    if isfield(Options, 'IndexDataType')
        IndexDataType = Options.IndexDataType;
    end
    if isfield(Options, 'BorderA')
        BorderA = Options.BorderA;
    end
    if isfield(Options, 'NumberOfThreads')
        NumberOfThreads = Options.NumberOfThreads;
    end
    if isfield(Options, 'IndexOfDevice')
        IndexOfDevice = Options.IndexOfDevice;
    end
end

%% Call mex
if nargout == 0 || nargout == 1
[Result] = blockMatchMex(A, B, BlockSize, ...
    'SearchWindow', SearchWindow, ...
    'StrideA', StrideA, ...
    'StrideB', StrideB, ...
    'BorderA', BorderA, ...
    'MeasureMethod', MeasureMethod, ...
    'PaddingA', PaddingA, ...
    'PaddingMethodA', PaddingMethodA, ...
    'PaddingB', PaddingB, ...
    'PaddingMethodB', PaddingMethodB, ...
    'Threshold', Threshold, 'Sort', Sort, 'Retain', Retain, ...
    'ResultDataType', ResultDataType, ...
    'IntermediateDataType', IntermediateDataType, ...
    'IndexDataType', IndexDataType, ...
    'NumberOfThreads', NumberOfThreads, ...
    'IndexOfDevice', IndexOfDevice);
end

if nargout == 2
[Result, Index] = blockMatchMex(A, B, BlockSize, ...
    'SearchWindow', SearchWindow, ...
    'StrideA', StrideA, ...
    'StrideB', StrideB, ...
    'BorderA', BorderA, ...
    'MeasureMethod', MeasureMethod, ...
    'PaddingA', PaddingA, ...
    'PaddingMethodA', PaddingMethodA, ...
    'PaddingB', PaddingB, ...
    'PaddingMethodB', PaddingMethodB, ...
    'Threshold', Threshold, 'Sort', Sort, 'Retain', Retain, ...
    'ResultDataType', ResultDataType, ...
    'IntermediateDataType', IntermediateDataType, ...
    'IndexDataType', IndexDataType, ...
    'NumberOfThreads', NumberOfThreads, ...
    'IndexOfDevice', IndexOfDevice);
end

if nargout == 3
[Result, Index, APadded] = blockMatchMex(A, B, BlockSize, ...
    'SearchWindow', SearchWindow, ...
    'StrideA', StrideA, ...
    'StrideB', StrideB, ...
    'BorderA', BorderA, ...
    'MeasureMethod', MeasureMethod, ...
    'PaddingA', PaddingA, ...
    'PaddingMethodA', PaddingMethodA, ...
    'PaddingB', PaddingB, ...
    'PaddingMethodB', PaddingMethodB, ...
    'Threshold', Threshold, 'Sort', Sort, 'Retain', Retain, ...
    'ResultDataType', ResultDataType, ...
    'IntermediateDataType', IntermediateDataType, ...
    'IndexDataType', IndexDataType, ...
    'NumberOfThreads', NumberOfThreads, ...
    'IndexOfDevice', IndexOfDevice);
end

if nargout == 4
[Result, Index, APadded, BPadded] = blockMatchMex(A, B, BlockSize, ...
    'SearchWindow', SearchWindow, ...
    'StrideA', StrideA, ...
    'StrideB', StrideB, ...
    'BorderA', BorderA, ...
    'MeasureMethod', MeasureMethod, ...
    'PaddingA', PaddingA, ...
    'PaddingMethodA', PaddingMethodA, ...
    'PaddingB', PaddingB, ...
    'PaddingMethodB', PaddingMethodB, ...
    'Threshold', Threshold, 'Sort', Sort, 'Retain', Retain, ...
    'ResultDataType', ResultDataType, ...
    'IntermediateDataType', IntermediateDataType, ...
    'IndexDataType', IndexDataType, ...
    'NumberOfThreads', NumberOfThreads, ...
    'IndexOfDevice', IndexOfDevice);
end