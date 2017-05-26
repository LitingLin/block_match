function [Result, Index, SequenceAPadded, SequenceBPadded] = blockMatch(SequenceA, SequenceB, BlockSize, Options)
%% Parameter Description
% Input:
%  SequenceA:
%   Sequence A, can be
%    MxN image,
%    MxNxC multi-channel image, [NOT IMPLEMENTED]
%  SequenceB:
%   Sequence B, the size must be the same as SequenceA, can be
%    MxN image, 
%    MxNxC multi-channel image,  [NOT IMPLEMENTED]
%  BlockSize:
%   Size of block, can be
%    scalar, 1x2 matrix
%  Options:
%   Options, can be
%    struct
%
% Output:
%  Result:
%   AxMxN
%  Index:
%   XxYxMxN matrix, stores the corresponding index of Result in matrix B

% Demo:
%  blockSize = [3,3];
%  opt.SearchBlock = [3,3];
%  res = blockMatch(A, B, blockSize, opt);

%% Check number of input parameter
if nargin < 3
    error('Too few input arguments');
end

%% Perform Full Search
SearchRegion = 'full';

%% Perform Local Search
% Define search region and stride of sequence B

% Size of search region, can be
%  scalar, 1x2 matrix
% SearchRegion = [5,5];

% Or define search region in blocks directly
%  In this case, SequenceBStride = BlockSize and
%  SearchRegion = SearchBlock.*SequenceBStride
% Size of search block, can be
%  scalar, 1x2 matrix
% SearchBlock = [3,3];

% Beginning index of B, can be
%  'topLeft', 'center'
SearchFrom = 'center';

%% Measure Method
% Measure method can be
%  'mse': Mean Sequare Error
%  'cc': Pearson Correlation Coefficient
MeasureMethod = 'mse';

%% Stride

% Stride size of sequence A, can be
%  scalar, 1x2 matrix
SequenceAStride = [1,1];

% Stride size of sequence B, can be
%  scalar, 1x2 matrix
SequenceBStride = [1,1];

%% Border

% Blocks in the borders 
%  can be
%  'normal'
%  'includeLastBlock'
SequenceABorder = 'includeLastBlock';

%% Padding Method
% Padding size of sequence A, can be
%  scalar, 1x2 matrix, 2x2 matrix,
%  'same', 'full'
SequenceAPadding = 0;
% Padding of sequence A, can be
%  'zero', 'circular', 'replicate', 'symmetric'
SequenceAPaddingMethod = 'symmetric';
% Padding size of sequence B, can be
%  scalar, 1x2 matrix, 2x2 matrix,
%  'same', 'full'
SequenceBPadding = 'same';
% Padding of sequence B, can be
%  'zero', 'circular', 'replicate', 'symmetric'
SequenceBPaddingMethod = 'symmetric';

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
% Sparse or not, can be
%  'auto': depends on results,
%  logical
Sparse = 'auto';

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
    if isfield(Options, 'SearchRegion')
        SearchRegion = Options.SearchRegion;
    end
    if isfield(Options, 'SearchFrom')
        SearchFrom = Options.SearchFrom;
    end
    if isfield(Options, 'SequenceAStride')
        SequenceAStride = Options.SequenceAStride;
    end
    if isfield(Options, 'SequenceBStride')
        SequenceBStride = Options.SequenceBStride;
    end
    if isfield(Options, 'SearchBlock')
        SearchBlock = Options.SearchBlock;
        SequenceBStride = BlockSize;
        SearchRegion = SequenceBStride.*SearchBlock;
    end
    if isfield(Options, 'MeasureMethod')
        MeasureMethod = Options.MeasureMethod;
    end
    if isfield(Options, 'SequenceAPadding')
        SequenceAPadding = Options.SequenceAPadding;
    end
    if isfield(Options, 'SequenceAPaddingMethod')
        SequenceAPaddingMethod = Options.SequenceAPaddingMethod;
    end
    if isfield(Options, 'SequenceBPadding')
        SequenceBPadding = Options.SequenceBPadding;
    end
    if isfield(Options, 'SequenceBPaddingMethod')
        SequenceBPaddingMethod = Options.SequenceBPaddingMethod;
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
    if isfield(Options, 'SequenceABorder')
        SequenceABorder = Options.SequenceABorder;
    end
    if isfield(Options, 'NumberOfThreads')
        NumberOfThreads = Options.NumberOfThreads;
    end
    if isfield(Options, 'IndexOfDevice')
        IndexOfDevice = Options.IndexOfDevice;
    end
end

%% Call mex
[Result, Index, SequenceAPadded, SequenceBPadded] = blockMatchMex(SequenceA, SequenceB, BlockSize, ...
    'SearchRegion', SearchRegion, ...
    'SearchFrom', SearchFrom, ...
    'SequenceAStride', SequenceAStride, ...
    'SequenceBStride', SequenceBStride, ...
    'SequenceABorder', SequenceABorder, ...
    'MeasureMethod', MeasureMethod, ...
    'SequenceAPadding', SequenceAPadding, ...
    'SequenceAPaddingMethod', SequenceAPaddingMethod, ...
    'SequenceBPadding', SequenceBPadding, ...
    'SequenceBPaddingMethod', SequenceBPaddingMethod, ...
    'Threshold', Threshold, 'Sort', Sort, 'Retain', Retain, ...
    'ResultDataType', ResultDataType, ...
    'IntermediateDataType', IntermediateDataType, ...
    'IndexDataType', IndexDataType, ...
    'NumberOfThreads', NumberOfThreads, ...
    'IndexOfDevice', IndexOfDevice);
