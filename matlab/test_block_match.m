rng(0);
A = randi(10, [20 25]);
% B = randi(10, 10);
block_size = [2 2];
% opt.SearchBlock = [3 3];
% opt.SequenceBPadding = [3,3];
opt.SequenceAStride = 3;
opt.Retain = 10;

res = blockMatch(A, A, block_size, opt);