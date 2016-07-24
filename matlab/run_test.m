opt.SequenceAStride = [6,6];
opt.SequenceBStride = [6,6];
opt.Retain = 10;
im = double(rgb2gray(imread('Lena256.png')));
blockMatch(im,im,[4,4],opt);