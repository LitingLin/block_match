function Result = v2tov1(ResultV2, IndexV2)
if nargin == 1
    Result = v2tov1_mex(ResultV2);
elseif nargin == 2
    Result = v2tov1_mex(ResultV2, IndexV2);
end
end