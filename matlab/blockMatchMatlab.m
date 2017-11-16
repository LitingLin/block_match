function result = blockMatchMatlab(im1, im2, blockSize, opt)
SearchWindow = opt.SearchWindow;
StrideA = opt.StrideA;
StrideB = opt.StrideB;
StrideBlockA = opt.StrideBlockA;
StrideBlockB = opt.StrideBlockB;

im1Size = size(im1);
im2Size = size(im2);

indexA_M = 1;
indexA_N = 1;
indexB_M = 1;
indexB_N = 1;
countA = 1;

while(true)
    indexA_N = 1;
    indexA_M_block_end = indexA_M + StrideBlockA(1) * blockSize(1) - 1;
    if (indexA_M_block_end > im1Size(1))
        break;
    end
    while (true)                
        indexA_N_block_end = indexA_N + StrideBlockA(2) * blockSize(2) - 1;
        if (indexA_N_block_end > im1Size(2))
            break;
        end
        patchA = im1(indexA_M: StrideBlockA(1): indexA_M_block_end,indexA_N: StrideBlockA(2): indexA_N_block_end);
        
        indexB_M = indexA_M - SearchWindow(1);
        if (indexB_M <= 0)
            indexB_M = 1;
        end
        indexB_M_end = indexA_M + SearchWindow(1);
        if (indexB_M_end > im2Size(1))
            indexB_M_end = im2Size(1);
        end
        
        countB = 1;
        
        while (true)
            indexB_N = indexA_N - SearchWindow(2);
            if (indexB_N <= 0)
                indexB_N = 1;
            end
            indexB_N_end = indexA_N + SearchWindow(2);
            if (indexB_N_end > im2Size(2))
                indexB_N_end = im2Size(2);
            end
                            
            indexB_M_block_end = indexB_M + StrideBlockB(1) * blockSize(1) - 1;
            if (indexB_M_block_end > indexB_M_end)
                break;
            end
            while (true)
                indexB_N_block_end = indexB_N + StrideBlockB(2) * blockSize(2) - 1;
                if (indexB_N_block_end > indexB_N_end)
                    break;
                end
                patchB = im2(indexB_M: StrideBlockB(1):indexB_M_block_end, indexB_N:StrideBlockB(2):indexB_N_block_end);
                
                result(countB,countA) = immse(patchA, patchB);
                
                countB = countB + 1;
                
                indexB_N = indexB_N + StrideB(2);
                if (indexB_N > indexB_N_end)
                    break;
                end
            end
            
            indexB_M = indexB_M + StrideB(1);
            if (indexB_M > indexB_M_end)
                break;
            end
        end
        if (countB ~= 1)
            countA = countA + 1;
        end
        indexA_N = indexA_N + StrideA(2);
        if (indexA_N > im1Size(2))
            break;
        end
    end
    
    indexA_M = indexA_M + StrideA(1);
    if (indexA_M > im1Size(1))
        break;
    end
        
end
