#pragma once

void tryToIncludeLastBlock(int *indexA, int strideA, int indexA_end);

void noIndexPostProcess(int *indexA, int strideA, int indexA_end);

void recordIndex(int *index_x_buffer, int *index_y_buffer, int index_x, int index_y);

void dummyRecordIndex(int *index_x_buffer, int *index_y_buffer, int index_x, int index_y);

void determineBlockB_index_local(int *indexB_begin, int *indexB_end, int matB, int block,
	int neighbour, int index_A);

void determineBlockB_index_local_topLeft(int *indexB_begin, int *indexB_end, int matB, int block,
	int neighbour, int index_A);

void determineBlockB_index_full(int *indexB_begin, int *indexB_end, int matB, int block,
	int neighbour, int index_A);