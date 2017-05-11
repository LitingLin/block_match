#pragma once

void determineGpuTaskConfiguration(const int maxNumberOfGpuThreads, const int numberOfGpuProcessors, const int numberOfBlockBPerBlockA,
	int *numberOfSubmitThreadsPerProcessor, int *numberOfSubmitProcessors, int *numberOfIterations);
int determineNumberOfThreads(bool sort,	const int numberOfTasks, const int maxNumberOfThreads);