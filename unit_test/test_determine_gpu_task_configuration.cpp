#include "test_common.h"

BOOST_AUTO_TEST_CASE(test_determineGpuTaskConfiguration)
{
	int maxNumberOfGpuThreads = 512;
	int numberOfGpuProcessors = 2;
	int numberOfBlockBPerBlockA = 32;

	int numberOfSubmitThreadsPerProcessor, numberOfSubmitProcessors, numberOfIterations;

	determineGpuTaskConfiguration(maxNumberOfGpuThreads, numberOfGpuProcessors, numberOfBlockBPerBlockA,
		&numberOfSubmitThreadsPerProcessor, &numberOfSubmitProcessors, &numberOfIterations);
	BOOST_CHECK_EQUAL(numberOfSubmitThreadsPerProcessor, 512);
	BOOST_CHECK_EQUAL(numberOfSubmitProcessors, 2);
	BOOST_CHECK_EQUAL(numberOfIterations, 32);

	numberOfBlockBPerBlockA = 3;

	determineGpuTaskConfiguration(maxNumberOfGpuThreads, numberOfGpuProcessors, numberOfBlockBPerBlockA,
		&numberOfSubmitThreadsPerProcessor, &numberOfSubmitProcessors, &numberOfIterations);
	BOOST_CHECK_EQUAL(numberOfSubmitThreadsPerProcessor, 510);
	BOOST_CHECK_EQUAL(numberOfSubmitProcessors, 2);
	BOOST_CHECK_EQUAL(numberOfIterations, 340);

	numberOfBlockBPerBlockA = 513;
	
	determineGpuTaskConfiguration(maxNumberOfGpuThreads, numberOfGpuProcessors, numberOfBlockBPerBlockA,
		&numberOfSubmitThreadsPerProcessor, &numberOfSubmitProcessors, &numberOfIterations);
	BOOST_CHECK_EQUAL(numberOfSubmitThreadsPerProcessor, 512);
	BOOST_CHECK_EQUAL(numberOfSubmitProcessors, 2);
	BOOST_CHECK_EQUAL(numberOfIterations, 1);

	numberOfGpuProcessors = 12;
	numberOfBlockBPerBlockA = 513;

	determineGpuTaskConfiguration(maxNumberOfGpuThreads, numberOfGpuProcessors, numberOfBlockBPerBlockA,
		&numberOfSubmitThreadsPerProcessor, &numberOfSubmitProcessors, &numberOfIterations);
	BOOST_CHECK_EQUAL(numberOfSubmitThreadsPerProcessor, 512);
	BOOST_CHECK_EQUAL(numberOfSubmitProcessors, 12);
	BOOST_CHECK_EQUAL(numberOfIterations, 11);

	numberOfBlockBPerBlockA = 1000;

	determineGpuTaskConfiguration(maxNumberOfGpuThreads, numberOfGpuProcessors, numberOfBlockBPerBlockA,
		&numberOfSubmitThreadsPerProcessor, &numberOfSubmitProcessors, &numberOfIterations);
	BOOST_CHECK_EQUAL(numberOfSubmitThreadsPerProcessor, 512);
	BOOST_CHECK_EQUAL(numberOfSubmitProcessors, 12);
	BOOST_CHECK_EQUAL(numberOfIterations, 6);

	numberOfBlockBPerBlockA = 6145;

	determineGpuTaskConfiguration(maxNumberOfGpuThreads, numberOfGpuProcessors, numberOfBlockBPerBlockA,
		&numberOfSubmitThreadsPerProcessor, &numberOfSubmitProcessors, &numberOfIterations);
	BOOST_CHECK_EQUAL(numberOfSubmitThreadsPerProcessor, 512);
	BOOST_CHECK_EQUAL(numberOfSubmitProcessors, 13);
	BOOST_CHECK_EQUAL(numberOfIterations, 1);
}