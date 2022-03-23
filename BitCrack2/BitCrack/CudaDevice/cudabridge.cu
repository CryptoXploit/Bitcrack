#include "cudabridge.h"


__global__ void keyFinderKernel(int points, int compression, int searchMode);
__global__ void keyFinderKernelWithDouble(int points, int compression, int searchMode);

void callKeyFinderKernel(int blocks, int threads, int points, bool useDouble, int compression, int searchMode)
{
	if (useDouble) {
		keyFinderKernelWithDouble << <blocks, threads >> > (points, compression, searchMode);
	}
	else {
		keyFinderKernel << <blocks, threads >> > (points, compression, searchMode);
	}
	waitForKernel();
}


void waitForKernel()
{
	// Check for kernel launch error
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		throw cuda::CudaException(err);
	}

	// Wait for kernel to complete
	err = cudaDeviceSynchronize();
	fflush(stdout);
	if (err != cudaSuccess) {
		throw cuda::CudaException(err);
	}
}