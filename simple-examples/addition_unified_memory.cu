#include <stdio.h>
#include <sys/time.h>

//////////////////////////////////////////////////////////////
//    Simple vector addition in CUDA with Unified Memory
//////////////////////////////////////////////////////////////

#define N 1024*1024 //Number of elements in the vector

// Definition of the kernel that will be executed by all threads on the GPU
__global__ void add(float *A, float *B, float *C, int n){
	int id = (blockDim.x * blockIdx.x)*n + threadIdx.x*n;

	for (int i = id; i < id+n; i++)
	{
		C[i] = A[i] + B[i];
	}
}

int main(void) {

	float *A, *B, *C;
	int size = N * sizeof(float);
	int numBlocks, numThreadsPerBlock;
	
	// Memory allocation as unified memory
	cudaMallocManaged(&A,size);
	cudaMallocManaged(&B,size);
	cudaMallocManaged(&C,size);

	// Initial values
	for (int i=0; i<N; i++)
	{
		A[i] = i+1;
		B[i] = (i+1)*2;
	}
	
	// Number of threads and blocks used to compte the kernel
	numThreadsPerBlock = 1024;
	numBlocks = 256;
	
	//Executing kernel function
	add<<<numBlocks,numThreadsPerBlock>>>(A,B,C,N/(numBlocks*numThreadsPerBlock));
	cudaDeviceSynchronize();
	

	//Display results
	printf("\n#########################\n");
	printf("Calculation results\n");
	printf("#########################\n");
	printf("Vector A : [%f,%f, ...,%f] \n",A[0],A[1],A[N-1]);
	printf("Vector B : [%f,%f, ...,%f] \n",B[0],B[1],B[N-1]);
	printf("Vector C (result A+B) : [%f,%f, ...,%f] \n",C[0],C[1],C[N-1]);
	
	cudaFree(A);
	cudaFree(B);

	return 0;

}
