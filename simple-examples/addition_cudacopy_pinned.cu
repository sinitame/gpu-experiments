#include <stdio.h>
#include <sys/time.h>

///////////////////////////////////////////////////////////
//   Simple vector addition in CUDA with pinned memory
///////////////////////////////////////////////////////////

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
	float *d_A, *d_B, *d_C;
	int size = N * sizeof(float);
	int numBlocks, numThreadsPerBlock;
	
	// Memory allocation on the HOST (pinned)
	cudaMallocHost(&A,size);
	cudaMallocHost(&B,size);
	cudaMallocHost(&C,size);

	// Initial values
	for (int i=0; i<N; i++)
	{
		A[i] = i+1;
		B[i] = (i+1)*2;
	}
	

	//Memory allocation on the GPU
	cudaMalloc((void**)&d_A, size); 
	cudaMalloc((void**)&d_B, size); 
	cudaMalloc((void**)&d_C, size); 

	//Copy data from HOST to GPU
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	// Number of threads and blocks used to compte the kernel
	numThreadsPerBlock = 1024;
	numBlocks = 256;
	
	//Executing kernel function
	add<<<numBlocks,numThreadsPerBlock>>>(d_A,d_B,d_C,N/(numBlocks*numThreadsPerBlock));
	

	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
	
	//Display results
	printf("\n#########################\n");
	printf("Calculation results\n");
	printf("#########################\n");
	printf("Vector A : [%f,%f, ...,%f] \n",A[0],A[1],A[N-1]);
	printf("Vector B : [%f,%f, ...,%f] \n",B[0],B[1],B[N-1]);
	printf("Vector C (result A+B) : [%f,%f, ...,%f] \n",C[0],C[1],C[N-1]);
	
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);

	return 0;

}
