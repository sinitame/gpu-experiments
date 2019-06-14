#include <stdio.h>
#include <sys/time.h>

//////////////////////////////////////////////////////
//	Simple vector addition in CUDA
//////////////////////////////////////////////////////

#define N 1024*1024 //Number of elements in the vector

// Definition of the kernel that will be executed by all threads on the GPU
__global__ void add(float *a, float *b, float *c, int n){
	int id = (blockDim.x * blockIdx.x)*n + threadIdx.x*n;

	for (int i = id; i < id+n; i++)
	{
		c[i] = a[i] + b[i];
	}
}

int main(void) {

	float *A, *B, *C;
	float *d_A, *d_B, *d_C;
	int size = N * sizeof(float);
	int numBlocks, numThreadsPerBlock;
	struct timeval copy_start, copy_end, process_start, process_end;
	int t_copy, t_process;
	
	// Memory allocation on the HOST
	A = (float*)malloc(size);
	B = (float*)malloc(size);
	C = (float*)malloc(size);

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
	gettimeofday(&copy_start, NULL);
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	gettimeofday(&copy_end, NULL);

	// Number of threads and blocks used to compute the kernel
	numThreadsPerBlock = 1024;
	numBlocks = 256;
	
	//Executing kernel function
	gettimeofday(&process_start, NULL);
	add<<<numBlocks,numThreadsPerBlock>>>(d_A,d_B,d_C,N/(numBlocks*numThreadsPerBlock));
	gettimeofday(&process_end, NULL);

	//Copy result from GPU to HOST
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
	
	//Compute process and copy time
	t_copy = (copy_end.tv_sec - copy_start.tv_sec)*1000000 + copy_end.tv_usec - copy_start.tv_usec;
	t_process = (process_end.tv_sec - process_start.tv_sec)*1000000 + process_end.tv_usec - process_start.tv_usec;

	//Display results
	printf("\n#########################\n");
	printf("Calculation results\n");
	printf("#########################\n");
	printf("Vector A : [%f,%f, ...,%f] \n",A[0],A[1],A[N-1]);
	printf("Vector B : [%f,%f, ...,%f] \n",B[0],B[1],B[N-1]);
	printf("Vector C (result A+B) : [%f,%f, ...,%f] \n",C[0],C[1],C[N-1]);
	
	printf("\n#########################\n");
	printf("Performance results\n");
	printf("#########################\n");
	printf("Copy HOST to GPU time : %d uS\n", t_copy);
	printf("Kernel process time : %d uS\n", t_process);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(A);
	free(B);
	free(C);

	return 0;

}
