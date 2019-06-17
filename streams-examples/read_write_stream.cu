#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include <unistd.h>
#include <assert.h>
#define MAX_STREAMS 3

uint32_t *bufferA[MAX_STREAMS], *bufferB[MAX_STREAMS];
int flags[MAX_STREAMS] = {1,1,1};
int max_iteration = 10;
pthread_mutex_t lock;

inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

// Data initialization kernel
__global__ void init_data(uint32_t *buff, const int vector_size){
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
	int my_idx = idx;
	while (my_idx < vector_size){
		buff[my_idx] = my_idx;
		my_idx += gridDim.x*blockDim.x; // grid-striding loop
	}
}

// Vector addition kernel
__global__ void vector_add(uint32_t *ibuff, uint32_t *obuff, const int vector_size){
	
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
	int my_idx = idx;
	while (my_idx < vector_size){
		obuff[my_idx] = ibuff[my_idx] + ibuff[my_idx];
		my_idx += gridDim.x*blockDim.x; // grid-striding loop
	}
}

// Read/write controller thread
void *read_write_controller(void *){
	int i = 0;
	printf("Starting read_write_controller\n");
	while (i<max_iteration) {
		sleep(0.0001);
		if (flags[i%MAX_STREAMS] == 1){

			// Printing bufferA and bufferB element
			printf("A : %d\n",bufferA[i%MAX_STREAMS][1]);
			printf("B : %d\n",bufferB[i%MAX_STREAMS][1]);
			
			// updating flag (mutex protected)
			pthread_mutex_lock(&lock);
			flags[i%MAX_STREAMS] = 0;
			printf("iteration (%d) : [%d,%d,%d]\n",i,flags[0],flags[1],flags[2]);
			pthread_mutex_unlock(&lock);

			i++;
		}
	}
	return NULL;	
}

int main(){

	uint32_t *ibuff[MAX_STREAMS], *obuff[MAX_STREAMS];
	int result=0, device_id=0;
	int numBlocks, numThreadsPerBlock = 1024;
	int vector_size = 1024*1024;
	size_t size = vector_size*sizeof(uint32_t);

	cudaStream_t streams[MAX_STREAMS];
	for (int stream = 0; stream < MAX_STREAMS; stream++){
		cudaStreamCreate(&streams[stream]);
	}

	////////////////////////////////////////////////////////////////
	//               MEMORY ALLOCATION ON GPU
	////////////////////////////////////////////////////////////////
	
	printf("Memory allocation GPU\n");
	cudaDeviceGetAttribute (&result, cudaDevAttrConcurrentManagedAccess, device_id);
	for (int stream = 0; stream < MAX_STREAMS; stream++){
		checkCuda(cudaMallocManaged(&ibuff[stream],size));
		checkCuda(cudaMallocManaged(&obuff[stream],size));	
	
		if (result) {
			checkCuda(cudaMemAdvise(ibuff[stream],size,cudaMemAdviseSetPreferredLocation,device_id));
			checkCuda(cudaMemAdvise(obuff[stream],size,cudaMemAdviseSetPreferredLocation,device_id));
		}
		
		checkCuda(cudaMemset(ibuff[stream], 0, size));
		checkCuda(cudaMemset(obuff[stream], 0, size));
	}
	////////////////////////////////////////////////////////////////
	//               MEMORY ALLOCATION ON HOST
	////////////////////////////////////////////////////////////////
	
	printf("Memory allocation HOST\n");
	for (int stream = 0; stream < MAX_STREAMS; stream++){
		checkCuda(cudaHostAlloc(&bufferA[stream], size, cudaHostAllocDefault));
		checkCuda(cudaHostAlloc(&bufferB[stream], size, cudaHostAllocDefault));
	}
	
	for (int i = 0; i < vector_size; i++){
		for (int stream = 0; stream < MAX_STREAMS; stream++){
			bufferA[stream][i] = i + 1000*stream;
		}
	}
	
	///////////////////////////////////////////////////////////////
	//	 RUNNING READ/WRITE CONTROLLER ON SPECIFIC THREAD
	///////////////////////////////////////////////////////////////

	pthread_t thread;
	printf("Running thread \n");
	if (pthread_create(&thread, NULL, &read_write_controller, NULL)){
		fprintf(stderr, "Error creating thread \n");
		return 1;
	}
 	
	if (pthread_mutex_init(&lock, NULL) != 0){
        	printf("Mutex initialization failed.\n");
       	 	return 1;
	}

	///////////////////////////////////////////////////////////////
	//             RUNNING GPU KERNEL PIPELINING
	//////////////////////////////////////////////////////////////
	int stream =0;
	cudaDeviceGetAttribute(&numBlocks, cudaDevAttrMultiProcessorCount, 0);	
	
	for (int iteration = 0; iteration < max_iteration; iteration++){
		stream = iteration % MAX_STREAMS;	
		
		//FPGA is writing data in buffer
		while(flags[stream] == 1){ 
			sleep(0.0001);
		}
		cudaMemcpyAsync(ibuff[stream],bufferA[stream], size, cudaMemcpyDeviceToHost, streams[stream]);
		vector_add<<<4*numBlocks, numThreadsPerBlock,0,streams[stream]>>>(ibuff[stream],obuff[stream],vector_size);
		cudaMemcpyAsync(bufferB[stream], obuff[stream], size, cudaMemcpyHostToDevice, streams[stream]);
	   	
		// FPGA can write new data	
		pthread_mutex_lock(&lock);	
		flags[stream] = 1;
		pthread_mutex_unlock(&lock);
	}
	
	pthread_join(thread, NULL);
	printf("Completed %d iterations successfully\n", max_iteration);

	for (int i = 0; i < MAX_STREAMS; i++){
		cudaFreeHost(bufferA[i]);
		cudaFreeHost(bufferB[i]);
		cudaFree(ibuff[i]);
		cudaFree(obuff[i]);
	}
}
