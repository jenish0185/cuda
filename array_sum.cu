#include<stdio.h>
#include<cuda_runtime.h>
#define MAX 100
void fill_array(int size, int *array){
	for(int i=0;i<size;i++){
		array[i] = i+1;
	} 
}
void display_array(int size, int *array){
        for(int i=0;i<size;i++){
                printf("%d,",array[i]); 
        } 
}

__global__ void add(int size, int *a, int *b, int *c){
	int index = blockIdx.y;
	c[index] = a[index]+b[index];
}	

int main(){
	int bytes = MAX * sizeof(int);
	int *ha, *hb, *hc;
	ha =  (int*)malloc(bytes);
	hb =  (int*)malloc(bytes);
	hc =  (int*)malloc(bytes);

	fill_array(MAX,ha);
	fill_array(MAX,hb);

//	display_array(MAX,ha);
//	display_array(MAX,hb);

	int *da, *db, *dc;
	cudaMalloc(&da,bytes);
	cudaMalloc(&db,bytes);
	cudaMalloc(&dc,bytes);

	cudaMemcpy(da,ha,bytes,cudaMemcpyHostToDevice);
	cudaMemcpy(db,hb,bytes,cudaMemcpyHostToDevice);

	dim3 gridSize(1,100,1);
	dim3 blockSize(1,1,1);

	add<<<gridSize,blockSize>>>(MAX, da, db, dc);
	cudaDeviceSynchronize();

	cudaMemcpy(hc,dc,bytes,cudaMemcpyDeviceToHost);
	display_array(MAX,hc);
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	cudaDeviceReset();
	free(ha);
	free(hb);
	free(hc);
	return 0;
}
