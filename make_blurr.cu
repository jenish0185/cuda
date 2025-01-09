// #include <stdio.h>
// #include <stdlib.h>
// #include "lodepng.h"
// #include <cuda_runtime.h>

// __global__ void manipulate_pixels(unsigned int h, unsigned int w, unsigned char *pixels) {
// 	int rows = blockIdx.y * blockDim.y + threadIdx.y;
// 	int cols = blockIdx.x * blockDim.x + threadIdx.x;
//     int index = (rows * w + cols) * 4;
// //	printf("index = %d, pixel[%d]= %d\n", index, index, pixels[index]);
// 	if(rows<h && cols<w){
//             		pixels[index] = 255 - pixels[index];		//Red
// 					pixels[index + 1] = 255 - pixels[index+1];	//Green
//             		pixels[index + 2] = 255 - pixels[index+2];	//Blue
//             		pixels[index + 3] = 255;          // Alpha (fully opaque)
//         }
// }

// int main() {
// 	unsigned char *h_pixels;
// 	unsigned int h, w;

// 	lodepng_decode32_file(&h_pixels, &w, &h,"eagle.png");

// 	unsigned char *d_pixels;
// 	int image_size = h*w*4;
// 	cudaMalloc(&d_pixels,image_size);
// 	cudaMemcpy(d_pixels, h_pixels, image_size, cudaMemcpyHostToDevice);

// 	dim3 gridSize(w,h,1);
// 	dim3 blockSize(4,1,1);

// 	manipulate_pixels<<<gridSize,blockSize>>>(h,w,d_pixels);
// 	cudaDeviceSynchronize();

// 	cudaMemcpy(h_pixels, d_pixels, image_size, cudaMemcpyDeviceToHost);

// 	lodepng_encode32_file("negative.png",h_pixels,w,h);

//     	free(h_pixels);
// 		cudaFree(d_pixels);
//     	return 0;
// }
#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"
#include <cuda_runtime.h>


__global__ void apply_box_blur(unsigned int h, unsigned int w, unsigned char *input_pixels, unsigned char *output_pixels) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = (row * w + col) * 4;

    if (row < h && col < w) {
        int r_sum = 0, g_sum = 0, b_sum = 0, count = 0;

       
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int neighbor_row = row + i;
                int neighbor_col = col + j;

               
                if (neighbor_row >= 0 && neighbor_row < h && neighbor_col >= 0 && neighbor_col < w) {
                    int neighbor_index = (neighbor_row * w + neighbor_col) * 4;
                    r_sum += input_pixels[neighbor_index];
                    g_sum += input_pixels[neighbor_index + 1];
                    b_sum += input_pixels[neighbor_index + 2];
                    count++;
                }
            }
        }

        
        output_pixels[index] = r_sum / count;         // Red
        output_pixels[index + 1] = g_sum / count;    // Green
        output_pixels[index + 2] = b_sum / count;    // Blue
        output_pixels[index + 3] = input_pixels[index + 3];
    }
}

int main() {
    unsigned char *h_pixels, *h_output_pixels;
    unsigned int h, w;

   
    lodepng_decode32_file(&h_pixels, &w, &h, "hck.png");

    
    int image_size = h * w * 4;
    h_output_pixels = (unsigned char *)malloc(image_size);

   
    unsigned char *d_input_pixels, *d_output_pixels;
    cudaMalloc(&d_input_pixels, image_size);
    cudaMalloc(&d_output_pixels, image_size);

    
    cudaMemcpy(d_input_pixels, h_pixels, image_size, cudaMemcpyHostToDevice);

    
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y, 1);

   
    apply_box_blur<<<gridSize, blockSize>>>(h, w, d_input_pixels, d_output_pixels);
    cudaDeviceSynchronize();

    
    cudaMemcpy(h_output_pixels, d_output_pixels, image_size, cudaMemcpyDeviceToHost);

    
    lodepng_encode32_file("blurred_hck.png", h_output_pixels, w, h);

    
    free(h_pixels);
    free(h_output_pixels);
    cudaFree(d_input_pixels);
    cudaFree(d_output_pixels);

    return 0;
}
