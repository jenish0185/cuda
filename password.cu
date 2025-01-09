#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Device-side encryption logic
__device__ void CudaEncrypt(const char* rawPassword, char* encryptedPassword) {
    encryptedPassword[0] = rawPassword[0] + 2;
    encryptedPassword[1] = rawPassword[0] - 2;
    encryptedPassword[2] = rawPassword[0] + 1;
    encryptedPassword[3] = rawPassword[1] + 3;
    encryptedPassword[4] = rawPassword[1] - 3;
    encryptedPassword[5] = rawPassword[1] - 1;
    encryptedPassword[6] = rawPassword[2] + 2;
    encryptedPassword[7] = rawPassword[2] - 2;
    encryptedPassword[8] = rawPassword[3] + 4;
    encryptedPassword[9] = rawPassword[3] - 4;
    encryptedPassword[10] = '\0';
}

// Device-side decryption logic
__device__ void CudaDecrypt(const char* encryptedPassword, char* decryptedPassword) {
    decryptedPassword[0] = encryptedPassword[0] - 2; // Reverse +2
    decryptedPassword[1] = encryptedPassword[3] - 3; // Reverse +3
    decryptedPassword[2] = encryptedPassword[6] - 2; // Reverse +2
    decryptedPassword[3] = encryptedPassword[8] - 4; // Reverse +4
    decryptedPassword[4] = '\0'; // Null-terminate the string
}

// Kernel function for encryption
__global__ void encryptKernel(char* alphabet, char* numbers) {
    char rawPassword[4];
    char encryptedPassword[11];

    rawPassword[0] = alphabet[blockIdx.x];
    rawPassword[1] = alphabet[blockIdx.y];
    rawPassword[2] = numbers[threadIdx.x];
    rawPassword[3] = numbers[threadIdx.y];

    CudaEncrypt(rawPassword, encryptedPassword);

    printf("Raw: %c%c%c%c -> Encrypted: %s\n", rawPassword[0], rawPassword[1], rawPassword[2], rawPassword[3], encryptedPassword);
}

// Kernel function for decryption
__global__ void decryptKernel(const char* inputEncryptedPassword, char* decryptedPassword) {
    CudaDecrypt(inputEncryptedPassword, decryptedPassword);
}

// Main function
int main() {
    // Alphabet and number set for encryption
    char cpuAlphabet[26] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
    char cpuNumbers[10] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};

    // Allocate and copy alphabet and numbers to the GPU
    char* gpuAlphabet;
    char* gpuNumbers;
    cudaMalloc((void**)&gpuAlphabet, sizeof(cpuAlphabet));
    cudaMalloc((void**)&gpuNumbers, sizeof(cpuNumbers));
    cudaMemcpy(gpuAlphabet, cpuAlphabet, sizeof(cpuAlphabet), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuNumbers, cpuNumbers, sizeof(cpuNumbers), cudaMemcpyHostToDevice);

    // Launch the encryption kernel
    printf("Encrypting passwords:\n");
    encryptKernel<<<dim3(26, 26, 1), dim3(10, 10, 1)>>>(gpuAlphabet, gpuNumbers);
    cudaDeviceSynchronize();

    // Decryption section
    char inputEncryptedPassword[11];
    printf("\nEnter the 10-character encrypted password to decrypt: ");
    scanf("%10s", inputEncryptedPassword);

    // Allocate memory for decryption
    char* gpuInputEncryptedPassword;
    char* gpuDecryptedPassword;
    cudaMalloc((void**)&gpuInputEncryptedPassword, sizeof(inputEncryptedPassword));
    cudaMalloc((void**)&gpuDecryptedPassword, sizeof(char) * 5);

    // Copy the encrypted password to the device
    cudaMemcpy(gpuInputEncryptedPassword, inputEncryptedPassword, sizeof(inputEncryptedPassword), cudaMemcpyHostToDevice);

    // Launch the decryption kernel
    decryptKernel<<<1, 1>>>(gpuInputEncryptedPassword, gpuDecryptedPassword);

    // Retrieve the decrypted password
    char decryptedPassword[5];
    cudaMemcpy(decryptedPassword, gpuDecryptedPassword, sizeof(decryptedPassword), cudaMemcpyDeviceToHost);

    printf("Decrypted password: %s\n", decryptedPassword);

    // Free GPU memory
    cudaFree(gpuAlphabet);
    cudaFree(gpuNumbers);
    cudaFree(gpuInputEncryptedPassword);
    cudaFree(gpuDecryptedPassword);

    return 0;
}