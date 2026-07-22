#include <iostream>

__global__ void naive_sgemm_kernel(float* A, float* B, float* C, int m, int k, int n, float alpha, float beta) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if ((row < m) && (col < n)) {
        float Pval = 0.0f;
        for (int i = 0; i < k; i++) {
            Pval += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = alpha * Pval + beta * C[row * n + col];
    }
}

void naive_sgemm_launcher(float* A_h, float* B_h, float* C_h, int m, int k, int n, float alpha, float beta) {
    size_t size_A = m * k * sizeof(float);
    size_t size_B = k * n * sizeof(float);
    size_t size_C = m * n * sizeof(float);

    float* A_d;
    float* B_d;
    float* C_d;

    cudaMalloc((void **)&A_d, size_A);
    cudaMalloc((void **)&B_d, size_B);
    cudaMalloc((void **)&C_d, size_C);

    cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h, size_C, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32, 1);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);

    std::cout << "Warming Up The GPU" << std::endl;
    naive_sgemm_kernel<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, m, k, n, alpha, beta);
    cudaDeviceSynchronize();
    std::cout << "Warm Up Done!" << std::endl;

    std::cout << "Launching Kernel" << std::endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    naive_sgemm_kernel<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, m, k, n, alpha, beta);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Naive SGEMM Kernel Time: " << milliseconds << " ms" << std::endl;

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(error) << std::endl;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    int m = 4096;
    int k = 4096;
    int n = 4096;

    float alpha = 1.0f;
    float beta = 0.0f;

    float* A = new float[m * k];
    float* B = new float[k * n];
    float* C = new float[m * n];

    for (int i = 0; i < m * k; i++) {
        A[i] = 1.0f;
    }
    for (int i = 0; i < k * n; i++) {
        B[i] = 2.0f;
    }
    for (int i = 0; i < m * n; i++) {
        C[i] = 0.0f;
    }

    naive_sgemm_launcher(A, B, C, m, k, n, alpha, beta);

    for (int i = 0; i < m * n; i++) {
        if (C[i] != A[0] * B[0] * k) {
            std::cout << "Error at C[" << i << "] | Expected: " << A[0] * B[0] * k << " | Got: " << C[i] << std::endl;
            break; 
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}