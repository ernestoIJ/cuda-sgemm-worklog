#include <iostream>
#include <cublas_v2.h>

void cublas_sgemm_launcher(float* A_h, float* B_h, float* C_h, int m, int k, int n, float alpha, float beta) {
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

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Row-major A(m,k), B(k,n), C(m,n) are, in memory, identical to column-major A^T(k,m), B^T(n,k), C^T(n,m).
    // We want C = A*B  =>  C^T = B^T * A^T
    // So we ask cuBLAS (column-major) to compute: C^T(n,m) = B^T(n,k) * A^T(k,m)
    // by passing B_d as the first operand and A_d as the second, swapping m/n.

    std::cout << "Warming up the GPU" << std::endl;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B_d, n, A_d, k, &beta, C_d, n);
    cudaDeviceSynchronize();
    std::cout << "Warm up done!" << std::endl;

    std::cout << "Launching cuBLAS SGEMM Kernel" << std::endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B_d, n, A_d, k, &beta, C_d, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "cuBLAS SGEMM Kernel Time: " << milliseconds << " ms" << std::endl;

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(error) << std::endl;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
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

    cublas_sgemm_launcher(A, B, C, m, k, n, alpha, beta);

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