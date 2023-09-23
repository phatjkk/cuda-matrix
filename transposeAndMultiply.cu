#include <iostream>
#include <cuda_runtime.h>

__global__ void transposeMatrix(float* result, const float* matrix, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    

    // TODO 9: Tiến hành lập trình thread chuyển vị
    if (row < rows && col < cols)
    {
        result[col * rows + row] = matrix[row * cols + col];
    }
}

__global__ void matrixMultiplication(float* result, const float* matrix1, const float* matrix2, int rows1, int cols1, int cols2)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO 10: Tiến hành lập trình thread nhân vô hướng vector hàng của matrix1 vào vector cột của matrix2
    if (row < rows1 && col < cols2)
    {
        float dotProduct = 0.0f;
        for (int k = 0; k < cols1; k++)
        {
            dotProduct += matrix1[row * cols1 + k] * matrix2[k * cols2 + col];
        }
        result[row * cols2 + col] = dotProduct;
    }
}

int main()
{
    // Số lượng dòng của ma trận
    int rows = 3;  
    // Số lượng cột của ma trận   
    int cols = 4;      
    
    // Định nghĩa ma trận
    float* matrix = new float[rows * cols];
    // Định nghĩa kết quả chuyển vị

    float* transpose_result = new float[cols * rows];

    // Định nghĩa kết quả nhân ma trận

    float* multiply_result = new float[cols * cols];
    
    // Khởi tạo giá trị ban đầu cho ma trận
    for (int i = 0; i < rows * cols; i++)
    {
        matrix[i] = i + 1;
    }
    
    // Khởi tạo con trỏ
    float* d_matrix;
    float* d_transpose_result;
    float* d_multiply_result;


    // TODO 1: Phân vùng nhớ trong GPU cho 3 biến d_matrix, d_transpose_result và d_multiply_result
    const int numElements = cols * cols;
    const int numBytes = numElements * sizeof(float);
    
    cudaMalloc((void**)&d_matrix,numBytes); 
    cudaMalloc((void**)&d_transpose_result,numBytes); 
    cudaMalloc((void**)&d_multiply_result,numBytes); 
	
    // TODO 2: Sao chép biến từ bộ nhớ của host đến bộ nhớ của device
    cudaMemcpy(d_matrix, matrix, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_transpose_result, transpose_result, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_multiply_result, multiply_result, numBytes, cudaMemcpyHostToDevice);
	
    // TODO 3: Chạy kernel transpose ma trận d_matrix trên GPU

    dim3 blockDim(16, 16);
    dim3 gridDim(ceil((float)rows / blockDim.y), ceil((float)cols / blockDim.x));

	  transposeMatrix<<<gridDim,blockDim>>>(d_transpose_result,d_matrix,rows,cols);
    // TODO 4: Chạy kernel nhân kết quả chuyển vị bên trên với ma trận ban đầu trên GPU
	  matrixMultiplication<<<gridDim,blockDim>>>(d_multiply_result,d_transpose_result,d_matrix,cols,rows,cols);
    // TODO 5: Sao chép kết quả chuyển vị về CPU
	  cudaMemcpy(transpose_result, d_transpose_result, numBytes, cudaMemcpyDeviceToHost);
    // TODO 6: Sao chép kết quả phép nhân ma trận về CPU
	  cudaMemcpy(multiply_result, d_multiply_result, numBytes, cudaMemcpyDeviceToHost);
    
    std::cout << "Ma trận ban đầu:\n";
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    // TODO 8: In kết quả chuyển vị và kết quả nhân ma trận như đề bài
    std::cout << "Ma trận chuyển vị:\n";
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            std::cout << transpose_result[i * rows + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Kết quả nhân Ma trận:\n";
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            std::cout << multiply_result[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    
    // Giải phóng bộ nhớ trên host
    cudaFree(d_matrix);
    cudaFree(d_transpose_result);
    cudaFree(d_multiply_result);
    
    // Giải phóng bộ nhớ trên device
    delete[] matrix;
    delete[] transpose_result;
    delete[] multiply_result;

    
    return 0;
}