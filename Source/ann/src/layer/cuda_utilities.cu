#include <iostream>
#include "cuda_utilities.h"
#include "../../config.h"

#define TILE_WIDTH 32

#define CHECK(call)                                                \
	{                                                              \
		const cudaError_t error = call;                            \
		if (error != cudaSuccess)                                  \
		{                                                          \
			fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
			fprintf(stderr, "code: %d, reason: %s\n", error,       \
					cudaGetErrorString(error));                    \
			exit(EXIT_FAILURE);                                    \
		}                                                          \
	}

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

static GpuTimer timer;
void startTimer()
{
    timer.Start();
}

float stopTimer()
{
    timer.Stop();

	return timer.Elapsed();
}

__host__ __device__ int idx1D(int r, int c, int colSz) // Create two verision: __host__ to be callable from CPU and run on CPU, __device__ to be callable from GPU and run on GPU
{
    return r * colSz + c;
}

__host__ __device__ int idx1D_col(int r, int c, int rowSz) // Create two verision: __host__ to be callable from CPU and run on CPU, __device__ to be callable from GPU and run on GPU
{
    return c * rowSz + r;
}

__global__ void unrollKernel_1(int C, int H, int W, int K, float* image, float* data_col)
{
	int c, s, h_out, w_out, h_unroll, w_unroll, w_base, p, q;
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int H_out = H - K + 1;
	int W_out = W - K + 1;
	int W_unroll = H_out * W_out;

	if (t < C * W_unroll)
	{
		c = t / W_unroll;
		s = t % W_unroll;
		h_out = s / W_out;
		w_out = s % W_out;
		h_unroll = h_out * W_out + w_out;
		w_base = c * (K * K);

		for (p = 0; p < K; p++)
		{
			for (q = 0; q < K; q++)
			{
				w_unroll = w_base + p * K + q;
				data_col[w_unroll * W_unroll + h_unroll] = image[c * H * W + (h_out + p) * W + (w_out + q)];
			}
		}
	}
}

__global__ void matrixMultiplicationKernel_1(float* A, float* B, float* C, int m, int n, int k, int image)
{
    // Xác định chỉ số hàng và cột trong ma trận kết quả
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Kiểm tra điều kiện biên
    if (row < m && col < k) {
        float val = 0.0f;

        // Tính tích vô hướng của hàng A và cột B
        for (int i = 0; i < n; i++) {
            val += A[row * n + i] * B[i * k + col];
        }

        // Ghi kết quả vào ma trận C
        C[row * k + col] = val;
    }
}



__global__ void matrixMultiplicationKernel_2(float* A, float* B, float* C, int m, int n, int k, int image)
{
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int row = threadIdx.y + blockIdx.y * TILE_WIDTH;
    int col = threadIdx.x + blockIdx.x * TILE_WIDTH;
    float val = 0.0f;

    // Duyệt qua các "tiled blocks" để thực hiện phép nhân ma trận
    for (int i = 0; i < (n + TILE_WIDTH - 1) / TILE_WIDTH; i++)
    {
        // Nạp dữ liệu từ A vào shared memory tile_A
        if (row < m && (i * TILE_WIDTH + threadIdx.x) < n) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * n + i * TILE_WIDTH + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Nạp dữ liệu từ B vào shared memory tile_B
        if (col < k && (i * TILE_WIDTH + threadIdx.y) < n) {
            tile_B[threadIdx.y][threadIdx.x] = B[(i * TILE_WIDTH + threadIdx.y) * k + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Đồng bộ hóa các thread trong block để đảm bảo mọi thread đã nạp xong dữ liệu vào shared memory
        __syncthreads();

        // Tính tích của tile_A và tile_B
        for (int j = 0; j < TILE_WIDTH; j++) {
            val += tile_A[threadIdx.y][j] * tile_B[j][threadIdx.x];
        }

        // Đồng bộ hóa các thread trong block trước khi tiếp tục lặp qua các tiles tiếp theo
        __syncthreads();
    }

    // Ghi giá trị tính được vào ma trận kết quả C
    if (row < m && col < k) {
        C[row * k + col] = val;
    }
}



void matrixMultiplicationCPU(float* A, float *B, float *C, int m, int n, int k)
{	
	for (int r = 0; r < m; r++)
        {
            for (int c = 0; c < k; c++)
            {
                for (int i = 0; i < n; i++) 
                {
                    C[idx1D(r, c, k)] += A[idx1D(r, i, n)] * B[idx1D(i, c, k)];
                }
            }
        }
}

void unrollGPUWrapper(int C, int H, int W, int K, float* image, float* data_col)
{
	int H_out = H - K + 1;
	int W_out = W - K + 1;
	int W_unroll = H_out * W_out;
	int num_threads = C * H_out * W_out;
	int block_size = 1024;
	int num_blocks = ceil((float)num_threads / block_size);
	
	// Copy image to device
	float* d_image;
	CHECK(cudaMalloc(&d_image, C * H * W * sizeof(float)));
	CHECK(cudaMemcpy(d_image, image, C * H * W * sizeof(float), cudaMemcpyHostToDevice));

	// Copy data_col to device
	float* d_data_col;
	CHECK(cudaMalloc(&d_data_col, C * K * K * W_unroll * sizeof(float)));

	unrollKernel_1<<<num_blocks, block_size>>>(C, H, W, K, d_image, d_data_col);
	CHECK(cudaGetLastError());

	// Copy data_col back to host
	CHECK(cudaMemcpy(data_col, d_data_col, C * K * K * W_unroll * sizeof(float), cudaMemcpyDeviceToHost));
	// Free memory
	CHECK(cudaFree(d_image));
	CHECK(cudaFree(d_data_col));
}

void matrixMultiplicationGPUWrapper(float* A, float *B, float *C, int m, int n, int k, int i, bool isOptimized)
{	
    // Kích thước block và grid
    dim3 blockSize(32, 32);
    dim3 gridSize((k + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);

    // Kích thước bộ nhớ
    const int size_A = m * n * sizeof(float);
    const int size_B = n * k * sizeof(float);
    const int size_C = m * k * sizeof(float);

    // Cấp phát bộ nhớ trên GPU
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, size_A));
    CHECK(cudaMalloc(&d_B, size_B));
    CHECK(cudaMalloc(&d_C, size_C));

    // Copy dữ liệu từ CPU sang GPU
    CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    // Gọi kernel
    if (isOptimized == false) {
        matrixMultiplicationKernel_1<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k, i);
    } else {
        matrixMultiplicationKernel_2<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k, i);
    }
    CHECK(cudaGetLastError());
    
    // Đồng bộ GPU để đảm bảo kernel hoàn thành
    CHECK(cudaDeviceSynchronize());

    // Copy kết quả từ GPU sang CPU
    CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    // Giải phóng bộ nhớ GPU
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
}


// void matrixMultiplicationGPUWrapper(float* A, float *B, float *C, int m, int n, int k, int i, bool isOptimized)
// {	
	
// 	memset(C, 0, m * k * sizeof(float));

// 	dim3 blockSize(32, 32);
// 	float *d_A, *d_B, *d_C;
// 	const int size_A = m * n * sizeof(float);
// 	const int size_B = n * k * sizeof(float);
// 	const int size_C = m * k * sizeof(float);
// 	CHECK(cudaMalloc(&d_A, size_A));
// 	CHECK(cudaMalloc(&d_B, size_B));
// 	CHECK(cudaMalloc(&d_C, size_C));

// 	CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
// 	CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));
// 	// CHECK(cudaMemcpy(d_C, C, size_C, cudaMemcpyHostToDevice));

// 	dim3 gridSize( (k - 1)/(blockSize.x) + 1, ( m - 1)/(blockSize.y) + 1);
// 	if (!isOptimized){
// 		matrixMultiplicationKernel_1<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k, i);
// 	}
// 	else{
// 		matrixMultiplicationKernel_2<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k, i);
// 	}
// 	CHECK(cudaGetLastError());
// 	CHECK(cudaDeviceSynchronize());
// 	CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

// 	CHECK(cudaFree(d_A));
// 	CHECK(cudaFree(d_B));
// 	CHECK(cudaFree(d_C));
// }