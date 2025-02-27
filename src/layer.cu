#include "layer.h"

#define THREADS_PER_BLOCK 512
#define BLOCK_SIZE 16

__global__ void __launch_bounds__(BLOCK_SIZE * BLOCK_SIZE) matmul(float *A, float *B, float *b, float *out,
                            size_t M, size_t K, size_t N) {
    // A: [M, K] , B: [N, K] , b: [M] , out: [M, N]
    // out idx
    int row = blockIdx.x * BLOCK_SIZE + threadIdx.x / BLOCK_SIZE;
    int col = blockIdx.y * BLOCK_SIZE + threadIdx.x % BLOCK_SIZE;
    // block idx
    int cRow = blockIdx.x;
    int cCol = blockIdx.y;
    // thread idx in block
    int threadRow = threadIdx.x / BLOCK_SIZE;
    int threadCol = threadIdx.x % BLOCK_SIZE;

    A += cRow * BLOCK_SIZE * K;
    B += cCol * BLOCK_SIZE * K;
    out += cRow * BLOCK_SIZE * N + cCol * BLOCK_SIZE;

    float val = 0.f;

    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE + 1];

    for (size_t blk_i = 0; blk_i < K; blk_i += BLOCK_SIZE) {
        A_shared[threadRow][threadCol] = 
            (cRow * BLOCK_SIZE + threadRow >= M || blk_i + threadCol >= K) ? 
            0.0f : A[threadRow * K + threadCol];
        B_shared[threadRow][threadCol] = 
            (cCol * BLOCK_SIZE + threadRow >= N || blk_i + threadCol >= K) ? 
            0.0f : B[threadRow * K + threadCol];
        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE;

        for (size_t dot_i = 0; dot_i < BLOCK_SIZE; dot_i++) {
          val += A_shared[threadRow][dot_i] * B_shared[threadCol][dot_i];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        out[threadRow * N + threadCol] = val + b[col];
    }
}
__global__ void __launch_bounds__(BLOCK_SIZE * BLOCK_SIZE) matmul_relu(float *A, float *B, float *b, float *out,
                            size_t M, size_t K, size_t N) {
    // A: [M, K] , B: [N, K] , b: [M] , out: [M, N]
    // out idx
    int row = blockIdx.x * BLOCK_SIZE + threadIdx.x / BLOCK_SIZE;
    int col = blockIdx.y * BLOCK_SIZE + threadIdx.x % BLOCK_SIZE;
    // block idx
    int cRow = blockIdx.x;
    int cCol = blockIdx.y;
    // thread idx in block
    int threadRow = threadIdx.x / BLOCK_SIZE;
    int threadCol = threadIdx.x % BLOCK_SIZE;

    A += cRow * BLOCK_SIZE * K;
    B += cCol * BLOCK_SIZE * K;
    out += cRow * BLOCK_SIZE * N + cCol * BLOCK_SIZE;

    float val = 0.f;

    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE + 1];

    for (size_t blk_i = 0; blk_i < K; blk_i += BLOCK_SIZE) {
        A_shared[threadRow][threadCol] = 
            (cRow * BLOCK_SIZE + threadRow >= M || blk_i + threadCol >= K) ? 
            0.0f : A[threadRow * K + threadCol];
        B_shared[threadRow][threadCol] = 
            (cCol * BLOCK_SIZE + threadRow >= N || blk_i + threadCol >= K) ? 
            0.0f : B[threadRow * K + threadCol];
        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE;

        for (size_t dot_i = 0; dot_i < BLOCK_SIZE; dot_i++) {
          val += A_shared[threadRow][dot_i] * B_shared[threadCol][dot_i];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        out[threadRow * N + threadCol] = MAX(val + b[col], 0);
    }
}
__global__  __launch_bounds__(256) void sgemm_medium(int M, int N, int K, float *A, float *B, float *b, float *C){
    // ms = ns = 32, ks = 8
    // mw = 16, nw = 32
    // mr = 4, nr = 4
    // blockId, warpId, and threadIdx
    int ms = 32, ns = 32, ks = 8, mw = 16, nw = 32, mr = 4, nr = 4;
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x; 
    // initial global read column
    int k = 0;
    // block row range: blockIdx.x * ms ~ blockIdx.x * ms + ms - 1
    // warp row id:  

    // global memory read
    // tile A size = ms x ks = 32 * 8, col major
    // tile B size = ns x ks = 32 * 8, row major
    // init double buffer with size ms * ks * 2 + ns * ks * 2 = 1024 in shared memory
    // [buffer_A_1, buffer_A_2, buffer_B_1, buffer_B_2]
    __shared__ float sAB[1024]; 
    int buffer_A_offset = 0;
    int buffer_B_offset = 2 * ms * ks;
    // tile A global offset
    // block bx read tile A with rows in [bx * ms, bx * ms + ms - 1]
    A += bx * ms;

    // tile B global offset
    // block bx read tile A with rows in [bx * ms, bx * ms + ms - 1]
    B += by * ns;

    // tile A inner offset.
    // Each thread load (32 * 8) / 64 = 4 floats from A.
    int load_tile_A_num_floats_one_thread = (int)((ms * ks) / blockDim.x);
    // number of threads to load a column of tile A: 32 floats / 4 floats = 8 threads,
    int load_tile_A_num_threads_one_col = (int)(ms / load_tile_A_num_floats_one_thread);
    // thread tx load 4 floats with rows = [(tx % 4 threads) * 4, (tx % 4  threads) * 4 + 3],
    //                              col  = (tx / 4 threads) of tile A
    A += (tx % load_tile_A_num_threads_one_col) * (load_tile_A_num_floats_one_thread) + (int)(tx / load_tile_A_num_threads_one_col) * M;

    // tile B inner offset.
    // each thread load (32 * 8) / 64 = 4 floats from B.
    int load_tile_B_num_floats_one_thread = (int)((ns * ks) / blockDim.x);
    // number of threads to load a column of tile B: 32 floats / 4 floats = 8 threads,
    int load_tile_B_num_threads_one_col = (int)(ns / load_tile_B_num_floats_one_thread);
    // thread tx load 4 floats with rows = [(tx % 4 threads) * 4, (tx % 4  threads) * 4 + 3],
    //                              col  = (tx / 4 threads) of tile A
    B += (tx % load_tile_B_num_threads_one_col) * (load_tile_B_num_floats_one_thread) + (int)(tx / load_tile_B_num_threads_one_col) * N;

    // prefetch the vector from A and B in global memory 
    float4 prefetch_vector_tile_A = *((float4*)A);
    float4 prefetch_vector_tile_B = *((float4*)B);

    // offset to store the prefetch vector
    int offset_store_prefetch = ((k / ks) & 1);
    
    // get the pointer to prefetched buffer A and prefetched buffer B
    float* buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_prefetch * ms * ks;
    float* buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_prefetch * ns * ks;

    // store the vectors in the prefetched buffer A and prefetched buffer B
    *(((float4*)buffer_A) + tx) = prefetch_vector_tile_A;
    *(((float4*)buffer_B) + tx) = prefetch_vector_tile_B;

    __syncthreads();
    
    // warp size mw x nw (16 x 32)
    //           -----------------
    //          |      vec B      |
    //           -----------------                 
    //  -----    -----------------    -             -
    // |     |  |     warp 0      |   | mw = 16     | ms = 32
    // | vec |  |                 |   |             | 
    // |     |   -----------------    -             |
    // |  A  |  |     warp 1      |                 | 
    // |     |  |                 |                 |
    //  -----    -----------------                  -
    //              ns = nw = 32

    // numbers of warp along A vector and B vector
    int num_warp_A = int(ms / mw);
    int num_warp_B = int(ns / nw);
    
    // 1D warp id =  tx / 32
    int id_warp = (int)(tx / 32);
    
    // 2D warp arrangement, row major
    // 2D warp idB = 1D warp id % num_warp_B
    //         idA = 1D warp id / num_warp_B    
    int idB_warp = id_warp % num_warp_B;
    int idA_warp = int(id_warp / num_warp_B);
    
    // offset for the warp tile
    // offset vec A = 2D warp idA * mw
    // offset vec B = 2D warp idB * nw
    int offset_vec_A_warp = idA_warp * mw;
    int offset_vec_B_warp = idB_warp * nw;

    // inner warp thread arrangement 1, row major
    //                warp 0
    //      --------------------------             -
    //     |  0  1  2  3  4  5  6  7  |  mr = 4    |  mw = 16  
    //     |  8  9 10 11 12 13 14 15  |            |
    //     | 16 17 18 19 20 21 22 23  |            |
    //     | 24 25 26 27 28 29 30 31  |            |
    //      --------------------------             -
    //      nr = 4
    //      nw = nr * 8 = 32

    //2D thread idB = tx % (nw / nr)
    //          idA = tx / (nw / nr)
    int idB_thread = ((tx & 31) % ((int)(nw / nr)));
    int idA_thread = int((tx & 31) / (nw / nr));

    // offset for the threads
    // offset vec A = 2D thread idA * mr
    // offset vec B = 2D thread idA * nr
    int offset_vec_A_thread = idA_thread * mr;
    int offset_vec_B_thread = idB_thread * nr;

    // load two vectors with size 4 from buffer A and buffer B into registers
    // initial the registers, to store two vectors with size mr and nr
    // prefetch with the double buffer
    float4 vec_A[2];
    float4 vec_B[2];
    float res[16];
    memset(res, 0, sizeof(res));
    // initial outer product column
    int kk = -1;
      
    // offset of register store for prefetching
    int offset_prefetch_register_kk = ((kk + 1) & 1);
    
    // offset of register to use 
    int offset_register_kk = 0;
    
    // offset of vec A and vec B w.r.t kk:
    int offset_load_vec_A_kk = ((kk + 1) % ks) * ms;
    int offset_load_vec_B_kk = ((kk + 1) % ks) * ns;
    
    // load the vectors from buffer to registers
    vec_A[offset_prefetch_register_kk] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk);
    vec_B[offset_prefetch_register_kk] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk);
    
    // K loop
    for(k = 0; k < K; k += ks){
        if (k + ks < K) {
          // tile A abd tile B global offsets move forward ks columns
          A += ks * M; 
          B += ks * N; 
          // prefetch the vector from A and B in global memory 
          prefetch_vector_tile_A = *((float4*)A);
          prefetch_vector_tile_B = *((float4*)B);
        }

        // inner k loop, 8
        for(kk = 0; kk < ks; ++kk){
            offset_register_kk = ((kk) & 1);
            offset_prefetch_register_kk = ((kk + 1) & 1);
    
            // offset of vec A and vec B w.r.t kk:
            offset_load_vec_A_kk = ((kk + 1) % ks) * ms;
            offset_load_vec_B_kk = ((kk + 1) % ks) * ns;
            
            // load the vectors from buffer to registers
            vec_A[offset_prefetch_register_kk] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk);
            vec_B[offset_prefetch_register_kk] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk);

            res[ 0] += vec_A[offset_register_kk].x * vec_B[offset_register_kk].x;
            res[ 1] += vec_A[offset_register_kk].x * vec_B[offset_register_kk].y;
            res[ 2] += vec_A[offset_register_kk].x * vec_B[offset_register_kk].z;
            res[ 3] += vec_A[offset_register_kk].x * vec_B[offset_register_kk].w;

            res[ 4] += vec_A[offset_register_kk].y * vec_B[offset_register_kk].x;
            res[ 5] += vec_A[offset_register_kk].y * vec_B[offset_register_kk].y;
            res[ 6] += vec_A[offset_register_kk].y * vec_B[offset_register_kk].z;
            res[ 7] += vec_A[offset_register_kk].y * vec_B[offset_register_kk].w;

            res[ 8] += vec_A[offset_register_kk].z * vec_B[offset_register_kk].x;
            res[ 9] += vec_A[offset_register_kk].z * vec_B[offset_register_kk].y;
            res[10] += vec_A[offset_register_kk].z * vec_B[offset_register_kk].z;
            res[11] += vec_A[offset_register_kk].z * vec_B[offset_register_kk].w;

            res[12] += vec_A[offset_register_kk].w * vec_B[offset_register_kk].x;
            res[13] += vec_A[offset_register_kk].w * vec_B[offset_register_kk].y;
            res[14] += vec_A[offset_register_kk].w * vec_B[offset_register_kk].z;
            res[15] += vec_A[offset_register_kk].w * vec_B[offset_register_kk].w;
        }
        
        // update offset to store the prefetch vector
        offset_store_prefetch = (((int)(k / ks) + 1) & 1);
        
        // update the pointer to prefetched buffer A and prefetched buffer B
        buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_prefetch * ms * ks;
        buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_prefetch * ns * ks;
        
        // store the vectors in the prefetched buffer A and prefetched buffer B
        *(((float4*)buffer_A) + tx) = prefetch_vector_tile_A;
        *(((float4*)buffer_B) + tx) = prefetch_vector_tile_B;
        __syncthreads();
        // initial outer product column
        kk = -1;
        
        // offset of register store for prefetching
        offset_prefetch_register_kk = ((kk + 1) & 1);
        
        // offset of vec A and vec B w.r.t kk:
        offset_load_vec_A_kk = ((kk + 1) % ks) * ms;
        offset_load_vec_B_kk = ((kk + 1) % ks) * ns;
        
        // load the vectors from buffer to registers
        vec_A[offset_prefetch_register_kk] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk);
        vec_B[offset_prefetch_register_kk] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk);

    }
    
    C += bx * ms + offset_vec_A_warp + offset_vec_A_thread;
    C += (by * ns + offset_vec_B_warp + offset_vec_B_thread) * M;
    int bias_start = bx * ms + offset_vec_A_warp + offset_vec_A_thread;
    float4 bias_vec = *((float4*)(b + bias_start));

    float4 C_res[4];
    C_res[0] = *((float4 *)C);
    C_res[1] = *((float4 *)(C + M));
    C_res[2] = *((float4 *)(C + 2 * M));
    C_res[3] = *((float4 *)(C + 3 * M));

    C_res[0].x = res[0 ] + bias_vec.x;
    C_res[0].y = res[4 ] + bias_vec.y;
    C_res[0].z = res[8 ] + bias_vec.z;
    C_res[0].w = res[12] + bias_vec.w;
    
    C_res[1].x = res[1 ] + bias_vec.x;
    C_res[1].y = res[5 ] + bias_vec.y;
    C_res[1].z = res[9 ] + bias_vec.z;
    C_res[1].w = res[13] + bias_vec.w;

    C_res[2].x = res[2 ] + bias_vec.x;
    C_res[2].y = res[6 ] + bias_vec.y;
    C_res[2].z = res[10] + bias_vec.z;
    C_res[2].w = res[14] + bias_vec.w;

    C_res[3].x = res[3 ] + bias_vec.x;
    C_res[3].y = res[7 ] + bias_vec.y;
    C_res[3].z = res[11] + bias_vec.z;
    C_res[3].w = res[15] + bias_vec.w;

    *((float4 *)C) = C_res[0];
    *((float4 *)(C + M)) = C_res[1];
    *((float4 *)(C + 2 * M)) = C_res[2];
    *((float4 *)(C + 3 * M)) = C_res[3];
}
__global__  __launch_bounds__(256) void sgemm_medium_relu(int M, int N, int K, float *A, float *B, float *b, float *C){
    // ms = ns = 32, ks = 8
    // mw = 16, nw = 32
    // mr = 4, nr = 4
    // blockId, warpId, and threadIdx
    int ms = 32, ns = 32, ks = 8, mw = 16, nw = 32, mr = 4, nr = 4;
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x; 
    // initial global read column
    int k = 0;
    // block row range: blockIdx.x * ms ~ blockIdx.x * ms + ms - 1
    // warp row id:  

    // global memory read
    // tile A size = ms x ks = 32 * 8, col major
    // tile B size = ns x ks = 32 * 8, row major
    // init double buffer with size ms * ks * 2 + ns * ks * 2 = 1024 in shared memory
    // [buffer_A_1, buffer_A_2, buffer_B_1, buffer_B_2]
    __shared__ float sAB[1025]; 
    int buffer_A_offset = 0;
    int buffer_B_offset = 2 * ms * ks;
    // tile A global offset
    // block bx read tile A with rows in [bx * ms, bx * ms + ms - 1]
    A += bx * ms;

    // tile B global offset
    // block bx read tile A with rows in [bx * ms, bx * ms + ms - 1]
    B += by * ns;

    // tile A inner offset.
    // Each thread load (32 * 8) / 64 = 4 floats from A.
    int load_tile_A_num_floats_one_thread = (int)((ms * ks) / blockDim.x);
    // number of threads to load a column of tile A: 32 floats / 4 floats = 8 threads,
    int load_tile_A_num_threads_one_col = (int)(ms / load_tile_A_num_floats_one_thread);
    // thread tx load 4 floats with rows = [(tx % 4 threads) * 4, (tx % 4  threads) * 4 + 3],
    //                              col  = (tx / 4 threads) of tile A
    A += (tx % load_tile_A_num_threads_one_col) * (load_tile_A_num_floats_one_thread) + (int)(tx / load_tile_A_num_threads_one_col) * M;

    // tile B inner offset.
    // each thread load (32 * 8) / 64 = 4 floats from B.
    int load_tile_B_num_floats_one_thread = (int)((ns * ks) / blockDim.x);
    // number of threads to load a column of tile B: 32 floats / 4 floats = 8 threads,
    int load_tile_B_num_threads_one_col = (int)(ns / load_tile_B_num_floats_one_thread);
    // thread tx load 4 floats with rows = [(tx % 4 threads) * 4, (tx % 4  threads) * 4 + 3],
    //                              col  = (tx / 4 threads) of tile A
    B += (tx % load_tile_B_num_threads_one_col) * (load_tile_B_num_floats_one_thread) + (int)(tx / load_tile_B_num_threads_one_col) * N;

    // prefetch the vector from A and B in global memory 
    float4 prefetch_vector_tile_A = *((float4*)A);
    float4 prefetch_vector_tile_B = *((float4*)B);

    // offset to store the prefetch vector
    int offset_store_prefetch = ((k / ks) & 1);
    
    // get the pointer to prefetched buffer A and prefetched buffer B
    float* buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_prefetch * ms * ks;
    float* buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_prefetch * ns * ks;

    // store the vectors in the prefetched buffer A and prefetched buffer B
    *(((float4*)buffer_A) + tx) = prefetch_vector_tile_A;
    *(((float4*)buffer_B) + tx) = prefetch_vector_tile_B;

    __syncthreads();
    
    // warp size mw x nw (16 x 32)
    //           -----------------
    //          |      vec B      |
    //           -----------------                 
    //  -----    -----------------    -             -
    // |     |  |     warp 0      |   | mw = 16     | ms = 32
    // | vec |  |                 |   |             | 
    // |     |   -----------------    -             |
    // |  A  |  |     warp 1      |                 | 
    // |     |  |                 |                 |
    //  -----    -----------------                  -
    //              ns = nw = 32

    // numbers of warp along A vector and B vector
    int num_warp_A = int(ms / mw);
    int num_warp_B = int(ns / nw);
    
    // 1D warp id =  tx / 32
    int id_warp = (int)(tx / 32);
    
    // 2D warp arrangement, row major
    // 2D warp idB = 1D warp id % num_warp_B
    //         idA = 1D warp id / num_warp_B    
    int idB_warp = id_warp % num_warp_B;
    int idA_warp = int(id_warp / num_warp_B);
    
    // offset for the warp tile
    // offset vec A = 2D warp idA * mw
    // offset vec B = 2D warp idB * nw
    int offset_vec_A_warp = idA_warp * mw;
    int offset_vec_B_warp = idB_warp * nw;

    // inner warp thread arrangement 1, row major
    //                warp 0
    //      --------------------------             -
    //     |  0  1  2  3  4  5  6  7  |  mr = 4    |  mw = 16  
    //     |  8  9 10 11 12 13 14 15  |            |
    //     | 16 17 18 19 20 21 22 23  |            |
    //     | 24 25 26 27 28 29 30 31  |            |
    //      --------------------------             -
    //      nr = 4
    //      nw = nr * 8 = 32

    //2D thread idB = tx % (nw / nr)
    //          idA = tx / (nw / nr)
    int idB_thread = ((tx & 31) % ((int)(nw / nr)));
    int idA_thread = int((tx & 31) / (nw / nr));

    // offset for the threads
    // offset vec A = 2D thread idA * mr
    // offset vec B = 2D thread idA * nr
    int offset_vec_A_thread = idA_thread * mr;
    int offset_vec_B_thread = idB_thread * nr;

    // load two vectors with size 4 from buffer A and buffer B into registers
    // initial the registers, to store two vectors with size mr and nr
    // prefetch with the double buffer
    float4 vec_A[2];
    float4 vec_B[2];
    float res[16];
    memset(res, 0, sizeof(res));
    // initial outer product column
    int kk = -1;
      
    // offset of register store for prefetching
    int offset_prefetch_register_kk = ((kk + 1) & 1);
    
    // offset of register to use 
    int offset_register_kk = 0;
    
    // offset of vec A and vec B w.r.t kk:
    int offset_load_vec_A_kk = ((kk + 1) % ks) * ms;
    int offset_load_vec_B_kk = ((kk + 1) % ks) * ns;
    
    // load the vectors from buffer to registers
    vec_A[offset_prefetch_register_kk] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk);
    vec_B[offset_prefetch_register_kk] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk);
    
    // K loop
    for(k = 0; k < K; k += ks){
        if (k + ks < K) {
          // tile A abd tile B global offsets move forward ks columns
          A += ks * M; 
          B += ks * N; 
          // prefetch the vector from A and B in global memory 
          prefetch_vector_tile_A = *((float4*)A);
          prefetch_vector_tile_B = *((float4*)B);
        }

        // inner k loop, 8
        for(kk = 0; kk < ks; ++kk){
            offset_register_kk = ((kk) & 1);
            offset_prefetch_register_kk = ((kk + 1) & 1);
    
            // offset of vec A and vec B w.r.t kk:
            offset_load_vec_A_kk = ((kk + 1) % ks) * ms;
            offset_load_vec_B_kk = ((kk + 1) % ks) * ns;
            
            // load the vectors from buffer to registers
            vec_A[offset_prefetch_register_kk] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk);
            vec_B[offset_prefetch_register_kk] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk);

            res[ 0] += vec_A[offset_register_kk].x * vec_B[offset_register_kk].x;
            res[ 1] += vec_A[offset_register_kk].x * vec_B[offset_register_kk].y;
            res[ 2] += vec_A[offset_register_kk].x * vec_B[offset_register_kk].z;
            res[ 3] += vec_A[offset_register_kk].x * vec_B[offset_register_kk].w;

            res[ 4] += vec_A[offset_register_kk].y * vec_B[offset_register_kk].x;
            res[ 5] += vec_A[offset_register_kk].y * vec_B[offset_register_kk].y;
            res[ 6] += vec_A[offset_register_kk].y * vec_B[offset_register_kk].z;
            res[ 7] += vec_A[offset_register_kk].y * vec_B[offset_register_kk].w;

            res[ 8] += vec_A[offset_register_kk].z * vec_B[offset_register_kk].x;
            res[ 9] += vec_A[offset_register_kk].z * vec_B[offset_register_kk].y;
            res[10] += vec_A[offset_register_kk].z * vec_B[offset_register_kk].z;
            res[11] += vec_A[offset_register_kk].z * vec_B[offset_register_kk].w;

            res[12] += vec_A[offset_register_kk].w * vec_B[offset_register_kk].x;
            res[13] += vec_A[offset_register_kk].w * vec_B[offset_register_kk].y;
            res[14] += vec_A[offset_register_kk].w * vec_B[offset_register_kk].z;
            res[15] += vec_A[offset_register_kk].w * vec_B[offset_register_kk].w;
        }
        
        // update offset to store the prefetch vector
        offset_store_prefetch = (((int)(k / ks) + 1) & 1);
        
        // update the pointer to prefetched buffer A and prefetched buffer B
        buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_prefetch * ms * ks;
        buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_prefetch * ns * ks;
        
        // store the vectors in the prefetched buffer A and prefetched buffer B
        *(((float4*)buffer_A) + tx) = prefetch_vector_tile_A;
        *(((float4*)buffer_B) + tx) = prefetch_vector_tile_B;
        __syncthreads();
        // initial outer product column
        kk = -1;
        
        // offset of register store for prefetching
        offset_prefetch_register_kk = ((kk + 1) & 1);
        
        // offset of vec A and vec B w.r.t kk:
        offset_load_vec_A_kk = ((kk + 1) % ks) * ms;
        offset_load_vec_B_kk = ((kk + 1) % ks) * ns;
        
        // load the vectors from buffer to registers
        vec_A[offset_prefetch_register_kk] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk);
        vec_B[offset_prefetch_register_kk] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk);

    }
    
    C += bx * ms + offset_vec_A_warp + offset_vec_A_thread;
    C += (by * ns + offset_vec_B_warp + offset_vec_B_thread) * M;
    int bias_start = bx * ms + offset_vec_A_warp + offset_vec_A_thread;
    float4 bias_vec = *((float4*)(b + bias_start));

    float4 C_res[4];
    C_res[0] = *((float4 *)C);
    C_res[1] = *((float4 *)(C + M));
    C_res[2] = *((float4 *)(C + 2 * M));
    C_res[3] = *((float4 *)(C + 3 * M));

    float tmp;
    tmp = res[0 ] + bias_vec.x;
    C_res[0].x = tmp > 0 ? tmp : 0;
    tmp = res[4 ] + bias_vec.y;
    C_res[0].y = tmp > 0 ? tmp : 0;
    tmp = res[8 ] + bias_vec.z;
    C_res[0].z = tmp > 0 ? tmp : 0;
    tmp = res[12] + bias_vec.w;
    C_res[0].w = tmp > 0 ? tmp : 0;
    
    tmp = res[1 ] + bias_vec.x;
    C_res[1].x = tmp > 0 ? tmp : 0;
    tmp = res[5 ] + bias_vec.y;
    C_res[1].y = tmp > 0 ? tmp : 0;
    tmp = res[9 ] + bias_vec.z;
    C_res[1].z = tmp > 0 ? tmp : 0;
    tmp = res[13] + bias_vec.w;
    C_res[1].w = tmp > 0 ? tmp : 0;

    tmp = res[2 ] + bias_vec.x;
    C_res[2].x = tmp > 0 ? tmp : 0;
    tmp = res[6 ] + bias_vec.y;
    C_res[2].y = tmp > 0 ? tmp : 0;
    tmp = res[10] + bias_vec.z;
    C_res[2].z = tmp > 0 ? tmp : 0;
    tmp = res[14] + bias_vec.w;
    C_res[2].w = tmp > 0 ? tmp : 0;

    tmp = res[3 ] + bias_vec.x;
    C_res[3].x = tmp > 0 ? tmp : 0;
    tmp = res[7 ] + bias_vec.y;
    C_res[3].y = tmp > 0 ? tmp : 0;
    tmp = res[11] + bias_vec.z;
    C_res[3].z = tmp > 0 ? tmp : 0;
    tmp = res[15] + bias_vec.w;
    C_res[3].w = tmp > 0 ? tmp : 0;

    *((float4 *)C) = C_res[0];
    *((float4 *)(C + M)) = C_res[1];
    *((float4 *)(C + 2 * M)) = C_res[2];
    *((float4 *)(C + 3 * M)) = C_res[3];
}

/* Conv1D 
 * @param [in1]  in: [C * K, BS, os] => [C * K, BS * os] 로 해석 가능
 * @param [in2]   w: [C * K, OC]
 * @param [in3]   b: [OC]
 * @param [out] out: [BS, os, OC] => [BS * os, OC] 로 해석 가능
 *    
 *    In this model, K is 3, 5, 7, or 9, 
 *    with stride = 1, pad = 0, dilation = 1.
 *    The formula for the output sequence length:
 *      os = (in - K + 2 * pad) / stride + 1
 *         = (s - K + 2 * 0) / 1 + 1
 *         = s - K + 1
 *
 * 'BS' is the batch size
 * 'C' is the input channel size
 * 's' is the input sequence length
 * 'OC' is the output channel size
 * 'os' is the output sequence length
 * 'K' is the kernel (or filter) size
 */
void Conv1D_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out, cudaStream_t &stream) {
  size_t CK = in->shape[0];
  size_t BS = in->shape[1];
  size_t os = in->shape[2];
  size_t OC = w->shape[1];

  dim3 blockDim(64);
  dim3 gridDim((OC + 32 - 1) / 32, (BS * os + 32 - 1) / 32);
  sgemm_medium<<<gridDim, blockDim, 0, stream>>>(OC, BS * os, CK, w->d_buf, in->d_buf, b->d_buf, out->d_buf);
  // CHECK_CUDA(cudaDeviceSynchronize());
}

/* Linear
 * @param [in] in: [N, BS]
 * @param [in] w: [N, M]
 * @param [in] b: [M]
 * @param [out] out: [BS, M]
 * 'N' is the input feature size
 * 'M' is the output feature size
 */
void Linear_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out, bool relu) {
  size_t N = in->shape[0];
  size_t BS = in->shape[1];
  size_t M = w->shape[1];
  dim3 blockDim(64);
  dim3 gridDim((M + 32 - 1) / 32, (BS + 32 - 1) / 32);
  if (relu) {
    sgemm_medium_relu<<<gridDim, blockDim>>>(M, BS, N, w->d_buf, in->d_buf, b->d_buf, out->d_buf);
  } else {
    sgemm_medium<<<gridDim, blockDim>>>(M, BS, N, w->d_buf, in->d_buf, b->d_buf, out->d_buf);
  }
  CHECK_CUDA(cudaDeviceSynchronize());
}

/* GetMax
 * @param [in]   in: [BS, s, C]
 * @param [out] out: [BS, C]
 *    
 *    This layer is to get the max value along the sequence dim.
 *    The formula for this layer: out = max(in, dim=-1)
 * 
 * 'BS' is the batch size
 * 's' is the sequence length
 * 'C' is the channel size
 */
__global__ void GetMax_Kernel(float *in, float *out, size_t BS, size_t s, size_t C) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= BS * C) return;
  
  size_t bs = idx / C;
  size_t c = idx % C;

  float max_val = in[bs * C * s + c];
  max_val = MAX(max_val, 0);
  for (size_t j = 1; j < s; j++) {
    float val = in[bs * C * s + j * C + c];
    val = MAX(val, 0);
    max_val = val > max_val ? val : max_val;
  }
  out[bs * C + c] = max_val;
}
void GetMax_CUDA(Tensor *in, Tensor *out) {
  size_t BS = in->shape[0];
  size_t s = in->shape[1];
  size_t C = in->shape[2];

  dim3 blockDim(THREADS_PER_BLOCK);
  dim3 gridDim((BS * C + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1);
  GetMax_Kernel<<<gridDim, blockDim>>>(in->d_buf, out->d_buf, BS, s, C);
  CHECK_CUDA(cudaDeviceSynchronize());
}

/* ReLU_GetMax
 * @param [in]   in: [BS, s, C]
 * @param [out] out: [BS, C]
 *    
 *    This layer is to get the max value along the sequence dim.
 *    The formula for this layer: out = max(in, dim=-1)
 * 
 * 'BS' is the batch size
 * 's' is the sequence length
 * 'C' is the channel size
 */
__global__ void ReLU_GetMax_Kernel(float *in, float *out, size_t BS, size_t s, size_t C) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= BS * C) return;
  
  size_t bs = idx / C;
  size_t c = idx % C;

  float max_val = in[bs * C * s + c];
  max_val = MAX(max_val, 0);
  for (size_t j = 1; j < s; j++) {
    float val = in[bs * C * s + j * C + c];
    val = MAX(val, 0);
    max_val = val > max_val ? val : max_val;
  }
  out[bs * C + c] = max_val;
}
void ReLU_GetMax_CUDA(Tensor *in, Tensor *out, cudaStream_t &stream) {
  size_t BS = in->shape[0];
  size_t s = in->shape[1];
  size_t C = in->shape[2];

  dim3 blockDim(THREADS_PER_BLOCK);
  dim3 gridDim((BS * C + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1);
  GetMax_Kernel<<<gridDim, blockDim, 0, stream>>>(in->d_buf, out->d_buf, BS, s, C);
  // CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void ConcatKernel(float *in1, float *in2, float *in3, float *in4, float *out,
                            size_t BS, size_t N1, size_t N2, size_t N3, size_t N4) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  idx *= 4;
  size_t total_N = N1 + N2 + N3 + N4;

  size_t bs = idx / total_N;
  size_t offset = idx % total_N;
  
  size_t id = bs * total_N + offset;
  *(float4*)(&out[id]) = offset < N1           ? *(float4*)(&in1[bs * N1 + offset]) :
                         offset < N1 + N2      ? *(float4*)(&in2[bs * N2 + (offset - N1)]) :
                         offset < N1 + N2 + N3 ? *(float4*)(&in3[bs * N3 + (offset - N1 - N2)]) :
                                                 *(float4*)(&in4[bs * N4 + (offset - N1 - N2 - N3)]);
}
/* Concat using CUDA */
void Concat_CUDA(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4,
                  Tensor *out) {
  size_t BS = in1->shape[0];
  size_t N1 = in1->shape[1];
  size_t N2 = in2->shape[1];
  size_t N3 = in3->shape[1];
  size_t N4 = in4->shape[1];
  dim3 blockDim(THREADS_PER_BLOCK);
  dim3 gridDim((BS * (N1 + N2 + N3 + N4) + (THREADS_PER_BLOCK * 4) - 1) / (THREADS_PER_BLOCK * 4), 1);
  // Tensor *tmp = new Tensor({N1 + N2 + N3 + N4, BS});
  ConcatKernel<<<gridDim, blockDim>>>(in1->d_buf, in2->d_buf, in3->d_buf, in4->d_buf, out->d_buf, BS, N1, N2, N3, N4);
  // Transpose_CUDA(tmp, out);
  CHECK_CUDA(cudaDeviceSynchronize());
}

/* Linear 
 * @param [in1]  in: [BS, N]
 * @param [in2]   w: [M, N]
 * @param [in3]   b: [M]
 * @param [out] out: [BS, M]
 * 'N' is the input feature size
 * 'M' is the output feature size
 */
void Linear_CUDA_slow(Tensor *in, Tensor *w, Tensor *b, Tensor *out, bool relu, cudaStream_t &stream) {
    size_t BS = in->shape[0];
    size_t N = in->shape[1];
    size_t M = w->shape[0];

    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridDim((BS + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    if (relu) {
        matmul_relu<<<gridDim, blockDim, 0, stream>>>(in->d_buf, w->d_buf, b->d_buf, out->d_buf, BS, N, M);
    } else {
        matmul<<<gridDim, blockDim, 0, stream>>>(in->d_buf, w->d_buf, b->d_buf, out->d_buf, BS, N, M);
    }
    // CHECK_CUDA(cudaDeviceSynchronize());
}

/* Softmax CUDA kernel */
__global__ void SoftmaxKernel(float *inout, size_t BS, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= BS) return;
  float4 vec = *(float4*)(&inout[idx * N]);
  float max_val = MAX(MAX(vec.x, vec.y), MAX(vec.z, vec.w));
  vec.x = exp(vec.x - max_val);
  vec.y = exp(vec.y - max_val);
  vec.z = exp(vec.z - max_val);
  vec.w = exp(vec.w - max_val);
  float sum = vec.x + vec.y + vec.z + vec.w;
  vec.x /= sum;
  vec.y /= sum;
  vec.z /= sum;
  vec.w /= sum;
  *(float4*)(&inout[idx * N]) = vec;
}
/* Softmax using CUDA */
void Softmax_CUDA(Tensor *inout, cudaStream_t &stream) {
  size_t BS = inout->shape[0];
  size_t N = inout->shape[1];

  // N is small, so we can parallelize over BS
  dim3 blockDim(THREADS_PER_BLOCK);
  dim3 gridDim((BS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1);
  SoftmaxKernel<<<gridDim, blockDim, 0, stream>>>(inout->d_buf, BS, N);
  // CHECK_CUDA(cudaDeviceSynchronize());
}

/* deprecated */
/* (Elemwise) Scaling
 * @param [in & out] inout: [BS, N]
 * @param [in]           gate: [BS, 4]
 * @param [in]           gate_col: [1]
 * 'N' is the number of elements in the tensor.
 */
void Scaling(Tensor *inout, float *gate, size_t gate_col) {
  size_t BS = inout->shape[0];
  size_t N = inout->shape[1];

  for (size_t bs = 0; bs < BS; bs++) {
    float scale = gate[bs * 4 + gate_col];
    for (size_t i = 0; i < N; i++) {
      inout->buf[bs * N + i] *= scale;
    }
  }
}

/* Scaling and (Elemwise) Addition
 * @param [in1] in1: [BS, N]
 * @param [in2] in2: [BS, N]
 * @param [in3] in3: [BS, N]
 * @param [in4] in4: [BS, N]
 * @param [out] out: [N, BS]
 * 'N' is the number of elements in the input tensor.
 */
__global__ void Scaling_Add_Kernel(float *in1, float *in2, float *in3, float *in4, float *gate,
                                   float *out, size_t BS, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= BS * N) return;

  size_t bs = idx / N;
  size_t n = idx % N;

  float sum = 0;

  sum += in1[bs * N + n] * gate[bs * 4];
  sum += in2[bs * N + n] * gate[bs * 4 + 1];
  sum += in3[bs * N + n] * gate[bs * 4 + 2];
  sum += in4[bs * N + n] * gate[bs * 4 + 3];
  
  // out[bs * N + n] = sum;
  out[n * BS + bs] = sum;
}
void Scaling_Add_CUDA(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, Tensor *gate,
                      Tensor *out) {
  size_t BS = in1->shape[0];
  size_t N = in1->shape[1];

  dim3 blockDim(THREADS_PER_BLOCK);
  dim3 gridDim((BS * N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1);
  // Tensor *tmp = new Tensor({N, BS});
  Scaling_Add_Kernel<<<gridDim, blockDim>>>(in1->d_buf, in2->d_buf, in3->d_buf, in4->d_buf,
                                            gate->d_buf, out->d_buf, BS, N);
  // Transpose_CUDA(tmp, out);
  CHECK_CUDA(cudaDeviceSynchronize());
}

/* im2col_1d_CUDA
 * @param [in]  in: [BS, C, s]
 * @param [out] out: [C * K, BS, os]
 * 'K' is the kernel size
 * 'os' is the output sequence length
 */
__global__ void im2col_1d_Kernel(const float *in, float *out,
                                    size_t BS, size_t C, size_t s, size_t K, size_t os) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= C * K * BS * os) return;

  // 출력 텐서를 [C * K, BS * os]로 해석
  size_t row = idx / (BS * os);
  size_t col = idx % (BS * os);

  // row를 분해: row = c * K + k
  size_t c = row / K;
  size_t k = row % K;

  // col을 분해: col = bs * os + j
  size_t bs = col / os;
  size_t j = col % os;

  // in[bs, c, j+k]
  size_t in_idx = bs * (C * s) + c * s + (j + k);
  out[idx] = in[in_idx];
}
void im2col_1d_CUDA(Tensor *in, Tensor *out, size_t K, cudaStream_t &stream) {
  size_t BS = in->shape[0];
  size_t C = in->shape[1];
  size_t s = in->shape[2];
  size_t os = s - K + 1;

  dim3 blockDim(THREADS_PER_BLOCK);
  dim3 gridDim((BS * os * C * K + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1);
  im2col_1d_Kernel<<<gridDim, blockDim, 0, stream>>>(in->d_buf, out->d_buf, BS, C, s, K, os);
  // CHECK_CUDA(cudaDeviceSynchronize());
}