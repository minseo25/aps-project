#include "layer.h"

#define THREADS_PER_BLOCK 256
#define BLOCK_SIZE 16

__global__ void __launch_bounds__(BLOCK_SIZE * BLOCK_SIZE) gemm_1(float *A, float *B, float *b, float *out,
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
__global__  __launch_bounds__(256) void sgemm_large(int M, int N, int K, float *A, float *B, float *b, float *C){
    // ms = ns = 64, ks = 8
    // mw = 32, nw = 64
    // mr = 8, nr = 8

    // mw x nw = 32 x mr x nr
    // (ms/mw) x (ns/nw) = (BLOCKDIM/32)
    // ms * ks / BLOCKDIM => 4의 배수
    // ns * ks / BLOCKDIM => 4의 배수
  
    // blockId, warpId, and threadIdx
    int ms = 64, ns = 64, ks = 8, mw = 32, nw = 64, mr = 8, nr = 8;
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x; 
    // initial global read column
    int k = 0;
    // block row range: blockIdx.x * ms ~ blockIdx.x * ms + ms - 1
    // warp row id:  

    // global memory read
    // tile A size = ms x ks = 64 * 8, col major
    // tile B size = ns x ks = 64 * 8, row major
    // init double buffer with size ms * ks * 2 + ns * ks * 2 = 2048 in shared memory
    // [buffer_A_1, buffer_A_2, buffer_B_1, buffer_B_2]
    __shared__ float sAB[2048]; 
    int buffer_A_offset = 0;
    int buffer_B_offset = 2 * ms * ks;
    // tile A global offset
    // block bx read tile A with rows in [bx * ms, bx * ms + ms - 1]
    A += bx * ms;

    // tile B global offset
    // block bx read tile A with rows in [bx * ms, bx * ms + ms - 1]
    B += by * ns;

    // tile A inner offset.
    // Each thread load (64 * 8) / 64 = 8 floats from A.
    int load_tile_A_num_floats_one_thread = (int)((ms * ks) / blockDim.x);
    // number of threads to load a column of tile A: 64 floats / 8 floats = 8 threads,
    int load_tile_A_num_threads_one_col = (int)(ms / load_tile_A_num_floats_one_thread);
    // thread tx load 8 floats with rows = [(tx % 8 threads) * 8, (tx % 8 threads) * 8 + 7],
    //                              col  = (tx / 8 threads) of tile A
    A += (tx % load_tile_A_num_threads_one_col) * (load_tile_A_num_floats_one_thread) + (int)(tx / load_tile_A_num_threads_one_col) * M;

    // tile B inner offset.
    // each thread load (64 * 8) / 64 = 8 floats from B.
    int load_tile_B_num_floats_one_thread = (int)((ns * ks) / blockDim.x);
    // number of threads to load a column of tile B: 64 floats / 8 floats = 8 threads,
    int load_tile_B_num_threads_one_col = (int)(ns / load_tile_B_num_floats_one_thread);
    // thread tx load 8 floats with rows = [(tx % 8 threads) * 8, (tx % 8 threads) * 8 + 7],
    //                              col  = (tx / 8 threads) of tile A
    B += (tx % load_tile_B_num_threads_one_col) * (load_tile_B_num_floats_one_thread) + (int)(tx / load_tile_B_num_threads_one_col) * N;

    // prefetch the vector from A and B in global memory 
    float4 prefetch_vector_tile_A[2], prefetch_vector_tile_B[2];
    prefetch_vector_tile_A[0] = *((float4*)A);
    prefetch_vector_tile_A[1] = *((float4*)A + 1);
    prefetch_vector_tile_B[0] = *((float4*)B);
    prefetch_vector_tile_B[1] = *((float4*)B + 1);

    // offset to store the prefetch vector
    int offset_store_prefetch = ((k / ks) & 1);
    
    // get the pointer to prefetched buffer A and prefetched buffer B
    float* buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_prefetch * ms * ks;
    float* buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_prefetch * ns * ks;

    // store the vectors in the prefetched buffer A and prefetched buffer B
    *(((float4*)buffer_A) + 2 * tx) = prefetch_vector_tile_A[0];
    *(((float4*)buffer_A) + 2 * tx + 1) = prefetch_vector_tile_A[1];
    *(((float4*)buffer_B) + 2 * tx) = prefetch_vector_tile_B[0];
    *(((float4*)buffer_B) + 2 * tx + 1) = prefetch_vector_tile_B[1];

    __syncthreads();
    
    // warp size mw x nw (32 x 64)
    //           -----------------
    //          |      vec B      |
    //           -----------------                 
    //  -----    -----------------    -             -
    // |     |  |     warp 0      |   | mw = 32     | ms = 64
    // | vec |  |                 |   |             | 
    // |     |   -----------------    -             |
    // |  A  |  |     warp 1      |                 | 
    // |     |  |                 |                 |
    //  -----    -----------------                  -
    //              ns = nw = 64

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
    float4 vec_A[4];
    float4 vec_B[4];
    float res[64];
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
    vec_A[offset_prefetch_register_kk * 2    ] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk);
    vec_A[offset_prefetch_register_kk * 2 + 1] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk + 4);
    vec_B[offset_prefetch_register_kk * 2    ] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk);
    vec_B[offset_prefetch_register_kk * 2 + 1] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk + 4);
    
    // K loop
    for(k = 0; k < K; k += ks){
        if (k + ks < K){
          // tile A abd tile B global offsets move forward ks columns
          A += ks * M; 
          B += ks * N; 
          // prefetch the vector from A and B in global memory 
          prefetch_vector_tile_A[0] = *((float4*)A);  
          prefetch_vector_tile_A[1] = *((float4*)A + 1);
          prefetch_vector_tile_B[0] = *((float4*)B);
          prefetch_vector_tile_B[1] = *((float4*)B + 1);
        }

        // inner k loop, 8
        for(kk = 0; kk < ks; ++kk){
            offset_register_kk = ((kk) & 1);
            offset_prefetch_register_kk = ((kk + 1) & 1);
    
            // offset of vec A and vec B w.r.t kk:
            offset_load_vec_A_kk = ((kk + 1) % ks) * ms;
            offset_load_vec_B_kk = ((kk + 1) % ks) * ns;
            
            // load the vectors from buffer to registers
            vec_A[offset_prefetch_register_kk * 2    ] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk);
            vec_A[offset_prefetch_register_kk * 2 + 1] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk + 4);
            vec_B[offset_prefetch_register_kk * 2    ] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk);
            vec_B[offset_prefetch_register_kk * 2 + 1] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk + 4);

            res[ 0] += vec_A[offset_register_kk * 2 + 0].x * vec_B[offset_register_kk * 2 + 0].x;
            res[ 1] += vec_A[offset_register_kk * 2 + 0].x * vec_B[offset_register_kk * 2 + 0].y;
            res[ 2] += vec_A[offset_register_kk * 2 + 0].x * vec_B[offset_register_kk * 2 + 0].z;
            res[ 3] += vec_A[offset_register_kk * 2 + 0].x * vec_B[offset_register_kk * 2 + 0].w;

            res[ 4] += vec_A[offset_register_kk * 2 + 0].x * vec_B[offset_register_kk * 2 + 1].x;
            res[ 5] += vec_A[offset_register_kk * 2 + 0].x * vec_B[offset_register_kk * 2 + 1].y;
            res[ 6] += vec_A[offset_register_kk * 2 + 0].x * vec_B[offset_register_kk * 2 + 1].z;
            res[ 7] += vec_A[offset_register_kk * 2 + 0].x * vec_B[offset_register_kk * 2 + 1].w;

            res[ 8] += vec_A[offset_register_kk * 2 + 0].y * vec_B[offset_register_kk * 2 + 0].x;
            res[ 9] += vec_A[offset_register_kk * 2 + 0].y * vec_B[offset_register_kk * 2 + 0].y;
            res[10] += vec_A[offset_register_kk * 2 + 0].y * vec_B[offset_register_kk * 2 + 0].z;
            res[11] += vec_A[offset_register_kk * 2 + 0].y * vec_B[offset_register_kk * 2 + 0].w;

            res[12] += vec_A[offset_register_kk * 2 + 0].y * vec_B[offset_register_kk * 2 + 1].x;
            res[13] += vec_A[offset_register_kk * 2 + 0].y * vec_B[offset_register_kk * 2 + 1].y;
            res[14] += vec_A[offset_register_kk * 2 + 0].y * vec_B[offset_register_kk * 2 + 1].z;
            res[15] += vec_A[offset_register_kk * 2 + 0].y * vec_B[offset_register_kk * 2 + 1].w;

            res[16] += vec_A[offset_register_kk * 2 + 0].z * vec_B[offset_register_kk * 2 + 0].x;
            res[17] += vec_A[offset_register_kk * 2 + 0].z * vec_B[offset_register_kk * 2 + 0].y;
            res[18] += vec_A[offset_register_kk * 2 + 0].z * vec_B[offset_register_kk * 2 + 0].z;
            res[19] += vec_A[offset_register_kk * 2 + 0].z * vec_B[offset_register_kk * 2 + 0].w;

            res[20] += vec_A[offset_register_kk * 2 + 0].z * vec_B[offset_register_kk * 2 + 1].x;
            res[21] += vec_A[offset_register_kk * 2 + 0].z * vec_B[offset_register_kk * 2 + 1].y;
            res[22] += vec_A[offset_register_kk * 2 + 0].z * vec_B[offset_register_kk * 2 + 1].z;
            res[23] += vec_A[offset_register_kk * 2 + 0].z * vec_B[offset_register_kk * 2 + 1].w;

            res[24] += vec_A[offset_register_kk * 2 + 0].w * vec_B[offset_register_kk * 2 + 0].x;
            res[25] += vec_A[offset_register_kk * 2 + 0].w * vec_B[offset_register_kk * 2 + 0].y;
            res[26] += vec_A[offset_register_kk * 2 + 0].w * vec_B[offset_register_kk * 2 + 0].z;
            res[27] += vec_A[offset_register_kk * 2 + 0].w * vec_B[offset_register_kk * 2 + 0].w;

            res[28] += vec_A[offset_register_kk * 2 + 0].w * vec_B[offset_register_kk * 2 + 1].x;
            res[29] += vec_A[offset_register_kk * 2 + 0].w * vec_B[offset_register_kk * 2 + 1].y;
            res[30] += vec_A[offset_register_kk * 2 + 0].w * vec_B[offset_register_kk * 2 + 1].z;
            res[31] += vec_A[offset_register_kk * 2 + 0].w * vec_B[offset_register_kk * 2 + 1].w;

            res[32] += vec_A[offset_register_kk * 2 + 1].x * vec_B[offset_register_kk * 2 + 0].x;
            res[33] += vec_A[offset_register_kk * 2 + 1].x * vec_B[offset_register_kk * 2 + 0].y;
            res[34] += vec_A[offset_register_kk * 2 + 1].x * vec_B[offset_register_kk * 2 + 0].z;
            res[35] += vec_A[offset_register_kk * 2 + 1].x * vec_B[offset_register_kk * 2 + 0].w;

            res[36] += vec_A[offset_register_kk * 2 + 1].x * vec_B[offset_register_kk * 2 + 1].x;
            res[37] += vec_A[offset_register_kk * 2 + 1].x * vec_B[offset_register_kk * 2 + 1].y;
            res[38] += vec_A[offset_register_kk * 2 + 1].x * vec_B[offset_register_kk * 2 + 1].z;
            res[39] += vec_A[offset_register_kk * 2 + 1].x * vec_B[offset_register_kk * 2 + 1].w;

            res[40] += vec_A[offset_register_kk * 2 + 1].y * vec_B[offset_register_kk * 2 + 0].x;
            res[41] += vec_A[offset_register_kk * 2 + 1].y * vec_B[offset_register_kk * 2 + 0].y;
            res[42] += vec_A[offset_register_kk * 2 + 1].y * vec_B[offset_register_kk * 2 + 0].z;
            res[43] += vec_A[offset_register_kk * 2 + 1].y * vec_B[offset_register_kk * 2 + 0].w;

            res[44] += vec_A[offset_register_kk * 2 + 1].y * vec_B[offset_register_kk * 2 + 1].x;
            res[45] += vec_A[offset_register_kk * 2 + 1].y * vec_B[offset_register_kk * 2 + 1].y;
            res[46] += vec_A[offset_register_kk * 2 + 1].y * vec_B[offset_register_kk * 2 + 1].z;
            res[47] += vec_A[offset_register_kk * 2 + 1].y * vec_B[offset_register_kk * 2 + 1].w;

            res[48] += vec_A[offset_register_kk * 2 + 1].z * vec_B[offset_register_kk * 2 + 0].x;
            res[49] += vec_A[offset_register_kk * 2 + 1].z * vec_B[offset_register_kk * 2 + 0].y;
            res[50] += vec_A[offset_register_kk * 2 + 1].z * vec_B[offset_register_kk * 2 + 0].z;
            res[51] += vec_A[offset_register_kk * 2 + 1].z * vec_B[offset_register_kk * 2 + 0].w;

            res[52] += vec_A[offset_register_kk * 2 + 1].z * vec_B[offset_register_kk * 2 + 1].x;
            res[53] += vec_A[offset_register_kk * 2 + 1].z * vec_B[offset_register_kk * 2 + 1].y;
            res[54] += vec_A[offset_register_kk * 2 + 1].z * vec_B[offset_register_kk * 2 + 1].z;
            res[55] += vec_A[offset_register_kk * 2 + 1].z * vec_B[offset_register_kk * 2 + 1].w;

            res[56] += vec_A[offset_register_kk * 2 + 1].w * vec_B[offset_register_kk * 2 + 0].x;
            res[57] += vec_A[offset_register_kk * 2 + 1].w * vec_B[offset_register_kk * 2 + 0].y;
            res[58] += vec_A[offset_register_kk * 2 + 1].w * vec_B[offset_register_kk * 2 + 0].z;
            res[59] += vec_A[offset_register_kk * 2 + 1].w * vec_B[offset_register_kk * 2 + 0].w;

            res[60] += vec_A[offset_register_kk * 2 + 1].w * vec_B[offset_register_kk * 2 + 1].x;
            res[61] += vec_A[offset_register_kk * 2 + 1].w * vec_B[offset_register_kk * 2 + 1].y;
            res[62] += vec_A[offset_register_kk * 2 + 1].w * vec_B[offset_register_kk * 2 + 1].z;
            res[63] += vec_A[offset_register_kk * 2 + 1].w * vec_B[offset_register_kk * 2 + 1].w;
        }
        
        // update offset to store the prefetch vector
        offset_store_prefetch = (((int)(k / ks) + 1) & 1);
        
        // update the pointer to prefetched buffer A and prefetched buffer B
        buffer_A = (float*)(sAB) + buffer_A_offset + offset_store_prefetch * ms * ks;
        buffer_B = (float*)(sAB) + buffer_B_offset + offset_store_prefetch * ns * ks;
        
        
        // store the vectors in the prefetched buffer A and prefetched buffer B
        *(((float4*)buffer_A) + 2 * tx) = prefetch_vector_tile_A[0];
        *(((float4*)buffer_A) + 2 * tx + 1) = prefetch_vector_tile_A[1];
        *(((float4*)buffer_B) + 2 * tx) = prefetch_vector_tile_B[0];
        *(((float4*)buffer_B) + 2 * tx + 1) = prefetch_vector_tile_B[1];
        __syncthreads();
        // initial outer product column
        kk = -1;
        
        // offset of register store for prefetching
        offset_prefetch_register_kk = ((kk + 1) & 1);
        
        // offset of vec A and vec B w.r.t kk:
        offset_load_vec_A_kk = ((kk + 1) % ks) * ms;
        offset_load_vec_B_kk = ((kk + 1) % ks) * ns;
        
        // load the vectors from buffer to registers
        vec_A[offset_prefetch_register_kk * 2    ] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk);
        vec_A[offset_prefetch_register_kk * 2 + 1] = *(float4*)(buffer_A + offset_vec_A_warp + offset_vec_A_thread + offset_load_vec_A_kk + 4);
        vec_B[offset_prefetch_register_kk * 2    ] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk);
        vec_B[offset_prefetch_register_kk * 2 + 1] = *(float4*)(buffer_B + offset_vec_B_warp + offset_vec_B_thread + offset_load_vec_B_kk + 4);

    }
    
    C += bx * ms + offset_vec_A_warp + offset_vec_A_thread;
    C += (by * ns + offset_vec_B_warp + offset_vec_B_thread) * M;

    float4 C_res[16];
    C_res[ 0] = *((float4 *)(C + 0 + M * 0));
    C_res[ 1] = *((float4 *)(C + 4 + M * 0));
    C_res[ 2] = *((float4 *)(C + 0 + M * 1));
    C_res[ 3] = *((float4 *)(C + 4 + M * 1));
    C_res[ 4] = *((float4 *)(C + 0 + M * 2));
    C_res[ 5] = *((float4 *)(C + 4 + M * 2));
    C_res[ 6] = *((float4 *)(C + 0 + M * 3));
    C_res[ 7] = *((float4 *)(C + 4 + M * 3));
    C_res[ 8] = *((float4 *)(C + 0 + M * 4));
    C_res[ 9] = *((float4 *)(C + 4 + M * 4));
    C_res[10] = *((float4 *)(C + 0 + M * 5));
    C_res[11] = *((float4 *)(C + 4 + M * 5));
    C_res[12] = *((float4 *)(C + 0 + M * 6));
    C_res[13] = *((float4 *)(C + 4 + M * 6));
    C_res[14] = *((float4 *)(C + 0 + M * 7));
    C_res[15] = *((float4 *)(C + 4 + M * 7));
    

    C_res[0].x = res[0 ];
    C_res[0].y = res[8 ];
    C_res[0].z = res[16];
    C_res[0].w = res[24];

    C_res[1].x = res[32];
    C_res[1].y = res[40];
    C_res[1].z = res[48];
    C_res[1].w = res[56];

    C_res[2].x = res[1 ];
    C_res[2].y = res[9 ];
    C_res[2].z = res[17];
    C_res[2].w = res[25];

    C_res[3].x = res[33];
    C_res[3].y = res[41];
    C_res[3].z = res[49];
    C_res[3].w = res[57];

    C_res[4].x = res[2 ];
    C_res[4].y = res[10];
    C_res[4].z = res[18];
    C_res[4].w = res[26];

    C_res[5].x = res[34];
    C_res[5].y = res[42];
    C_res[5].z = res[50];
    C_res[5].w = res[58];

    C_res[6].x = res[3 ];
    C_res[6].y = res[11];
    C_res[6].z = res[19];
    C_res[6].w = res[27];

    C_res[7].x = res[35];
    C_res[7].y = res[43];
    C_res[7].z = res[51];
    C_res[7].w = res[59];

    C_res[8].x = res[4 ];
    C_res[8].y = res[12];
    C_res[8].z = res[20];
    C_res[8].w = res[28];

    C_res[9].x = res[36];
    C_res[9].y = res[44];
    C_res[9].z = res[52];
    C_res[9].w = res[60];

    C_res[10].x = res[5 ];
    C_res[10].y = res[13];
    C_res[10].z = res[21];
    C_res[10].w = res[29];

    C_res[11].x = res[37];
    C_res[11].y = res[45];
    C_res[11].z = res[53];
    C_res[11].w = res[61];

    C_res[12].x = res[6 ];
    C_res[12].y = res[14];
    C_res[12].z = res[22];
    C_res[12].w = res[30];

    C_res[13].x = res[38];
    C_res[13].y = res[46];
    C_res[13].z = res[54];
    C_res[13].w = res[62];

    C_res[14].x = res[7 ];
    C_res[14].y = res[15];
    C_res[14].z = res[23];
    C_res[14].w = res[31];

    C_res[15].x = res[39];
    C_res[15].y = res[47];
    C_res[15].z = res[55];
    C_res[15].w = res[63];

    *((float4 *)(C + 0 + M * 0)) = C_res[ 0];
    *((float4 *)(C + 4 + M * 0)) = C_res[ 1];
    *((float4 *)(C + 0 + M * 1)) = C_res[ 2];
    *((float4 *)(C + 4 + M * 1)) = C_res[ 3];
    *((float4 *)(C + 0 + M * 2)) = C_res[ 4];
    *((float4 *)(C + 4 + M * 2)) = C_res[ 5];
    *((float4 *)(C + 0 + M * 3)) = C_res[ 6];
    *((float4 *)(C + 4 + M * 3)) = C_res[ 7];
    *((float4 *)(C + 0 + M * 4)) = C_res[ 8];
    *((float4 *)(C + 4 + M * 4)) = C_res[ 9];
    *((float4 *)(C + 0 + M * 5)) = C_res[10];
    *((float4 *)(C + 4 + M * 5)) = C_res[11];
    *((float4 *)(C + 0 + M * 6)) = C_res[12];
    *((float4 *)(C + 4 + M * 6)) = C_res[13];
    *((float4 *)(C + 0 + M * 7)) = C_res[14];
    *((float4 *)(C + 4 + M * 7)) = C_res[15];
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
void Conv1D_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t CK = in->shape[0];
  size_t BS = in->shape[1];
  size_t os = in->shape[2];
  size_t OC = w->shape[1];

  dim3 blockDim(64);
  dim3 gridDim((OC + 32 - 1) / 32, (BS * os + 32 - 1) / 32);
  sgemm_medium<<<gridDim, blockDim>>>(OC, BS * os, CK, w->d_buf, in->d_buf, b->d_buf, out->d_buf);
  CHECK_CUDA(cudaDeviceSynchronize());
}

/* ReLU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
/* ReLU CUDA kernel */
__global__ void ReLUKernel(float *inout, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    inout[i] = inout[i] > 0 ? inout[i] : 0;
  }
}
/* ReLU using CUDA */
void ReLU_CUDA(Tensor *inout) {
  size_t N = inout->num_elem();

  dim3 blockDim(THREADS_PER_BLOCK);
  dim3 gridDim((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1);
  ReLUKernel<<<gridDim, blockDim>>>(inout->d_buf, N);
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
  for (size_t j = 1; j < s; j++) {
    float val = in[bs * C * s + j * C + c];
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

/* Concat
 * @param [in1] in1: [BS, N1]
 * @param [in2] in2: [BS, N2]
 * @param [in3] in3: [BS, N3]
 * @param [in4] in4: [BS, N4]
 * @param [out] out: [BS, N1 + N2 + N3 + N4]
 * 'N1', 'N2', 'N3', and 'N4' are the num of elems in the tensors.
 */
void Concat(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
            Tensor *out) {
  size_t BS = in1->shape[0];
  size_t N1 = in1->shape[1];
  size_t N2 = in2->shape[1];
  size_t N3 = in3->shape[1];
  size_t N4 = in4->shape[1];

  for (size_t bs = 0; bs < BS; bs++) {
    for (size_t i = 0; i < N1; i++) {
      out->buf[bs * (N1 + N2 + N3 + N4) + i] = in1->buf[bs * N1 + i];
    }
    for (size_t i = 0; i < N2; i++) {
      out->buf[bs * (N1 + N2 + N3 + N4) + N1 + i] = in2->buf[bs * N2 + i];
    }
    for (size_t i = 0; i < N3; i++) {
      out->buf[bs * (N1 + N2 + N3 + N4) + N1 + N2 + i] = 
        in3->buf[bs * N3 + i];
    }
    for (size_t i = 0; i < N4; i++) {
      out->buf[bs * (N1 + N2 + N3 + N4) + N1 + N2 + N3 + i] = 
        in4->buf[bs * N4 + i];
    }
  }
}

/* Concat CUDA kernel */
__global__ void ConcatKernel(float *in1, float *in2, float *in3, float *in4, float *out,
                            size_t BS, size_t N1, size_t N2, size_t N3, size_t N4) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_N = N1 + N2 + N3 + N4;
  
  for (size_t i = idx; i < BS * total_N; i += blockDim.x * gridDim.x) {
    size_t bs = i / total_N;
    size_t offset = i % total_N;
    
    if (offset < N1) {
        out[i] = in1[bs * N1 + offset];
    } else if (offset < N1 + N2) {
        out[i] = in2[bs * N2 + (offset - N1)];
    } else if (offset < N1 + N2 + N3) {
        out[i] = in3[bs * N3 + (offset - N1 - N2)];
    } else {
        out[i] = in4[bs * N4 + (offset - N1 - N2 - N3)];
    }
  }
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
  dim3 gridDim((BS * (N1 + N2 + N3 + N4) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1);
  ConcatKernel<<<gridDim, blockDim>>>(in1->d_buf, in2->d_buf, in3->d_buf, in4->d_buf, out->d_buf, BS, N1, N2, N3, N4);
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
void Linear_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
    size_t BS = in->shape[0];
    size_t N = in->shape[1];
    size_t M = w->shape[0];

    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridDim((BS + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    gemm_1<<<gridDim, blockDim>>>(in->d_buf, w->d_buf, b->d_buf, out->d_buf, BS, N, M);
    CHECK_CUDA(cudaDeviceSynchronize());
}

/* [Advanced Example] Linear in Half precision on CPU */
// void Linear_Half(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
//   size_t N = in->shape[0];
//   size_t M = w->shape[0];

//   for (size_t i = 0; i < M; i++) {
//     float val = 0.f;
//     for (size_t j = 0; j < N; j++) {
//       val += static_cast<float>(half_cpu(in->buf[j]) * 
//         half_cpu(w->buf[i * N + j]));
//     }
//     out->buf[i] = val + b->buf[i];
//   }
// }

/* Softmax (w/ Max Trick)
 * @param [in & out] inout: [BS, N]
 * 'N' is the number of elements in the tensor.
 * 'N' is fixed to 4 in this implementation (for 4 experts)
 */
void Softmax(Tensor *inout) {
  size_t BS = inout->shape[0];
  size_t N = inout->shape[1];

  for (size_t bs = 0; bs < BS; bs++) {
    float max_val = -INFINITY;
    for (size_t n = 0; n < N; n++) {
      max_val = inout->buf[bs * N + n] > max_val ? inout->buf[bs * N + n] : max_val;
    }

    float sum = 0.f;
    for (size_t n = 0; n < N; n++) {
      inout->buf[bs * N + n] = exp(inout->buf[bs * N + n] - max_val);
      sum += inout->buf[bs * N + n];
    }

    for (size_t n = 0; n < N; n++) { inout->buf[bs * N + n] /= sum; }
  }
}
/* Softmax CUDA kernel */
__global__ void SoftmaxKernel(float *inout, size_t BS, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= BS) return;

  size_t offset = idx * N;
  float max_val = inout[offset];
  for (size_t i = 1; i < N; i++) {
    float val = inout[offset + i];
    max_val = val > max_val ? val : max_val;
  }

  float sum = 0.f;
  for (size_t i = 0; i < N; i++) {
    float exp_val = exp(inout[offset + i] - max_val);
    inout[offset + i] = exp_val;
    sum += exp_val;
  }

  for (size_t i = 0; i < N; i++) { inout[offset + i] /= sum; }
}
/* Softmax using CUDA */
void Softmax_CUDA(Tensor *inout) {
  size_t BS = inout->shape[0];
  size_t N = inout->shape[1];

  // N is small, so we can parallelize over BS
  dim3 blockDim(THREADS_PER_BLOCK);
  dim3 gridDim((BS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1);
  SoftmaxKernel<<<gridDim, blockDim>>>(inout->d_buf, BS, N);
  CHECK_CUDA(cudaDeviceSynchronize());
}

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
/* Scaling CUDA kernel */
__global__ void ScalingKernel(float *inout, float *gate, size_t gate_col, size_t BS, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= BS * N) return;

  size_t bs = idx / N;
  inout[idx] *= gate[bs * 4 + gate_col];
}
/* Scaling using CUDA */
void Scaling_CUDA(Tensor *inout, Tensor *gate, size_t gate_col) {
  size_t BS = inout->shape[0];
  size_t N = inout->shape[1];

  dim3 blockDim(THREADS_PER_BLOCK);
  dim3 gridDim((BS * N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1);
  ScalingKernel<<<gridDim, blockDim>>>(inout->d_buf, gate->d_buf, gate_col, BS, N);
  CHECK_CUDA(cudaDeviceSynchronize());
}

/* (Elemwise) Addition
 * @param [in1] in1: [BS, N]
 * @param [in2] in2: [BS, N]
 * @param [in3] in3: [BS, N]
 * @param [in4] in4: [BS, N]
 * @param [out] out: [BS, N]
 * 'N' is the number of elements in the input tensor.
 */
void Add(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
         Tensor *out) {
  size_t BS = in1->shape[0];
  size_t N = in1->shape[1];

  for (size_t bs = 0; bs < BS; bs++) {
    for (size_t n = 0; n < N; n++) {
      out->buf[bs * N + n] = in1->buf[bs * N + n] + in2->buf[bs * N + n] + 
        in3->buf[bs * N + n] + in4->buf[bs * N + n];
    }
  }
}
/* Add CUDA kernel */
__global__ void AddKernel(float *in1, float *in2, float *in3, float *in4, float *out, size_t BS, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= BS * N) return;
  out[idx] = in1[idx] + in2[idx] + in3[idx] + in4[idx];
}
/* Add using CUDA */
void Add_CUDA(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
              Tensor *out) {
  size_t BS = in1->shape[0];
  size_t N = in1->shape[1];

  dim3 blockDim(THREADS_PER_BLOCK);
  dim3 gridDim((BS * N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1);
  AddKernel<<<gridDim, blockDim>>>(in1->d_buf, in2->d_buf, in3->d_buf, in4->d_buf, out->d_buf, BS, N);
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
void im2col_1d_CUDA(Tensor *in, Tensor *out, size_t K) {
  size_t BS = in->shape[0];
  size_t C = in->shape[1];
  size_t s = in->shape[2];
  size_t os = s - K + 1;

  dim3 blockDim(THREADS_PER_BLOCK);
  dim3 gridDim((BS * os * C * K + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1);
  im2col_1d_Kernel<<<gridDim, blockDim>>>(in->d_buf, out->d_buf, BS, C, s, K, os);
  CHECK_CUDA(cudaDeviceSynchronize());
}