#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuda_fp16.h>

// 块内规约辅助函数
__device__ __forceinline__ float blockReduceMax(float val) {
    static __shared__ float shared[32]; 
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // 1. Warp 内规约
    for (int offset = 16; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // 2. 将各 Warp 的结果再次规约
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -1e20f;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2)
            val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void fused_logp_forward_kernel(
    const float* __restrict__ logits,      // [TotalTokens, VocabSize]
    const int64_t* __restrict__ token_ids, // [TotalTokens]
    float* __restrict__ output,            // [TotalTokens]
    int vocab_size) {

    int row = blockIdx.x;
    const float* row_logits = logits + row * vocab_size;

    // Step 1: Find Max
    float local_max = -1e20f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        local_max = max(local_max, row_logits[i]);
    }
    float max_val = blockReduceMax(local_max);
    __shared__ float res_max;
    if (threadIdx.x == 0) res_max = max_val;
    __syncthreads();

    // Step 2: Sum Exp
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        local_sum += expf(row_logits[i] - res_max);
    }
    float sum_val = blockReduceSum(local_sum);
    __shared__ float res_sum;
    if (threadIdx.x == 0) res_sum = sum_val;
    __syncthreads();

    // Step 3: Final Logprob
    if (threadIdx.x == 0) {
        int64_t target_id = token_ids[row];
        // 加上越界检查
        if (target_id >= 0 && target_id < vocab_size) {
            float target_logit = row_logits[target_id];
            output[row] = target_logit - res_max - logf(res_sum);
        } else {
            output[row] = 0.0f; 
        }
    }
}

torch::Tensor fused_logp_forward(torch::Tensor logits, torch::Tensor token_ids) {
    // 确保输入是连续的，CUDA 算子最怕非连续内存
    auto logits_contig = logits.contiguous();
    auto token_ids_contig = token_ids.contiguous();
    
    int64_t total_tokens = logits.size(0);
    int64_t vocab_size = logits.size(1);
    auto output = torch::empty({total_tokens}, logits.options());

    const int threads = 256;
    fused_logp_forward_kernel<<<total_tokens, threads>>>(
        logits_contig.data_ptr<float>(),
        token_ids_contig.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        vocab_size);

    return output;
}