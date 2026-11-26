// One binary and one kernel for all cache preferences and dynamic TSM sizes.
#include "probe_common.cuh"

constexpr int threads = 128;
constexpr int passes = 100;
constexpr int stride_words = 64 / sizeof(unsigned);
constexpr int max_workset_kib = 512;

__global__ void kvd_capacity_read(const unsigned *data, unsigned *out, int lines, int tsm_bytes) {
    extern __shared__ volatile unsigned char tsm[];
    if (tsm_bytes && threadIdx.x == 0) tsm[tsm_bytes - 1] = 1;
    unsigned sum = 0;
    for (int pass = 0; pass < passes; ++pass) {
        for (int line = threadIdx.x; line < lines; line += blockDim.x) {
            unsigned value;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(value)
                         : "l"(data + line * stride_words) : "memory");
            sum += value;
        }
        __syncthreads();
    }
    out[threadIdx.x] = sum;
}

int main(int argc, char **argv) {
    if (argc != 5) return 2;
    const bool prefer_shared = std::strcmp(argv[1], "prefer_shared") == 0;
    const int tsm_kib = std::atoi(argv[2]);
    const int workset_kib = std::atoi(argv[3]);
    if ((!prefer_shared && std::strcmp(argv[1], "default") != 0) ||
        tsm_kib < 0 || tsm_kib > 256 || workset_kib < 1 || workset_kib > max_workset_kib) return 2;
    CHECK(cudaSetDevice(0));
    int limit;
    CHECK(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    cudaFuncAttributes attr{};
    CHECK(cudaFuncGetAttributes(&attr, kvd_capacity_read));
    const int ceiling = limit - static_cast<int>(attr.sharedSizeBytes);
    if (tsm_kib * 1024 > ceiling) {
        std::fprintf(stderr, "TSM request exceeds available %d bytes\n", ceiling);
        return 1;
    }
    // Each case is a fresh process. Leave the default preference untouched.
    if (prefer_shared) CHECK(cudaFuncSetCacheConfig(kvd_capacity_read, cudaFuncCachePreferShared));
    CHECK(cudaFuncSetAttribute(kvd_capacity_read, cudaFuncAttributeMaxDynamicSharedMemorySize, ceiling));

    // Keep allocation size, alignment within the allocation and input identical across the sweep.
    std::vector<unsigned> input(max_workset_kib * 1024 / sizeof(unsigned));
    for (size_t i = 0; i < input.size(); ++i) input[i] = unsigned(i * 2654435761u + 17u);
    unsigned *data, *out;
    CHECK(cudaMalloc(&data, input.size() * sizeof(unsigned)));
    CHECK(cudaMalloc(&out, threads * sizeof(unsigned)));
    CHECK(cudaMemcpy(data, input.data(), input.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
    const int lines = workset_kib * 1024 / 64;
    CHECK(cudaProfilerStart());
    kvd_capacity_read<<<1, threads, tsm_kib * 1024>>>(data, out, lines, tsm_kib * 1024);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaProfilerStop());

    unsigned actual[threads];
    CHECK(cudaMemcpy(actual, out, sizeof(actual), cudaMemcpyDeviceToHost));
    for (int tid = 0; tid < threads; ++tid) {
        unsigned expected = 0;
        for (int line = tid; line < lines; line += threads) expected += input[line * stride_words];
        if (actual[tid] != expected * passes) {
            std::fprintf(stderr, "checksum mismatch for thread %d\n", tid);
            return 1;
        }
    }
    FILE *check = std::fopen(argv[4], "w");
    if (!check) return 1;
    std::fprintf(check, "{\"checksum\":\"passed\",\"dynamic_limit_bytes\":%d,"
                        "\"data_address\":\"%p\",\"registers_per_thread\":%d}\n",
                        ceiling, static_cast<void *>(data), attr.numRegs);
    if (std::fclose(check) != 0) return 1;
    CHECK(cudaFree(out));
    CHECK(cudaFree(data));
    return 0;
}
