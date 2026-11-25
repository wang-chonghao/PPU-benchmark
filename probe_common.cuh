#pragma once
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); \
    std::exit(1); } } while (0)

struct Buffers {
    float *a, *b;
    explicit Buffers(size_t bytes) {
        CHECK(cudaMalloc(&a, bytes));
        CHECK(cudaMalloc(&b, bytes));
    }
    ~Buffers() { cudaFree(b); cudaFree(a); }
};

template<class Launch>
float elapsed(Launch launch, int repetitions) {
    cudaEvent_t first, last;
    CHECK(cudaEventCreate(&first));
    CHECK(cudaEventCreate(&last));
    CHECK(cudaEventRecord(first));
    for (int i = 0; i < repetitions; ++i) launch();
    CHECK(cudaEventRecord(last));
    CHECK(cudaEventSynchronize(last));
    CHECK(cudaGetLastError());
    float ms;
    CHECK(cudaEventElapsedTime(&ms, first, last));
    CHECK(cudaEventDestroy(first));
    CHECK(cudaEventDestroy(last));
    if (!(ms > 0)) { std::fprintf(stderr, "invalid event time\n"); std::exit(1); }
    return ms * 1000.0f / repetitions;
}

template<class Launch>
void capture_one(Launch launch) {
    launch();
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaProfilerStart());
    launch();
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaProfilerStop());
}
