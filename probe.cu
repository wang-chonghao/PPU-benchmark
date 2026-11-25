// Single-device probes for the PPU SDK's CUDA-compatible runtime.
#include "probe_common.cuh"

template<int Mode>
__global__ void stream_kernel(const float *a, float *b, int count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    for (int i = tid; i < count; i += gridDim.x * blockDim.x) {
        if (Mode == 0) sum += a[i];
        if (Mode == 1) b[i] = float(i & 15);
        if (Mode == 2) b[i] = a[i];
    }
    if (Mode == 0 && tid < count) b[tid] = sum;
}

template<int Mode>
void stream_case(Buffers &buf, int mib, int cus, bool profile = false) {
    int count = mib * 262144;
    int blocks = (count + 255) / 256;
    if (blocks > cus * 32) blocks = cus * 32;
    auto launch = [&]() { stream_kernel<Mode><<<blocks, 256>>>(buf.a, buf.b, count); };
    // The repeating 0..15 input gives an exactly representable reference sum.
    stream_kernel<1><<<blocks, 256>>>(buf.b, buf.a, count);
    launch();
    CHECK(cudaDeviceSynchronize());
    if (profile) {
        CHECK(cudaProfilerStart());
        launch();
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaProfilerStop());
        return;
    }
    for (int sample = 0; sample < 5; ++sample) {
        const int reps = mib < 16 ? 64 : 8;
        float us = elapsed(launch, reps);
        long long read = Mode == 1 ? 0 : 4LL * count;
        long long written = Mode == 0 ? 4LL * blocks * 256 : 4LL * count;
        std::printf("stream,%s,%d,%d,%d,%d,%lld,%lld,%.6f\n",
            Mode == 0 ? "read" : Mode == 1 ? "write" : "copy",
            mib, blocks, sample, reps, read, written, us);
    }
    CHECK(cudaMemset(buf.b, 0xff, size_t(count) * sizeof(float)));
    launch();
    CHECK(cudaGetLastError());
    for (int idx : {1, blocks * 256 - 1}) {
        float actual;
        CHECK(cudaMemcpy(&actual, buf.b + idx, sizeof(float), cudaMemcpyDeviceToHost));
        const int visits = (count - 1 - idx) / (blocks * 256) + 1;
        float expected = float(idx & 15) * (Mode == 0 ? visits : 1);
        if (actual != expected) { std::fprintf(stderr, "stream check failed\n"); std::exit(1); }
    }
}

// Keep independent live values across a runtime-length recurrence.
// Use the compiler's actual numRegs and localSizeBytes to interpret each variant.
template<int Op, int Chains>
__global__ void math_kernel(float *output, int steps, float scale) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float v[Chains];
    #pragma unroll
    for (int j = 0; j < Chains; ++j) v[j] = 0.125f + (tid % 127 + j) * 0.0001f;
    for (int i = 0; i < steps; ++i) {
        #pragma unroll
        for (int j = 0; j < Chains; ++j) {
            if (Op == 0) v[j] = fmaf(v[j], scale, 0.0001f);
            if (Op == 1) v[j] = exp2f(-v[j]);
            if (Op == 2) v[j] = scale / (1.0f + v[j]);
        }
    }
    float sum = 0;
    #pragma unroll
    for (int j = 0; j < Chains; ++j) sum += v[j];
    output[tid] = sum;
}

template<int Op, int Chains>
void math_case(float *out, int blocks, int threads) {
    for (int steps : {32, 128, 512}) {
        auto launch = [&]() { math_kernel<Op, Chains><<<blocks, threads>>>(out, steps, 0.999f); };
        launch();
        CHECK(cudaDeviceSynchronize());
        for (int sample = 0; sample < 5; ++sample) {
            float us = elapsed(launch, 8);
            std::printf("math,%s,%d,%d,%d,%d,%d,%d,%.6f\n",
                Op == 0 ? "fma" : Op == 1 ? "exp2_neg" : "div_add",
                Chains, blocks, threads, steps, sample, 8, us);
        }
        float value;
        CHECK(cudaMemcpy(&value, out, sizeof(float), cudaMemcpyDeviceToHost));
        if (!std::isfinite(value) || value <= 0) { std::fprintf(stderr, "math check failed\n"); std::exit(1); }
    }
}

template<int Chains>
void occupancy_case() {
    cudaFuncAttributes attr{};
    CHECK(cudaFuncGetAttributes(&attr, math_kernel<0, Chains>));
    for (int threads : {64, 128, 256, 512}) {
        for (int dynamic : {0, 8192, 32768}) {
            int blocks;
            CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, math_kernel<0, Chains>, threads, dynamic));
            std::printf("occupancy,%d,%d,%d,%d,%zu,%zu,%d\n", Chains, threads, dynamic,
                attr.numRegs, attr.localSizeBytes, attr.sharedSizeBytes, blocks);
        }
    }
}

__global__ void shared_read(float *out, int stride, int steps) {
    __shared__ volatile float words[1024];
    int lane = threadIdx.x % 32;
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) words[i] = float(i & 15);
    __syncthreads();
    float sum = 0;
    for (int r = 0; r < steps; ++r) sum += words[(lane * stride + r) & 1023];
    out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

#ifdef PPU_PROBE_MMA
// SDK compatibility ISA. Inspect PASM before converting fragment rates to TC02 II.
template<bool Fp8, int Chains>
__global__ void mma_kernel(float *out, int steps, unsigned packed_one) {
    float c[Chains][4] = {};
    for (int i = 0; i < steps; ++i) {
        #pragma unroll
        for (int j = 0; j < Chains; ++j) {
            if (Fp8) {
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                             "{%0,%1,%2,%3},{%4,%4,%4,%4},{%4,%4},{%0,%1,%2,%3};"
                             : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                             : "r"(packed_one));
            } else {
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                             "{%0,%1,%2,%3},{%4,%4,%4,%4},{%4,%4},{%0,%1,%2,%3};"
                             : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                             : "r"(packed_one));
            }
        }
    }
    float sum = 0;
    #pragma unroll
    for (int j = 0; j < Chains; ++j) sum += c[j][0];
    out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

template<bool Fp8, int Chains>
void mma_case(float *out, int blocks, int threads) {
    for (int steps : {32, 128, 512}) {
        auto launch = [&]() {
            mma_kernel<Fp8, Chains><<<blocks, threads>>>(out, steps, Fp8 ? 0x38383838u : 0x3f803f80u);
        };
        launch();
        CHECK(cudaDeviceSynchronize());
        for (int sample = 0; sample < 5; ++sample) {
            float us = elapsed(launch, 8);
            std::printf("mma,%s,%d,%d,%d,%d,%d,%d,%.6f\n", Fp8 ? "fp8" : "bf16",
                        Chains, blocks, threads, steps, sample, 8, us);
        }
        float value;
        CHECK(cudaMemcpy(&value, out, sizeof(float), cudaMemcpyDeviceToHost));
        if (value != float((Fp8 ? 32 : 16) * steps * Chains)) {
            std::fprintf(stderr, "mma check failed: %g\n", value); std::exit(1);
        }
    }
}
#endif

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    CHECK(cudaSetDevice(0));
    cudaDeviceProp p{};
    CHECK(cudaGetDeviceProperties(&p, 0));
    char pci[32]{};
    CHECK(cudaDeviceGetPCIBusId(pci, sizeof(pci), 0));
    std::fprintf(stderr, "device=%s pci=%s CU=%d clock_kHz=%d\n", p.name, pci, p.multiProcessorCount, p.clockRate);
    if (!std::strcmp(argv[1], "device")) {
        std::printf("name,pci,cu,core_clock_khz,memory_clock_khz,hbm_bytes,warp_threads,max_threads_per_cu,"
                    "smem_bytes_per_cu,regs_per_cu_runtime_units,regs_per_block_runtime_units,l2_bytes_runtime\n");
        std::printf("%s,%s,%d,%d,%d,%zu,%d,%d,%zu,%d,%d,%d\n", p.name, pci, p.multiProcessorCount,
                    p.clockRate, p.memoryClockRate, p.totalGlobalMem, p.warpSize, p.maxThreadsPerMultiProcessor,
                    p.sharedMemPerMultiprocessor, p.regsPerMultiprocessor, p.regsPerBlock, p.l2CacheSize);
    } else if (!std::strcmp(argv[1], "occupancy")) {
        std::puts("kind,chains,threads,dynamic_smem_bytes,regs_per_thread,local_bytes_per_thread,static_smem_bytes,api_blocks_per_cu");
        occupancy_case<1>(); occupancy_case<8>(); occupancy_case<32>(); occupancy_case<64>();
    } else if (!std::strcmp(argv[1], "stream")) {
        const int max_mib = std::atoi(argv[2]);
        if (max_mib < 1 || max_mib > 2048) return 2;
        Buffers buf(size_t(max_mib) << 20);
        std::puts("kind,operation,mib_per_array,blocks,sample,repetitions,logical_read_bytes,logical_write_bytes,us_per_launch");
        for (int mib : {1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024, 2048}) {
            if (mib > max_mib) continue;
            stream_case<0>(buf, mib, p.multiProcessorCount);
            stream_case<1>(buf, mib, p.multiProcessorCount);
            stream_case<2>(buf, mib, p.multiProcessorCount);
        }
    } else if (!std::strcmp(argv[1], "profile-read") || !std::strcmp(argv[1], "profile-copy")) {
        const int mib = std::atoi(argv[2]);
        if (mib < 1 || mib > 2048) return 2;
        Buffers buf(size_t(mib) << 20);
        if (!std::strcmp(argv[1], "profile-read")) stream_case<0>(buf, mib, p.multiProcessorCount, true);
        else stream_case<2>(buf, mib, p.multiProcessorCount, true);
    } else if (!std::strcmp(argv[1], "profile-reg8") || !std::strcmp(argv[1], "profile-reg64")) {
        int blocks = p.multiProcessorCount * 16;
        Buffers buf(size_t(blocks) * 256 * sizeof(float));
        if (!std::strcmp(argv[1], "profile-reg8"))
            capture_one([&]() { math_kernel<0,8><<<blocks, 256>>>(buf.a, 128, 0.999f); });
        else
            capture_one([&]() { math_kernel<0,64><<<blocks, 256>>>(buf.a, 128, 0.999f); });
    } else if (!std::strcmp(argv[1], "shared")) {
        int blocks = p.multiProcessorCount * 8;
        Buffers buf(size_t(blocks) * 256 * sizeof(float));
        std::puts("kind,stride_words,blocks,threads,steps,sample,repetitions,us_per_launch");
        for (int stride : {0, 1, 2, 4, 8, 16, 32}) {
            for (int steps : {128, 512}) {
                auto launch = [&]() { shared_read<<<blocks, 256>>>(buf.a, stride, steps); };
                launch();
                CHECK(cudaDeviceSynchronize());
                for (int sample = 0; sample < 5; ++sample)
                    std::printf("shared,%d,%d,256,%d,%d,8,%.6f\n", stride, blocks, steps, sample, elapsed(launch, 8));
                float value;
                CHECK(cudaMemcpy(&value, buf.a + 1, sizeof(float), cudaMemcpyDeviceToHost));
                if (value != float(steps / 16 * 120)) { std::fprintf(stderr, "TSM check failed\n"); return 1; }
            }
        }
    } else if (!std::strcmp(argv[1], "math")) {
        int blocks = p.multiProcessorCount * 8;
        Buffers buf(size_t(blocks) * 256 * sizeof(float));
        std::puts("kind,operation,chains,blocks,threads,steps,sample,repetitions,us_per_launch");
        math_case<0,1>(buf.a, 1, 32); math_case<1,1>(buf.a, 1, 32); math_case<2,1>(buf.a, 1, 32);
        math_case<0,4>(buf.a, blocks, 256); math_case<1,4>(buf.a, blocks, 256); math_case<2,4>(buf.a, blocks, 256);
#ifdef PPU_PROBE_MMA
    } else if (!std::strcmp(argv[1], "profile-bf16") || !std::strcmp(argv[1], "profile-fp8")) {
        int blocks = p.multiProcessorCount * 8;
        Buffers buf(size_t(blocks) * 256 * sizeof(float));
        if (!std::strcmp(argv[1], "profile-bf16"))
            capture_one([&]() { mma_kernel<false,4><<<blocks, 256>>>(buf.a, 128, 0x3f803f80u); });
        else
            capture_one([&]() { mma_kernel<true,4><<<blocks, 256>>>(buf.a, 128, 0x38383838u); });
    } else if (!std::strcmp(argv[1], "mma")) {
        int blocks = p.multiProcessorCount * 8;
        Buffers buf(size_t(blocks) * 256 * sizeof(float));
        std::puts("kind,operation,chains,blocks,threads,steps,sample,repetitions,us_per_launch");
        mma_case<false,1>(buf.a, 1, 32); mma_case<true,1>(buf.a, 1, 32);
        mma_case<false,4>(buf.a, blocks, 256); mma_case<true,4>(buf.a, blocks, 256);
#endif
    } else return 2;
}
