// Supplemental probes. Build with the site's PPU compatibility compiler, not a host CUDA install.
// CAL_VECTOR=16/32/64 isolates optional arithmetic support from the memory probes.
#include "probe_common.cuh"
#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <map>
#include <cstdint>
#ifdef CAL_VECTOR
#if CAL_VECTOR == 16
#include <cuda_fp16.h>
using Value = unsigned;
static constexpr int packed = 2;
__device__ Value seed_value() { return 0x34003400u; }
__device__ Value calc(Value x, int op) {
    // Inline volatile instructions prevent constant folding of the independent inputs.
    const unsigned a = 0x38003800u, b = 0x34003400u;
    unsigned y;
    if (op == 0) asm volatile("add.rn.f16x2 %0,%1,%2;" : "=r"(y) : "r"(x), "r"(b));
    else if (op == 1) asm volatile("mul.rn.f16x2 %0,%1,%2;" : "=r"(y) : "r"(x), "r"(a));
    else asm volatile("fma.rn.f16x2 %0,%1,%2,%3;" : "=r"(y) : "r"(x), "r"(a), "r"(b));
    return y;
}
__device__ double to_double(Value x) {
    return double(__half2float(__ushort_as_half(x & 0xffffu))) +
           double(__half2float(__ushort_as_half(x >> 16)));
}
#else
#if CAL_VECTOR == 32
using Value = float;
#else
using Value = double;
#endif
static constexpr int packed = 1;
__device__ Value seed_value() { return Value(0.25); }
__device__ Value calc(Value x, int op) {
    Value y;
#if CAL_VECTOR == 32
    if (op == 0) asm volatile("add.rn.f32 %0,%1,%2;" : "=f"(y) : "f"(x), "f"(0.25f));
    else if (op == 1) asm volatile("mul.rn.f32 %0,%1,%2;" : "=f"(y) : "f"(x), "f"(0.5f));
    else asm volatile("fma.rn.f32 %0,%1,%2,%3;" : "=f"(y) : "f"(x), "f"(0.5f), "f"(0.25f));
#else
    if (op == 0) asm volatile("add.rn.f64 %0,%1,%2;" : "=d"(y) : "d"(x), "d"(0.25));
    else if (op == 1) asm volatile("mul.rn.f64 %0,%1,%2;" : "=d"(y) : "d"(x), "d"(0.5));
    else asm volatile("fma.rn.f64 %0,%1,%2,%3;" : "=d"(y) : "d"(x), "d"(0.5), "d"(0.25));
#endif
    return y;
}
__device__ double to_double(Value x) { return double(x); }
#endif

template<int Slots, bool Chain>
__global__ void vector_probe(double* out, int steps, int active, int op) {
    int t = threadIdx.x;
    if (t * packed >= active) return;
    Value values[Slots];
    #pragma unroll
    for (int j = 0; j < Slots; ++j) values[j] = seed_value();
    // Non-recurrent target inputs; each result is made observable to the compiler.
    // The inline asm sink is a compiler barrier, NOT proof that an SDK retained every opcode.
    #pragma unroll 1
    for (int i = 0; i < steps; ++i) {
        #pragma unroll
        for (int j = 0; j < Slots; ++j) {
            values[j] = calc(Chain ? values[j] : seed_value(), op);
#if CAL_VECTOR == 16
            asm volatile("" : : "r"(values[j]));
#elif CAL_VECTOR == 32
            asm volatile("" : : "f"(values[j]));
#else
            asm volatile("" : : "d"(values[j]));
#endif
        }
    }
    double sum = 0;
    #pragma unroll
    for (int j = 0; j < Slots; ++j) sum += to_double(values[j]);
    out[blockIdx.x * blockDim.x + t] = sum;
}
#endif

#ifdef CAL_MATRIX
template<int Slots, bool Chain>
__global__ void cube_probe(double* out, int steps) {
    float c[Slots][4];
    #pragma unroll
    for (int j = 0; j < Slots; ++j) for (int z = 0; z < 4; ++z) c[j][z] = 0.25f;
#if CAL_MATRIX == 8
    const unsigned one = 0x38383838u;
#elif CAL_MATRIX == 17
    const unsigned one = 0x3f803f80u;
#else
    const unsigned one = 0x3c003c00u;
#endif
    #pragma unroll 1
    for (int i = 0; i < steps; ++i) {
        #pragma unroll
        for (int j = 0; j < Slots; ++j) {
            float a0 = Chain ? c[j][0] : 0.25f, a1 = Chain ? c[j][1] : 0.25f;
            float a2 = Chain ? c[j][2] : 0.25f, a3 = Chain ? c[j][3] : 0.25f;
#if CAL_MATRIX == 8
#define MMA "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
#elif CAL_MATRIX == 17
#define MMA "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
#else
#define MMA "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
#endif
            asm volatile(MMA "{%0,%1,%2,%3},{%4,%4,%4,%4},{%4,%4},{%5,%6,%7,%8};"
                : "=f"(c[j][0]), "=f"(c[j][1]), "=f"(c[j][2]), "=f"(c[j][3])
                : "r"(one), "f"(a0), "f"(a1), "f"(a2), "f"(a3));
#undef MMA
        }
    }
    double sum = 0;
    #pragma unroll
    for (int j = 0; j < Slots; ++j) for (int z = 0; z < 4; ++z) sum += c[j][z];
    out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}
#endif

// Each warp owns a disjoint shared region. Readers never read a writer's region.
template<int Width>
__global__ void tsm_mix(unsigned* out, int steps, int read_warps) {
    extern __shared__ unsigned data[];
    volatile unsigned* s = data;
    int t = threadIdx.x;
    #pragma unroll
    for (int j = 0; j < Width; ++j) s[t + j * blockDim.x] = (t & 31) + 1;
    __syncthreads();
    unsigned sums[Width] = {};
    if (t / 32 < read_warps) {
        #pragma unroll 1
        for (int i = 0; i < steps; ++i) {
            #pragma unroll
            for (int j = 0; j < Width; ++j) sums[j] += s[t + j * blockDim.x];
        }
    } else {
        #pragma unroll 1
        for (int i = 0; i < steps; ++i) {
            #pragma unroll
            for (int j = 0; j < Width; ++j) s[t + j * blockDim.x] = unsigned(t + i + j);
        }
    }
    __syncthreads();
    unsigned sum = 0;
    #pragma unroll
    for (int j = 0; j < Width; ++j)
        sum += t / 32 < read_warps ? sums[j] : s[t + j * blockDim.x];
    out[blockIdx.x * blockDim.x + t] = sum;
}

__global__ void tsm_chain(unsigned* out, int steps, int words) {
    extern __shared__ unsigned data[];
    volatile unsigned* s = data;
    for (int i = threadIdx.x; i < words; i += blockDim.x) s[i] = (i + 17) & (words - 1);
    __syncthreads();
    unsigned x = threadIdx.x;
    #pragma unroll 1
    for (int i = 0; i < steps; ++i) x = s[x];
    out[threadIdx.x] = x;
}

__global__ void shuffle_chain(unsigned* out, int steps) {
    unsigned x = threadIdx.x;
    #pragma unroll 1
    for (int i = 0; i < steps; ++i)
        asm volatile("shfl.sync.bfly.b32 %0,%1,1,31,-1;" : "=r"(x) : "r"(x));
    out[threadIdx.x] = x;
}

__global__ void barrier_probe(unsigned* out, int steps) {
    #pragma unroll 1
    for (int i = 0; i < steps; ++i) __syncthreads();
    out[threadIdx.x] = threadIdx.x;
}

// Padded random ring, one index per 64-byte slot. All lanes use different starts.
// The benchmark measures effective dependent-load cost including addressing/loop overhead.
__global__ void global_chain(const unsigned* links, unsigned* out, int steps) {
    unsigned x = threadIdx.x;
    #pragma unroll 1
    for (int i = 0; i < steps; ++i) {
        const unsigned* p = links + size_t(x) * 16;
        asm volatile("ld.global.ca.u32 %0,[%1];" : "=r"(x) : "l"(p) : "memory");
    }
    out[threadIdx.x] = x;
}

struct Options {
    std::map<std::string, std::string> values;
    Options(int argc, char** argv) {
        for (int i = 2; i < argc; i += 2) {
            if (i + 1 >= argc || std::string(argv[i]).rfind("--", 0) != 0) std::exit(2);
            values[argv[i] + 2] = argv[i + 1];
        }
    }
    int get(const char* name, int fallback) const {
        auto p = values.find(name); return p == values.end() ? fallback : std::stoi(p->second);
    }
};

template<class Kernel, class Launch, class Validate>
void measure(Kernel kernel, Launch launch, Validate validate, bool profile, int shared_bytes) {
    cudaFuncAttributes attr{};
    CHECK(cudaFuncGetAttributes(&attr, kernel));
    std::fprintf(stderr, "regs_per_thread=%d local_bytes=%zu static_shared_bytes=%zu dynamic_shared_bytes=%d\n",
                 attr.numRegs, attr.localSizeBytes, attr.sharedSizeBytes, shared_bytes);
    launch(); CHECK(cudaDeviceSynchronize()); validate();
    if (profile) { capture_one(launch); return; }
    std::puts("sample,us_per_launch,regs_per_thread,local_bytes,static_shared_bytes,dynamic_shared_bytes");
    for (int sample = 0; sample < 7; ++sample)
        std::printf("%d,%.9f,%d,%zu,%zu,%d\n", sample, elapsed(launch, 8), attr.numRegs,
                    attr.localSizeBytes, attr.sharedSizeBytes, shared_bytes);
}

#ifdef CAL_VECTOR
template<int Slots, bool Chain>
void vector_case(double* out, int blocks, int threads, int steps, int active, int op, bool profile) {
    auto launch = [&]() { vector_probe<Slots, Chain><<<blocks, threads>>>(out, steps, active, op); };
    auto validate = [&]() {
        std::vector<double> host(size_t(blocks) * threads);
        CHECK(cudaMemcpy(host.data(), out, host.size() * sizeof(double), cudaMemcpyDeviceToHost));
        double expected = 0.25;
        for (int i = 0; i < (Chain ? steps : 1); ++i)
            expected = op == 0 ? expected + 0.25 : op == 1 ? expected * 0.5 : expected * 0.5 + 0.25;
        // Only FMA is offered for recurrent half arithmetic (bounded and finite).
        for (int b = 0; b < blocks; ++b) for (int t = 0; t * packed < active; ++t)
            if (!std::isfinite(host[b * threads + t]) ||
                std::abs(host[b * threads + t] - packed * Slots * expected) > packed * Slots * 0.001) {
                std::fprintf(stderr, "vector CPU reference failed\n"); std::exit(1);
            }
    };
    measure(vector_probe<Slots, Chain>, launch, validate, profile, 0);
}
#endif

#ifdef CAL_MATRIX
template<int Slots, bool Chain>
void cube_case(double* out, int blocks, int threads, int steps, bool profile) {
    auto launch = [&]() { cube_probe<Slots, Chain><<<blocks,threads>>>(out, steps); };
    auto validate = [&]() {
        std::vector<double> host(size_t(blocks) * threads);
        CHECK(cudaMemcpy(host.data(), out, host.size() * sizeof(double), cudaMemcpyDeviceToHost));
        double expected = Slots * 4.0 * (0.25 + (CAL_MATRIX == 8 ? 32 : 16) * (Chain ? steps : 1));
        for (double x : host) if (x != expected) {
            std::fprintf(stderr, "matrix CPU reference failed\n"); std::exit(1);
        }
    };
    measure(cube_probe<Slots, Chain>, launch, validate, profile, 0);
}
#endif

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    Options o(argc, argv);
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{}; CHECK(cudaGetDeviceProperties(&prop, 0));
    char pci[32]{}; CHECK(cudaDeviceGetPCIBusId(pci, sizeof(pci), 0));
    if (std::string(argv[1]) == "info") {
        std::printf("{\"cu_count\":%d,\"nominal_core_khz\":%d,\"pci\":\"%s\",\"warp_size\":%d}\n",
                    prop.multiProcessorCount, prop.clockRate, pci, prop.warpSize); return 0;
    }
    const std::string mode = argv[1];
    int blocks = o.get("blocks", 1), threads = o.get("threads", 256), steps = o.get("steps", 2048);
    int width = o.get("width", 1), readers = o.get("read-warps", threads / 32);
    int kib = o.get("workset-kib", 4), slots = o.get("slots", 1);
    bool profile = o.get("profile", 0) != 0;
    if (prop.warpSize != 32 || blocks < 1 || blocks > 65536 || threads < 32 || threads > 1024 ||
        threads % 32 || steps < 1 || steps > (1 << 24) || kib < 1 || kib > 262144 ||
        readers < 0 || readers > threads / 32) return 2;
    if (mode == "vector" || mode == "vector-chain") {
#ifdef CAL_VECTOR
        int active = o.get("active", threads * packed), op = o.get("op", 2);
        if (active < 1 || active > threads * packed || op < 0 || op > 2 ||
            (mode == "vector-chain" && op != 2)) return 2;
        double* out; CHECK(cudaMalloc(&out, size_t(blocks) * threads * sizeof(double)));
#define CALL(S) if (mode == "vector-chain") vector_case<S, true>(out, blocks, threads, steps, active, op, profile); \
                else vector_case<S, false>(out, blocks, threads, steps, active, op, profile)
        if (slots == 1) { CALL(1); } else if (slots == 4) { CALL(4); }
        else if (slots == 16) { CALL(16); } else { cudaFree(out); return 2; }
#undef CALL
        CHECK(cudaFree(out)); return 0;
#else
        return 2;
#endif
    }
    if (mode == "cube" || mode == "cube-chain") {
#ifdef CAL_MATRIX
        double* out; CHECK(cudaMalloc(&out, size_t(blocks) * threads * sizeof(double)));
#define CUBE(S) if (mode == "cube-chain") cube_case<S,true>(out,blocks,threads,steps,profile); \
                else cube_case<S,false>(out,blocks,threads,steps,profile)
        if (slots == 1) { CUBE(1); } else if (slots == 4) { CUBE(4); }
        else if (slots == 16) { CUBE(16); } else { cudaFree(out); return 2; }
#undef CUBE
        CHECK(cudaFree(out)); return 0;
#else
        return 2;
#endif
    }
    unsigned* out; CHECK(cudaMalloc(&out, size_t(blocks) * threads * sizeof(unsigned)));
    auto check = [&](auto expected) {
        std::vector<unsigned> host(size_t(blocks) * threads);
        CHECK(cudaMemcpy(host.data(), out, host.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
        for (int b = 0; b < blocks; ++b) for (int t = 0; t < threads; ++t)
            if (host[b * threads + t] != expected(t)) {
                std::fprintf(stderr, "%s CPU reference failed at %d/%d\n", mode.c_str(), b, t); std::exit(1);
            }
    };
    if (mode == "tsm-mix") {
        int bytes = threads * width * 4;
        auto valid = [&]() { check([&](int t) {
            if (t / 32 < readers) return unsigned(steps) * width * ((t & 31) + 1);
            unsigned sum = 0; for (int j = 0; j < width; ++j) sum += unsigned(t + steps - 1 + j); return sum;
        }); };
#define MIX(W) measure(tsm_mix<W>, [&]() { tsm_mix<W><<<blocks,threads,bytes>>>(out, steps, readers); }, valid, profile, bytes)
        if (width == 1) { MIX(1); } else if (width == 4) { MIX(4); } else if (width == 16) { MIX(16); }
        else { cudaFree(out); return 2; }
#undef MIX
    } else if (mode == "tsm-chain") {
        int words = kib * 256;
        if (blocks != 1 || threads != 32 || (words & (words - 1)) || kib > 48) return 2;
        measure(tsm_chain, [&]() { tsm_chain<<<1,32,kib*1024>>>(out,steps,words); },
                [&]() { check([&](int t) { return unsigned(t + 17LL * steps) & (words - 1); }); }, profile, kib * 1024);
    } else if (mode == "shuffle-chain") {
        if (blocks != 1 || threads != 32) return 2;
        measure(shuffle_chain, [&]() { shuffle_chain<<<1,32>>>(out,steps); },
                [&]() { check([&](int t) { return unsigned(t ^ (steps & 1)); }); }, profile, 0);
    } else if (mode == "barrier") {
        if (blocks != 1) return 2;
        measure(barrier_probe, [&]() { barrier_probe<<<1,threads>>>(out,steps); },
                [&]() { check([](int t) { return unsigned(t); }); }, profile, 0);
    } else if (mode == "global-chain") {
        if (blocks != 1 || threads != 32) return 2;
        int nodes = kib * 16;
        std::vector<unsigned> order(nodes), host(size_t(nodes) * 16, 0);
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 gen(7); std::shuffle(order.begin(), order.end(), gen);
        for (int i = 0; i < nodes; ++i) host[size_t(order[i]) * 16] = order[(i + 1) % nodes];
        unsigned* links; CHECK(cudaMalloc(&links, host.size() * sizeof(unsigned)));
        CHECK(cudaMemcpy(links, host.data(), host.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
        auto warm = [&]() { global_chain<<<1,32>>>(links,out,nodes); CHECK(cudaDeviceSynchronize()); };
        warm();
        auto launch = [&]() { global_chain<<<1,32>>>(links,out,steps); };
        measure(global_chain, launch, [&]() { check([&](int t) {
            unsigned x = t; for (int i = 0; i < steps; ++i) x = host[size_t(x) * 16]; return x;
        }); }, profile, 0);
        CHECK(cudaFree(links));
    } else { cudaFree(out); return 2; }
    CHECK(cudaFree(out));
    return 0;
}
