// Independent work for throughput measurements on the SDK compatibility ISA.
#include "probe_common.cuh"

enum class Op { fma, ex2, div, bf16, fp8 };

// Each input is independent of every previously computed result.
__host__ __device__ unsigned input_bits(int tid, int index) {
    return 0x3f800000u | (unsigned(tid & 511) << 13) | unsigned(index);
}

template<Op op>
__device__ float calculate(float x, float scale, unsigned packed_one) {
    float y;
    if constexpr (op == Op::fma)
        asm volatile("fma.rn.f32 %0,%1,%2,%2;" : "=f"(y) : "f"(x), "f"(scale));
    if constexpr (op == Op::ex2)
        asm volatile("ex2.approx.ftz.f32 %0,%1;" : "=f"(y) : "f"(x));
    if constexpr (op == Op::div)
        asm volatile("div.rn.f32 %0,%1,%2;" : "=f"(y) : "f"(scale), "f"(x));
#ifdef PPU_PROBE_MMA
    if constexpr (op == Op::bf16 || op == Op::fp8) {
        // C comes from the independent input, never from a previous D.
        float d1, d2, d3;
        if constexpr (op == Op::bf16)
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                         "{%0,%1,%2,%3},{%4,%4,%4,%4},{%4,%4},{%5,%5,%5,%5};"
                         : "=f"(y),"=f"(d1),"=f"(d2),"=f"(d3) : "r"(packed_one),"f"(x));
        else
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                         "{%0,%1,%2,%3},{%4,%4,%4,%4},{%4,%4},{%5,%5,%5,%5};"
                         : "=f"(y),"=f"(d1),"=f"(d2),"=f"(d3) : "r"(packed_one),"f"(x));
    }
#endif
    return y;
}

template<Op op, int Slots>
__global__ void independent(unsigned *out, int operations, float scale, unsigned packed_one) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned checks[Slots] = {};
    #pragma unroll 1
    for (int base = 0; base < operations; base += Slots) {
        #pragma unroll
        for (int j = 0; j < Slots; ++j) {
            float x = __uint_as_float(input_bits(tid, base + j));
            float y = calculate<op>(x, scale, packed_one);
            // Integer sinks keep every result observable without feeding the target FU's inputs.
            checks[j] += __float_as_uint(y) >> 16;
        }
    }
    unsigned sum = 0;
    #pragma unroll
    for (int j = 0; j < Slots; ++j) sum += checks[j];
    out[tid] = sum;
}

template<Op op, int Slots>
void register_case(unsigned *out, const char *name, int blocks, int operations, bool profile) {
    constexpr int threads = 256;
    unsigned packed = op == Op::fp8 ? 0x38383838u : 0x3f803f80u;
    auto launch = [&]() { independent<op, Slots><<<blocks,threads>>>(out, operations, 0.875f, packed); };
    if (profile) { capture_one(launch); return; }
    launch();
    CHECK(cudaDeviceSynchronize());
    constexpr bool matrix = op == Op::bf16 || op == Op::fp8;
    long long count = 1LL * blocks * threads * operations / (matrix ? 32 : 1);
    int flops = matrix ? 2 * 16 * 8 * (op == Op::fp8 ? 32 : 16) : op == Op::fma ? 2 : 0;
    for (int sample = 0; sample < 5; ++sample)
        std::printf("register,%s,%d,256,%d,%d,0,%lld,%d,%d,8,%.6f\n",
                    name,blocks,Slots,operations,count,flops,sample,elapsed(launch,8));
    for (int tid : {0, 127, blocks * threads - 1}) {
        unsigned actual, expected = 0;
        CHECK(cudaMemcpy(&actual, out + tid, sizeof(actual), cudaMemcpyDeviceToHost));
        for (int i = 0; i < operations; ++i) {
            unsigned bits = input_bits(tid, i);
            float x, y;
            std::memcpy(&x, &bits, sizeof(x));
            if constexpr (op == Op::fp8 || op == Op::bf16) y = (op == Op::fp8 ? 32.f : 16.f) + x;
            if constexpr (op == Op::fma) y = std::fma(x, 0.875f, 0.875f);
            if constexpr (op == Op::div) y = 0.875f / x;
            if constexpr (op == Op::ex2) y = std::exp2(x);
            std::memcpy(&bits, &y, sizeof(bits));
            expected += bits >> 16;
        }
        // Approximate EX2 may cross a truncated-mantissa boundary.
        unsigned tolerance = op == Op::ex2 ? unsigned(operations) : 0;
        if ((actual > expected ? actual - expected : expected - actual) > tolerance) {
            std::fprintf(stderr, "%s checksum failed: %u vs %u\n", name, actual, expected); std::exit(1);
        }
    }
}

template<Op op>
void sweep(unsigned *out, const char *name, int cus, bool profile) {
    if (profile) { register_case<op,16>(out,name,cus*8,512,true); return; }
    for (int blocks_per_cu : {1, 8, 16}) {
        for (int operations : {128, 512}) {
            register_case<op,4>(out,name,cus*blocks_per_cu,operations,false);
            register_case<op,16>(out,name,cus*blocks_per_cu,operations,false);
        }
    }
}

__global__ void init_array(float *a, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        a[i] = float(i & 255) / 256;
}

template<bool Neg>
__global__ void exp2_array(const float *a, float *b, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        b[i] = exp2f(Neg ? -a[i] : a[i]);
}

template<bool Neg>
void array_case(Buffers &buf, int mib, int blocks, bool profile) {
    int n = mib * 262144;
    init_array<<<blocks,256>>>(buf.a,n);
    auto launch = [&]() { exp2_array<Neg><<<blocks,256>>>(buf.a,buf.b,n); };
    if (profile) { capture_one(launch); return; }
    launch();
    CHECK(cudaDeviceSynchronize());
    for (int sample = 0; sample < 5; ++sample)
        std::printf("array,%s,%d,256,0,0,%d,%d,0,%d,8,%.6f\n",
                    Neg ? "exp2f_neg" : "exp2f",blocks,mib,n,sample,elapsed(launch,8));
    for (int idx : {0, 123, n - 1}) {
        float actual;
        CHECK(cudaMemcpy(&actual, buf.b + idx, sizeof(float), cudaMemcpyDeviceToHost));
        float x = float(idx & 255) / 256;
        if (!(std::abs(actual - std::exp2(Neg ? -x : x)) < 1e-5f)) {
            std::fprintf(stderr, "exp2 array check failed\n"); std::exit(1);
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    CHECK(cudaSetDevice(0));
    cudaDeviceProp p{};
    CHECK(cudaGetDeviceProperties(&p, 0));
    int cus = p.multiProcessorCount;
    char pci[32]{};
    CHECK(cudaDeviceGetPCIBusId(pci, sizeof(pci), 0));
    std::fprintf(stderr, "device=%s pci=%s CU=%d nominal_core_kHz=%d\n", p.name, pci, cus, p.clockRate);
    int max_mib = std::atoi(argv[2]);
    if (max_mib < 1 || max_mib > 2048) return 2;
    const bool profile = std::strncmp(argv[1], "profile-", 8) == 0;
    const char *name = argv[1] + (profile ? 8 : 0);
    std::puts("kind,operation,blocks,threads,slots,ops_per_thread,mib_per_array,target_ops_per_launch,flop_per_op,sample,repetitions,us_per_launch");
    if (std::strcmp(name,"array") == 0) {
        Buffers buf(size_t(max_mib) << 20);
        if (profile) array_case<true>(buf,max_mib,cus*16,true);
        else for (int mib : {1, 16, 32, 128, 512, 1024, 2048}) {
            if (mib > max_mib) continue;
            array_case<false>(buf,mib,cus*16,false);
            array_case<true>(buf,mib,cus*16,false);
        }
        return 0;
    }
    unsigned *out;
    CHECK(cudaMalloc(&out,size_t(cus)*16*256*sizeof(unsigned)));
    if (std::strcmp(name,"fma") == 0) sweep<Op::fma>(out,name,cus,profile);
    else if (std::strcmp(name,"ex2") == 0) sweep<Op::ex2>(out,name,cus,profile);
    else if (std::strcmp(name,"div") == 0) sweep<Op::div>(out,name,cus,profile);
#ifdef PPU_PROBE_MMA
    else if (std::strcmp(name,"bf16") == 0) sweep<Op::bf16>(out,name,cus,profile);
    else if (std::strcmp(name,"fp8") == 0) sweep<Op::fp8>(out,name,cus,profile);
#endif
    else { cudaFree(out); return 2; }
    CHECK(cudaFree(out));
    return 0;
}
