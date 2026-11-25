"""Compile, export and run a single TileLang kernel from one JIT artifact."""
import argparse
import hashlib
import json
import os
from pathlib import Path

CASES = ("gemm_aiu_on", "gemm_aiu_off", "gemm_multiwave", "vector")


def workload(case):
    try:
        import tilelang.ppu.language as T
    except ModuleNotFoundError as exc:
        if exc.name not in ("tilelang.ppu", "tilelang.ppu.language"):
            raise
        import sys
        import tilelang
        raise RuntimeError(
            f"paired requires the SAIL PPU dialect ({exc.name} missing). "
            f"{sys.executable} loaded tilelang {getattr(tilelang, '__version__', 'unknown')} "
            f"from {tilelang.__file__}. Check the container environment against the README."
        ) from exc

    if case == "vector":
        n = 10240

        @T.prim_func
        def vector_kernel(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"),
                          C: T.Tensor((n,), "float32")):
            with T.Kernel(n // 128, threads=128) as bx:
                i = bx * 128 + T.get_thread_binding()
                C[i] = (A[i] + B[i]) * (A[i] - B[i])

        return vector_kernel, [(n,), (n,), (n,)], "float32"

    m, n, k = (640, 512, 96) if case == "gemm_multiwave" else (128, 64, 96)

    @T.prim_func
    def gemm_kernel(A: T.Tensor((m, k), "bfloat16"), B: T.Tensor((k, n), "bfloat16"),
                    C: T.Tensor((m, n), "bfloat16")):
        with T.Kernel(n // 64, m // 64, threads=128) as (bx, by):
            a = T.alloc_shared((64, 32), "bfloat16")
            b = T.alloc_shared((32, 64), "bfloat16")
            c = T.alloc_fragment((64, 64), "float32")
            T.clear(c)
            for ki in T.Pipelined(k // 32, num_stages=2):
                T.copy(A[by * 64, ki * 32], a)
                T.copy(B[ki * 32, bx * 64], b)
                T.gemm(a, b, c)
            T.copy(c, C[by * 64, bx * 64])

    return gemm_kernel, [(m, k), (k, n), (m, n)], "bfloat16"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("out", type=Path)
    parser.add_argument("--profile-one", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["TILELANG_CACHE_DIR"] = str(args.out.resolve() / "jit-cache")
    os.environ["TILELANG_DISABLE_CACHE"] = "1"
    import tilelang
    import torch
    from tilelang import tvm

    torch.cuda.set_device(0)
    prim, shapes, dtype = workload(args.case)
    cfg = {"tl.disable_aiu_lower": args.case == "gemm_aiu_off", "tl.disable_ldmat_swzl": False}
    target = {"kind": "ppu", "arch": "ppu_15"}
    kernel = tilelang.compile(prim, target=target, execution_backend="tvm_ffi", pass_configs=cfg)
    artifact = kernel.artifact
    if artifact is None or artifact.device_mod is None:
        raise RuntimeError("JIT did not retain device_mod; check fork/version and disable the JIT cache")
    tir = tvm.ir.save_json(artifact.device_mod).encode()
    (args.out / "device.json").write_bytes(tir)
    (args.out / "device.tir").write_text(artifact.device_mod.script())
    (args.out / "kernel.ppu.cc").write_text(kernel.get_kernel_source())
    (args.out / "input.tir").write_text(prim.script())
    rng = torch.Generator().manual_seed(7)
    # Small integers give exactly representable products and deterministic CPU references.
    host = [torch.randint(-2, 3, s, generator=rng).float() for s in shapes[:2]]
    expected = ((host[0] + host[1]) * (host[0] - host[1]) if args.case == "vector"
                else host[0] @ host[1])
    dt = getattr(torch, dtype)
    a, b = [x.to(device="cuda", dtype=dt) for x in host]
    c = torch.full(shapes[2], float("nan"), device="cuda", dtype=dt)

    def launch():
        kernel(a, b, c)

    launch()
    torch.testing.assert_close(c.cpu(), expected.to(dt), rtol=0, atol=0)
    for _ in range(20):
        launch()
    torch.cuda.synchronize()
    if args.profile_one:
        api = torch.cuda.cudart()
        torch.cuda.check_error(api.cudaProfilerStart())
        launch()
        torch.cuda.synchronize()
        torch.cuda.check_error(api.cudaProfilerStop())
        return

    def measure(count):
        first = torch.cuda.Event(enable_timing=True)
        last = torch.cuda.Event(enable_timing=True)
        first.record()
        for _ in range(count):
            launch()
        last.record()
        last.synchronize()
        return 1000 * first.elapsed_time(last) / count

    hot = [measure(100) for _ in range(7)]
    # A large intervening write is an eviction attempt. ACU traffic determines actual residency.
    scrub = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    after_scrub = []
    for i in range(7):
        scrub.fill_(i)
        torch.cuda.synchronize()
        after_scrub.append(measure(1))
    record = dict(case=args.case, target=target, pass_configs=cfg, shapes=shapes, dtype=dtype,
                  torch=str(torch.__version__), tilelang=str(getattr(tilelang, "__version__", "unknown")),
                  device=torch.cuda.get_device_name(0), tir_sha256=hashlib.sha256(tir).hexdigest(),
                  source_sha256=hashlib.sha256((args.out / "kernel.ppu.cc").read_bytes()).hexdigest(),
                  check="exact_cpu_reference_passed", hot_us=hot, hot_launches_per_sample=100,
                  after_scrub_us=after_scrub, scrub_bytes=scrub.numel())
    (args.out / "hardware.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
