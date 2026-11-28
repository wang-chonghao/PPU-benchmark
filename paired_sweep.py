"""One shape of a same-source TileLang edge/FU test. All artifacts stay local."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def definition(args):
    # No fallback to generic CUDA TileLang: that changes the compilation path.
    import tilelang.ppu.language as T
    dtype = args.dtype
    if args.family == "cube":
        m, n, k = args.m, args.n, args.k
        bm, bn, bk = args.bm, args.bn, args.bk

        @T.prim_func
        def kernel(A: T.Tensor((m, k), dtype), B: T.Tensor((k, n), dtype),
                   C: T.Tensor((m, n), "float32")):
            with T.Kernel((n + bn - 1) // bn, (m + bm - 1) // bm, threads=128) as (bx, by):
                a = T.alloc_shared((bm, bk), dtype)
                b = T.alloc_shared((bk, bn), dtype)
                c = T.alloc_fragment((bm, bn), "float32")
                T.clear(c)
                for ki in T.Pipelined((k + bk - 1) // bk, num_stages=2):
                    # The compiler must implement these boundary copies correctly.
                    # CPU checking rejects missing padding / out-of-range results.
                    T.copy(A[by * bm, ki * bk], a)
                    T.copy(B[ki * bk, bx * bn], b)
                    T.gemm(a, b, c)
                T.copy(c, C[by * bm, bx * bn])

        return kernel, [(m, k), (k, n)], (m, n), "float32"
    rows, valid, tile, op = args.rows, args.valid, args.tile, args.op

    @T.prim_func
    def kernel(A: T.Tensor((rows, valid), dtype), B: T.Tensor((rows, valid), dtype),
               C: T.Tensor((rows, valid), dtype)):
        with T.Kernel(rows, threads=128) as bx:
            for j in T.Parallel(tile):
                if j < valid:
                    if op == "add":
                        C[bx, j] = A[bx, j] + B[bx, j]
                    elif op == "exp2":
                        C[bx, j] = T.exp2(A[bx, j])
                    else:
                        C[bx, j] = (A[bx, j] + B[bx, j]) * T.exp2(A[bx, j])

    return kernel, [(rows, valid), (rows, valid)], (rows, valid), dtype


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("family", choices=("vector", "cube"))
    p.add_argument("out", type=Path)
    p.add_argument("--dtype", choices=("float16", "bfloat16", "float32", "float64"), default="float16")
    p.add_argument("--op", choices=("add", "exp2", "forkjoin"), default="add")
    for name, default in (("m", 128), ("n", 128), ("k", 128), ("bm", 64), ("bn", 64), ("bk", 32),
                          ("rows", 312), ("valid", 200), ("tile", 256)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--aiu", choices=("on", "off"), default="on")
    p.add_argument("--profile-one", action="store_true")
    args = p.parse_args()
    for name in ("m", "n", "k", "bm", "bn", "bk", "rows", "valid", "tile"):
        if getattr(args, name) < 1:
            p.error(name + " must be positive")
    if args.family == "vector" and args.valid > args.tile:
        p.error("valid must not exceed the fixed launch tile")
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["TILELANG_CACHE_DIR"] = str(args.out.resolve() / "jit-cache")
    os.environ["TILELANG_DISABLE_CACHE"] = "1"
    spec = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.out / "request.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    import torch
    import tilelang
    from tilelang import tvm
    torch.cuda.set_device(0)
    prim, inputs, output, out_dtype = definition(args)
    cfg = {"tl.disable_aiu_lower": args.aiu == "off", "tl.disable_ldmat_swzl": False}
    compiled = tilelang.compile(prim, target={"kind": "ppu", "arch": "ppu_15"},
                               execution_backend="tvm_ffi", pass_configs=cfg)
    if compiled.artifact is None or compiled.artifact.device_mod is None:
        raise RuntimeError("device TIR unavailable; no same-source comparison possible")
    (args.out / "input.tir").write_text(prim.script(), encoding="utf-8")
    (args.out / "device.tir").write_text(compiled.artifact.device_mod.script(), encoding="utf-8")
    (args.out / "device.json").write_text(tvm.ir.save_json(compiled.artifact.device_mod), encoding="utf-8")
    (args.out / "kernel.ppu.cc").write_text(compiled.get_kernel_source(), encoding="utf-8")
    gen = torch.Generator().manual_seed(7)
    host = [torch.randint(-2, 3, shape, generator=gen).double() for shape in inputs]
    if args.family == "vector":
        host = [h / 4 for h in host]
    dtype = getattr(torch, args.dtype)
    a, b = [h.to(device="cuda", dtype=dtype) for h in host]
    c = torch.full(output, math.nan, device="cuda", dtype=getattr(torch, out_dtype))
    if args.family == "cube":
        reference = host[0] @ host[1]
    elif args.op == "add":
        reference = host[0] + host[1]
    elif args.op == "exp2":
        reference = torch.exp2(host[0])
    else:
        reference = (host[0] + host[1]) * torch.exp2(host[0])
    compiled(a, b, c)
    torch.cuda.synchronize()
    approximate = args.family == "vector" and args.op != "add"
    tolerance = (0.02 if args.dtype == "bfloat16" else 0.005 if args.dtype == "float16" else 1e-5) if approximate else 0
    torch.testing.assert_close(c.cpu().double(), reference, atol=tolerance, rtol=tolerance)
    max_error = (c.cpu().double() - reference).abs().max().item()
    for _ in range(20):
        compiled(a, b, c)
    torch.cuda.synchronize()
    times = []
    if args.profile_one:
        api = torch.cuda.cudart()
        torch.cuda.check_error(api.cudaProfilerStart())
        compiled(a, b, c)
        torch.cuda.synchronize()
        torch.cuda.check_error(api.cudaProfilerStop())
    else:
        for _ in range(7):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                compiled(a, b, c)
            end.record(); end.synchronize()
            times.append(start.elapsed_time(end) * 1000 / 100)
    record = dict(request=spec, shapes=inputs + [output], output_dtype=out_dtype, pass_configs=cfg,
                  check="cpu_reference_passed", atol=tolerance, rtol=tolerance, max_abs_error=max_error,
                  hot_us=times, launches_per_sample=100, torch=str(torch.__version__),
                  tilelang=str(getattr(tilelang, "__version__", "unknown")),
                  device=torch.cuda.get_device_name(0),
                  sha256={f: hashlib.sha256((args.out / f).read_bytes()).hexdigest()
                          for f in ("device.json", "kernel.ppu.cc")},
                  note="Correctness does not prove the native opcode/dtype, execution granule, cache state or peak rate.")
    (args.out / "hardware.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
