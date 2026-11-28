"""Same-artifact PPU tests for all six kernels in the pinned DeepSeek reference.

The vendored kernel is unmodified. `plan` needs only Python's standard library;
`run` needs the site's SAIL TileLang, PyTorch and PPU SDK inside its container.
"""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "third_party/deepseek_v41"
KERNELS = {"act_quant_kernel", "fp4_quant_kernel", "fp8_gemm_kernel",
           "fp4_gemm_kernel", "sparse_attn_kernel", "hc_split_sinkhorn_kernel"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def environment_snapshot(out, tilelang):
    commands = {"ppu-smi": ["ppu-smi", "-q"], "sdk": ["nvcc", "--version"],
                "tilelang-git": ["git", "-C", str(Path(tilelang.__file__).parent), "rev-parse", "HEAD"]}
    status = {}
    for name, cmd in commands.items():
        try:
            result = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
            (out / (name + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
            status[name] = dict(command=cmd, returncode=result.returncode)
        except (OSError, subprocess.TimeoutExpired) as error:
            status[name] = dict(command=cmd, error=str(error))
    status["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    save(out / "environment.json", status)


def check_source():
    provenance = json.loads((SOURCE / "provenance.json").read_text())
    for name, digest in provenance["sha256"].items():
        if sha(SOURCE / name) != digest:
            raise ValueError("pinned upstream file changed: " + name)
    tree = ast.parse((SOURCE / "kernel.py").read_text())
    found = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)
             and any(isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
                     and d.func.attr == "jit" for d in n.decorator_list)}
    if found != KERNELS:
        raise ValueError("kernel inventory changed: " + str(found))
    return provenance


def cases(suite):
    cfg = json.loads((SOURCE / "config.json").read_text())
    result = []

    def add(name, family, shapes, **options):
        result.append(dict(name=name, family=family, shapes=shapes, options=options,
                           shape_origin="benchmark-selected; model widths from pinned config where applicable"))

    if suite in ("smoke", "all"):
        for name, family, group, inplace, scale in (
            ("act_fp8", "act_quant_kernel", 32, False, "float8_e8m0fnu"),
            ("act_fp8_qdq", "act_quant_kernel", 32, True, "float8_e8m0fnu"),
            ("act_fp4_qdq", "fp4_quant_kernel", 32, True, "float8_e8m0fnu"),
            ("act_fp4_latent_qdq", "fp4_quant_kernel", 16, True, "float8_e4m3"),
            ("act_fp4_packed", "fp4_quant_kernel", 32, False, "float8_e8m0fnu")):
            add(name, family, dict(M=32, N=128), group=group, inplace=inplace, scale=scale)
        for kind in ("fp8", "fp4"):
            add("gemm_" + kind, kind + "_gemm_kernel", dict(M=32, N=128, K=128), group=32)
        for name, topk in (("sparse_window", cfg["window_size"]),
                           ("sparse_compressed", cfg["window_size"] + cfg["index_topk"])):
            add(name, "sparse_attn_kernel", dict(b=1, m=1, n=1024, topk=topk,
                h=cfg["n_heads"], d=cfg["head_dim"]), pattern="repeat_pad")
        add("mhc_sinkhorn", "hc_split_sinkhorn_kernel", dict(n=1, hc=cfg["hc_mult"]),
            iters=cfg["hc_sinkhorn_iters"], eps=cfg["hc_eps"])
        add("act_fp8_hidden", "act_quant_kernel", dict(M=32, N=cfg["dim"]),
            group=32, inplace=False, scale="float8_e8m0fnu")
        add("gemm_fp4_expert", "fp4_gemm_kernel", dict(M=32, N=cfg["moe_inter_dim"], K=cfg["dim"]), group=32)
    if suite in ("model", "all"):
        for m in (1, 32, 256):
            for label, kind, n, k in (("query", "fp8", cfg["q_lora_rank"], cfg["dim"]),
                                     ("expert_up", "fp4", cfg["moe_inter_dim"], cfg["dim"]),
                                     ("expert_down", "fp4", cfg["dim"], cfg["moe_inter_dim"])):
                add(f"{label}_m{m}", kind + "_gemm_kernel", dict(M=m, N=n, K=k), group=32)
            add(f"quant_hidden_m{m}", "act_quant_kernel", dict(M=m, N=cfg["dim"]),
                group=32, inplace=False, scale="float8_e8m0fnu")
            add(f"sinkhorn_n{m}", "hc_split_sinkhorn_kernel", dict(n=m, hc=cfg["hc_mult"]),
                iters=cfg["hc_sinkhorn_iters"], eps=cfg["hc_eps"])
        for b, m in ((1, 16), (8, 1)):
            for topk in (cfg["window_size"], cfg["window_size"] + cfg["index_topk"]):
                add(f"attention_b{b}_m{m}_k{topk}", "sparse_attn_kernel",
                    dict(b=b, m=m, n=8192, topk=topk, h=cfg["n_heads"], d=cfg["head_dim"]), pattern="uniform")
        add("attention_all_invalid", "sparse_attn_kernel",
            dict(b=1, m=1, n=1024, topk=128, h=cfg["n_heads"], d=cfg["head_dim"]), pattern="all_invalid")
    return result


def load_kernel():
    # Preserve upstream language import and algorithms. Explicit PPU target is
    # applied during elaboration and compilation, as in the archived GTSim run.
    spec = importlib.util.spec_from_file_location("deepseek_v41_reference", SOURCE / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def elaborate(module, case):
    s, o, name = case["shapes"], case["options"], case["family"]
    if name in ("act_quant_kernel", "fp4_quant_kernel"):
        positional = (s["N"], o["group"])
        keyword = dict(scale_dtype=o["scale"], inplace=o["inplace"])
        if name == "act_quant_kernel":
            keyword["round_scale"] = True
        bindings = {"M": s["M"]}
    elif name in ("fp8_gemm_kernel", "fp4_gemm_kernel"):
        positional, keyword = (s["N"], s["K"], o["group"]), dict(scale_dtype=module.FE8M0)
        bindings = {"M": s["M"]}
    elif name == "sparse_attn_kernel":
        positional, keyword = (s["h"], s["d"]), {}
        bindings = {k: s[k] for k in ("b", "m", "n", "topk")}
    else:
        positional, keyword = (s["hc"], o["iters"], o["eps"]), {}
        bindings = {"n": s["n"]}
    operator = getattr(module, name)
    # Bind only T.symbolic declarations, leaving tiling, loops and math intact.
    # This is intentionally recorded as static specialization, not dynamic TIR.
    with patch.object(module.T, "symbolic", side_effect=bindings.__getitem__):
        prim = getattr(operator, "jit_impl", operator).get_tir(*positional, **keyword)
    return prim, bindings


def run_case(args, case):
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    provenance = check_source()
    save(out / "request.json", dict(case=case, arguments=vars(args) | {"out": str(out)}, source=provenance))
    os.environ["TILELANG_DISABLE_CACHE"] = "1"
    os.environ["TILELANG_CACHE_DIR"] = str(out.resolve() / "jit-cache")
    import torch
    import tilelang
    from tilelang import tvm
    from deepseek_v41_data import make_data, check_outputs
    torch.set_num_threads(args.cpu_threads)
    torch.cuda.set_device(0)
    environment_snapshot(out, tilelang)
    module = load_kernel()
    target = {"kind": "ppu", "arch": "ppu_15"}
    pass_configs = dict(module.pass_configs)
    if args.aiu != "default":
        pass_configs["tl.disable_aiu_lower"] = args.aiu == "off"
    with tvm.target.Target(target):
        prim, bindings = elaborate(module, case)
    (out / "input.tir").write_text(prim.script(), encoding="utf-8")
    compiled = tilelang.compile(prim, target=target, execution_backend="tvm_ffi", pass_configs=pass_configs)
    if compiled.artifact is None or compiled.artifact.device_mod is None:
        raise RuntimeError("missing device IR; cannot establish same-artifact simulation")
    device = compiled.artifact.device_mod
    (out / "device.json").write_text(tvm.ir.save_json(device), encoding="utf-8")
    (out / "device.tir").write_text(device.script(), encoding="utf-8")
    (out / "kernel.ppu.cc").write_text(compiled.get_kernel_source(), encoding="utf-8")
    host, output_ids, expected = make_data(case, args.seed)
    # Exact generated buffers stay on the intranet. Needed for later index/cache replay.
    torch.save(dict(case=case, tensors=host, output_ids=output_ids), out / "tensors.pt")
    integer_inputs = dict(bindings)
    if case["family"] == "sparse_attn_kernel":
        integer_inputs["topk_idxs"] = host[4].flatten().tolist()
    save(out / "inputs.json", integer_inputs)
    buffers = [x.to("cuda") for x in host]

    def launch():
        compiled(*buffers)

    launch()
    torch.cuda.synchronize()
    # Export this compiled executable, never recompile generated source with
    # another set of flags just to obtain a disassembly.
    binary = {}
    try:
        compiled.adapter.get_exportable_executable().export_library(str(out / "kernel.so"))
        binary = dict(status="exported", sha256=sha(out / "kernel.so"))
    except Exception as error:
        binary = dict(status="export_unavailable", error=str(error))
    save(out / "binary.json", binary)
    validation = check_outputs(case, [buffers[i].cpu() for i in output_ids], expected)
    repeated_checks = [validation]
    for _ in range(args.check_repeat - 1):
        for i in output_ids:
            buffers[i].view(torch.uint8).fill_(255)
        launch()
        torch.cuda.synchronize()
        repeated_checks.append(check_outputs(case, [buffers[i].cpu() for i in output_ids], expected))
    validation = dict(status="passed", repetitions=repeated_checks)
    save(out / "validation.json", validation)
    record = dict(case=case, target=target, pass_configs={str(k): v for k, v in pass_configs.items()},
                  static_bindings=bindings, source=provenance,
                  torch=str(torch.__version__), tilelang=str(getattr(tilelang, "__version__", "unknown")),
                  tilelang_file=str(tilelang.__file__), device=torch.cuda.get_device_name(0),
                  device_properties=str(torch.cuda.get_device_properties(0)), seed=args.seed,
                  clock_note=args.clock_note, validation=validation, binary=binary,
                  buffers=[dict(shape=list(x.shape), dtype=str(x.dtype), bytes=x.numel()*x.element_size()) for x in host],
                  sha256={f: sha(out / f) for f in ("device.json", "kernel.ppu.cc", "input.tir", "inputs.json", "tensors.pt")})
    save(out / "result.json", record)
    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    if args.profile_one:
        # No graph/scrub inside the ACU window: exactly one selected kernel call.
        runtime = torch.cuda.cudart()
        torch.cuda.check_error(runtime.cudaProfilerStart())
        launch()
        torch.cuda.synchronize()
        torch.cuda.check_error(runtime.cudaProfilerStop())
        record.update(status="profiled", cache_state="after_validation_and_warmup")
        save(out / "result.json", record)
        return

    if args.timing == "graph":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                launch()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.repeat):
                launch()
        single_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(single_graph):
            launch()
        hot_call, cold_call = graph.replay, single_graph.replay
    else:
        def hot_call():
            for _ in range(args.repeat):
                launch()
        cold_call = launch

    def elapsed(call):
        first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        first.record()
        call()
        last.record()
        last.synchronize()
        return first.elapsed_time(last) * 1000

    hot = [elapsed(hot_call) / args.repeat for _ in range(args.samples)]
    scrub = torch.empty(args.scrub_mib << 20, dtype=torch.uint8, device="cuda")
    after_scrub = []
    for i in range(args.samples):
        scrub.fill_(i)
        torch.cuda.synchronize()
        after_scrub.append(elapsed(cold_call))
    # Recheck after all launches; no failed or overwritten output becomes a timing result.
    check_outputs(case, [buffers[i].cpu() for i in output_ids], expected)
    record.update(status="passed", timing=args.timing, hot_us=hot, hot_median_us=statistics.median(hot),
                  after_scrub_us=after_scrub, after_scrub_median_us=statistics.median(after_scrub),
                  hot_launches_per_sample=args.repeat, warmup=args.warmup, scrub_bytes=scrub.numel(),
                  cold_cache_guaranteed=False)
    save(out / "result.json", record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "run", "case"))
    p.add_argument("out", type=Path, nargs="?")
    p.add_argument("--suite", choices=("smoke", "model", "all"), default="smoke")
    p.add_argument("--cases", nargs="+")
    p.add_argument("--aiu", choices=("default", "on", "off"), default="default")
    p.add_argument("--timing", choices=("graph", "event"), default="graph")
    p.add_argument("--clock-note", default="not_verified")
    p.add_argument("--profile-one", action="store_true")
    p.add_argument("--acu", action="store_true", help="run a separate ACU process after each successful timing")
    p.add_argument("--acu-cache", choices=("none", "all"), default="none")
    for key, default in (("samples", 7), ("repeat", 50), ("warmup", 20), ("scrub-mib", 256),
                         ("seed", 7), ("timeout", 1800), ("cpu-threads", 4), ("check-repeat", 3)):
        p.add_argument("--" + key, type=int, default=default)
    args = p.parse_args()
    for key in ("samples", "repeat", "warmup", "scrub_mib", "timeout", "cpu_threads", "check_repeat"):
        if getattr(args, key) < 1:
            p.error(key + " must be positive")
    check_source()
    selected = cases(args.suite)
    if args.cases:
        unknown = set(args.cases) - {c["name"] for c in selected}
        if unknown:
            p.error("unknown cases for selected suite: " + str(sorted(unknown)))
        selected = [c for c in selected if c["name"] in args.cases]
    if args.action == "plan":
        text = json.dumps(selected, indent=2, ensure_ascii=False)
        if args.out:
            args.out.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return
    if args.out is None:
        p.error("out directory required")
    if args.action == "case":
        if args.acu:
            p.error("--acu belongs to run; use case --profile-one inside an ACU command")
        if len(selected) != 1:
            p.error("case requires exactly one --cases name")
        try:
            run_case(args, selected[0])
        except Exception:
            if args.out.is_dir():
                (args.out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        return
    if args.profile_one:
        p.error("use case --profile-one for one ACU invocation")
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / "plan.json", selected)
    manifest = []
    for case in selected:
        name = case["name"]
        cmd = [sys.executable, str(Path(__file__).resolve()), "case", str(args.out / name),
               "--suite", args.suite, "--cases", name, "--aiu", args.aiu, "--timing", args.timing,
               "--clock-note", args.clock_note]
        for key in ("samples", "repeat", "warmup", "scrub_mib", "seed", "cpu_threads", "check_repeat"):
            cmd += ["--" + key.replace("_", "-"), str(getattr(args, key))]
        save(args.out / (name + ".command.json"), cmd)
        started = time.monotonic()
        with (args.out / (name + ".log")).open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                status = "passed" if process.returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                status = "timeout"
        row = dict(case=name, status=status, elapsed_s=time.monotonic()-started)
        result = args.out / name / "result.json"
        if result.exists():
            report = json.loads(result.read_text())
            if status == "passed" and report.get("status") == "passed":
                row.update({k: report[k] for k in ("hot_median_us", "after_scrub_median_us")})
            else:
                row["status"] = status if status != "passed" else "incomplete"
        elif status == "passed":
            row["status"] = "incomplete"
        if row["status"] == "passed" and args.acu:
            profile_dir = args.out / (name + "-profile")
            profile_cmd = list(cmd)
            profile_cmd[3] = str(profile_dir)
            profile_cmd.append("--profile-one")
            prefix = args.out / (name + "-acu")
            acu_cmd = ["acu", "--clock-control", "none", "--cache-control", args.acu_cache,
                       "--profile-from-start", "off", "--set=full", "--target-processes", "all",
                       "-o", str(prefix), *profile_cmd]
            save(args.out / (name + "-acu.command.json"), acu_cmd)
            profile_status = dict(case=name, cache_control=args.acu_cache)
            try:
                with (args.out / (name + "-acu.log")).open("w", encoding="utf-8") as log:
                    subprocess.run(acu_cmd, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=True)
                for filename in ("device.json", "kernel.ppu.cc", "inputs.json"):
                    if sha(profile_dir / filename) != sha(args.out / name / filename):
                        raise ValueError("profile artifact differs: " + filename)
                export_cmd = ["acu", "-i", str(prefix) + ".acurep", "--page", "raw", "--csv",
                              "--csv-file", str(prefix) + ".csv"]
                with (args.out / (name + "-acu-export.log")).open("w", encoding="utf-8") as log:
                    subprocess.run(export_cmd, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=True)
                profile_status["status"] = "passed"
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                profile_status.update(status="failed", error=str(error))
                row["status"] = "timing_passed_profile_failed"
            save(args.out / (name + "-acu.status.json"), profile_status)
        manifest.append(row)
        save(args.out / "manifest.json", manifest)
        print(name, row["status"], flush=True)
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("case", "status", "elapsed_s", "hot_median_us", "after_scrub_median_us"))
        writer.writeheader()
        writer.writerows(manifest)
    if any(row["status"] != "passed" for row in manifest):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
