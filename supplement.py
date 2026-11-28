"""Offline supplemental PPU probes and conservative slope analysis (stdlib only)."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from collections import defaultdict

HERE = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(suite, cus):
    if cus < 1:
        raise ValueError("cu_count must be positive")
    cases = []

    def add(mode, binary="memory", steps=(256, 1024, 4096), **params):
        for n in steps:
            p = dict(blocks=1, threads=32, steps=n)
            p.update(params)
            cases.append(dict(mode=mode, binary=binary, params=p))

    if suite in ("smoke", "tsm", "all"):
        for blocks in ([1] if suite == "smoke" else [1, cus, cus * 8]):
            for width in ([4] if suite == "smoke" else [1, 4, 16]):
                for readers in ([0, 4, 8] if suite == "smoke" else [0, 2, 4, 6, 8]):
                    add("tsm-mix", blocks=blocks, threads=256, width=width, read_warps=readers)
    if suite in ("smoke", "latency", "all"):
        add("tsm-chain", workset_kib=4)
        add("shuffle-chain")
        for threads in ([128] if suite == "smoke" else [32, 128, 256, 512]):
            add("barrier", threads=threads)
        for kib in ([64] if suite == "smoke" else [16, 32, 64, 96, 128, 256, 512, 1024, 1536,
                                                    2048, 4096, 16384, 32768, 65536, 131072]):
            add("global-chain", workset_kib=kib)
    if suite in ("smoke", "vector", "all"):
        for dtype in ([32] if suite == "smoke" else [16, 32, 64]):
            binary = "vector" + str(dtype)
            add("vector-chain", binary, slots=1, op=2)
            for blocks in ([cus * 8] if suite == "smoke" else [1, cus, cus * 8, cus * 16]):
                for slots in ([4] if suite == "smoke" else [1, 4, 16]):
                    for op in ([2] if suite == "smoke" else [0, 1, 2]):
                        add("vector", binary, blocks=blocks, threads=256, slots=slots, op=op)
        if suite != "smoke":
            # Fixed launch envelope; masks change valid elements, not block/thread count.
            for active in [63, 64, 65, 127, 128, 129, 199, 200, 255, 256, 257, 300, 511, 512]:
                add("vector", "vector16", blocks=cus * 8, threads=256, slots=4, op=0, active=active)
    if suite in ("cube", "all"):
        for dtype in ("16", "bf16", "8"):
            binary = "cube" + dtype
            add("cube-chain", binary, slots=1)
            for blocks in (1, cus, cus * 8, cus * 16):
                for slots in (1, 4, 16):
                    add("cube", binary, blocks=blocks, threads=256, slots=slots)
    for i, case in enumerate(cases):
        # Stable descriptive identity independent of case order and of the output directory.
        digest = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()[:12]
        case["id"] = case["mode"] + "-" + case["binary"] + "-" + digest
    return cases


def logged(command, dest, name, timeout=180):
    command = list(map(str, command))
    save(dest / (name + ".command.json"), command)
    with (dest / (name + ".stdout")).open("w", encoding="utf-8") as out:
        with (dest / (name + ".stderr")).open("w", encoding="utf-8") as err:
            subprocess.run(command, stdout=out, stderr=err, check=True, timeout=timeout)


def command_for(exe, case, profile=False):
    cmd = [exe, case["mode"]]
    for key, value in case["params"].items():
        cmd.extend(["--" + key.replace("_", "-"), str(value)])
    if profile:
        cmd.extend(["--profile", "1"])
    return cmd


def build_flags(binary):
    if binary.startswith("vector"):
        return ["-DCAL_VECTOR=" + binary[6:]]
    if binary.startswith("cube"):
        return ["-DCAL_MATRIX=" + {"16": "16", "bf16": "17", "8": "8"}[binary[4:]]]
    return []


def collect(args):
    dest = args.out.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    manifest = dict(started=time.time(), suite=args.suite, visibility=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    clock_note=args.clock_note, frequency_changed_by_runner=False,
                    source_sha256={p: sha(HERE / p) for p in ("calibration_probe.cu", "probe_common.cuh", "supplement.py")},
                    steps={}, cases=[])
    records = manifest["steps"]

    def step(name, action):
        try:
            action()
            records[name] = {"status": "ok"}
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            records[name] = {"status": "failed", "reason": str(exc)}
        save(dest / "manifest.json", manifest)
        print(name, records[name]["status"], flush=True)
        return records[name]["status"] == "ok"

    def build(binary):
        logged([args.nvcc, "-O3", "-std=c++17", "-arch=" + args.arch, *build_flags(binary),
                HERE / "calibration_probe.cu", "-o", dest / binary], dest, "build-" + binary, args.timeout)
        manifest.setdefault("binary_sha256", {})[binary] = sha(dest / binary)

    monitor = None
    clock_log = (dest / "clocks.log").open("w", encoding="utf-8")
    rows = []
    try:
        step("compiler", lambda: logged([args.nvcc, "--version"], dest, "compiler"))
        step("inventory", lambda: logged(["ppu-smi", "-q"], dest, "inventory"))
        if args.smi_id is not None:
            try:
                monitor = subprocess.Popen(["ppu-smi", "dmon", "-i", str(args.smi_id), "-s", "pucv"],
                                           stdout=clock_log, stderr=subprocess.STDOUT)
            except OSError as exc:
                records["clock_monitor"] = dict(status="failed", reason=str(exc))
        if not step("build-memory", lambda: build("memory")):
            return 1
        if not step("device-info", lambda: logged([dest / "memory", "info"], dest, "device-info")):
            return 1
        device = json.loads((dest / "device-info.stdout").read_text())
        manifest["device"] = device
        cases = plan(args.suite, device["cu_count"])
        if args.case:
            requested = set(args.case)
            unknown = requested - {c["id"] for c in cases}
            if unknown:
                raise ValueError("unknown case IDs; plan with the board's CU count: " + ", ".join(sorted(unknown)))
            cases = [c for c in cases if c["id"] in requested]
        manifest["cases"] = cases
        save(dest / "plan.json", cases)
        built = {"memory": True}
        for case in cases:
            binary, name = case["binary"], case["id"]
            if binary not in built:
                built[binary] = step("build-" + binary, lambda: build(binary))
            if not built[binary]:
                records[name] = dict(status="blocked", reason="compiler did not support/build this binary; no fallback dtype")
                continue

            def one():
                logged(command_for(dest / binary, case), dest, name, args.timeout)
                with (dest / (name + ".stdout")).open(newline="") as f:
                    samples = list(csv.DictReader(f))
                if len(samples) != 7 or any(not math.isfinite(float(r["us_per_launch"])) or
                                            float(r["us_per_launch"]) <= 0 for r in samples):
                    raise ValueError("missing or invalid seven timing samples")
                rows.extend(dict(case_id=name, **r) for r in samples)

            if step(name, one) and args.acu:
                def profile():
                    prefix = dest / ("acu-" + name)
                    logged(["acu", "--clock-control", "none", "--cache-control", args.acu_cache,
                            "--profile-from-start", "off", "--set=full", "--target-processes", "all",
                            "-o", prefix, *command_for(dest / binary, case, True)], dest, "acu-" + name, args.timeout)
                    logged(["acu", "-i", str(prefix) + ".acurep", "--page", "raw", "--csv", "--csv-file",
                            str(prefix) + ".csv"], dest, "acu-export-" + name, args.timeout)
                step("acu-" + name, profile)
            if rows:
                with (dest / "samples.csv").open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    finally:
        if monitor is not None:
            early = monitor.poll()
            if early is None:
                monitor.terminate()
                try:
                    monitor.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    monitor.kill(); monitor.wait()
            clock_log.flush()
            records["clock_monitor"] = dict(status="ok" if early is None and (dest / "clocks.log").stat().st_size else "failed")
        clock_log.close()
        manifest["finished"] = time.time()
        manifest["acu_cache"] = args.acu_cache if args.acu else None
        save(dest / "manifest.json", manifest)
    return int(any(v["status"] != "ok" for v in records.values()))


def linear_fit(points):
    if len({x for x, _ in points}) < 3:
        raise ValueError("at least three distinct step counts are required")
    xs, ys = zip(*points)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    xx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in points) / xx
    intercept = my - slope * mx
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
    total = sum((y - my) ** 2 for y in ys)
    return slope, intercept, 1 - residual / total if total > 0 else 0


def work_per_step(case):
    p, mode = case["params"], case["mode"]
    blocks, threads, slots = p["blocks"], p["threads"], p.get("slots", 1)
    if mode.startswith("vector"):
        pack = 2 if case["binary"] == "vector16" else 1
        valid = p.get("active", threads * pack)
        executed = math.ceil(valid / pack) * pack  # source-level packing only, NOT inferred CU granularity
        return dict(valid_elements=blocks * valid * slots, source_elements=blocks * executed * slots)
    if mode.startswith("cube"):
        count = blocks * (threads // 32) * slots
        k = 32 if case["binary"] == "cube8" else 16
        return dict(compat_fragments=count, source_flops=count * 2 * 16 * 8 * k)
    if mode == "tsm-mix":
        readers = p["read_warps"] * 32
        return dict(read_bytes=blocks * readers * 4 * p["width"],
                    write_bytes=blocks * (threads - readers) * 4 * p["width"])
    return dict(chain_steps=1)


def analyze(args):
    root = args.results
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in manifest["cases"]}
    raw = defaultdict(list)
    with (root / "samples.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if row["case_id"] not in cases:
                raise ValueError("sample identity not in saved plan")
            raw[row["case_id"]].append(row)
    groups = defaultdict(list)
    for name, rows in raw.items():
        if manifest["steps"].get(name, {}).get("status") != "ok":
            continue
        case = cases[name]
        p = {k: v for k, v in case["params"].items() if k != "steps"}
        key = json.dumps(dict(mode=case["mode"], binary=case["binary"], params=p), sort_keys=True)
        times = [float(r["us_per_launch"]) for r in rows]
        if not times or any(not math.isfinite(t) or t <= 0 for t in times):
            raise ValueError("timings must be finite and positive")
        median = statistics.median(times)
        spread = (max(times) - min(times)) / median
        groups[key].append((case, median, spread, max(int(r["local_bytes"]) for r in rows)))
    output = []
    for key, values in sorted(groups.items()):
        record = json.loads(key)
        points = sorted((v[0]["params"]["steps"], v[1]) for v in values)
        flags = ["PASM_and_dynamic_counts_need_local_review", "not_a_calibrated_parameter"]
        record.update(points=points, case_ids=[v[0]["id"] for v in values], flags=flags)
        try:
            slope, intercept, r2 = linear_fit(points)
        except ValueError as exc:
            flags.append(str(exc)); output.append(record); continue
        record.update(us_per_step=slope, intercept_us=intercept, r_squared=r2)
        if slope <= 0 or r2 < 0.98:
            flags.append("invalid_or_non_linear_slope")
        if any(v[2] > 0.05 for v in values):
            flags.append("sample_range_exceeds_5_percent")
        if any(v[3] > 0 for v in values):
            flags.append("local_storage_reported_check_spill")
        rates = work_per_step(values[0][0])
        record["source_work_per_step"] = rates
        if slope > 0:
            record["source_work_per_second"] = {k: v * 1e6 / slope for k, v in rates.items()}
            if args.core_mhz:
                record["assumed_or_measured_core_mhz"] = args.core_mhz
                record["frequency_evidence_note"] = args.frequency_note
                record["cycles_per_step"] = slope * args.core_mhz
                scope = 1 if record["params"]["blocks"] == 1 else manifest["device"]["cu_count"]
                record["rate_normalization_cus"] = scope
                record["source_work_per_cu_cycle"] = {k: v / (slope * args.core_mhz * scope) for k, v in rates.items()}
                flags.append("verify_clock_and_active_CUs_before_using_cycle_rates")
        if record["mode"] == "global-chain":
            flags.append("effective_dependency_cost_cache_level_unclassified")
        if record["mode"] in ("vector", "cube", "tsm-mix"):
            flags.append("compare_concurrency_sweep_no_automatic_peak_or_II_claim")
        output.append(record)
    dest = args.out or root / "analysis.json"
    if dest.exists():
        raise ValueError("analysis output already exists; choose a new --out")
    save(dest, output)
    print(str(dest))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for action in ("plan", "run"):
        p = sub.add_parser(action)
        p.add_argument("out", type=Path)
        p.add_argument("--suite", choices=("smoke", "vector", "cube", "tsm", "latency", "all"), default="smoke")
        if action == "plan":
            p.add_argument("--cu-count", type=int, default=39)
        else:
            p.add_argument("--nvcc", default="nvcc")
            p.add_argument("--arch", default="sm_89")
            p.add_argument("--smi-id", type=int)
            p.add_argument("--clock-note", default="site policy unchanged; verify clocks.log")
            p.add_argument("--case", nargs="+", help="exact IDs from plan with the actual CU count")
            p.add_argument("--timeout", type=int, default=180)
            p.add_argument("--acu", action="store_true", help="profile each selected case separately; use --case to bound overhead")
            p.add_argument("--acu-cache", choices=("all", "none"), default="none")
    p = sub.add_parser("analyze")
    p.add_argument("results", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--core-mhz", type=float, help="verified core frequency, never memory MHz")
    p.add_argument("--frequency-note", default="not independently verified by analyzer")
    args = parser.parse_args()
    if args.command == "plan":
        if args.out.exists():
            raise ValueError("plan output already exists")
        cases = plan(args.suite, args.cu_count)
        save(args.out, cases); print(str(args.out), len(cases), "cases"); return 0
    if args.command == "analyze":
        if args.core_mhz is not None and (not math.isfinite(args.core_mhz) or args.core_mhz <= 0):
            raise ValueError("core MHz must be finite and positive")
        return analyze(args)
    return collect(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr); sys.exit(1)
