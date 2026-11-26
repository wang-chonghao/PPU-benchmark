"""Offline board acquisition and host-side replay for PPU 1.5."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
import bm_bw

from paired import CASES

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_logged(command, out, name, stdout_path=None):
    command = [str(x) for x in command]
    save(out / (name + ".command.json"), command)
    with (stdout_path or out / (name + ".log")).open("w") as stdout:
        with (out / (name + ".stderr")).open("w") as stderr:
            subprocess.run(command, stdout=stdout, stderr=stderr, check=True)


def summarize(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    grouped = defaultdict(list)
    for row in rows:
        key = tuple((k, v) for k, v in row.items() if k not in ("sample", "us_per_launch"))
        grouped[key].append(float(row["us_per_launch"]))
    result = []
    for key, samples in grouped.items():
        row = dict(key)
        us = statistics.median(samples)
        row.update(median_us=us, min_us=min(samples), max_us=max(samples), samples=len(samples))
        if row["kind"] == "stream":
            total = int(row["logical_read_bytes"]) + int(row["logical_write_bytes"])
            row["logical_total_GB_s"] = total / (us * 1000)
        elif row["kind"] in ("register", "array"):
            count = int(row["target_ops_per_launch"])
            row["target_ops_per_second"] = count * 1e6 / us
            if int(row["flop_per_op"]):
                row["TFLOP_s"] = count * int(row["flop_per_op"]) / (us * 1e6)
            if row["kind"] == "array":
                row["logical_total_GB_s"] = 8 * count / (us * 1000)
        elif row["kind"] in ("mma", "math"):
            ops = int(row["blocks"]) * int(row["threads"]) * int(row["chains"]) * int(row["steps"])
            if row["kind"] == "mma":
                k = 32 if row["operation"] == "fp8" else 16
                row["issued_fragment_TFLOP_s"] = (ops // 32) * (2 * 16 * 8 * k) / (us * 1e6)
            else:
                row["recurrence_Telement_s"] = ops / (us * 1e6)
        result.append(row)
    return result


def board(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = {}
    manifest = dict(platform=platform.platform(), python=sys.version, started_unix=time.time(),
                    visibility=os.environ.get("CUDA_VISIBLE_DEVICES"), runtime_device=0,
                    clock_note=args.clock_note, smi_id=args.smi_id,
                    sources={name: digest(HERE / name) for name in
                             ("probe.cu", "probe_common.cuh", "throughput.cu", "paired.py", "bm_bw.py", "run.py")},
                    steps=status)

    def step(name, action):
        started = time.time()
        try:
            action()
        except (OSError, ValueError, subprocess.CalledProcessError, RuntimeError) as exc:
            status[name] = {"status": "failed", "error": str(exc)}
        else:
            status[name] = {"status": "ok"}
        status[name].update(started_unix=started, finished_unix=time.time())
        save(out / "manifest.json", manifest)
        print(name, status[name]["status"], flush=True)
        return status[name]["status"] == "ok"

    def acu(name, command):
        prefix = out / ("acu-" + name)
        run_logged(["acu", "--clock-control", "none", "--cache-control", args.acu_cache,
                    "--profile-from-start", "off", "--set=full",
                    "--target-processes", "all", "-o", prefix, *command], out, "acu-" + name)
        run_logged(["acu", "-i", str(prefix) + ".acurep", "--page", "raw", "--csv",
                    "--csv-file", str(prefix) + ".csv"], out, "acu-export-" + name)

    monitor = None
    with (out / "clocks.log").open("w") as log:
        if shutil.which("ppu-smi"):
            step("inventory", lambda: run_logged(["ppu-smi", "-q"], out, "inventory"))
            if args.smi_id is not None:
                monitor = subprocess.Popen(["ppu-smi", "dmon", "-i", str(args.smi_id), "-s", "pucv"],
                                           stdout=log, stderr=subprocess.STDOUT)
        try:
            if args.acu:
                step("acu-version", lambda: run_logged(["acu", "--version"], out, "acu-version"))
            if args.suite in ("all", "native", "throughput", "bm-bw", "bm-ramp"):
                step("compiler", lambda: run_logged([args.nvcc, "--version"], out, "compiler"))
            if args.suite in ("bm-bw", "bm-ramp"):
                bm_bw.collect(args, out, step, run_logged, acu, save)
            if args.suite in ("all", "native"):
                base = [args.nvcc, "-O3", "-std=c++17", "-arch=" + args.arch, HERE / "probe.cu"]
                exe = out / "probe"
                if step("build", lambda: run_logged([*base, "-o", exe], out, "build")):
                    for case in ("device", "occupancy", "stream", "shared", "math"):
                        raw = out / (case + ".csv")
                        def collect():
                            run_logged([exe, case, args.max_mib], out, case, raw)
                            if case in ("stream", "shared", "math"):
                                save(out / (case + "-summary.json"), summarize(raw))
                        step(case, collect)
                    if args.acu:
                        for op in ("reg8", "reg64"):
                            step("acu-" + op, lambda: acu(op, [exe, "profile-" + op, 0]))
                        for mib in sorted({min(32, args.max_mib), args.max_mib}):
                            for op in ("read", "copy"):
                                name = f"{op}-{mib}MiB"
                                step("acu-" + name, lambda: acu(name, [exe, "profile-" + op, mib]))
                mma = out / "probe-mma"
                if step("build-mma", lambda: run_logged([*base, "-DPPU_PROBE_MMA", "-o", mma], out, "build-mma")):
                    def collect_mma():
                        raw = out / "mma.csv"
                        run_logged([mma, "mma", 0], out, "mma", raw)
                        save(out / "mma-summary.json", summarize(raw))
                    step("mma", collect_mma)
                    if args.acu:
                        for op in ("bf16", "fp8"):
                            step("acu-" + op, lambda: acu(op, [mma, "profile-" + op, 0]))
            if args.suite in ("all", "native", "throughput"):
                for binary, flags, operations in (
                    ("independent", [], ("array", "fma", "ex2", "div")),
                    ("independent-mma", ["-DPPU_PROBE_MMA"], ("bf16", "fp8")),
                ):
                    exe = out / binary
                    build = [args.nvcc, "-O3", "-std=c++17", "-arch=" + args.arch, *flags,
                             HERE / "throughput.cu", "-o", exe]
                    if not step("build-" + binary, lambda: run_logged(build, out, "build-" + binary)):
                        continue
                    for op in operations:
                        name = "independent-" + op
                        def collect_independent():
                            raw = out / (name + ".csv")
                            run_logged([exe, op, args.max_mib], out, name, raw)
                            save(out / (name + "-summary.json"), summarize(raw))
                        if step(name, collect_independent) and args.acu:
                            step("acu-" + name, lambda: acu(name, [exe, "profile-" + op, args.max_mib]))
            if args.suite in ("all", "paired"):
                for case in args.cases:
                    dest = out / case
                    cmd = [sys.executable, HERE / "paired.py", case]
                    if step(case, lambda: run_logged([*cmd, dest], out, case)) and args.acu:
                        def profile_pair():
                            prof = out / (case + "-profile")
                            acu(case, [*cmd, prof, "--profile-one"])
                            for filename in ("device.json", "kernel.ppu.cc"):
                                if digest(prof / filename) != digest(dest / filename):
                                    raise ValueError("profile build differs from timed build: " + filename)
                        step("acu-" + case, profile_pair)
        finally:
            early_exit = None
            if monitor is not None:
                early_exit = monitor.poll()
                if early_exit is None:
                    monitor.terminate()
                    try:
                        monitor.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        monitor.kill()
                        monitor.wait()
                log.flush()
            manifest["clock_log_bytes"] = (out / "clocks.log").stat().st_size
            if args.smi_id is not None:
                clock_ok = monitor is not None and early_exit in (None, 0) and manifest["clock_log_bytes"] > 0
                status["clock_monitor"] = dict(status="ok" if clock_ok else "failed", early_exit=early_exit,
                                               diagnostic="Check pclk samples and PCI identity in clocks.log and inventory.log")
            manifest["acu_cache_control"] = args.acu_cache if args.acu else None
            manifest["native_executables"] = {name: digest(out / name) for name in
                                              ("probe", "probe-mma", "independent", "independent-mma",
                                               *("bm-" + key for key in bm_bw.SOURCES))
                                              if (out / name).is_file()}
            manifest["finished_unix"] = time.time()
            save(out / "manifest.json", manifest)
    return int(any(x["status"] != "ok" for x in status.values()))


def replay(args):
    cases = [p.parent for p in sorted(args.results.glob("*/hardware.json"))]
    if not cases:
        raise ValueError("no paired hardware.json files found")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in cases:
        dest = out / case.name
        dest.mkdir()
        try:
            hw = json.loads((case / "hardware.json").read_text())
            tir = case / "device.json"
            if (hw["case"] != case.name or hw["check"] != "exact_cpu_reference_passed"
                    or digest(tir) != hw["tir_sha256"] or digest(case / "kernel.ppu.cc") != hw["source_sha256"]):
                raise ValueError("case identity, correctness check or artifact hashes do not match")
            run_logged([sys.executable, args.adaptor, tir, "--sim", args.sim.resolve(),
                        "--timeline", dest / "timeline.tsv", "--timeline-blocks", "0",
                        "--residency", dest / "residency.tsv", "--report", dest / "sim.json"], dest, "simulation")
            report = json.loads((dest / "sim.json").read_text())
            if report["tir_sha256"] != hw["tir_sha256"]:
                raise ValueError("simulator consumed a different TIR")
            cfg = dict(line.split("=", 1) for line in report["profile"]["text"].splitlines()
                       if "=" in line and not line.lstrip().startswith("#"))
            mhz = float(cfg["core_frequency_mhz"])
            cycles = report["completion_cycle"]
            rows.append(dict(case=case.name, status="ok", tir_sha256=hw["tir_sha256"],
                             profile_sha256=report["profile"]["sha256"], cycles=cycles,
                             sim_MHz=mhz, sim_us=cycles / mhz,
                             event_hot_median_us=statistics.median(hw["hot_us"]),
                             event_after_scrub_median_us=statistics.median(hw["after_scrub_us"])))
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
            rows.append(dict(case=case.name, status="failed", diagnostic=str(exc)))
        save(out / "comparison.json", rows)
    print(json.dumps(rows, indent=2))
    return int(any(row["status"] != "ok" for row in rows))


def smoke(args):
    """Exercise the real importer/simulator with previously exported on/off fixtures."""
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ("aiu_on", "aiu_off"):
        dest = out / name
        dest.mkdir()
        run_logged([sys.executable, args.adaptor, args.fixtures / (name + ".json.gz"),
                    "--sim", args.sim.resolve(), "--report", dest / "sim.json",
                    "--timeline", dest / "timeline.tsv", "--residency", dest / "residency.tsv"], dest, "simulation")
        report = json.loads((dest / "sim.json").read_text())
        if report["completion_cycle"] <= 0 or report["threads_per_block"] != 128:
            raise ValueError("unexpected replay result for " + name)
        print(name, report["grid"], report["completion_cycle"], "cycles")
    return 0


def pack(args):
    root = args.results.resolve()
    if not (root / "manifest.json").is_file():
        raise ValueError("expected board results containing manifest.json")
    archive = args.archive.resolve()
    if root in archive.parents:
        raise ValueError("place the archive outside the results directory")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if path.is_file() and not path.is_symlink() and "jit-cache" not in relative.parts:
                z.write(path, relative.as_posix())
    print(archive)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("board")
    p.add_argument("out", type=Path)
    p.add_argument("--suite", choices=("all", "native", "paired", "throughput", "bm-bw", "bm-ramp"), default="all")
    p.add_argument("--bm-root", type=Path, default=HERE / "BM", help="pinned BM source tree containing BW/")
    p.add_argument("--bm-blocks", type=int, nargs="+", choices=bm_bw.RAMP_BLOCKS, default=bm_bw.RAMP_BLOCKS,
                   help="subset of original 14 GiB ramp grids (bm-ramp only)")
    p.add_argument("--smi-id", type=int, help="physical ppu-smi index matching runtime device 0")
    p.add_argument("--clock-note", default="unverified")
    p.add_argument("--nvcc", default="nvcc", help="PPU SDK CUDA-compatible compiler")
    p.add_argument("--arch", default="sm_89")
    p.add_argument("--max-mib", type=int, choices=(1, 16, 32, 128, 512, 1024, 2048), default=512)
    p.add_argument("--acu", action="store_true")
    p.add_argument("--acu-cache", choices=("none", "all"), default="none")
    p.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    p = sub.add_parser("replay")
    p.add_argument("results", type=Path)
    commands = [p, sub.add_parser("smoke", help="host-only GTSim check using historical TIR fixtures")]
    commands[1].add_argument("--fixtures", type=Path, default=HERE.parent / "GTSim-ppu/tests/data/tilelang")
    for p in commands:
        p.add_argument("out", type=Path)
        p.add_argument("--adaptor", type=Path, default=HERE.parent / "GTSim-ppu/scripts/gtsim_tilelang.py")
        p.add_argument("--sim", type=Path, required=True)
    p = sub.add_parser("pack", help="archive board results, omitting transient JIT caches")
    p.add_argument("results", type=Path)
    p.add_argument("archive", type=Path)
    args = parser.parse_args()
    return {"board": board, "smoke": smoke, "replay": replay, "pack": pack}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
