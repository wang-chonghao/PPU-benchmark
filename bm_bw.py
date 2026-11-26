"""Run unmodified, pinned GitCode BM bandwidth kernels on the allocated board."""
import csv
import hashlib
import re
from pathlib import Path

REVISION = "9221f6624d7c3525bb8cd31fd1c7f50967accf01"
SOURCES = {
    "dram": ("test_peak/dram_bw.cu", "7a96b5422f803c7b11b5ccd24e98b0845906394b86d8a6ceb260f84bf73819aa"),
    "llc": ("test_peak/llc_bw.cu", "133aa7e4d246d705cc15a554223acb632e7ea103d832f97c8f74a785e694715b"),
    "l2": ("test_peak/l2_bw.cu", "db45747235ee24a36114c2ccf05d9ffa2790ff1cb084b99ee193b04bb85bd368"),
    "ramp": ("test_dram_climb/dram_bw.cu", "773272a55d0479d162de815e735172170c797d5c65aabdcbb2b40f3afa545f76"),
}
# Grids in the actual 14 GiB reports under acu_dram_core_ramp_full_2.
RAMP_BLOCKS = (*range(2, 41, 2), 48, 56, 64, 80, 96, 112, 128, 136, 144, 152,
               160, 168, 176, 184, 192, 200, 216, 232, 248, 256)
METRICS = {
    "dram_read_GB_s": "dram__bytes_read.sum.per_second",
    "dram_write_GB_s": "dram__bytes_write.sum.per_second",
    "llc_fabric_read_GB_s": "llc__bytes_load_pipe_fabric.sum.per_second",
    "l2_read_GB_s": "l2__bytes_load_pipe_l1.sum.per_second",
    "llc_hit_pct": "llc__requests_hit_rate.pct",
    "l2_hit_pct": "l2__requests_hit_rate.pct",
}


def cases(suite, grids):
    if suite == "bm-ramp":
        for blocks in grids:
            for op in ("read", "write"):
                yield f"ramp-{op}-{blocks}", "ramp", f"{op}_kernel_u4", [14, 20, op, blocks, 1024, 100]
        return
    for op in ("read", "write", "copy"):
        yield "dram-auto-" + op, "dram", op + "_kernel_u4", [16, 20, op, 0, 512, 100]
    yield "dram-fixed-read", "dram", "read_kernel_u4", [16, 20, "read", 1024, 256, 100]
    yield "dram-512MiB-read", "dram", "read_kernel_u4", [0.5, 20, "read", 1248, 256, 100]
    yield "llc-read", "llc", "llc_read_kernel_u4", [32, 5000, 0]
    yield "l2-read", "l2", "l2_bw_private_kernel", [
        "--blocks", 64, "--threads", 512, "--workset-kb", 100, "--iters", 5000, "--smem-kb", 160]


def kernel_counter_row(path, kernel):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [row for row in csv.DictReader(f) if kernel in (row.get("Kernel Name") or "")]
    if len(rows) != 1:
        raise ValueError(f"{path.name}: expected one raw counter row for {kernel}, got {len(rows)}")
    return rows[0]


def read_counters(path, kernel):
    row = kernel_counter_row(path, kernel)
    values = {}
    for field, metric in METRICS.items():
        raw = row.get(metric)
        if raw and raw.strip() not in ("n/a", "N/A", "--"):
            values[field] = float(raw.replace(",", "")) / (1e9 if field.endswith("GB_s") else 1)
    required = {"llc_read_kernel_u4": "llc_fabric_read_GB_s", "l2_bw_private_kernel": "l2_read_GB_s",
                "write_kernel_u4": "dram_write_GB_s"}.get(kernel, "dram_read_GB_s")
    if required not in values:
        raise ValueError(f"{path.name}: missing {METRICS[required]}")
    return values


def collect(args, out, step, run_logged, acu, save):
    jobs = list(cases(args.suite, args.bm_blocks))
    built = set()
    source_ids = sorted({job[1] for job in jobs})
    save(out / "bm-source.json", {"revision": REVISION, "sources": {i: SOURCES[i] for i in source_ids}})
    for key in source_ids:
        rel, expected = SOURCES[key]
        src = args.bm_root / "BW" / rel
        def build():
            if hashlib.sha256(src.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != expected:
                raise ValueError(f"{src} differs from BM {REVISION}")
            run_logged([args.nvcc, "-O3", "-arch=" + args.arch, src, "-o", out / ("bm-" + key)],
                       out, "build-bm-" + key)
        if step("build-bm-" + key, build):
            built.add(key)

    with (out / "bm-bandwidth.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "kernel", "args", "event_ms", "logical_GB_s", *METRICS])
        writer.writeheader()
        for name, key, kernel, params in jobs:
            if key not in built:
                continue
            cmd = [out / ("bm-" + key), *params]
            row = {"case": name, "kernel": kernel, "args": " ".join(map(str, params))}
            def measure():
                run_logged(cmd, out, name)
                log = (out / (name + ".log")).read_text()
                match = re.search(r"kernel_time=([\d.]+) ms, requested_(?:user|read)_BW=([\d.]+) GB/s", log)
                if not match:
                    raise ValueError(f"{name}: missing BM event time or bandwidth")
                row["event_ms"], row["logical_GB_s"] = map(float, match.groups())
            if step(name, measure) and args.acu:
                def profile():
                    acu(name, cmd)
                    row.update(read_counters(out / ("acu-" + name + ".csv"), kernel))
                step("acu-" + name, profile)
            writer.writerow(row)
            f.flush()
