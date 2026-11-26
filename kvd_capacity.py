"""Compare the KVD working-set curve across dynamic TSM sizes and cache preferences."""
import csv
import json
import math
from pathlib import Path

from bm_bw import kernel_counter_row

METRICS = {
    "dynamic_tsm_bytes": "launch__shared_mem_per_block_dynamic",
    "kvd_load_hit_pct": "kvd__requests_load_hit_rate.pct",
    "kvd_load_requests": "kvd__requests_pipe_lsu_mem_global_op_ld.sum",
    "l2_load_transactions_from_l1": "l2__transactions_pipe_l1_mem_global_op_ld.sum",
}


def measurements(path, tsm_kib):
    raw = kernel_counter_row(path, "kvd_capacity_read")
    result = {}
    for name, metric in METRICS.items():
        try:
            value = float(raw[metric].replace(",", ""))
            if not math.isfinite(value) or value < 0:
                raise ValueError
        except (KeyError, AttributeError, ValueError) as exc:
            raise ValueError(f"{path.name}: missing or invalid {metric}") from exc
        result[name] = value
    if result["dynamic_tsm_bytes"] != tsm_kib * 1024:
        raise ValueError(f"{path.name}: ACU dynamic shared size differs from requested {tsm_kib} KiB")
    if result["kvd_load_requests"] == 0 or result["kvd_load_hit_pct"] > 100:
        raise ValueError(f"{path.name}: invalid KVD load counters")
    return result


def collect(args, out, step, run_logged, acu):
    exe = out / "kvd-capacity"
    source = Path(__file__).with_suffix(".cu")
    build = [args.nvcc, "-O3", "-std=c++17", "-arch=" + args.arch, source, "-o", exe]
    if not step("build-kvd-capacity", lambda: run_logged(build, out, "build-kvd-capacity")):
        return
    with (out / "kvd-capacity.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tsm_kib", "workset_kib", "preference", "status",
                                               *METRICS, "hit_delta_from_default_pp"])
        writer.writeheader()
        for tsm in dict.fromkeys(args.tsm_kib):
            for ws in dict.fromkeys(args.kvd_kib):
                baseline = None
                for preference in ("default", "prefer_shared"):
                    name = f"kvd-{tsm}-{ws}-{preference}"
                    check = out / (name + "-check.json")
                    row = dict(tsm_kib=tsm, workset_kib=ws, preference=preference)

                    def profile():
                        acu(name, [exe, preference, tsm, ws, check], metrics=METRICS.values())
                        if json.loads(check.read_text()).get("checksum") != "passed":
                            raise ValueError("device result did not pass the CPU reference")
                        row.update(measurements(out / ("acu-" + name + ".csv"), tsm))

                    ok = step(name, profile)
                    row["status"] = "ok" if ok else "failed"
                    if ok and preference == "default":
                        baseline = row["kvd_load_hit_pct"]
                    elif ok and baseline is not None:
                        row["hit_delta_from_default_pp"] = row["kvd_load_hit_pct"] - baseline
                    writer.writerow(row)
                    f.flush()
