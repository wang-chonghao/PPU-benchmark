"""Host-only checks; none of these claims SDK compilation or board validation."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import supplement as s


class SupplementalTests(unittest.TestCase):
    def test_plan_ids_unique_and_steps_fit_without_crossing_parameters(self):
        cases = s.plan("all", 39)
        self.assertEqual(len(cases), len({c["id"] for c in cases}))
        for c in cases:
            p = c["params"]
            self.assertEqual(p["threads"] % 32, 0)
            if c["mode"] in ("global-chain", "tsm-chain", "shuffle-chain"):
                self.assertEqual((p["blocks"], p["threads"]), (1, 32))
        smoke = s.plan("smoke", 39)
        ids = {c["id"] for c in cases}
        self.assertTrue(all(c["id"] in ids for c in smoke))
        self.assertEqual(s.build_flags("cubebf16"), ["-DCAL_MATRIX=17"])

    def test_counts_are_per_warp_or_per_packed_element_not_per_thread_flops(self):
        c = dict(mode="cube", binary="cube8", params=dict(blocks=2, threads=64, slots=4))
        self.assertEqual(s.work_per_step(c), dict(compat_fragments=16, source_flops=16 * 8192))
        c = dict(mode="vector", binary="vector16", params=dict(blocks=2, threads=256, slots=4, active=199))
        self.assertEqual(s.work_per_step(c), dict(valid_elements=1592, source_elements=1600))
        c = dict(mode="tsm-mix", binary="memory", params=dict(blocks=1, threads=256, width=4, read_warps=2))
        self.assertEqual(s.work_per_step(c), dict(read_bytes=1024, write_bytes=3072))

    def test_slope_cancels_fixed_launch_overhead(self):
        slope, intercept, r2 = s.linear_fit([(256, 3 + 256 * .02), (1024, 3 + 1024 * .02), (4096, 3 + 4096 * .02)])
        self.assertAlmostEqual(slope, .02)
        self.assertAlmostEqual(intercept, 3)
        self.assertAlmostEqual(r2, 1)
        with self.assertRaises(ValueError):
            s.linear_fit([(1, 1), (2, 2)])

    def test_analysis_units_and_no_automatic_hardware_claim(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [c for c in s.plan("smoke", 39) if c["mode"] == "tsm-mix" and c["params"]["read_warps"] == 4]
            s.save(root / "manifest.json", dict(cases=cases, device=dict(cu_count=39),
                                                steps={c["id"]: {"status": "ok"} for c in cases}))
            with (root / "samples.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["case_id", "sample", "us_per_launch", "local_bytes"])
                w.writeheader()
                for c in cases:
                    for sample in range(7):
                        w.writerow(dict(case_id=c["id"], sample=sample,
                                        us_per_launch=3 + c["params"]["steps"] * .02, local_bytes=0))
            args = SimpleNamespace(results=root, out=None, core_mhz=1000, frequency_note="synthetic fixture")
            self.assertEqual(s.analyze(args), 0)
            record, = json.loads((root / "analysis.json").read_text())
            self.assertAlmostEqual(record["cycles_per_step"], 20)
            self.assertEqual(record["rate_normalization_cus"], 1)
            self.assertAlmostEqual(record["source_work_per_cu_cycle"]["read_bytes"], 102.4)
            self.assertIn("not_a_calibrated_parameter", record["flags"])
            with self.assertRaises(ValueError):
                s.analyze(args)

    def test_profile_uses_same_parameters_and_binary(self):
        case = s.plan("smoke", 39)[0]
        normal = s.command_for("/local/memory", case)
        self.assertEqual(s.command_for("/local/memory", case, True), normal + ["--profile", "1"])


if __name__ == "__main__":
    unittest.main()
