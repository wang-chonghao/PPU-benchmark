"""Host-only checks for units, artifact identity and offline result packaging."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from zipfile import ZipFile

import run


class RunnerTest(unittest.TestCase):
    def test_independent_ops_are_counted_once_per_launch(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "independent.csv"
            path.write_text("kind,operation,target_ops_per_launch,flop_per_op,sample,repetitions,us_per_launch\n"
                            "register,bf16,128,4096,0,8,2\narray,exp2f,2048,0,0,8,4\n")
            mma, array = run.summarize(path)
            self.assertEqual(mma["target_ops_per_second"], 64e6)
            self.assertAlmostEqual(mma["TFLOP_s"], 128 * 4096 / 2e6)
            self.assertEqual(array["target_ops_per_second"], 512e6)
            self.assertEqual(array["logical_total_GB_s"], 4.096)
            self.assertNotIn("TFLOP_s", array)

    def test_copy_counts_both_directions(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream.csv"
            path.write_text("kind,sample,logical_read_bytes,logical_write_bytes,us_per_launch\n"
                            "stream,0,2000,2000,1\nstream,1,2000,2000,3\n")
            row, = run.summarize(path)
            self.assertEqual(row["median_us"], 2)
            self.assertEqual(row["logical_total_GB_s"], 2)

    def test_mma_is_per_warp(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "mma.csv"
            path.write_text("kind,operation,blocks,threads,chains,steps,sample,us_per_launch\n"
                            "mma,fp8,2,64,4,32,0,8\n")
            row, = run.summarize(path)
            self.assertAlmostEqual(row["issued_fragment_TFLOP_s"], 4 * 4 * 32 * 8192 / 8e6)

    def test_replay_rejects_mismatched_ir(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / "board" / "vector"
            case.mkdir(parents=True)
            (case / "device.json").write_text("{}")
            run.save(case / "hardware.json", dict(case="vector", check="exact_cpu_reference_passed", tir_sha256="wrong"))
            out = root / "replay"
            self.assertEqual(run.replay(SimpleNamespace(results=case.parent, out=out)), 1)
            self.assertIn("hashes", json.loads((out / "comparison.json").read_text())[0]["diagnostic"])

    def test_pack_keeps_partial_results(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "board"
            (root / "vector" / "jit-cache").mkdir(parents=True)
            (root / "vector" / "jit-cache" / "object").write_text("cached")
            run.save(root / "manifest.json", {"steps": {"vector": {"status": "failed"}}})
            (root / "vector.stderr").write_text("diagnostic")
            dest = root.parent / "partial.zip"
            run.pack(SimpleNamespace(results=root, archive=dest))
            with ZipFile(dest) as archive:
                self.assertEqual(set(archive.namelist()), {"manifest.json", "vector.stderr"})


if __name__ == "__main__":
    unittest.main()
