"""Host-only checks; numeric helpers additionally run when PyTorch is installed."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import deepseek_v41 as bench


class SourceContract(unittest.TestCase):
    def test_pinned_source_and_complete_kernel_inventory(self):
        source = bench.check_source()
        self.assertTrue(source["files_unmodified"])
        self.assertEqual({c["family"] for c in bench.cases("all")}, bench.KERNELS)

    def test_shapes_and_names(self):
        cases = bench.cases("all")
        self.assertEqual(len(cases), len({c["name"] for c in cases}))
        old = {c["name"]: c for c in cases}
        self.assertEqual(old["gemm_fp4_expert"]["shapes"], dict(M=32, N=2304, K=5120))
        self.assertEqual(old["sparse_compressed"]["shapes"]["topk"], 640)
        for case in cases:
            s, o = case["shapes"], case["options"]
            self.assertTrue(all(x > 0 for x in s.values()))
            if "group" in o:
                self.assertEqual(s.get("K", s["N"]) % o["group"], 0)
            if "gemm" in case["family"]:
                self.assertEqual(s["N"] % 128, 0)


class ResultIntegrity(unittest.TestCase):
    def test_zero_exit_without_result_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "run"
            argv = ["deepseek_v41.py", "run", str(out), "--cases", "gemm_fp8"]
            with patch.object(sys, "argv", argv), patch.object(bench.subprocess, "run",
                    return_value=subprocess.CompletedProcess([], 0)):
                with self.assertRaises(SystemExit):
                    bench.main()
            self.assertEqual(json.loads((out / "manifest.json").read_text())[0]["status"], "incomplete")

    def test_acu_must_match_timed_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "run"

            def fake_run(cmd, **kwargs):
                if cmd[0] == "acu":
                    # Find the child Python case command following ACU options.
                    child = cmd[cmd.index("case"):]
                    dest = Path(child[1])
                    dest.mkdir()
                    (dest / "device.json").write_text("different compiled IR")
                else:
                    dest = Path(cmd[3])
                    dest.mkdir()
                    (dest / "device.json").write_text("timed compiled IR")
                    bench.save(dest / "result.json", dict(status="passed", hot_median_us=1., after_scrub_median_us=2.))
                return subprocess.CompletedProcess(cmd, 0)

            argv = ["deepseek_v41.py", "run", str(out), "--cases", "gemm_fp8", "--acu"]
            with patch.object(sys, "argv", argv), patch.object(bench.subprocess, "run", side_effect=fake_run):
                with self.assertRaises(SystemExit):
                    bench.main()
            row = json.loads((out / "manifest.json").read_text())[0]
            self.assertEqual(row["status"], "timing_passed_profile_failed")
            self.assertIn("differs", json.loads((out / "gemm_fp8-acu.status.json").read_text())["error"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires CPU PyTorch")
class NumericContracts(unittest.TestCase):
    def test_fp4_round_ties_and_pack(self):
        import torch
        from deepseek_v41_data import fp4_round, pack_fp4, unpack_fp4
        x = torch.tensor([[.25, .75, 1.25, 1.75, 2.5, 3.5, 5., -5.]])
        values, codes = fp4_round(x)
        torch.testing.assert_close(values, torch.tensor([[0., 1., 1., 2., 2., 4., 4., -4.]]))
        torch.testing.assert_close(unpack_fp4(pack_fp4(codes)), values)

    def test_scale_raw_bytes(self):
        import torch
        from deepseek_v41_data import scale_tensor, decode_scale
        x = torch.tensor([[2.**-126, .5, 1., 2., 128.]])
        encoded = scale_tensor(x, "float8_e8m0fnu")
        self.assertEqual(encoded.view(torch.uint8).tolist(), [[1, 126, 127, 128, 134]])
        torch.testing.assert_close(decode_scale(encoded, "float8_e8m0fnu"), x, rtol=0, atol=0)

    def test_invalid_attention_is_zero(self):
        import torch
        from deepseek_v41_data import make_data
        case = next(c for c in bench.cases("model") if c["name"] == "attention_all_invalid")
        _, _, expected = make_data(case, 7)
        torch.testing.assert_close(expected[0], torch.zeros_like(expected[0]), rtol=0, atol=0)

    def test_output_count_must_match(self):
        from deepseek_v41_data import check_outputs
        case = next(c for c in bench.cases("smoke") if c["name"] == "gemm_fp8")
        with self.assertRaises(ValueError):
            check_outputs(case, [], [])

    def test_quantized_buffers_can_be_archived_without_conversion(self):
        import torch
        from deepseek_v41_data import make_data
        names = {"act_fp4_packed", "gemm_fp4", "act_fp8"}
        with tempfile.TemporaryDirectory() as folder:
            for case in bench.cases("smoke"):
                if case["name"] not in names:
                    continue
                tensors, ids, expected = make_data(case, 7)
                path = Path(folder) / (case["name"] + ".pt")
                torch.save(tensors, path)
                restored = torch.load(path, weights_only=True)
                for before, after in zip(tensors, restored):
                    self.assertEqual(before.dtype, after.dtype)
                    torch.testing.assert_close(before.view(torch.uint8), after.view(torch.uint8), rtol=0, atol=0)
                self.assertTrue(all(bool(torch.isfinite(x).all()) for x in expected))


if __name__ == "__main__":
    unittest.main()
