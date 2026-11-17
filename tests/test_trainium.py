"""Trainium (hardware steps 6 to 8): AWS Trn1 (calibrated), Trn2 and Trn3 (uncalibrated).

    python3 -m pytest tests/test_trainium.py

The link-model change (link tiers for Trainium) must leave every GPU result as it was: Part 1 proves it against a copy of the old
function and the existing pins. Neuron rules are tied to a Neuron SDK version in the code comments.
"""
from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import mfu_model as mm  # noqa: E402
from utils import phase9_mfu_estimate as p9  # noqa: E402
from utils import validation_runs as vr  # noqa: E402
from utils.architecture_loader import load_variant_from_hf_config  # noqa: E402

DOC_MFU = Path(__file__).parent.parent / "docs" / "CALCULATOR-MFU.md"
DOC_HW = Path(__file__).parent.parent / "docs" / "CALCULATOR-HARDWARE.md"


def preset(name):
    from configuration.variants_config import VARIANTS
    return next(v for v in VARIANTS if v["name"] == name)


def run(model, **config):
    with redirect_stdout(io.StringIO()):
        return p9.run_estimate(config, preset(model))


def read(path):
    return open(path).read() if not hasattr(path, "read_text") else path.read_text()


def old_span_bw(hw, inner, group):
    """The link model of main before the Trainium tiers (copied): two tiers, NVLink inside a node, min(NVLink, r x NIC) across."""
    if inner * group <= hw["gpn"]:
        return hw["nvlink"] * mm.LINK_EFFICIENCY["nvlink"] * 1e9
    r = max(1, hw["gpn"] // inner)
    return min(hw["nvlink"] * mm.LINK_EFFICIENCY["nvlink"], r * hw["net"] * mm.LINK_EFFICIENCY["net"]) * 1e9


class LinkModelLeavesGpusUnchanged(unittest.TestCase):
    """Step 6, part 1: the link-tier change must not move a GPU number. The pins of test_hardware_support.py (p5en, B200, B300, GB200,
    GB300), the 38-run summary and parity.py are checked by their own tests; this proves the function itself."""

    def test_every_gpu_system_uses_the_old_function_result(self):
        gpu_systems = {k: hw for k, hw in mm.HARDWARE.items() if not mm.is_trainium(hw)}
        self.assertGreaterEqual(len(gpu_systems), 18)
        for key, hw in gpu_systems.items():
            self.assertNotIn("tiers", hw, key)
            for inner in (1, 2, 4, 8, 16, 72):
                for group in (1, 2, 3, 4, 8, 9, 16, 36, 72, 144, 256):
                    self.assertEqual(mm._span_bw(hw, inner, group), old_span_bw(hw, inner, group), (key, inner, group))

    def test_the_38_run_summary_and_the_gpu_rows_are_unchanged(self):
        s = vr.summary(vr.evaluate())
        self.assertEqual((s["all"]["runs"], s["all"]["median_abs_error_pct"], s["all"]["max_abs_error_pct"]), (38, 6.1, 58.9))
        self.assertEqual((s["hold_out"]["median_abs_error_pct"], s["fit"]["median_abs_error_pct"]), (7.4, 5.8))
        self.assertEqual(vr.rack_summary(vr.evaluate(vr.RACK_RUNS)), {"runs": 8, "median_abs_error_pct": 10.4, "max_abs_error_pct": 82.3})
        hold = vr.pair_summary([r for r in vr.evaluate_pairs(vr.ALL_GB300_PAIRS) if not r["fit"]])
        self.assertEqual((hold["median_abs_error_pct"], hold["max_abs_error_pct"]), (3.8, 7.1))

    def test_the_fit_of_the_38_runs_does_not_include_a_trainium_row(self):
        self.assertFalse({r["id"] for r in vr.TRN1_RUNS} & {r["id"] for r in vr.RUNS})
        self.assertFalse({r["id"] for r in vr.TRN1_RUNS} & set(vr.fit()["fit_runs"]))
        for key in vr.fit()["base_efficiency"]:
            self.assertFalse(key.startswith("trainium"), key)

    def test_candidate_plans_of_the_gpus_are_unchanged(self):
        self.assertEqual(mm.candidate_plans(64, 32, 0, 8, 4096), mm.candidate_plans(64, 32, 0, 8, 4096, tp_options=(1, 2, 4, 8)))
        self.assertEqual({c["tp"] for c in mm.candidate_plans(64, 32, 0, 8, 4096)}, {1, 2, 4, 8})
        self.assertEqual(mm.auto_gpu_counts(mm.HARDWARE["p6-b300.48xlarge"], 8, 100), [8, 16, 32, 64])


class Trainium1Hardware(unittest.TestCase):
    """Step 6: Trn1 and Trn1n. The specs are those of the Neuron and AWS pages cited in mfu_model.HARDWARE."""

    def test_specs(self):
        for key, net in (("trn1.32xlarge", 3.125), ("trn1n.32xlarge", 6.25)):
            hw = mm.HARDWARE[key]
            self.assertEqual((hw["family"], hw["ranks_per_chip"], hw["bf16"], hw["fp8"]), ("trainium1", 2, 95.0, None))
            self.assertEqual((hw["domain_gpus"], hw["instance_gpus"], hw["billing_gpus"], hw["gpn"]), (16, 16, 16, 32))
            self.assertEqual(hw["net"], net)  # 800 / 1600 Gbps per instance / 32 NeuronCores
            self.assertEqual(hw["tiers"], ((32, 96.0, "torus"),))  # NeuronLink-v2 384 GB/s per chip / 2 / 2 cores
            self.assertIn("no hold-out", hw["calibration"])
        self.assertEqual(mm.HARDWARE["trn1.32xlarge"]["bf16"] * 2, 190.0)  # 190 BF16 TFLOPS per chip

    def test_link_tiers_torus_inside_the_instance_efa_across(self):
        hw = mm.HARDWARE["trn1.32xlarge"]
        torus = 96.0 * mm.LINK_EFFICIENCY["torus"] * 1e9
        for inner, group in ((1, 32), (1, 2), (8, 4), (16, 2)):
            self.assertEqual(mm._span_bw(hw, inner, group), torus, (inner, group))
        # two instances: a group of 2 ranks 32 apart has one rank per instance, one NeuronCore share of the EFA
        self.assertEqual(mm._span_bw(hw, 32, 2), 3.125 * mm.LINK_EFFICIENCY["net"] * 1e9)
        # 32 ranks per instance share the instance's adapters: more than the torus, so the torus limits
        self.assertEqual(mm._span_bw(hw, 1, 64), torus)

    def test_the_two_neuron_rows_are_the_only_fit_and_have_no_hold_out(self):
        self.assertEqual(len(vr.TRN1_RUNS), 2)
        self.assertTrue(all(r["fit"] and r["hardware"] == "trn1.32xlarge" and r["precision"] == "bf16" for r in vr.TRN1_RUNS))
        self.assertEqual({r["plan"]["num_gpus"] for r in vr.TRN1_RUNS}, {1024})  # 32 x 32 NeuronCores
        self.assertEqual([r["reported"] for r in vr.TRN1_RUNS], [383.6, 63.52])  # seq/s x 8192 / 1024 ranks
        self.assertEqual(vr.fit_trainium1(), mm.BASE_EFFICIENCY[("trainium1", "bf16")])
        self.assertEqual(vr.fit()["trainium1_bf16"], 0.33)

    def test_error_of_the_two_rows_is_pinned_and_in_sample(self):
        rows = vr.evaluate(vr.TRN1_RUNS)
        self.assertEqual([r["error_pct"] for r in rows], [13.5, -12.6])
        s = vr.trn1_summary(rows)
        self.assertEqual((s["median_abs_error_pct"], s["max_abs_error_pct"]), (13.1, 13.5))
        v = mm.VALIDATION_SUMMARY
        self.assertEqual((v["trainium1_rows"], v["trainium1_median_abs_error_pct"], v["trainium1_max_abs_error_pct"]), (2, 13.1, 13.5))

    def test_fp8_is_refused_by_the_model(self):
        shape = mm.model_shape(vr.ARCH["Llama-3-8B"])
        plan = dict(num_gpus=1024, tp=32, gbs=1024, mbs=1, seq=8192)
        with self.assertRaises(ValueError) as cm:
            mm.estimate(shape, mm.HARDWARE["trn1.32xlarge"], plan, "fp8")
        self.assertIn("not sized", str(cm.exception))
        self.assertGreater(mm.estimate(shape, mm.HARDWARE["trn1.32xlarge"], plan, "bf16")["mfu"], 0.2)

    def test_tp_up_to_32_neuroncores_only_for_trainium(self):
        trn = {c["tp"] for c in mm.candidate_plans(1024, 80, 0, 32, 8192, tp_options=(1, 2, 4, 8, 16, 32))}
        self.assertEqual(trn, {1, 2, 4, 8, 16, 32})
        self.assertEqual(mm.auto_gpu_counts(mm.HARDWARE["trn1.32xlarge"], 16, 200), [16, 32, 64, 128])
        self.assertEqual(mm.rack_alternatives(mm.HARDWARE["trn1.32xlarge"], 48), [])  # no 64-of-72 layouts

    def test_docs_table_is_current(self):
        rows = vr.evaluate(vr.TRN1_RUNS)
        self.assertIn(vr.markdown(rows), read(DOC_MFU))


class Trainium1Phase9(unittest.TestCase):
    def test_names(self):
        for given, want in (("trn1", "trn1.32xlarge"), ("Trn1.32xlarge", "trn1.32xlarge"), ("Trainium1", "trn1.32xlarge"),
                            ("trn1n", "trn1n.32xlarge"), ("trn1n.32xlarge", "trn1n.32xlarge"), ("AWS Trn1", "trn1.32xlarge")):
            self.assertEqual(p9.normalise_hardware(given), want, given)

    def test_the_family_name_is_refused_with_the_choices_not_guessed(self):
        with self.assertRaises(ValueError) as cm:
            p9.normalise_hardware("Trainium")
        self.assertIn("trn1", str(cm.exception))
        self.assertIn("trn3", str(cm.exception))
        for given in ("nd96p", "inferentia2", "tpu-v5"):
            with self.assertRaises(ValueError, msg=given):
                p9.normalise_hardware(given)

    def test_chips_in_the_request_neuroncores_in_the_plan(self):
        b = run("Llama-3.1-70B", hardware="trn1", num_gpus=512, total_tokens_b=1000, sequence_length=8192,
                tensor_parallel=32, pipeline_parallel=8)
        self.assertNotIn("error", b, b)
        p, h = b["parallelism"], b["hardware"]
        self.assertEqual((p["total_gpus"], p["total_ranks"], p["num_nodes"]), (512, 1024, 32))
        self.assertEqual((p["tensor_parallel"], p["pipeline_parallel"], p["data_parallel"]), (32, 8, 4))
        self.assertEqual((h["chips"], h["ranks"], h["ranks_per_chip"], h["gpus_per_node"]), (512, 1024, 2, 16))
        self.assertEqual(b["memory_breakdown"]["gpu_capacity_gb"], 16)
        self.assertEqual(p["zero_stage"], 1)

    def test_the_published_plan_fits_and_the_estimate_is_near_the_published_run(self):
        b = run("Llama-3.1-70B", hardware="trn1", num_gpus=512, total_tokens_b=1000, sequence_length=8192,
                tensor_parallel=32, pipeline_parallel=8, global_batch_size=1024, micro_batch_size=1)
        self.assertTrue(b["fits_in_memory"])
        self.assertAlmostEqual(b["training_time"]["tokens_per_second"] / (7.94 * 8192), 1.0, delta=0.15)

    def test_the_calibration_label_is_in_the_result_and_the_disclaimer(self):
        b = run("Llama-3.1-8B", hardware="trn1", num_gpus=32, total_tokens_b=100)
        self.assertIn("no hold-out", b["hardware"]["calibration"])
        self.assertIn("Calibration: calibrated on the 2 published Neuron Trn1 rows", b["disclaimer"])
        self.assertNotIn("Calibration:", run("Llama-3.1-8B", hardware="B300", num_gpus=8, total_tokens_b=100)["disclaimer"])

    def test_price_is_the_aws_trn1_page_and_billing_is_per_instance(self):
        b = run("Llama-3.1-8B", hardware="trn1", num_gpus=40, total_tokens_b=100)
        self.assertEqual(b["parallelism"]["num_nodes"], 3)
        self.assertEqual(b["hardware"]["price_per_instance_hour_usd"], 21.5)
        self.assertEqual(run("Llama-3.1-8B", hardware="trn1n", num_gpus=16, total_tokens_b=100)["hardware"]["price_per_instance_hour_usd"], 24.78)

    def test_fp8_is_refused_with_the_reason(self):
        b = run("Llama-3.1-8B", hardware="trn1", num_gpus=16, total_tokens_b=100, precision="fp8")
        self.assertIn("FP8 / MXFP8 training on Trainium is not sized", b["error"])
        self.assertNotIn("error", run("Llama-3.1-8B", hardware="trn1", num_gpus=16, total_tokens_b=100, precision="bf16"))

    def test_plan_is_dp_tp_pp_zero1_only(self):
        b = run("Llama-3.1-70B", hardware="trn1", num_gpus=512, total_tokens_b=1000, sequence_length=8192)
        p = b["parallelism"]
        self.assertEqual((p["zero_stage"], p["expert_parallel"]), (1, 1))
        self.assertLessEqual(p["tensor_parallel"], 32)
        self.assertIn("error", run("Qwen3-30B-A3B", hardware="trn1", num_gpus=512, total_tokens_b=1000, expert_parallel=8))
        moe = run("Qwen3-30B-A3B", hardware="trn1", num_gpus=512, total_tokens_b=1000)
        self.assertEqual(moe["parallelism"]["expert_parallel"], 1)

    def test_a_model_that_does_not_fit_is_oom_not_moved_to_zero_2_or_3(self):
        b = run("Llama-3.1-8B", hardware="trn1", num_gpus=16, total_tokens_b=100)
        self.assertFalse(b["fits_in_memory"])
        self.assertEqual(b["parallelism"]["zero_stage"], -1)

    def test_bf16_sr_memory_mode_needs_less_memory(self):
        kw = dict(hardware="trn1", num_gpus=512, total_tokens_b=1000, sequence_length=8192, tensor_parallel=32, pipeline_parallel=8)
        a = run("Llama-3.1-70B", **kw)
        s = run("Llama-3.1-70B", **kw, neuron_precision="bf16_sr")
        self.assertEqual((a["hardware"]["memory_mode"], s["hardware"]["memory_mode"]), ("bf16_fp32_optimizer", "bf16_sr"))
        self.assertLess(s["memory_breakdown"]["gradients_gb"], a["memory_breakdown"]["gradients_gb"])
        self.assertLess(s["memory_breakdown"]["optimizer_states_gb"], a["memory_breakdown"]["optimizer_states_gb"])
        self.assertEqual(s["training_time"]["tokens_per_second"], a["training_time"]["tokens_per_second"])
        self.assertIn("Unknown neuron_precision", run("Llama-3.1-70B", **kw, neuron_precision="fp32")["error"])

    def test_memory_mode_does_not_leak_into_the_next_gpu_request(self):
        kw = dict(hardware="B300", num_gpus=8, total_tokens_b=100)
        before = run("Llama-3.1-8B", **kw)
        run("Llama-3.1-70B", hardware="trn1", num_gpus=512, total_tokens_b=1000, tensor_parallel=32, pipeline_parallel=8, neuron_precision="bf16_sr")
        self.assertEqual(before["memory_breakdown"], run("Llama-3.1-8B", **kw)["memory_breakdown"])

    def test_auto_search_counts_chips_in_whole_instances(self):
        b = run("Llama-3.1-8B", hardware="trn1", total_tokens_b=100, num_gpus="auto", target_days=30)
        counts = [t["num_gpus"] for t in b["gpu_count_search"]["tried"]]
        self.assertEqual(counts[:2], [16, 32])
        self.assertIn("16-chip instances", b["gpu_count_search"]["basis"])
        self.assertLessEqual(b["training_time"]["training_days"], 30)


if __name__ == "__main__":
    unittest.main()
