"""mfu_model: generic config parsing, the validation set of published runs, and Phase 9.

    python3 -m pytest tests/test_mfu_model.py

The validation set (validation_runs.py) pins the model's accuracy: a change that makes any
published run worse than the bounds below fails here.
"""
from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import mfu_model as mm  # noqa: E402
from utils import validation_runs as vr  # noqa: E402
from utils.architecture_loader import load_variant_from_hf_config  # noqa: E402
from utils import phase9_mfu_estimate as p9  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"

# Runs allowed above the 30% bound, each with its reason (see docs/CALCULATOR-MFU.md).
EXPLAINED_OUTLIERS = {
    "deepseek-v3-paper": 35.0,  # GPU-hours include batch warm-up and DeepSeek's own kernels; DualPipe approximated
}


def preset(name):
    from configuration.variants_config import VARIANTS
    return next(v for v in VARIANTS if v["name"] == name)


def call(model, **config):
    """Phase 9 for a catalog variant; config keys as run_estimate."""
    with redirect_stdout(io.StringIO()):
        return p9.run_estimate(config, preset(model))


class GenericParser(unittest.TestCase):
    def test_every_validation_architecture_matches_its_published_size(self):
        for name, cfg in vr.ARCH.items():
            s = mm.model_shape(cfg)
            total, active = vr.PUBLISHED_PARAMS[name]
            self.assertAlmostEqual(s["total_b"] / total, 1, delta=0.02, msg=name)
            self.assertAlmostEqual(s["active_b"] / active, 1, delta=0.03, msg=name)

    def test_attention_and_ffn_kinds(self):
        kinds = {n: mm.model_shape(c)["attention"]["kind"] for n, c in vr.ARCH.items()}
        self.assertEqual(kinds["DeepSeek-V3"], "MLA")
        self.assertEqual(kinds["GPT-3-175B"], "MHA")
        self.assertEqual(kinds["Llama-3-70B"], "GQA")
        self.assertEqual(mm.model_shape({"num_hidden_layers": 2, "hidden_size": 256, "num_attention_heads": 4,
                                         "num_key_value_heads": 1})["attention"]["kind"], "MQA")
        self.assertEqual(mm.ffn_matrices(vr.ARCH["GPT-3-175B"]), 2)        # GELU, plain
        self.assertEqual(mm.ffn_matrices(vr.ARCH["Nemotron-4-340B"]), 2)   # ReLU^2, plain
        self.assertEqual(mm.ffn_matrices(vr.ARCH["Llama-3-70B"]), 3)       # SwiGLU

    def test_moe_layer_placement(self):
        base = {"num_hidden_layers": 48, "hidden_size": 1024, "num_experts": 8, "num_experts_per_tok": 2}
        self.assertEqual(mm.model_shape(base)["moe_layers"], 48)
        self.assertEqual(mm.model_shape({**base, "decoder_sparse_step": 2})["moe_layers"], 24)
        self.assertEqual(mm.model_shape({**base, "first_k_dense_replace": 3})["moe_layers"], 45)
        self.assertEqual(mm.model_shape({**base, "moe_layers": 40})["moe_layers"], 40)


class Formulas(unittest.TestCase):
    def test_bubble_is_pp_minus_one_over_vp_m_plus_pp_minus_one(self):
        shape = mm.model_shape(vr.ARCH["Llama-3-8B"])
        est = mm.estimate(shape, mm.HARDWARE["p5en.48xlarge"],
                          {"num_gpus": 32, "tp": 1, "pp": 4, "vp": 1, "mbs": 1, "seq": 4096, "gbs": 96})
        self.assertEqual(est["plan"]["micro_batches"], 12)
        self.assertAlmostEqual(est["bubble_fraction"], round(3 / 15, 4))
        est2 = mm.estimate(shape, mm.HARDWARE["p5en.48xlarge"],
                           {"num_gpus": 32, "tp": 1, "pp": 4, "vp": 2, "mbs": 1, "seq": 4096, "gbs": 96})
        self.assertAlmostEqual(est2["bubble_fraction"], round(3 / 27, 4))

    def test_default_global_batch_is_16m_tokens(self):
        self.assertEqual(mm.default_global_batch(4096, 8, 1), 4096)
        self.assertEqual(mm.default_global_batch(8192, 64, 1), 2048)

    def test_flops_per_token_palm_convention(self):
        s = mm.model_shape(vr.ARCH["Llama-3-70B"])
        f = mm.flops_per_token(s, 8192)
        a = s["attention"]
        self.assertAlmostEqual(f["attention"], 12 * a["heads"] * 128 * 8192 * 80, delta=1)


class Validation(unittest.TestCase):
    """Predicted vs published, per run and overall (median and max absolute % error)."""

    @classmethod
    def setUpClass(cls):
        cls.rows = vr.evaluate()
        cls.summary = vr.summary(cls.rows)

    def test_set_covers_many_families(self):
        families = {r["model"].split("-")[0] for r in vr.RUNS}
        self.assertGreaterEqual(len(families), 6, families)
        self.assertGreaterEqual(len(vr.RUNS), 25)
        self.assertTrue(all(r["source"] for r in vr.RUNS))
        self.assertGreater(sum(1 for r in vr.RUNS if not r["fit"]), sum(1 for r in vr.RUNS if r["fit"]))

    def test_median_error_within_15_percent(self):
        self.assertLessEqual(self.summary["all"]["median_abs_error_pct"], 15)
        self.assertLessEqual(self.summary["hold_out"]["median_abs_error_pct"], 15)

    def test_no_run_off_by_more_than_30_percent_unless_explained(self):
        for r in self.rows:
            bound = EXPLAINED_OUTLIERS.get(r["id"], 30.0)
            self.assertLessEqual(abs(r["error_pct"]), bound, r)

    def test_constants_are_the_fit_of_the_fit_runs(self):
        fit = vr.fit()
        for key, value in fit["base_efficiency"].items():
            fam, prec = key.split("/")
            self.assertEqual(mm.BASE_EFFICIENCY[(fam, prec)], value, key)
        self.assertEqual(mm.MOE_EXPERT_FACTOR, fit["moe_expert_factor"])
        self.assertEqual(mm.MOE_EXPERT_HALF_FFN, fit["moe_expert_half_ffn"])

    def test_pinned_summary_is_current(self):
        s = self.summary
        self.assertEqual((mm.VALIDATION_SUMMARY["runs"], mm.VALIDATION_SUMMARY["hold_out_runs"]),
                         (s["all"]["runs"], s["hold_out"]["runs"]))
        self.assertEqual(mm.VALIDATION_SUMMARY["median_abs_error_pct"], s["all"]["median_abs_error_pct"])
        self.assertEqual(mm.VALIDATION_SUMMARY["max_abs_error_pct"], s["all"]["max_abs_error_pct"])
        self.assertEqual(mm.VALIDATION_SUMMARY["hold_out_median_abs_error_pct"], s["hold_out"]["median_abs_error_pct"])

    def test_docs_table_is_current(self):
        doc = (Path(__file__).parent.parent / "docs" / "CALCULATOR-MFU.md").read_text()
        self.assertIn(vr.markdown(self.rows), doc)

    def test_known_runs_llama3_405b_and_deepseek_v3(self):
        by_id = {r["id"]: r for r in self.rows}
        for rid in ("meta-llama3-405b-8k-gpus", "meta-llama3-405b-16k-gpus", "deepseek-v3-paper",
                    "nemo-h100-deepseek-v3", "nemo-h100-llama31-405b"):
            self.assertIn(rid, by_id)


class Phase9UsesTheModel(unittest.TestCase):
    """These fail on the old calculator (fixed 40% MFU, no plan search, cost the same at every GPU count)."""

    def test_mfu_is_estimated_not_a_fixed_40_percent(self):
        b = call("Llama-3.1-70B", hardware="p5en", total_tokens_b=1000, num_gpus=64)
        self.assertIn("mfu_breakdown", b)
        self.assertNotEqual(b["training_time"]["mfu"], 0.4)
        self.assertEqual(b["inputs"]["mfu_source"], "estimated (mfu_model)")
        self.assertAlmostEqual(b["training_time"]["mfu"], b["mfu_breakdown"]["mfu_estimated"], places=4)

    def test_bigger_plans_can_cost_more(self):
        small = call("Llama-3.1-70B", hardware="p5en", total_tokens_b=15000, num_gpus=64)
        big = call("Llama-3.1-70B", hardware="p5en", total_tokens_b=15000, num_gpus=4096)
        self.assertGreater(big["cost_estimate"]["total_cost_usd"], small["cost_estimate"]["total_cost_usd"])
        self.assertLess(big["training_time"]["mfu"], small["training_time"]["mfu"])

    def test_bubble_uses_a_realistic_batch(self):
        b = call("Llama-3.1-405B", hardware="p5en", total_tokens_b=15000, num_gpus=1024)
        self.assertGreaterEqual(b["parallelism"]["global_batch_tokens"], 16 * 1024 * 1024 * 0.9)
        if b["parallelism"]["pipeline_parallel"] > 1:
            self.assertLess(b["pipeline_communication"]["pipeline_bubble_pct"], 25)

    def test_days_follow_throughput_and_goodput(self):
        b = call("Llama-3.1-8B", hardware="p5en", total_tokens_b=1000, num_gpus=64)
        t = b["training_time"]
        days = 1000e9 / (t["tokens_per_second"] * mm.GOODPUT) / 86400
        self.assertAlmostEqual(t["training_days"], round(days, 1), delta=0.11)

    def test_every_gpu_is_used_and_plan_divides(self):
        for n in (64, 256, 2048):
            p = call("Qwen3-235B-A22B", hardware="p5en", total_tokens_b=1000, num_gpus=n)["parallelism"]
            self.assertEqual(p["tensor_parallel"] * p["pipeline_parallel"] * p["data_parallel"], n, (n, p))
            self.assertEqual(p["data_parallel"] % p["expert_parallel"], 0, (n, p))

    def test_fixed_degrees_are_respected_and_impossible_ones_rejected(self):
        b = call("Llama-3.1-70B", hardware="p5en", total_tokens_b=100, num_gpus=64,
                 tensor_parallel=4, pipeline_parallel=4)
        self.assertEqual((b["parallelism"]["tensor_parallel"], b["parallelism"]["pipeline_parallel"]), (4, 4))
        bad = call("Llama-3.1-70B", hardware="p5en", total_tokens_b=100, num_gpus=64, tensor_parallel=16)
        self.assertIn("error", bad)

    def test_memory_is_checked_with_the_chosen_tp(self):
        auto = call("Llama-3.1-70B", hardware="p5en", total_tokens_b=100, num_gpus=64)
        p = auto["parallelism"]
        fixed = call("Llama-3.1-70B", hardware="p5en", total_tokens_b=100, num_gpus=64,
                     tensor_parallel=p["tensor_parallel"], pipeline_parallel=p["pipeline_parallel"],
                     expert_parallel=p["expert_parallel"])
        self.assertEqual(auto["memory_breakdown"]["model_params_gb"], fixed["memory_breakdown"]["model_params_gb"])

    def test_auto_gpu_count_meets_target_days(self):
        b = call("Llama-3.1-8B", hardware="p5en", total_tokens_b=1000, num_gpus="auto", target_days=10)
        self.assertLessEqual(b["training_time"]["training_days"], 10)
        self.assertTrue(b["fits_in_memory"])
        self.assertGreaterEqual(len(b["gpu_count_search"]["tried"]), 1)

    def test_project_config_is_restored(self):
        import configuration.project_config as pc
        before = (pc.TP, pc.PP, pc.EP, pc.PRECISION, pc.N_EXPERTS, pc.SEQ_LEN, pc.MFU)
        call("Qwen3-235B-A22B", hardware="p5en", total_tokens_b=100, num_gpus=64, precision="fp8")
        self.assertEqual(before, (pc.TP, pc.PP, pc.EP, pc.PRECISION, pc.N_EXPERTS, pc.SEQ_LEN, pc.MFU))

    def test_hardware_aliases(self):
        self.assertEqual(p9.normalise_hardware("NVIDIA B300"), "p6-b300.48xlarge")
        self.assertEqual(p9.normalise_hardware("h200"), "p5en.48xlarge")
        with self.assertRaises(ValueError):
            p9.normalise_hardware("tpu-v5")

    def test_instance_specs_match_hardware_config(self):
        from configuration.hardware_config import HARDWARE
        for instance, short in (("p5.48xlarge", "p5"), ("p5en.48xlarge", "p5en"),
                                ("p6-b200.48xlarge", "p6-b200"), ("p6-b300.48xlarge", "p6-b300")):
            hw = next(h for h in HARDWARE if h["name"].endswith(" " + short))
            spec = p9.INSTANCE_SPECS[instance]
            for key in ("mem_gb", "inter_node_bw_gb", "intra_node_bw_gbps", "peak_tflops_bf16"):
                self.assertEqual(spec[key], hw[key], (instance, key))


class ReferenceModels(unittest.TestCase):
    """Param2-17B-A2.4B and GLM-5.3-Flash through the whole path. The values are the ones the portal
    calculator returns for the same inputs (see the PR description)."""

    def variant(self, name):
        cfg, total = {"param2": ("param2_17b_a2_4b_config.json", None),
                      "glm": ("glm_5_3_flash_config.json", 321.32)}[name]
        return load_variant_from_hf_config(FIXTURES / cfg, name, params_total_b=total)

    def run_it(self, name, **kw):
        with redirect_stdout(io.StringIO()):
            return p9.run_estimate({"hardware": "p5en", "total_tokens_b": 1000, **kw}, self.variant(name))

    def test_param2_64_gpus(self):
        r = self.run_it("param2", num_gpus=64)
        p, t = r["parallelism"], r["training_time"]
        self.assertEqual((p["tensor_parallel"], p["pipeline_parallel"], p["expert_parallel"]), (1, 1, 1))
        self.assertEqual((t["mfu"], t["tokens_per_second"], t["training_days"]), (0.4425, 1605888, 8.0))

    def test_glm_fixed_plan(self):
        r = self.run_it("glm", num_gpus=64, tensor_parallel=2, pipeline_parallel=16)
        t = r["training_time"]
        self.assertEqual((t["mfu"], t["tokens_per_second"], t["training_days"]), (0.4456, 240580, 53.5))
        self.assertEqual(r["cost_estimate"]["total_cost_usd"], 1436855.0)

    def test_glm_expert_parallel_plan(self):
        t = self.run_it("glm", num_gpus=256, tensor_parallel=2, pipeline_parallel=8,
                        expert_parallel=8)["training_time"]
        self.assertEqual((t["mfu"], t["tokens_per_second"]), (0.3027, 653618))


class ExpertOptimizerSharding(unittest.TestCase):
    """Fails on the old core_calculations: with dp <= ep the expert optimizer state was sharded over
    the full DP instead of dp / ep (1 when EP = DP)."""

    def test_expert_optimizer_state_is_not_sharded_beyond_dp_over_ep(self):
        from configuration.advanced_config import (
            EMPIRICAL_ACT_MULTIPLIER, SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER, FWD_BWD_ROUTING_BUFF_PASSES,
            NCCL_MEM_BUF, OPTIM_BYTES, NCCL_EP_SCALING_FACTOR, FRAGMENTATION_FACTOR, PARAM_BYTES)
        from utils.core_calculations import calculate_memory_for_micro
        v = preset("Qwen3-235B-A22B")

        def optim(dp, ep):
            return calculate_memory_for_micro(
                micro=1, model_mem_gb=40.0, variant=v, dp=dp, layers_per_gpu=v["layers"] // 8,
                zero_strategy=1, seq_len=4096, topk=8, param_bytes=PARAM_BYTES,
                empirical_act_multiplier=EMPIRICAL_ACT_MULTIPLIER,
                selective_act_checkpointing_multiplier=SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER,
                fwd_bwd_routing_buff_passes=FWD_BWD_ROUTING_BUFF_PASSES, nccl_mem_buf=NCCL_MEM_BUF,
                optim_bytes=OPTIM_BYTES, ep=ep, pp=8, vp=1, nccl_ep_scaling_factor=NCCL_EP_SCALING_FACTOR,
                fragmentation_factor=FRAGMENTATION_FACTOR, experts_per_gpu=v["num_experts"] // ep)

        # More data parallelism can only shard optimizer state further, never grow it. Under the old
        # rule dp=8 (= EP) used the full dp (1/8 of the expert state) while dp=16 used dp // ep = 2
        # (1/2), so memory ROSE from 8 to 16 ranks.
        o = [optim(dp=d, ep=8)["optim"] for d in (8, 16, 32, 64)]
        self.assertEqual(o, sorted(o, reverse=True), o)
        self.assertGreater(o[0], o[1])


class Cli(unittest.TestCase):
    def test_estimate_mode_runs_end_to_end(self):
        import ai_model_calculator
        buf = io.StringIO()
        with redirect_stdout(buf):
            ai_model_calculator.main(["--model", "Llama-3.1-8B", "--estimate", "--instance", "h200",
                                      "--num-gpus", "64", "--tokens-b", "1000"])
        out = buf.getvalue()
        self.assertIn("MFU", out)
        self.assertIn("estimated (mfu_model)", out)

    def test_estimate_json(self):
        import ai_model_calculator
        buf = io.StringIO()
        with redirect_stdout(buf):
            ai_model_calculator.main(["--model", "Llama-3.1-8B", "--estimate", "--num-gpus", "64",
                                      "--format", "json"])
        self.assertIn("training_time", json.loads(buf.getvalue()))


if __name__ == "__main__":
    unittest.main()
