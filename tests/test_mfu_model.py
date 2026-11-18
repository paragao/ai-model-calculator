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
    # Megatron alltoall dispatcher (no DeepEP/HybridEP) at v0.4.2; the dispatcher kind is recorded, not modelled
    "mb2604-b200-qwen3-235b": 60.0,
    # DeepEP with moe_a2a_overlap hides more all-to-all than the default EP overlap (0.3); -30.8% in mfu-model-4, -35.6% since the
    # Blackwell MoE half width (mfu-model-5) lowers every Blackwell MoE estimate
    "mb2604-b200-dsv3": 40.0,
    # Our p6-b300 dispatcher A/B runs (TP 8, EP 32 over 4 nodes of EFA, VP 2): the measured all-to-all over EFA costs several
    # seconds a step, the model 0.5 to 1.0 s (docs/CALCULATOR-HARDWARE.md section 6b). +26.3% and +29.9% in mfu-model-5.
    "repo-kimi-k2-b300-32n": 35.0,
    "repo-dsv3-b300-32n": 35.0,
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
            if r["superseded"]:
                continue  # printed, not scored
            bound = EXPLAINED_OUTLIERS.get(r["id"], 30.0)
            self.assertLessEqual(abs(r["error_pct"]), bound, r)

    def test_constants_are_the_fit_of_the_fit_runs(self):
        fit = vr.fit()
        for key, value in fit["base_efficiency"].items():
            fam, prec = key.split("/")
            self.assertEqual(mm.BASE_EFFICIENCY[(fam, prec)], value, key)
        self.assertEqual(mm.MOE_EXPERT_FACTOR, fit["moe_expert_factor"])
        self.assertEqual(mm.MOE_EXPERT_HALF_FFN, fit["moe_expert_half_ffn"])
        self.assertEqual(mm.OVERLAP["ep"], fit["overlap_ep"])

    def test_pinned_summary_is_current(self):
        s = self.summary
        self.assertEqual((mm.VALIDATION_SUMMARY["runs"], mm.VALIDATION_SUMMARY["hold_out_runs"]),
                         (s["all"]["runs"], s["hold_out"]["runs"]))
        self.assertEqual(mm.VALIDATION_SUMMARY["median_abs_error_pct"], s["all"]["median_abs_error_pct"])
        self.assertEqual(mm.VALIDATION_SUMMARY["max_abs_error_pct"], s["all"]["max_abs_error_pct"])
        self.assertEqual(mm.VALIDATION_SUMMARY["hold_out_median_abs_error_pct"], s["hold_out"]["median_abs_error_pct"])
        for group in ("original", "new_moe"):
            self.assertEqual(mm.VALIDATION_SUMMARY[f"{group}_runs"], s[group]["runs"], group)
            self.assertEqual(mm.VALIDATION_SUMMARY[f"{group}_median_abs_error_pct"], s[group]["median_abs_error_pct"], group)
            self.assertEqual(mm.VALIDATION_SUMMARY[f"{group}_max_abs_error_pct"], s[group]["max_abs_error_pct"], group)
        self.assertEqual(mm.VALIDATION_SUMMARY["superseded_runs"], s["superseded"]["runs"])

    def test_original_set_within_15_percent_typical_and_30_worst(self):
        self.assertLessEqual(self.summary["original"]["median_abs_error_pct"], 15)
        self.assertLessEqual(self.summary["original"]["max_abs_error_pct"], 30)

    def test_new_moe_rows_are_scored_as_their_own_group(self):
        self.assertEqual(self.summary["new_moe"]["runs"], 11)
        self.assertGreater(self.summary["new_moe_hold_out"]["runs"], 0)
        models = {r["model"] for r in vr.RUNS if r["id"] in vr.NEW_MOE_RUNS}
        self.assertTrue({"DeepSeek-V3", "Qwen3-30B-A3B", "Qwen3-235B-A22B", "Kimi-K2"} <= models, models)

    def test_superseded_rows_are_not_fitted_or_scored(self):
        sup = [r for r in vr.RUNS if r.get("superseded")]
        self.assertEqual({r["id"] for r in sup}, {"nemo-h100-deepseek-v3", "nemo-b200-deepseek-v3"})
        self.assertFalse(any(r["fit"] for r in sup))
        self.assertNotIn("nemo-h100-deepseek-v3", vr.fit()["fit_runs"])
        scored = sum(1 for r in self.rows if not r["superseded"])
        self.assertEqual(self.summary["all"]["runs"], scored)
        self.assertEqual(scored, len(vr.RUNS) - len(sup))

    def test_every_moe_row_states_its_layout(self):
        for r in vr.RUNS:
            if not mm.model_shape(vr.ARCH[r["model"]])["n_experts"]:
                continue
            self.assertTrue(r["layout"].startswith(("pinned: ", "unverified: ")), r["id"])
            if r["layout"].startswith("pinned: ") and r["source"] != vr.DSV3:
                self.assertIn("etp", r["plan"], r["id"])  # expert TP comes from the recipe, not the default

    def test_nemo_deepseek_rows_use_the_recipe_expert_tp(self):
        # NeMo r2.4.0 recommended configs: etp_size 1 for DeepSeek-V3 on H100 and B200. With the default
        # (expert TP = TP = 2) the B200 run spread EP 8 over two nodes and hid +135% behind network time.
        by_id = {r["id"]: r for r in vr.RUNS}
        for rid in ("nemo-h100-deepseek-v3", "nemo-b200-deepseek-v3"):
            self.assertEqual(by_id[rid]["plan"].get("etp"), 1, rid)

    def test_deepseek_v3_paper_within_10_percent(self):
        # 20 of 132 SMs run DualPipe's all-to-all (paper Sec. 3.2.2) and RMSNorm / MLA up-projection are
        # recomputed (Sec. 3.2.3): with both counted the run is within 10% (+31% in mfu-model-1).
        row = next(r for r in self.rows if r["id"] == "deepseek-v3-paper")
        self.assertLessEqual(abs(row["error_pct"]), 10.0, row)

    def test_dgx_b300_hardware(self):
        hw = mm.HARDWARE["dgx-b300"]
        self.assertEqual((hw["fp8"], hw["nvlink"], hw["net"]), (4500.0, 900.0, 100.0))

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
        self.assertEqual((t["mfu"], t["tokens_per_second"], t["training_days"]), (0.3866, 1402960, 9.2))

    def test_glm_fixed_plan(self):
        r = self.run_it("glm", num_gpus=64, tensor_parallel=2, pipeline_parallel=16)
        t = r["training_time"]
        self.assertEqual((t["mfu"], t["tokens_per_second"], t["training_days"]), (0.3949, 213197, 60.3))
        self.assertEqual(r["cost_estimate"]["total_cost_usd"], 1621399.0)

    def test_glm_expert_parallel_plan(self):
        t = self.run_it("glm", num_gpus=256, tensor_parallel=2, pipeline_parallel=8,
                        expert_parallel=8)["training_time"]
        self.assertEqual((t["mfu"], t["tokens_per_second"]), (0.3036, 655762))


class RecomputeAndCommSMs(unittest.TestCase):
    """Activation recompute adds hardware FLOPs (not model FLOPs); an overlapped SM-based all-to-all
    (DeepEP / HybridEP) takes its SMs from compute."""

    DSV3 = vr.ARCH["DeepSeek-V3"]
    PLAN = {"num_gpus": 256, "gbs": 4096, "mbs": 1, "seq": 4096, "pp": 8, "vp": 2, "ep": 32, "etp": 1}

    def est(self, **extra):
        return mm.estimate(mm.model_shape(self.DSV3), mm.HARDWARE["dgx-b200"], {**self.PLAN, **extra}, "fp8")

    def test_mla_up_projection_size(self):
        # q_lora x H x (nope + rope) + kv_lora x H x (nope + v) = 1536 x 128 x 192 + 512 x 128 x 256
        self.assertEqual(mm.model_shape(self.DSV3)["attention"]["mla_up_per_layer"], 1536 * 128 * 192 + 512 * 128 * 256)

    def test_recompute_adds_its_forward_flops_to_hardware_time_only(self):
        base, rec = self.est(), self.est(recompute=["mla_up_proj", "layernorm"])
        extra = 2.0 * (1536 * 128 * 192 + 512 * 128 * 256) * 61 / base["flops_per_token"]
        self.assertAlmostEqual(rec["time_s"]["compute"] / base["time_s"]["compute"], 1 + extra, places=9)
        self.assertEqual(rec["flops_per_token"], base["flops_per_token"])  # MFU keeps model FLOPs
        self.assertLess(rec["mfu"], base["mfu"])
        with self.assertRaises(ValueError):
            self.est(recompute=["not_a_module"])

    def test_overlapped_deepep_takes_20_sms_from_compute(self):
        base = self.est()["time_s"]["compute"]
        on = self.est(dispatcher="deepep", a2a_overlap=True)["time_s"]["compute"]
        self.assertAlmostEqual(on / base, 148 / (148 - 20), places=9)  # B200: 148 SMs
        self.assertEqual(self.est(dispatcher="deepep", a2a_overlap=False)["time_s"]["compute"], base)
        self.assertEqual(self.est(dispatcher="alltoall", a2a_overlap=True)["time_s"]["compute"], base)


class ZeroThree(unittest.TestCase):
    """ZeRO-3 / FSDP: params gathered for forward and backward every micro-batch (ZeRO paper sec. 7.2.2),
    overlap capped per layer, prefetch from the memory model's live-parameter cap. Same tests as the
    portal's test_mfu_model.ZeroThree."""

    LLAMA70B = vr.ARCH["Llama-3-70B"]

    @classmethod
    def setUpClass(cls):
        cls.glm = ReferenceModels.variant(None, "glm")

    def est(self, shape, hw, **plan):
        return mm.estimate(shape, mm.HARDWARE[hw], {"seq": 8192, **plan}, "bf16")

    def test_zero3_moves_one_and_a_half_times_the_bytes_every_micro_batch(self):
        s = mm.model_shape(self.LLAMA70B)
        plan = dict(num_gpus=64, mbs=1, gbs=128)  # 2 micro-batches per DP rank
        z2 = self.est(s, "p5en.48xlarge", zero=2, **plan)["time_s"]["dp_total"]
        z3 = self.est(s, "p5en.48xlarge", zero=3, **plan)["time_s"]["dp_total"]
        m = 2
        # ZeRO-2: grads every micro-batch + params once = t/2 (1 + m); ZeRO-3: 1.5 m t.
        self.assertAlmostEqual(z3 / z2, 1.5 * m / ((1 + m) / 2), delta=0.02)

    def test_zero3_and_fsdp_are_the_same_schedule(self):
        s = mm.model_shape(self.LLAMA70B)
        plan = dict(num_gpus=64, mbs=1, gbs=128)
        a = self.est(s, "p5en.48xlarge", zero=3, **plan)
        b = self.est(s, "p5en.48xlarge", fsdp=True, **plan)
        self.assertEqual(a["time_s"], b["time_s"])

    def test_overlap_is_capped_by_each_layers_compute(self):
        # One MoE layer in four, 64 experts at EP 1: an MoE layer gathers all 64 experts but computes
        # with 2, so its gather cannot hide behind the spare compute of the dense layers.
        cfg = {"num_hidden_layers": 32, "hidden_size": 4096, "intermediate_size": 14336, "num_attention_heads": 32,
               "num_key_value_heads": 8, "vocab_size": 128256, "num_experts": 64, "num_experts_per_tok": 2,
               "moe_intermediate_size": 14336, "decoder_sparse_step": 4, "tie_word_embeddings": False}
        t = self.est(mm.model_shape(cfg), "p5en.48xlarge", num_gpus=64, mbs=2, gbs=512, seq=4096, zero=3)["time_s"]
        step_level = max(t["dp_total"] * (1 - mm.OVERLAP["fsdp"]), t["dp_total"] - t["compute"])
        self.assertGreater(t["dp_exposed"], 1.3 * step_level)

    def test_no_prefetch_exposes_every_gather(self):
        t = self.est(mm.model_shape(self.LLAMA70B), "p5en.48xlarge", num_gpus=64, mbs=1, gbs=128, zero=3,
                     zero3_prefetch=0)["time_s"]
        self.assertAlmostEqual(t["dp_exposed"], t["dp_total"], delta=1e-9 * t["dp_total"])

    def test_prefetch_comes_from_the_memory_live_parameter_cap(self):
        import utils.phase1_memory as p1m
        # One MoE layer at EP 1 (~7.4B params) is larger than stage3_max_live_parameters (1e9): no prefetch.
        self.assertEqual(p1m.zero3_prefetch_fraction(self.glm, self.glm["num_experts"], 1), 0.0)
        # At EP 64 the largest unit is the 154,880 x 4,096 embedding (634M); the 1e9 live cap leaves
        # room for 58% of the next unit's gather.
        emb = 154880 * 4096
        self.assertAlmostEqual(p1m.zero3_prefetch_fraction(self.glm, self.glm["num_experts"] // 64, 1),
                               (1e9 - emb) / emb, places=6)

    def b300(self, n):
        with redirect_stdout(io.StringIO()):
            return p9.run_estimate({"hardware": "p6-b300.48xlarge", "num_gpus": n, "total_tokens_b": 20000,
                                    "sequence_length": 4096}, self.glm)

    def test_plan_search_does_not_pick_zero3_for_glm_on_b300(self):
        for n in (128, 512, 2048):
            self.assertNotEqual(self.b300(n)["parallelism"]["zero_stage"], 3, msg=n)

    def test_search_reports_the_memory_of_the_stage_it_chose(self):
        p = self.b300(128)["parallelism"]
        self.assertIn(p["micro_batch_size"], p[f"viable_micros_z{p['zero_stage']}"])

    def test_zero3_is_available_from_the_memory_modules(self):
        # The memory modules still offer ZeRO-3 for this request; timed as ZeRO-3 it no longer wins.
        p = ReferenceModels.run_it(ReferenceModels(), "glm", num_gpus=256)["parallelism"]
        self.assertTrue(p["viable_micros_z3"])
        self.assertNotEqual(p["zero_stage"], 3)


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
