"""Hardware support: GB200 and GB300 (docs/CALCULATOR-HARDWARE.md has the sources and the plan).

    python3 -m pytest tests/test_hardware_support.py

Part 1 pins today's results for every hardware entry that existed before GB200 was added, so no
default (GPUs per node, price, bandwidth, search) can leak into p5, p5en, B200, B300 or A100.
The numbers below were produced by main 0ac7828, before any of these changes. Exception: the Blackwell (B200, B300) entries
were re-pinned for mfu-model-5 (B300 board-power term and joint refit, docs/CALCULATOR-HARDWARE.md section 6a); the Hopper and A100
entries are still the 0ac7828 numbers.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import mfu_model as mm  # noqa: E402
from utils import phase9_mfu_estimate as p9  # noqa: E402
from utils import validation_runs as vr  # noqa: E402
from utils.architecture_loader import load_variant_from_hf_config  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
DOC_MFU = Path(__file__).parent.parent / "docs" / "CALCULATOR-MFU.md"


def preset(name):
    from configuration.variants_config import VARIANTS
    return next(v for v in VARIANTS if v["name"] == name)


def variant(model):
    if model == "param2":
        return load_variant_from_hf_config(FIXTURES / "param2_17b_a2_4b_config.json", "param2", params_total_b=None)
    if model == "glm":
        return load_variant_from_hf_config(FIXTURES / "glm_5_3_flash_config.json", "glm", params_total_b=321.32)
    return preset(model)


def run(model, **config):
    with redirect_stdout(io.StringIO()):
        return p9.run_estimate(config, variant(model))


def summary_row(r):
    p, t, c = r["parallelism"], r["training_time"], r["cost_estimate"]
    return [p["total_gpus"], p["tensor_parallel"], p["pipeline_parallel"], p["expert_parallel"], p["zero_stage"],
            p["micro_batch_size"], p["num_nodes"], t["mfu"], t["tokens_per_second"], t["training_days"],
            c["total_cost_usd"], r["fits_in_memory"]]


# key: hardware|model|precision -> [mfu, tokens/s/GPU]. mfu-model-6 re-pinned the DeepSeek-V3 rows of the AWS (EFA) instances only:
# EP 32 x ETP 1 spans 4 nodes, and the 3/4 of its all-to-all that crosses EFA is now timed at the calibrated EP-over-EFA efficiency
# (mfu_model.EP_EFA_KERNEL x EP_EFA_TRAINING, docs/CALCULATOR-HARDWARE.md section 6b). Dense rows and every DGX row are unchanged.
PIN_MM = {
    "p5.48xlarge|Llama-3-70B|bf16": [0.5421, 1113.7],
    "p5.48xlarge|Llama-3-70B|fp8": [0.4213, 1731.9],
    "p5.48xlarge|DeepSeek-V3|bf16": [0.224, 761.6],  # mfu-model-5 [0.2775, 943.7]: EP over EFA (mfu-model-6)
    "p5.48xlarge|DeepSeek-V3|fp8": [0.1486, 1011.0],  # mfu-model-5 [0.1997, 1358.9]: EP over EFA (mfu-model-6)
    "p5e.48xlarge|Llama-3-70B|bf16": [0.5421, 1113.7],
    "p5e.48xlarge|Llama-3-70B|fp8": [0.4213, 1731.9],
    "p5e.48xlarge|DeepSeek-V3|bf16": [0.224, 761.6],  # mfu-model-5 [0.2775, 943.7]: EP over EFA (mfu-model-6)
    "p5e.48xlarge|DeepSeek-V3|fp8": [0.1486, 1011.0],  # mfu-model-5 [0.1997, 1358.9]: EP over EFA (mfu-model-6)
    "p5en.48xlarge|Llama-3-70B|bf16": [0.5421, 1113.7],
    "p5en.48xlarge|Llama-3-70B|fp8": [0.4213, 1731.9],
    "p5en.48xlarge|DeepSeek-V3|bf16": [0.224, 761.6],  # mfu-model-5 [0.2775, 943.7]: EP over EFA (mfu-model-6)
    "p5en.48xlarge|DeepSeek-V3|fp8": [0.1486, 1011.0],  # mfu-model-5 [0.1997, 1358.9]: EP over EFA (mfu-model-6)
    "p6-b200.48xlarge|Llama-3-70B|bf16": [0.3954, 1847.7],
    "p6-b200.48xlarge|Llama-3-70B|fp8": [0.389, 3636.1],
    "p6-b200.48xlarge|DeepSeek-V3|bf16": [0.1273, 985.1],  # mfu-model-5 [0.1697, 1312.6]: EP over EFA (mfu-model-6)
    "p6-b200.48xlarge|DeepSeek-V3|fp8": [0.0899, 1391.2],  # mfu-model-5 [0.1388, 2148.1]: EP over EFA (mfu-model-6)
    "p6-b300.48xlarge|Llama-3-70B|bf16": [0.445, 2079.8],
    "p6-b300.48xlarge|Llama-3-70B|fp8": [0.4375, 4089.3],
    "p6-b300.48xlarge|DeepSeek-V3|bf16": [0.1768, 1367.6],  # mfu-model-5 [0.2138, 1654.1]: EP over EFA (mfu-model-6)
    "p6-b300.48xlarge|DeepSeek-V3|fp8": [0.1372, 2122.2],  # mfu-model-5 [0.1876, 2902.0]: EP over EFA (mfu-model-6)
    "p4d.24xlarge|Llama-3-70B|bf16": [0.4497, 291.4],
    "p4d.24xlarge|Llama-3-70B|fp8": [0.4497, 291.4],
    "p4d.24xlarge|DeepSeek-V3|bf16": [0.134, 143.8],  # mfu-model-5 [0.1891, 202.9]: EP over EFA (mfu-model-6)
    "p4d.24xlarge|DeepSeek-V3|fp8": [0.134, 143.8],  # mfu-model-5 [0.1891, 202.9]: EP over EFA (mfu-model-6)
    "p4de.24xlarge|Llama-3-70B|bf16": [0.4497, 291.4],
    "p4de.24xlarge|Llama-3-70B|fp8": [0.4497, 291.4],
    "p4de.24xlarge|DeepSeek-V3|bf16": [0.134, 143.8],  # mfu-model-5 [0.1891, 202.9]: EP over EFA (mfu-model-6)
    "p4de.24xlarge|DeepSeek-V3|fp8": [0.134, 143.8],  # mfu-model-5 [0.1891, 202.9]: EP over EFA (mfu-model-6)
    "dgx-h100|Llama-3-70B|bf16": [0.5421, 1113.7],
    "dgx-h100|Llama-3-70B|fp8": [0.4213, 1731.9],
    "dgx-h100|DeepSeek-V3|bf16": [0.2775, 943.7],
    "dgx-h100|DeepSeek-V3|fp8": [0.1997, 1358.9],
    "dgx-b200|Llama-3-70B|bf16": [0.3954, 1847.7],
    "dgx-b200|Llama-3-70B|fp8": [0.389, 3636.1],
    "dgx-b200|DeepSeek-V3|bf16": [0.1697, 1312.6],
    "dgx-b200|DeepSeek-V3|fp8": [0.1388, 2148.1],
    "dgx-b300|Llama-3-70B|bf16": [0.445, 2079.8],
    "dgx-b300|Llama-3-70B|fp8": [0.4375, 4089.3],
    "dgx-b300|DeepSeek-V3|bf16": [0.2138, 1654.1],
    "dgx-b300|DeepSeek-V3|fp8": [0.1876, 2902.0],
    "dgx-a100-40g|Llama-3-70B|bf16": [0.4506, 292.0],
    "dgx-a100-40g|Llama-3-70B|fp8": [0.4506, 292.0],
    "dgx-a100-40g|DeepSeek-V3|bf16": [0.2381, 255.5],
    "dgx-a100-40g|DeepSeek-V3|fp8": [0.2381, 255.5],
    "meta-h100-roce|Llama-3-70B|bf16": [0.5421, 1113.7],
    "meta-h100-roce|Llama-3-70B|fp8": [0.4213, 1731.9],
    "meta-h100-roce|DeepSeek-V3|bf16": [0.2775, 943.7],
    "meta-h100-roce|DeepSeek-V3|fp8": [0.1997, 1358.9],
    "deepseek-h800|Llama-3-70B|bf16": [0.5308, 1090.4],
    "deepseek-h800|Llama-3-70B|fp8": [0.4078, 1676.1],
    "deepseek-h800|DeepSeek-V3|bf16": [0.2775, 943.6],
    "deepseek-h800|DeepSeek-V3|fp8": [0.1997, 1358.7],
}

PIN_CANDIDATES = {
    "(64, 32, 0, 8, 4096)": [
        19,
        "729e71906f5d"
    ],
    "(256, 61, 256, 8, 4096)": [
        140,
        "cb863d9763ee"
    ],
    "(72, 48, 128, 8, 4096)": [
        59,
        "f9671c66ebd8"
    ],
    "(128, 80, 0, 8, 65536)": [
        70,
        "28faa110c635"
    ]
}

PIN_CASES = {  # id: (model, run_estimate config)
    "llama8b-p5en-8": ("Llama-3.1-8B", {"hardware": "p5en", "num_gpus": 8, "total_tokens_b": 2000}),
    "llama70b-b200-64": ("Llama-3.1-70B", {"hardware": "B200", "num_gpus": 64, "total_tokens_b": 2000}),
    "llama70b-b300-128": ("Llama-3.1-70B", {"hardware": "B300", "num_gpus": 128, "total_tokens_b": 2000}),
    "qwen30b-p5en-64": ("Qwen3-30B-A3B", {"hardware": "p5en", "num_gpus": 64, "total_tokens_b": 2000}),
    "dsv3-b200-256": ("DeepSeek-V3", {"hardware": "B200", "num_gpus": 256, "total_tokens_b": 2000}),
    "param2-p5en-64": ("param2", {"hardware": "p5en", "num_gpus": 64, "total_tokens_b": 2000}),
    "glm-b200-64": ("glm", {"hardware": "B200", "num_gpus": 64, "total_tokens_b": 2000}),
    "glm-b300-auto30": ("glm", {"hardware": "B300", "target_days": 30, "num_gpus": "auto", "total_tokens_b": 2000}),
    "qwen8b-b300-auto": ("Qwen3-8B", {"hardware": "B300", "target_days": 30, "num_gpus": "auto", "total_tokens_b": 2000}),
    "llama8b-b300-12": ("Llama-3.1-8B", {"hardware": "B300", "num_gpus": 12, "total_tokens_b": 2000}),
}

PIN_RESULTS = {  # id: [gpus, tp, pp, ep, zero, mbs, nodes, mfu, tokens/s, days, cost_usd, fits]
    "llama8b-p5en-8": [8, 1, 1, 1, 1, 1, 1, 0.573, 88077, 292.0, 981180.0, True],
    "llama70b-b200-64": [64, 1, 4, 1, 1, 1, 8, 0.4121, 132017, 194.8, 6733131.0, True],
    "llama70b-b300-128": [128, 1, 4, 1, 1, 1, 16, 0.4589, 294028, 87.5, 8397614.0, True],
    "qwen30b-p5en-64": [64, 1, 2, 1, 1, 1, 8, 0.3201, 726242, 35.4, 951966.0, True],
    "dsv3-b200-256": [256, 2, 16, 1, 1, 1, 32, 0.241, 587843, 43.8, 6048469.0, True],
    "param2-p5en-64": [64, 1, 1, 1, 1, 1, 8, 0.3866, 1402960, 18.3, 492785.0, True],
    "glm-b200-64": [64, 1, 16, 1, 1, 1, 8, 0.2565, 315077, 81.6, 2821176.0, True],
    "glm-b300-auto30": [256, 2, 8, 1, 1, 1, 32, 0.2816, 1383398, 18.6, 3569668.0, True],
    "qwen8b-b300-auto": [64, 1, 1, 1, 1, 1, 8, 0.4696, 1306561, 19.7, 944898.0, True],
    "llama8b-b300-12": [12, 1, 1, 1, 1, 8, 2, 0.4697, 246407, 104.4, 1252568.0, True],
}

PIN_TRIED = {"glm-b300-auto30": [[8, True, 619.3], [16, True, 298.6], [32, True, 143.9], [64, True, 72.5], [128, True, 36.7], [256, True, 18.6]], "qwen8b-b300-auto": [[8, True, 157.4], [16, True, 78.7], [32, True, 39.4], [64, True, 19.7]]}

LLAMA70B_PLAN = dict(num_gpus=64, tp=2, pp=4, cp=1, vp=5, ep=1, mbs=1, seq=8192, gbs=128)
DSV3_PLAN = dict(num_gpus=256, tp=1, pp=8, cp=1, vp=2, ep=32, etp=1, mbs=1, seq=4096, gbs=2048,
                 dispatcher="deepep", a2a_overlap=True, dedup_by_group=True)


class PinnedExistingHardware(unittest.TestCase):
    """p5, p5e, p5en, A100 and the Hopper validation systems give the numbers they gave on main; B200 and B300 give the mfu-model-5
    numbers (re-pinned once, for the B300 refit)."""

    def test_mfu_model_estimates_are_unchanged(self):
        shapes = {m: mm.model_shape(vr.ARCH[m]) for m in ("Llama-3-70B", "DeepSeek-V3")}
        plans = {"Llama-3-70B": LLAMA70B_PLAN, "DeepSeek-V3": DSV3_PLAN}
        for key, (mfu, tps) in PIN_MM.items():
            hw, model, prec = key.split("|")
            e = mm.estimate(shapes[model], mm.HARDWARE[hw], plans[model], prec)
            self.assertEqual(round(e["mfu"], 4), mfu, key)
            self.assertEqual(round(e["tokens_per_second_per_gpu"], 1), tps, key)

    def test_candidate_plans_are_unchanged(self):
        for key, (count, digest) in PIN_CANDIDATES.items():
            c = mm.candidate_plans(*eval(key))
            self.assertEqual(len(c), count, key)
            self.assertEqual(hashlib.sha1(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12], digest, key)

    def test_phase9_estimates_are_unchanged(self):
        for cid, (model, config) in PIN_CASES.items():
            r = run(model, **config)
            self.assertNotIn("error", r, cid)
            self.assertEqual(summary_row(r), PIN_RESULTS[cid], cid)
            if cid in PIN_TRIED:
                tried = [[t["num_gpus"], t["fits"], t["training_days"]] for t in r["gpu_count_search"]["tried"]]
                self.assertEqual(tried, PIN_TRIED[cid], cid)

    def test_eight_gpu_defaults(self):
        for hw in ("p5.48xlarge", "p5en.48xlarge", "p6-b200.48xlarge", "p6-b300.48xlarge", "p4d.24xlarge"):
            self.assertEqual(mm.HARDWARE[hw]["gpn"], 8, hw)
        r = run("Llama-3.1-8B", hardware="B300", num_gpus="auto", total_tokens_b=100)
        self.assertEqual(r["hardware"]["gpus_per_node"], 8)


class Gb200Reachable(unittest.TestCase):
    """Step 1: GB200 (p6e-gb200) can be sized. Every spec is the AWS instance table's (see mfu_model.HARDWARE)."""

    def test_names_map_to_the_ultraserver(self):
        for given in ("GB200", "gb200", "p6e-gb200", "P6E-GB200", "p6e-gb200.36xlarge", "NVIDIA GB200",
                      "gb200-nvl72", "u-p6e-gb200x72", "GB200s"):
            self.assertEqual(p9.normalise_hardware(given), "u-p6e-gb200x72", given)
        for given in ("u-p6e-gb200x36", "gb200x36"):
            self.assertEqual(p9.normalise_hardware(given), "u-p6e-gb200x36", given)

    def test_other_accelerators_are_still_refused(self):
        for given in ("trainium", "tpu-v5", "b800", "nd96p"):
            with self.assertRaises(ValueError, msg=given):
                p9.normalise_hardware(given)

    def test_specs_are_the_instance_table_numbers(self):
        hw = mm.HARDWARE["u-p6e-gb200x72"]
        # p6e-gb200.36xlarge: 4 GPUs, 3200 Gigabit EFA -> 800 Gbps = 100 GB/s per GPU
        self.assertEqual((hw["bf16"], hw["fp8"], hw["nvlink"], hw["net"]), (2500.0, 5000.0, 900.0, 100.0))
        self.assertEqual(hw["family"], "blackwell")
        specs = p9.instance_specs("u-p6e-gb200x72")
        self.assertEqual(specs["mem_gb"], 185)  # 740 GiB / 4 GPUs
        self.assertEqual(specs["inter_node_bw_gb"], 7200)  # 100 GB/s x 72
        self.assertEqual(p9.instance_specs("u-p6e-gb200x36")["inter_node_bw_gb"], 3600)

    def test_hardware_config_matches_phase9(self):
        from configuration.hardware_config import HARDWARE
        entries = [h for h in HARDWARE if "p6e-gb200" in h["name"]]
        self.assertEqual(len(entries), 4)
        specs = p9.instance_specs("u-p6e-gb200x72")
        for entry in entries:
            for key in ("mem_gb", "inter_node_bw_gb", "intra_node_bw_gbps", "peak_tflops_bf16"):
                self.assertEqual(entry[key], specs[key], (entry["name"], key))

    def test_domain_instance_and_billing_gpus_are_three_numbers(self):
        for key, domain in (("u-p6e-gb200x72", 72), ("u-p6e-gb200x36", 36)):
            hw = mm.HARDWARE[key]
            self.assertEqual((mm.domain_gpus(hw), mm.instance_gpus(hw), mm.billing_gpus(hw)), (domain, 4, domain))
        hw = mm.HARDWARE["p6-b300.48xlarge"]  # an HGX node: all three are 8
        self.assertEqual((mm.domain_gpus(hw), mm.instance_gpus(hw), mm.billing_gpus(hw)), (8, 8, 8))

    def test_default_gpus_per_node_comes_from_the_hardware(self):
        self.assertEqual(p9.default_gpus_per_node("u-p6e-gb200x72"), 72)
        self.assertEqual(p9.default_gpus_per_node("u-p6e-gb200x36"), 36)
        self.assertEqual(p9.default_gpus_per_node("p5en.48xlarge"), 8)
        r = run("Llama-3.1-8B", hardware="GB200", num_gpus=8, total_tokens_b=100)
        self.assertEqual(r["hardware"]["gpus_per_node"], 72)

    def test_gpus_in_one_ultraserver_are_billed_as_one_unit(self):
        r = run("Llama-3.1-8B", hardware="GB200", num_gpus=8, total_tokens_b=100)
        self.assertEqual(r["parallelism"]["num_nodes"], 1)
        r = run("Llama-3.1-8B", hardware="GB200", num_gpus=80, total_tokens_b=100)
        self.assertEqual(r["parallelism"]["num_nodes"], 2)
        h = r["hardware"]
        self.assertEqual((h["domain_gpus"], h["instance_gpus"], h["billing_gpus"], h["billed_units"]), (72, 4, 72, 2))
        self.assertEqual(h["instance"], "u-p6e-gb200x72")
        self.assertEqual(h["gpu"], "GB200 185GB")
        self.assertEqual(r["memory_breakdown"]["gpu_capacity_gb"], 185)

    def test_price_is_unknown_not_invented(self):
        r = run("Llama-3.1-8B", hardware="GB200", num_gpus=8, total_tokens_b=100)
        c = r["cost_estimate"]
        self.assertIsNone(c["cost_per_hour_usd"])
        self.assertIsNone(c["total_cost_usd"])
        self.assertIs(c["price_known"], False)
        self.assertIn("Price unknown", c["note"])
        self.assertIsNone(r["hardware"]["price_per_instance_hour_usd"])
        self.assertGreater(r["training_time"]["training_days"], 0)  # time is still estimated
        self.assertIn("Cost**: unknown", p9.format_estimate(r))

    def test_hgx_hardware_keeps_a_price(self):
        r = run("Llama-3.1-8B", hardware="B300", num_gpus=8, total_tokens_b=100)
        self.assertEqual(r["cost_estimate"]["cost_per_hour_usd"], 250.0)
        self.assertNotIn("price_known", r["cost_estimate"])
        self.assertNotIn("billing_gpus", r["hardware"])

    def test_nvlink_domain_reaches_72_gpus(self):
        gb200, b200 = mm.HARDWARE["u-p6e-gb200x72"], mm.HARDWARE["p6-b200.48xlarge"]
        nvlink = 900.0 * mm.LINK_EFFICIENCY["nvlink"] * 1e9
        self.assertEqual(mm._span_bw(gb200, 1, 72), nvlink)  # a 72-GPU group stays on NVLink
        self.assertEqual(mm._span_bw(gb200, 1, 9), nvlink)
        self.assertLess(mm._span_bw(b200, 1, 16), nvlink)  # two HGX nodes: EFA-limited
        # one rank per UltraServer: its single EFA link (100 GB/s x 0.8) carries the traffic
        self.assertEqual(mm._span_bw(gb200, 72, 2), 100.0 * mm.LINK_EFFICIENCY["net"] * 1e9)

    def test_pipeline_of_2_to_9_stays_in_the_domain_on_72_gpus(self):
        plans = mm.candidate_plans(72, 72, 0, 72, 4096)
        pps = sorted({c["pp"] for c in plans})
        for pp in (2, 3, 4, 6, 8, 9):
            self.assertIn(pp, pps)
        shape = mm.model_shape(vr.ARCH["Llama-3-70B"])
        for pp in (2, 4, 8, 9):
            e = mm.estimate(shape, mm.HARDWARE["u-p6e-gb200x72"],
                            dict(num_gpus=72, tp=1, pp=pp, cp=1, vp=1, ep=1, mbs=1, seq=4096, gbs=144), "bf16")
            self.assertGreater(e["mfu"], 0.2, pp)

    def test_estimate_runs_on_gb200_for_dense_and_moe(self):
        dense = run("Llama-3.1-70B", hardware="GB200", num_gpus=72, total_tokens_b=2000)
        self.assertTrue(dense["fits_in_memory"])
        self.assertEqual(dense["parallelism"]["total_gpus"], 72)
        moe = run("param2", hardware="gb200", num_gpus=64, total_tokens_b=2000)
        self.assertTrue(moe["fits_in_memory"])
        self.assertEqual(moe["mfu_breakdown"]["base_efficiency_key"], "blackwell/bf16")

    def test_cli_estimate_accepts_gb200(self):
        import subprocess
        out = subprocess.run([sys.executable, str(Path(__file__).parent.parent / "ai_model_calculator.py"), "--estimate",
                              "--instance", "gb200", "--model", "Llama-3.1-8B", "--num-gpus", "8"],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        self.assertIn("GB200", out.stdout)
        self.assertIn("Cost**: unknown", out.stdout)

    def test_legacy_preset_uses_the_instance_table_memory(self):
        from configuration.hardware_config import HARDWARE
        gb = [h for h in HARDWARE if "p6e-gb200" in h["name"]]
        self.assertTrue(all(h["mem_gb"] == 185 and h["inter_node_bw_gb"] == 7200 for h in gb))

    def test_72_gpu_request_with_ep_64_uses_64_of_72_and_bills_one_ultraserver(self):
        r = run("DeepSeek-V3", hardware="gb200", total_tokens_b=2000, num_gpus=72, expert_parallel=64)
        self.assertNotIn("error", r, r)
        p, rack = r["parallelism"], r["rack_layout"]
        self.assertEqual((p["total_gpus"], p["expert_parallel"], p["num_nodes"]), (64, 64, 1))
        self.assertEqual((rack["requested_gpus"], rack["used_gpus"], rack["idle_gpus"], rack["billed_units"]), (72, 64, 8, 1))
        self.assertEqual([c["gpus"] for c in rack["considered"]], [72, 64])
        self.assertIn("error", rack["considered"][0])  # EP 64 cannot divide 72 GPUs
        self.assertEqual(r["hardware"]["billed_units"], 1)
        self.assertIn("64 of the 72 GPUs", p9.format_estimate(r))

    def test_a_faster_72_gpu_plan_is_kept(self):
        r = run("Llama-3.1-70B", hardware="GB200", num_gpus=72, total_tokens_b=2000)
        self.assertEqual(r["parallelism"]["total_gpus"], 72)
        self.assertEqual(r["rack_layout"]["idle_gpus"], 0)
        self.assertEqual([c["gpus"] for c in r["rack_layout"]["considered"]], [72, 64])

    def test_power_of_two_counts_have_no_alternative(self):
        r = run("Llama-3.1-8B", hardware="GB200", num_gpus=64, total_tokens_b=100)
        self.assertNotIn("rack_layout", r)
        self.assertEqual(r["parallelism"]["num_nodes"], 1)
        r = run("Llama-3.1-8B", hardware="GB200", num_gpus=128, total_tokens_b=100)
        self.assertEqual(r["parallelism"]["num_nodes"], 2)  # 128 GPUs: two UltraServers

    def test_hgx_requests_have_no_rack_layout(self):
        r = run("Llama-3.1-8B", hardware="B300", num_gpus=72, total_tokens_b=100)
        self.assertNotIn("rack_layout", r)
        self.assertEqual(r["parallelism"]["total_gpus"], 72)

    def test_ep_64_on_256_gpus_is_allowed_and_all_to_all_stays_on_nvlink(self):
        gb = run("DeepSeek-V3", hardware="gb200", total_tokens_b=2000, num_gpus=256, expert_parallel=64)
        self.assertEqual((gb["parallelism"]["total_gpus"], gb["parallelism"]["expert_parallel"]), (256, 64))
        self.assertTrue(gb["alltoall_communication"]["comm_type"].startswith("NVLink only"))
        self.assertEqual(gb["alltoall_communication"]["efa_share"], 0.0)
        hgx = run("DeepSeek-V3", hardware="b200", total_tokens_b=2000, num_gpus=256, expert_parallel=64)
        self.assertTrue(hgx["alltoall_communication"]["comm_type"].startswith("EFA: "))
        self.assertGreater(hgx["alltoall_communication"]["efa_share"], 0.8)  # EP 64 x ETP 1 on 8-GPU nodes: 7/8 leaves the node

    def test_auto_search_steps_inside_the_domain_then_by_ultraserver(self):
        r = run("Llama-3.1-70B", hardware="GB200", total_tokens_b=2000, num_gpus="auto", target_days=60)
        search = r["gpu_count_search"]
        counts = [t["num_gpus"] for t in search["tried"]]
        self.assertEqual(counts, [4, 8, 16, 32, 64, 72, 128, 144, 256][:len(counts)])
        self.assertEqual(counts[0], 4)
        self.assertIn("72-GPU NVLink domain", search["basis"])
        self.assertLessEqual(r["training_time"]["training_days"], 60)
        self.assertTrue(r["fits_in_memory"])
        self.assertEqual(r["parallelism"]["num_nodes"], -(-r["parallelism"]["total_gpus"] // 72))

    # ---- GB300 (step 5) ----

    def test_gb300_names_map_to_the_ultraserver_and_no_instance_name_is_invented(self):
        """u-p6e-gb300x72 and u-p6e-gb300x36 (paragao, 2026-11-18, AWS P6 page); the old aliases land on x72."""
        for given in ("GB300", "gb300", "p6e-gb300", "NVIDIA GB300", "gb300-nvl72", "u-p6e-gb300x72", "GB300s", "gb300x72",
                      "p6e-gb300-ultraserver"):
            self.assertEqual(p9.normalise_hardware(given), "u-p6e-gb300x72", given)
        for given in ("u-p6e-gb300x36", "gb300x36", "gb300-nvl36"):
            self.assertEqual(p9.normalise_hardware(given), "u-p6e-gb300x36", given)
        self.assertEqual(p9.GPU_OF_INSTANCE["u-p6e-gb300x72"], "GB300 279GB")
        self.assertEqual(p9.GPU_OF_INSTANCE["u-p6e-gb300x36"], "GB300 279GB")
        self.assertNotIn("p6e-gb300-ultraserver", p9.GPU_OF_INSTANCE)  # never a canonical name or an output
        self.assertNotIn("p6e-gb300-ultraserver", mm.HARDWARE)
        for bad in ("p6e-gb300.36xlarge", "u-p6e-gb300x18", "gb300x144"):
            with self.assertRaises(ValueError, msg=bad):
                p9.normalise_hardware(bad)

    def test_gb300_72_gpus_is_sized_with_unknowns_and_user_figures_listed(self):
        r = run("Llama-3.1-70B", hardware="GB300", num_gpus=72, total_tokens_b=2000)
        self.assertNotIn("error", r, r)
        self.assertTrue(r["fits_in_memory"])
        h, c = r["hardware"], r["cost_estimate"]
        self.assertEqual((h["gpu"], h["domain_gpus"], h["billing_gpus"], h["billed_units"]), ("GB300 279GB", 72, 72, 1))
        self.assertEqual(r["memory_breakdown"]["gpu_capacity_gb"], 279)
        unknown = " ".join(h["unknowns"]).lower()
        self.assertIn("price", unknown)
        self.assertNotIn("efa", unknown)
        self.assertIn("paragao, 2026-11-17", " ".join(h["user_provided"]))
        self.assertEqual(h["network_gb_s_per_gpu"], 100.0)
        self.assertIs(c["price_known"], False)
        self.assertIsNone(c["total_cost_usd"])
        self.assertEqual(r["mfu_breakdown"]["base_efficiency_key"], "blackwell/bf16")

    def test_gb200_does_not_list_unknowns(self):
        r = run("Llama-3.1-8B", hardware="GB200", num_gpus=8, total_tokens_b=100)
        self.assertNotIn("unknowns", r["hardware"])

    def test_gb300_scales_out_by_whole_ultraservers(self):
        for n, units in ((72, 1), (144, 2), (288, 4)):
            r = run("Llama-3.1-70B", hardware="GB300", num_gpus=n, total_tokens_b=2000)
            self.assertNotIn("error", r, (n, r))
            self.assertEqual((r["parallelism"]["total_gpus"], r["parallelism"]["num_nodes"]), (n, units))
            self.assertEqual(r["hardware"]["billed_units"], units)
            self.assertTrue(r["fits_in_memory"])

    def test_gb300_144_gpus_is_a_bit_slower_per_gpu_than_72(self):
        """Same model and global batch rule at 72 and 144 GPUs: 144 uses the EFA between the two UltraServers."""
        a = run("Llama-3.1-70B", hardware="GB300", num_gpus=72, total_tokens_b=2000)
        b = run("Llama-3.1-70B", hardware="GB300", num_gpus=144, total_tokens_b=2000)
        per_gpu = lambda r: r["training_time"]["tokens_per_second"] / r["parallelism"]["total_gpus"]  # noqa: E731
        self.assertGreater(b["training_time"]["tokens_per_second"], 1.8 * a["training_time"]["tokens_per_second"])
        self.assertLessEqual(per_gpu(b), per_gpu(a))
        self.assertLess(b["training_time"]["training_days"], a["training_time"]["training_days"])

    def test_gb300_auto_search_goes_past_one_ultraserver(self):
        r = run("Llama-3.1-70B", hardware="GB300", total_tokens_b=2000, num_gpus="auto", target_days=30)
        self.assertNotIn("error", r, r)
        counts = [t["num_gpus"] for t in r["gpu_count_search"]["tried"]]
        self.assertEqual(counts[:7], [4, 8, 16, 32, 64, 72, 128])
        self.assertGreater(counts[-1], 72)
        self.assertLessEqual(r["training_time"]["training_days"], 30)
        self.assertNotIn("note", r["gpu_count_search"])
    def test_cli_estimate_accepts_gb300(self):
        import subprocess
        out = subprocess.run([sys.executable, str(Path(__file__).parent.parent / "ai_model_calculator.py"), "--estimate",
                              "--instance", "gb300", "--model", "Llama-3.1-8B", "--num-gpus", "8"],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        self.assertIn("GB300", out.stdout)


class RackAwareCounts(unittest.TestCase):
    """Step 2: GPU counts in steps of 4 inside a 72-GPU domain, 64 of 72, billing by UltraServer."""

    def test_auto_counts_for_hgx_are_unchanged_doubling(self):
        hw = mm.HARDWARE["p6-b300.48xlarge"]
        self.assertEqual(mm.auto_gpu_counts(hw, 8, 100), [8, 16, 32, 64])
        self.assertEqual(mm.auto_gpu_counts(hw, 4, 40), [4, 8, 16, 32])

    def test_auto_counts_for_gb200(self):
        x72, x36 = mm.HARDWARE["u-p6e-gb200x72"], mm.HARDWARE["u-p6e-gb200x36"]
        self.assertEqual(mm.auto_gpu_counts(x72, 72, 600), [4, 8, 16, 32, 64, 72, 128, 144, 256, 288, 512, 576])
        self.assertEqual(mm.auto_gpu_counts(x36, 36, 300), [4, 8, 16, 32, 36, 64, 72, 128, 144, 256, 288])

    def test_power_of_two_share_of_the_domain(self):
        self.assertEqual(mm.rack_share(mm.HARDWARE["u-p6e-gb200x72"]), 64)
        self.assertEqual(mm.rack_share(mm.HARDWARE["u-p6e-gb200x36"]), 32)
        self.assertEqual(mm.rack_share(mm.HARDWARE["p6-b300.48xlarge"]), 0)

    def test_alternatives_are_the_largest_power_of_two_below(self):
        g = mm.HARDWARE["u-p6e-gb200x72"]
        got = {n: mm.rack_alternatives(g, n) for n in (4, 8, 64, 72, 100, 128, 144, 288, 576)}
        self.assertEqual(got, {4: [], 8: [], 64: [], 72: [64], 100: [64], 128: [], 144: [128], 288: [256], 576: [512]})
        self.assertEqual(mm.rack_alternatives(mm.HARDWARE["p6-b300.48xlarge"], 72), [])  # an HGX node never changes

    def test_128_gpus_run_as_64_per_ultraserver(self):
        g = mm.HARDWARE["u-p6e-gb200x72"]
        self.assertEqual({n: mm.rack_hw(g, n)["gpn"] for n in (8, 64, 72, 100, 128, 144, 256, 288)},
                         {8: 72, 64: 72, 72: 72, 100: 72, 128: 64, 144: 72, 256: 64, 288: 72})
        self.assertIs(mm.rack_hw(mm.HARDWARE["p6-b200.48xlarge"], 128), mm.HARDWARE["p6-b200.48xlarge"])

    def test_ep_64_stays_on_nvlink_inside_a_64_gpu_share(self):
        g = mm.HARDWARE["u-p6e-gb200x72"]
        slow = {**g, "net": 0.001}  # a network 100,000 times slower must not matter to an all-NVLink all-to-all
        shape = mm.model_shape(vr.ARCH["DeepSeek-V3"])
        plan = dict(num_gpus=128, tp=1, pp=2, cp=1, vp=1, ep=64, etp=1, mbs=1, seq=4096, gbs=2048, dispatcher="deepep")
        a, b = mm.estimate(shape, g, plan, "fp8"), mm.estimate(shape, slow, plan, "fp8")
        self.assertEqual(a["time_s"]["ep_total"], b["time_s"]["ep_total"])
        wide = mm.estimate(shape, g, {**plan, "num_gpus": 256, "ep": 128, "pp": 1, "gbs": 2048}, "fp8")
        wide_slow = mm.estimate(shape, slow, {**plan, "num_gpus": 256, "ep": 128, "pp": 1, "gbs": 2048}, "fp8")
        self.assertGreater(wide_slow["time_s"]["ep_total"], wide["time_s"]["ep_total"])  # EP 128 does not fit a 64-GPU share


class Gb200Validation(unittest.TestCase):
    """Step 3: the DGX GB200 rows of NeMo 25.07 are a hold-out for the 72-GPU NVLink domain model. Nothing is
    refitted: these rows are not in vr.RUNS, so the 38-run summary, the fit and every existing number are unchanged."""

    # Rows above the 30% bar, and why (docs/CALCULATOR-MFU.md, section 7a). Bounds pin them so they cannot drift unseen.
    EXPLAINED = {
        "nemo-gb200-deepseek-v3": 70.0,   # NeMo 25.07 software; NeMo 26.08.01 runs this model on the same GPUs 2.1x faster
                                          # (82.3% in mfu-model-4, 64.7% since the Blackwell MoE half width of mfu-model-5)
        "nemo-gb200-llama4-maverick": 40.0,  # dispatcher and layout not recorded (unverified)
    }

    @classmethod
    def setUpClass(cls):
        cls.rows = vr.evaluate(vr.RACK_RUNS)
        cls.summary = vr.rack_summary(cls.rows)

    def test_eight_rows_all_hold_out_and_not_in_the_fit_set(self):
        self.assertEqual(len(vr.RACK_RUNS), 8)
        self.assertTrue(all(not r["fit"] for r in vr.RACK_RUNS))
        self.assertFalse({r["id"] for r in vr.RACK_RUNS} & {r["id"] for r in vr.RUNS})
        self.assertFalse({r["id"] for r in vr.RACK_RUNS} & set(vr.fit()["fit_runs"]))
        self.assertEqual({r["hardware"] for r in vr.RACK_RUNS}, {"dgx-gb200"})
        self.assertTrue(all(r["source"] == vr.NEMO_2507 and r["precision"] == "fp8" for r in vr.RACK_RUNS))

    def test_the_38_run_summary_is_the_mfu_model_5_one(self):
        s = vr.summary(vr.evaluate())
        self.assertEqual((s["all"]["runs"], s["all"]["median_abs_error_pct"], s["all"]["max_abs_error_pct"]), (38, 6.0, 38.2))
        self.assertEqual((s["hold_out"]["runs"], s["hold_out"]["median_abs_error_pct"]), (23, 7.4))
        self.assertEqual(len(vr.RUNS), 40)  # 38 scored + 2 superseded

    def test_typical_error_within_15_percent(self):
        self.assertLessEqual(self.summary["median_abs_error_pct"], 15)

    def test_each_row_within_30_percent_unless_explained(self):
        for r in self.rows:
            self.assertLessEqual(abs(r["error_pct"]), self.EXPLAINED.get(r["id"], 30.0), r)

    def test_pinned_summary_is_current(self):
        v = mm.VALIDATION_SUMMARY
        self.assertEqual((v["rack_scale_runs"], v["rack_scale_median_abs_error_pct"], v["rack_scale_max_abs_error_pct"]),
                         (self.summary["runs"], self.summary["median_abs_error_pct"], self.summary["max_abs_error_pct"]))

    def test_dgx_gb200_hardware(self):
        hw = mm.HARDWARE["dgx-gb200"]
        # 4 x ConnectX-7 NDR 400 Gb/s per 4-GPU compute tray = 50 GB/s per GPU (NVIDIA DGX SuperPOD GB200 reference architecture)
        self.assertEqual((hw["bf16"], hw["fp8"], hw["nvlink"], hw["net"], hw["gpn"]), (2500.0, 5000.0, 900.0, 50.0, 72))
        self.assertEqual(hw["family"], "blackwell")

    def test_new_architectures_match_their_published_size(self):
        for name in ("Llama-4-Scout", "Llama-4-Maverick", "GPT-OSS-120B"):
            s = mm.model_shape(vr.ARCH[name])
            total, active = vr.PUBLISHED_PARAMS[name]
            self.assertAlmostEqual(s["total_b"] / total, 1, delta=0.02, msg=name)
            self.assertAlmostEqual(s["active_b"] / active, 1, delta=0.03, msg=name)
        self.assertEqual(mm.model_shape(vr.ARCH["Llama-4-Maverick"])["moe_layers"], 24)  # every second layer
        self.assertEqual(mm.model_shape(vr.ARCH["Llama-4-Scout"])["moe_layers"], 48)

    def test_every_moe_row_states_its_layout(self):
        for r in vr.RACK_RUNS:
            if mm.model_shape(vr.ARCH[r["model"]])["n_experts"]:
                self.assertTrue(r["layout"].startswith(("pinned: ", "unverified: ")), r["id"])
                self.assertIn("etp", r["plan"], r["id"])

    def test_same_recipe_on_b200_and_gb200_keeps_its_ratio(self):
        """The change under test is a bigger NVLink domain on the same GPU: predicted GB200/B200 vs published."""
        by = {r["id"]: r for r in self.rows}
        b200 = {r["id"]: r for r in vr.evaluate()}
        for gb, b in (("nemo-gb200-llama3-70b", "nemo-b200-llama3-70b"), ("nemo-gb200-mixtral-8x7b", "nemo-b200-mixtral-8x7b")):
            published = by[gb]["reported"] / b200[b]["reported"]
            predicted = by[gb]["predicted"] / b200[b]["predicted"]
            self.assertAlmostEqual(predicted / published, 1, delta=0.15, msg=gb)

    def test_docs_table_is_current(self):
        doc = DOC_MFU.read_text() if hasattr(DOC_MFU, "read_text") else open(DOC_MFU).read()
        self.assertIn(vr.markdown(self.rows), doc)


class Gb300Hardware(unittest.TestCase):
    """Step 5: GB300 with the numbers AWS and NVIDIA publish, and the ones they do not left unknown."""

    def test_specs_are_gb200_class_compute_with_the_published_uplift_only_in_memory(self):
        g3, g2 = mm.HARDWARE["u-p6e-gb300x72"], mm.HARDWARE["u-p6e-gb200x72"]
        # NVIDIA GB300 NVL72: 360 PF BF16 and 720 PF FP8 with sparsity for 72 GPUs = 2500 / 5000 dense per GPU, as GB200;
        # AWS: the 1.5x uplift is FP4 only. NVLink 130 TB/s / 72 = 1.8 TB/s per GPU.
        for key in ("bf16", "fp8", "nvlink", "family", "domain_gpus", "billing_gpus"):
            self.assertEqual(g3[key], g2[key], key)
        self.assertEqual((g3["bf16"], g3["fp8"], g3["nvlink"], g3["domain_gpus"]), (2500.0, 5000.0, 900.0, 72))
        self.assertEqual(g3["gpu"], "GB300")
        self.assertEqual(mm.SMS_PER_GPU["GB300"], 160)

    def test_unknowns_stay_unknown(self):
        g3 = mm.HARDWARE["u-p6e-gb300x72"]
        self.assertEqual(g3["net"], 100.0)  # given by paragao, 2026-11-17: 18 servers x 8 x 400 Gbps / 72 GPUs
        text = " ".join(g3["unknowns"]).lower()
        for word in ("price", "memory"):
            self.assertIn(word, text)
        self.assertNotIn("efa", text)  # no longer unknown
        self.assertNotIn("instance type names", text)  # given by paragao, 2026-11-18
        self.assertIn("paragao, 2026-11-17", " ".join(g3["user_provided"]))
        self.assertIn("paragao, 2026-11-18", " ".join(g3["user_provided"]))
        names = [k for k in mm.HARDWARE if "gb300" in k]
        self.assertEqual(sorted(names), ["dgx-gb300", "u-p6e-gb300x36", "u-p6e-gb300x72"])
        x36 = mm.HARDWARE["u-p6e-gb300x36"]
        for key in ("gpu", "family", "bf16", "fp8", "nvlink", "net", "instance_gpus", "tgp_w", "tgp_ref_w", "unknowns"):
            self.assertEqual(x36[key], g3[key], key)  # same per-GPU specs as x72
        self.assertEqual((x36["gpn"], x36["domain_gpus"], x36["billing_gpus"]), (36, 36, 36))  # 9 instances of 4 GPUs
        self.assertEqual(mm.HARDWARE["dgx-gb300"]["net"], 100.0)  # NVIDIA's own DGX: 800 Gb/s ConnectX-8 per GPU
        for key, hw in mm.HARDWARE.items():
            self.assertIsNotNone(hw["net"], key)

    def test_a_group_inside_one_domain_needs_no_network_number(self):
        g3 = mm.HARDWARE["u-p6e-gb300x72"]
        shape = mm.model_shape(vr.ARCH["Llama-3-70B"])
        e = mm.estimate(shape, g3, dict(num_gpus=72, tp=2, pp=4, cp=1, vp=1, ep=1, mbs=1, seq=4096, gbs=144), "bf16")
        self.assertGreater(e["mfu"], 0.2)
        moe = mm.model_shape(vr.ARCH["Qwen3-30B-A3B"])
        e = mm.estimate(moe, g3, dict(num_gpus=64, tp=1, pp=1, cp=1, vp=1, ep=64, etp=1, mbs=1, seq=4096, gbs=512), "bf16")
        self.assertGreater(e["time_s"]["ep_total"], 0)  # all-to-all over NVLink only

    def test_scales_out_by_whole_ultraservers(self):
        g3 = mm.HARDWARE["u-p6e-gb300x72"]
        shape = mm.model_shape(vr.ARCH["Llama-3-70B"])
        e = mm.estimate(shape, g3, dict(num_gpus=144, tp=2, pp=1, cp=1, vp=1, ep=1, mbs=1, seq=4096, gbs=288), "bf16")
        self.assertGreater(e["mfu"], 0.2)
        self.assertEqual(mm.auto_gpu_counts(g3, 72, 600)[-3:], [288, 512, 576])  # the search is no longer capped at 72

    def test_hgx_and_gb200_networks_are_untouched(self):
        for key, hw in mm.HARDWARE.items():
            if "gb300" not in key:
                self.assertIsNotNone(hw["net"], key)
        self.assertEqual(mm.auto_gpu_counts(mm.HARDWARE["u-p6e-gb200x72"], 72, 600)[-1], 576)


class NvlinkInsideDomainEfaBetween(unittest.TestCase):
    """GB200 x72 and GB300: inside one 72-GPU NVLink domain no EFA is used; EFA carries only traffic between UltraServers.
    "EFA time" is measured by making the EFA 10^8 times slower and seeing which times move."""

    HARDWARE = ("u-p6e-gb200x72", "u-p6e-gb300x72")
    MOE288 = {**vr.ARCH["DeepSeek-V3"], "n_routed_experts": 288, "n_group": 1, "topk_group": 1}  # 288 experts, as GLM-5.3-Flash
    DENSE = dict(tp=2, pp=1, cp=1, vp=1, ep=1, mbs=1, seq=4096, zero=3)  # TP on NVLink, ZeRO-3 gathers across the whole DP group

    @staticmethod
    def slow(hw):
        return {**hw, "net": 1e-6}

    def times(self, hw, shape, n, **plan):
        return mm.estimate(shape, hw, dict(num_gpus=n, gbs=2 * n, **plan), "fp8")["time_s"]

    def moe_plan(self, n):
        return dict(tp=1, pp=1, cp=1, vp=1, ep=n, etp=1, mbs=1, seq=4096)

    def test_the_two_systems_share_the_network_numbers(self):
        for key in self.HARDWARE:
            hw = mm.HARDWARE[key]
            self.assertEqual((hw["net"], hw["nvlink"], hw["domain_gpus"], hw["instance_gpus"]), (100.0, 900.0, 72, 4), key)

    def test_72_gpus_all_comm_on_nvlink_zero_efa_time(self):
        dense, moe = mm.model_shape(vr.ARCH["Llama-3-70B"]), mm.model_shape(self.MOE288)
        for key in self.HARDWARE:
            hw = mm.HARDWARE[key]
            cases = [(dense, 72, dict(self.DENSE)),                                    # TP 2 x DP 36, ZeRO-3
                     (dense, 72, dict(tp=2, pp=4, cp=1, vp=1, ep=1, mbs=1, seq=4096)),  # TP 2 x PP 4 x DP 9
                     (moe, 72, self.moe_plan(72))]                                      # EP 72: all-to-all inside the domain
            for shape, n, plan in cases:
                real, slow = self.times(hw, shape, n, **plan), self.times(self.slow(hw), shape, n, **plan)
                self.assertEqual(real, slow, (key, plan))  # nothing depends on the EFA
            t = self.times(hw, dense, 72, **self.DENSE)
            self.assertGreater(t["tp_total"], 0)
            self.assertGreater(t["dp_total"], 0)   # ZeRO-3 gathers, over NVLink
            t = self.times(hw, moe, 72, **self.moe_plan(72))
            self.assertGreater(t["ep_total"], 0)   # all-to-all, over NVLink

    def test_72_gpus_group_bandwidth_is_nvlink(self):
        nvlink = 900.0 * mm.LINK_EFFICIENCY["nvlink"] * 1e9
        for key in self.HARDWARE:
            hw = mm.HARDWARE[key]
            for inner, group in ((1, 72), (2, 36), (8, 9), (1, 4)):
                self.assertEqual(mm._span_bw(hw, inner, group), nvlink, (key, inner, group))

    def test_144_gpus_efa_only_for_the_groups_that_span_both_ultraservers(self):
        dense, moe = mm.model_shape(vr.ARCH["Llama-3-70B"]), mm.model_shape(self.MOE288)
        for key in self.HARDWARE:
            hw = mm.HARDWARE[key]
            # (a) TP 2 stays inside a server: its time does not move with the EFA
            real = self.times(hw, dense, 144, **self.DENSE)
            slow = self.times(self.slow(hw), dense, 144, **self.DENSE)
            self.assertEqual(real["tp_total"], slow["tp_total"], key)
            # (b) the ZeRO-3 / DP group of 72 x 2 ranks spans both UltraServers: it uses the EFA
            self.assertGreater(slow["dp_total"], 100 * real["dp_total"], key)
            # (c) EP 144 spans both: the off-domain half of the all-to-all goes over EFA, and the EFA is what limits it
            real = self.times(hw, moe, 144, **self.moe_plan(144))
            slow = self.times(self.slow(hw), moe, 144, **self.moe_plan(144))
            fast = self.times({**hw, "net": 1e6}, moe, 144, **self.moe_plan(144))
            self.assertGreater(real["ep_total"], 0, key)
            self.assertGreater(slow["ep_total"], 100 * real["ep_total"], key)
            self.assertGreater(real["ep_total"], fast["ep_total"], key)
            # (d) EP 72 on the same 144 GPUs (one group per UltraServer) stays on NVLink
            ep72 = dict(self.moe_plan(72))
            real = self.times(hw, moe, 144, **ep72)
            slow = self.times(self.slow(hw), moe, 144, **ep72)
            self.assertEqual(real["ep_total"], slow["ep_total"], key)

    def test_144_gpus_group_bandwidth(self):
        nvlink = 900.0 * mm.LINK_EFFICIENCY["nvlink"] * 1e9
        for key in self.HARDWARE:
            hw = mm.HARDWARE[key]
            # one rank per UltraServer: its single 100 GB/s adapter set, x 0.8 link efficiency
            self.assertEqual(mm._span_bw(hw, 72, 2), 100.0 * mm.LINK_EFFICIENCY["net"] * 1e9)
            # 72 ranks per UltraServer share 72 x 100 GB/s of EFA: more than NVLink, so NVLink is the limit
            self.assertEqual(mm._span_bw(hw, 1, 144), nvlink)
            # 4 ranks per UltraServer: 4 x 80 GB/s of EFA is the limit
            self.assertEqual(mm._span_bw(hw, 18, 8), 4 * 100.0 * mm.LINK_EFFICIENCY["net"] * 1e9)

    def test_gb200_and_gb300_give_the_same_comm_times(self):
        dense, moe = mm.model_shape(vr.ARCH["Llama-3-70B"]), mm.model_shape(self.MOE288)
        for n in (72, 144):
            for shape, plan in ((dense, self.DENSE), (moe, self.moe_plan(n))):
                a = self.times(mm.HARDWARE["u-p6e-gb200x72"], shape, n, **plan)
                b = self.times(mm.HARDWARE["u-p6e-gb300x72"], shape, n, **plan)
                for k in ("tp_total", "cp_total", "ep_total", "dp_total"):  # same links; the compute differs by the power term
                    self.assertEqual(a[k], b[k], (n, plan, k))


class Gb300Validation(unittest.TestCase):
    """GB300 / GB200 throughput ratio of the same NeMo recipe, 12 pairs. The power exponent is fitted on 2 (dense FP8, 26.04.01);
    the other 10 are hold-out: dense MXFP8, the MoE pairs of 26.04.01 and the four MoE pairs of 26.08.01."""

    @classmethod
    def setUpClass(cls):
        cls.rows = vr.evaluate_pairs(vr.ALL_GB300_PAIRS)
        cls.groups = vr.pair_groups(cls.rows)
        cls.before = vr.evaluate_pairs(vr.ALL_GB300_PAIRS, tgp_exponent=0.0)

    def test_twelve_pairs_two_fit_ten_hold_out(self):
        self.assertEqual(len(self.rows), 12)
        self.assertEqual([r["id"] for r in self.rows if r["fit"]],
                         ["gb300-vs-gb200-llama3-70b-fp8-2604", "gb300-vs-gb200-llama31-405b-fp8-2604"])
        run_ids = {r["id"] for r in vr.RUNS}
        for p in vr.ALL_GB300_PAIRS:
            for side in ("gb300", "gb200"):
                self.assertFalse(p[side]["fit"])  # not part of the 38-run fit
                self.assertNotIn(p[side]["id"], run_ids)
                self.assertNotIn(p[side]["id"], vr.fit()["fit_runs"])
            self.assertEqual((p["gb300"]["hardware"], p["gb200"]["hardware"]), ("dgx-gb300", "dgx-gb200"))
            self.assertEqual(p["gb300"]["plan"]["gbs"], p["gb200"]["plan"]["gbs"])  # same global batch
        dense_fit = [p for p in vr.ALL_GB300_PAIRS if p["fit"]]
        self.assertTrue(all(p["model"].startswith("Llama") and p["gb300"]["precision"] == "fp8" for p in dense_fit))

    def test_ratio_error_after_the_power_term_meets_the_bar(self):
        g = self.groups
        # bar: 15% typical, 30% worst. Pinned numbers: hold-out median 3.8%, worst 7.8% (7.1% before the Blackwell MoE half width).
        self.assertEqual((g["hold_out"]["pairs"], g["hold_out"]["median_abs_error_pct"], g["hold_out"]["max_abs_error_pct"]), (10, 3.8, 7.8))
        self.assertEqual((g["fit"]["median_abs_error_pct"], g["fit"]["max_abs_error_pct"]), (2.1, 3.1))
        for r in self.rows:
            self.assertLessEqual(abs(r["ratio_error_pct"]), 30.0, r)
        for key in ("hold_out_dense", "hold_out_moe_2604", "hold_out_moe_2608"):
            self.assertLessEqual(g[key]["median_abs_error_pct"], 15, key)
            self.assertLessEqual(g[key]["max_abs_error_pct"], 30, key)

    def test_before_the_term_the_ratio_was_one_and_the_typical_error_above_the_bar(self):
        """Fails on the branch before this change (the model gave GB300 the same speed as GB200 per plan)."""
        for r in self.before:
            self.assertAlmostEqual(r["predicted_ratio"], 1.0, delta=0.05, msg=r["id"])  # plan effects only
            self.assertGreater(r["published_ratio"], 1.08, r)
        hold = vr.pair_summary([r for r in self.before if not r["fit"]])
        self.assertEqual((hold["median_abs_error_pct"], hold["max_abs_error_pct"]), (15.8, 22.7))
        self.assertGreater(hold["median_abs_error_pct"], 15)
        before_2608 = vr.pair_summary([r for r in self.before if r["release"] == "2608"])
        self.assertEqual((before_2608["median_abs_error_pct"], before_2608["max_abs_error_pct"]), (15.3, 21.2))

    def test_the_exponent_is_the_fit_of_the_fit_pairs(self):
        self.assertEqual(vr.fit_tgp_exponent(), mm.TGP_EXPONENT)
        self.assertEqual(mm.TGP_EXPONENT, 1.25)
        self.assertEqual(vr.fit()["tgp_exponent"], mm.TGP_EXPONENT)

    def test_the_term_is_board_power_on_gb300_and_b300(self):
        powered = [k for k, hw in mm.HARDWARE.items() if hw.get("tgp_ref_w")]
        self.assertEqual(sorted(powered), ["dgx-b300", "dgx-gb300", "p6-b300.48xlarge", "u-p6e-gb300x36", "u-p6e-gb300x72"])
        gb300 = [k for k in powered if "gb300" in k]
        for k in gb300:  # NVIDIA Blackwell Ultra table: up to 1,400 W against up to 1,200 W at the same 5 PF dense FP8
            self.assertEqual((mm.HARDWARE[k]["tgp_w"], mm.HARDWARE[k]["tgp_ref_w"]), (1400.0, 1200.0))
            self.assertEqual(mm.HARDWARE[k]["fp8"], mm.HARDWARE["u-p6e-gb200x72"]["fp8"])
        shape = mm.model_shape(vr.ARCH["Llama-3-70B"])
        plan = dict(num_gpus=64, tp=1, pp=1, cp=1, vp=1, ep=1, mbs=2, seq=8192, gbs=256, fsdp=True)
        a = mm.estimate(shape, mm.HARDWARE["dgx-gb200"], plan, "fp8")
        b = mm.estimate(shape, mm.HARDWARE["dgx-gb300"], plan, "fp8")
        self.assertEqual((a["tgp_gain"], b["tgp_gain"]), (1.0, round((1400 / 1200) ** 1.25, 4)))
        self.assertAlmostEqual(b["tokens_per_second"] / a["tokens_per_second"], (1400 / 1200) ** 1.25, delta=0.01)
        self.assertEqual(a["time_s"]["dp_total"], b["time_s"]["dp_total"])  # communication is untouched

    def test_the_other_causes_do_not_explain_the_gap(self):
        """Peaks and HBM bandwidth are equal (NVIDIA table); the faster attention SFU can save at most 1.5 points of 18-24."""
        for p in vr.GB300_FIT_PAIRS:
            shape = mm.model_shape(vr.ARCH[p["model"]])
            seq = p["gb200"]["plan"]["seq"]
            share = vr.attention_exp_share(shape, seq, p["gb200"]["reported"], vr.SFU_TERA_EXP["blackwell"])
            ultra = vr.attention_exp_share(shape, seq, p["gb300"]["reported"], vr.SFU_TERA_EXP["blackwell-ultra"])
            self.assertLess(share, 0.04, p["id"])           # attention exponentials: under 4% of a GB200 second at SFU peak
            self.assertLess(share * (1 - 5.0 / 10.7), 0.02, p["id"])  # so a 2.14x faster SFU saves under 2%
            self.assertLess(ultra, share, p["id"])
            gap = p["gb300"]["reported"] / p["gb200"]["reported"] - 1
            self.assertGreater(gap, 0.15, p["id"])
        # same plan, same micro batch, 70B FP8: the gap is not capacity (larger micro batches) and not the plan
        p = vr.GB300_FIT_PAIRS[0]
        keys = ("tp", "pp", "cp", "vp", "ep", "mbs", "gbs", "seq")
        self.assertEqual({k: p["gb300"]["plan"][k] for k in keys}, {k: p["gb200"]["plan"][k] for k in keys})

    def test_b300_hgx_check_the_term_is_applied_and_right(self):
        off, on = vr.evaluate_b300_pairs(False), vr.evaluate_b300_pairs()
        self.assertEqual(len(off), 3)
        self.assertGreater(vr.pair_summary(off)["median_abs_error_pct"], 10)   # ratio predicted 1.0, published 1.11-1.13
        self.assertLessEqual(vr.pair_summary(on)["max_abs_error_pct"], 3.0)    # with 1,100 W / 1,000 W: within 3%
        self.assertEqual(mm.HARDWARE["dgx-b300"]["tgp_ref_w"], 1000.0)          # evaluate_b300_pairs(False) restored it

    def test_the_summaries_are_the_mfu_model_5_ones(self):
        s = vr.summary(vr.evaluate())
        self.assertEqual((s["all"]["runs"], s["all"]["median_abs_error_pct"], s["all"]["max_abs_error_pct"]), (38, 6.0, 38.2))
        self.assertEqual((s["hold_out"]["median_abs_error_pct"], s["fit"]["median_abs_error_pct"]), (7.4, 5.7))
        self.assertEqual(vr.rack_summary(vr.evaluate(vr.RACK_RUNS)),
                         {"runs": 8, "median_abs_error_pct": 8.9, "max_abs_error_pct": 64.7})

    def test_pinned_summary_is_current(self):
        v = mm.VALIDATION_SUMMARY
        h = self.groups["hold_out"]
        self.assertEqual((v["gb300_fit_pairs"], v["gb300_pairs"], v["gb300_pair_median_abs_error_pct"], v["gb300_pair_max_abs_error_pct"]),
                         (self.groups["fit"]["pairs"], h["pairs"], h["median_abs_error_pct"], h["max_abs_error_pct"]))

    def test_gpt_oss_120b_matches_its_model_card(self):
        s = mm.model_shape(vr.ARCH["GPT-OSS-120B"])
        self.assertEqual((s["layers"], s["n_experts"], s["topk"]), (36, 128, 4))

    def test_docs_table_is_current(self):
        doc = DOC_MFU.read_text() if hasattr(DOC_MFU, "read_text") else open(DOC_MFU).read()
        self.assertIn(vr.pairs_markdown(self.rows), doc)


class Gb300PowerTermInTheResult(unittest.TestCase):
    """The board-power term shows in the phase 9 result for GB300 and nowhere else."""

    def test_gb300_result_has_the_power_gain_and_is_faster_than_gb200_on_the_same_plan(self):
        kw = dict(num_gpus=72, total_tokens_b=2000, tensor_parallel=1, pipeline_parallel=4)
        a = run("Llama-3.1-70B", hardware="GB200", **kw)
        b = run("Llama-3.1-70B", hardware="GB300", **kw)
        self.assertNotIn("power_gain", a["mfu_breakdown"])
        self.assertEqual(b["mfu_breakdown"]["power_gain"], round((1400 / 1200) ** 1.25, 4))
        ratio = b["training_time"]["tokens_per_second"] / a["training_time"]["tokens_per_second"]
        self.assertGreater(ratio, 1.15)
        self.assertLess(ratio, 1.25)

    def test_b300_has_it_against_the_b200(self):
        r = run("Llama-3.1-8B", hardware="B300", num_gpus=8, total_tokens_b=100)
        self.assertEqual(r["mfu_breakdown"]["power_gain"], round((1100 / 1000) ** 1.25, 4))

    def test_no_other_hardware_has_it(self):
        for hw in ("B200", "H200", "H100", "A100", "GB200"):
            r = run("Llama-3.1-8B", hardware=hw, num_gpus=8, total_tokens_b=100)
            self.assertNotIn("power_gain", r["mfu_breakdown"], hw)


class B300PowerRefit(unittest.TestCase):
    """mfu-model-5: the board-power term on HGX B300 (1,100 W against 1,000 W) with a joint refit of the Blackwell constants.
    Every test here fails on main d6051e2 (no term on the B300, VP 1 on our B300 MoE rows, no Blackwell MoE half width)."""

    # Rows that move more than 2 points: mfu-model-4 error -> mfu-model-5 error (percent). docs/CALCULATOR-HARDWARE.md 6a.
    MOVED = {
        "repo-llama70b-bf16-b200": (5.3, -0.7), "repo-qwen3-8b-bf16-b300": (-5.0, 0.7), "nemo-b200-mixtral-8x7b": (2.0, -1.4),
        "mb2604-b200-dsv3": (-30.8, -35.6), "mb2604-b200-qwen3-30b": (15.0, -3.6), "mb2604-b300-qwen3-30b": (1.6, -5.9),
        "mb2604-b200-qwen3-235b": (58.9, 38.2), "mb2604-b300-qwen3-235b": (-1.2, -4.6),
        "repo-kimi-k2-b300-32n": (19.0, 26.3), "repo-dsv3-b300-32n": (18.7, 29.9), "nemo-b200-deepseek-v3": (136.3, 113.0),
    }
    RACK_MOVED = {"nemo-gb200-llama4-scout": (7.2, 4.5), "nemo-gb200-mixtral-8x7b": (7.4, 3.8), "nemo-gb200-deepseek-v3": (82.3, 64.7)}

    @classmethod
    def setUpClass(cls):
        # "after" is mfu-model-5: the shipped model with the EP over EFA term of mfu-model-5 (mfu-model-6 changed only that term;
        # class EpOverEfaCalibration tests it).
        with vr.mfu_model_5_ep():
            cls.after = {r["id"]: r for r in vr.evaluate()}
            cls.rack_after = {r["id"]: r for r in vr.evaluate(vr.RACK_RUNS)}
        with getattr(vr, "mfu_model_4", contextlib.nullcontext)():  # (nullcontext on main, so each test fails on its own)
            cls.before = {r["id"]: r for r in vr.evaluate()}
            cls.rack_before = {r["id"]: r for r in vr.evaluate(vr.RACK_RUNS)}
            cls.pairs_before = vr.evaluate_pairs(vr.ALL_GB300_PAIRS)

    def test_b300_systems_carry_the_board_power_against_the_b200(self):
        for key in ("p6-b300.48xlarge", "dgx-b300"):  # NVIDIA HGX B200 datasheet 1,000 W; B300 1,100 W (mfu_model.HARDWARE comment)
            hw = mm.HARDWARE[key]
            self.assertEqual((hw["tgp_w"], hw["tgp_ref_w"], hw["fp8"], hw["bf16"]), (1100.0, 1000.0, 4500.0, 2250.0), key)
        for key in ("p6-b200.48xlarge", "dgx-b200"):
            self.assertNotIn("tgp_ref_w", mm.HARDWARE[key], key)
        shape = mm.model_shape(vr.ARCH["Llama-3-70B"])
        a = mm.estimate(shape, mm.HARDWARE["p6-b200.48xlarge"], LLAMA70B_PLAN, "bf16")
        b = mm.estimate(shape, mm.HARDWARE["p6-b300.48xlarge"], LLAMA70B_PLAN, "bf16")
        self.assertEqual(b["tgp_gain"], round(1.1 ** 1.25, 4))
        self.assertAlmostEqual(b["tokens_per_second"] / a["tokens_per_second"], 1.1 ** 1.25, delta=0.015)  # B300's 2x EFA adds a little
        self.assertEqual(mm.MODEL_VERSION, "mfu-model-6")

    def test_b300_vs_b200_published_ratios_within_5_percent(self):
        rows = vr.evaluate_b300_pairs()
        for r in rows:
            self.assertLessEqual(abs(r["ratio_error_pct"]), 5.0, r)
            self.assertGreater(r["predicted_ratio"], 1.09, r)
        s = vr.pair_summary(rows)
        self.assertEqual((s["pairs"], s["median_abs_error_pct"], s["max_abs_error_pct"]), (3, 0.2, 2.3))
        v = mm.VALIDATION_SUMMARY
        self.assertEqual((v["b300_pairs"], v["b300_pair_median_abs_error_pct"], v["b300_pair_max_abs_error_pct"]), (3, 0.2, 2.3))
        self.assertFalse(any(p["fit"] for p in vr.B300_PAIRS))  # a hold-out: TGP_EXPONENT comes from the GB300 fit pairs

    def test_constants_are_the_joint_refit(self):
        f = vr.fit()
        self.assertEqual(f["base_efficiency"]["blackwell/bf16"], 0.417)  # 0.443 before: its B300 fit row now carries the term
        self.assertEqual(mm.BASE_EFFICIENCY[("blackwell", "bf16")], 0.417)
        for key in ("hopper/bf16", "hopper/fp8", "blackwell/fp8", "ampere/bf16"):
            self.assertEqual(f["base_efficiency"][key], {"hopper/bf16": 0.573, "hopper/fp8": 0.45, "blackwell/fp8": 0.417,
                                                         "ampere/bf16": 0.471}[key], key)  # unchanged
        self.assertEqual(f["moe_expert_half_ffn_family"], {"blackwell": 1664})
        self.assertEqual(mm.MOE_EXPERT_HALF_FFN_FAMILY, f["moe_expert_half_ffn_family"])
        self.assertEqual((f["moe_expert_factor"], f["moe_expert_half_ffn"], f["overlap_ep"], f["tgp_exponent"]), (0.78, 1024, 0.3, 1.25))
        self.assertEqual(mm.moe_expert_factor(2048, "hopper"), mm.moe_expert_factor(2048))
        self.assertLess(mm.moe_expert_factor(2048, "blackwell"), mm.moe_expert_factor(2048, "hopper"))

    def test_shared_moe_constants_are_the_mfu_model_4_fit(self):
        with vr.mfu_model_4():
            self.assertEqual(vr.fit_moe_shared(), {"moe_expert_factor": 0.78, "moe_expert_half_ffn": 1024, "overlap_ep": 0.3})
        # rerun on the new model (Blackwell half width 1664 in place) the same grid would move the shared half width to 768,
        # which is the joint alternative: not taken (next test)
        self.assertEqual(vr.fit_moe_shared(), {"moe_expert_factor": 0.78, "moe_expert_half_ffn": 768, "overlap_ep": 0.3})

    def test_the_joint_alternative_is_worse_on_the_hold_out_and_moves_hopper(self):
        joint = vr.fit_moe_joint()
        self.assertEqual(joint, {"moe_expert_factor": 0.79, "moe_expert_half_ffn": 768, "overlap_ep": 0.3, "blackwell_moe_half_ffn": 1728})
        shipped = vr.summary(list(self.after.values()))
        with vr.moe_constants(**joint):
            alt_rows = vr.evaluate()
        alt = vr.summary(alt_rows)
        self.assertLess(alt["fit"]["median_abs_error_pct"], shipped["fit"]["median_abs_error_pct"])           # better in-sample
        self.assertGreater(alt["hold_out"]["median_abs_error_pct"], shipped["hold_out"]["median_abs_error_pct"])  # worse on hold-out
        moved = [r["id"] for r in alt_rows if "p5en" in r["id"] and r["error_pct"] != self.after[r["id"]]["error_pct"]]
        self.assertEqual(moved, ["repo-qwen3-30b-a3b-bf16-p5en"])  # a Hopper (p5en) MoE answer would move

    def test_hopper_and_ampere_rows_do_not_move(self):
        for rid, r in self.after.items():
            if mm.HARDWARE[r["hardware"]]["family"] in ("hopper", "ampere"):
                self.assertEqual(r["error_pct"], self.before[rid]["error_pct"], rid)

    def test_rows_that_move_more_than_2_points(self):
        moved = {k: (self.before[k]["error_pct"], self.after[k]["error_pct"]) for k in self.after
                 if abs(self.after[k]["error_pct"] - self.before[k]["error_pct"]) > 2}
        self.assertEqual(moved, self.MOVED)
        rack = {k: (self.rack_before[k]["error_pct"], self.rack_after[k]["error_pct"]) for k in self.rack_after
                if abs(self.rack_after[k]["error_pct"] - self.rack_before[k]["error_pct"]) > 2}
        self.assertEqual(rack, self.RACK_MOVED)
        pairs = {r["id"]: r["ratio_error_pct"] for r in vr.evaluate_pairs(vr.ALL_GB300_PAIRS)}
        for r in self.pairs_before:  # GB300 / GB200 ratios: both sides Blackwell, no pair moves more than 1 point
            self.assertLessEqual(abs(pairs[r["id"]] - r["ratio_error_pct"]), 1.0, r["id"])

    def test_targets(self):
        s = vr.summary(list(self.after.values()))
        self.assertLessEqual(abs(s["all"]["median_abs_error_pct"] - 6.1), 1.0)   # within about 1 point of mfu-model-4
        new_over_30 = [k for k, r in self.after.items() if not r["superseded"] and abs(r["error_pct"]) > 30
                       and abs(self.before[k]["error_pct"]) <= 30]
        self.assertEqual(new_over_30, [])
        g = vr.pair_groups(vr.evaluate_pairs(vr.ALL_GB300_PAIRS))
        self.assertLess(g["hold_out"]["median_abs_error_pct"], 15)
        for rid in ("repo-qwen3-8b-bf16-b300",):            # our dense p6-b300 row: within 15% (0.7%)
            self.assertLessEqual(abs(self.after[rid]["error_pct"]), 15, rid)

    def test_our_b300_moe_rows_record_the_bench_plan(self):
        """VP 2 and no recompute: the overlap-on bench sets them (awsome-distributed-ai megatron-bridge bench scripts)."""
        by = {r["id"]: r for r in vr.RUNS}
        for rid in vr.VP2_CORRECTED:
            plan = by[rid]["plan"]
            self.assertEqual((plan["tp"], plan["pp"], plan["vp"], plan["ep"], plan["mbs"], plan["gbs"]), (8, 8, 2, 32, 4, 256), rid)
            self.assertNotIn("recompute", plan, rid)
            self.assertIn("VPP", by[rid]["note"], rid)
            self.assertFalse(by[rid]["fit"], rid)

    def test_our_b300_moe_rows_are_high_because_of_the_efa_all_to_all(self):
        """The same bench with overlap off: swapping only the dispatcher (NCCL alltoall -> DeepEP) moves the step by 3.3 to 3.5 s, so
        the EP all-to-all over EFA costs seconds a step; the model's whole EP time is 0.5 s (DeepSeek-V3) and 1.0 s (Kimi-K2), and
        its overlap-off step is 1.6 to 2.4 s shorter than the measured DeepEP step."""
        measured = {"repo-dsv3-b300-32n": (6.263, 9.774), "repo-kimi-k2-b300-32n": (6.041, 9.303)}  # deepep, alltoall (s/step)
        by = {r["id"]: r for r in vr.RUNS}
        for rid, (deepep, alltoall) in measured.items():
            r = json.loads(json.dumps(by[rid]))
            r["plan"].update(vp=1, a2a_overlap=False, overlap_ep=0.0,
                             recompute=["core_attn", "mla_up_proj", "mlp", "moe", "layernorm"])  # overlap off: full recompute, VPP 1
            with vr.mfu_model_5_ep():  # the mfu-model-5 term; mfu-model-6 calibrates it (EpOverEfaCalibration)
                t = vr.predict(r)[1]["time_s"]
            self.assertLess(t["ep_total"], (alltoall - deepep) / 3, rid)
            self.assertLess(t["step"], deepep - 1.5, rid)


def call(**ev):
    """The portal Lambda's request shape on run_estimate: (status, body), 400 when the result is an error."""
    from utils.architecture_loader import variant_from_architecture
    cfg = {k: v for k, v in ev.items() if k not in ("model_name", "architecture", "model_params_b", "params_total_b", "total_tokens")}
    cfg["total_tokens_b"] = ev.get("total_tokens", 1e12) / 1e9
    v = variant_from_architecture(ev["architecture"], ev.get("model_name") or "moe", params_total_b=ev.get("params_total_b"))
    with redirect_stdout(io.StringIO()):
        r = p9.run_estimate(cfg, v)
    return (400, r) if "error" in r else (200, r)

def _moe(n_experts):
    """Qwen3-30B-A3B with another expert count: a MoE that fits every layout below."""
    return {**vr.ARCH["Qwen3-30B-A3B"], "num_experts": n_experts}


class EpGroupInsideTheNvlinkDomain(unittest.TestCase):
    """mfu-model-6, paragao's rules of 2026-11-18. Expert traffic stays on NVLink only when the whole EP group (EP x expert TP GPUs)
    sits in one NVLink domain: the 8-GPU node of p5en, the whole UltraServer (72 or 36 GPUs) of GB200 and GB300. Groups may share a
    domain, never straddle one; EP must divide the experts. One test per row of the scenario table; each row is checked three ways:
    the layout (ep_domain_layout), the step-time model (estimate's ep_efa_share) and the Lambda (GPUs used, EP, EFA share).
    Every test fails on the MR 42 head fcc79ea (no layout function, no EFA share, no x36 GB300, EP 72 never tried, 144 GPUs with EP 16
    kept as 144 with the straddling group on NVLink)."""

    X72 = ("u-p6e-gb200x72", "u-p6e-gb300x72")
    X36 = ("u-p6e-gb200x36", "u-p6e-gb300x36")

    def check(self, hw_key, experts, bought, ep, etp, used, share):
        hw = mm.HARDWARE[hw_key]
        lay = mm.ep_domain_layout(hw, bought, ep, etp, experts)
        self.assertIsNone(lay["refused"], lay)
        self.assertEqual((lay["used_gpus"], lay["idle_gpus"]), (used, bought - used), (hw_key, ep, lay))
        self.assertAlmostEqual(lay["efa_share"], share, places=6, msg=(hw_key, ep))
        self.assertEqual(lay["nvlink_only"], share == 0, (hw_key, ep))
        e = mm.estimate(mm.model_shape(_moe(experts)), hw, dict(num_gpus=used, tp=1, pp=1, ep=ep, etp=etp, mbs=1, seq=4096), "bf16")
        self.assertAlmostEqual(e["ep_efa_share"], share, places=4, msg=(hw_key, ep))
        self.assertEqual(e["ep_link"] == "NVLink only", share == 0, (hw_key, ep, e["ep_link"]))
        ev = dict(model_name="moe", architecture=_moe(experts), hardware=hw_key, num_gpus=bought, expert_parallel=ep,
                  total_tokens=1e12)
        if etp > 1:
            ev["expert_tensor_parallel"] = etp
        status, body = call(**ev)
        self.assertEqual(status, 200, body)
        p, a2a = body["parallelism"], body["alltoall_communication"]
        self.assertEqual((p["total_gpus"], p["expert_parallel"], p["expert_tensor_parallel"]), (used, ep, etp), (hw_key, ep))
        self.assertAlmostEqual(a2a["efa_share"], share, places=4, msg=(hw_key, ep))
        self.assertEqual(a2a["comm_type"].startswith("NVLink only"), share == 0, a2a)
        if used != bought:
            self.assertEqual((body["rack_layout"]["used_gpus"], body["rack_layout"]["idle_gpus"]), (used, bought - used))
        return body

    def refused(self, hw_key, experts, bought, ep, why):
        lay = mm.ep_domain_layout(mm.HARDWARE[hw_key], bought, ep, 1, experts)
        self.assertIn(why, lay["refused"] or "", lay)
        status, body = call(model_name="moe", architecture=_moe(experts), hardware=hw_key, num_gpus=bought, expert_parallel=ep,
                            total_tokens=1e12)
        self.assertEqual(status, 400, body)
        self.assertIn(why, body["error"])

    # --- 128 experts on 4 p5en (32 GPUs), domain = the 8-GPU node ---
    def test_p5en_128_experts_ep8_nvlink_only(self):
        self.check("p5en.48xlarge", 128, 32, 8, 1, 32, 0.0)

    def test_p5en_128_experts_ep16_efa_half(self):
        self.check("p5en.48xlarge", 128, 32, 16, 1, 32, 1 / 2)

    def test_p5en_128_experts_ep32_efa_three_quarters(self):
        self.check("p5en.48xlarge", 128, 32, 32, 1, 32, 3 / 4)

    def test_p5en_128_experts_ep8_expert_tp2_efa_half(self):
        self.check("p5en.48xlarge", 128, 32, 8, 2, 32, 1 / 2)  # the group is EP x ETP = 16 GPUs: two nodes

    # --- 256 experts on 2 x72 (144 GPUs bought): 64 GPUs used per UltraServer ---
    def test_x72_256_experts_ep16_nvlink_only_64_per_ultraserver(self):
        for k in self.X72:
            self.check(k, 256, 144, 16, 1, 128, 0.0)

    def test_x72_256_experts_ep32_nvlink_only_64_per_ultraserver(self):
        for k in self.X72:
            self.check(k, 256, 144, 32, 1, 128, 0.0)

    def test_x72_256_experts_ep64_nvlink_only_64_per_ultraserver(self):
        for k in self.X72:
            self.check(k, 256, 144, 64, 1, 128, 0.0)

    def test_x72_256_experts_ep72_refused_not_whole_experts(self):
        for k in self.X72:
            self.refused(k, 256, 144, 72, "EP 72 does not divide the 256 experts")

    def test_x72_256_experts_ep128_efa_half(self):
        for k in self.X72:
            self.check(k, 256, 144, 128, 1, 128, 1 / 2)

    # --- 288 experts ---
    def test_x72_288_experts_ep72_nvlink_only_all_72_used(self):
        for k in self.X72:
            self.check(k, 288, 72, 72, 1, 72, 0.0)  # 4 experts per GPU

    def test_two_x72_288_experts_ep144_efa_half(self):
        for k in self.X72:
            self.check(k, 288, 144, 144, 1, 144, 1 / 2)

    def test_x36_288_experts_ep36_nvlink_only(self):
        for k in self.X36:
            self.check(k, 288, 36, 36, 1, 36, 0.0)

    # --- 256 experts on 1 x36: 4 GPUs idle ---
    def test_x36_256_experts_ep32_nvlink_only_4_idle(self):
        for k in self.X36:
            self.check(k, 256, 36, 32, 1, 32, 0.0)

    def test_x36_256_experts_ep16_nvlink_only_4_idle(self):
        for k in self.X36:
            self.check(k, 256, 36, 16, 1, 32, 0.0)

    # --- the rules behind the table ---
    def test_a_straddling_group_counts_its_cross_domain_part_as_efa(self):
        """Rule B: 144 GPUs used with EP 16 is 9 groups, one across the boundary. estimate() counts it as EFA (its worst rank has
        half of its peers in the other UltraServer); the search never builds it (it uses 64 per UltraServer instead)."""
        for k in self.X72:
            hw = mm.HARDWARE[k]
            self.assertAlmostEqual(mm.ep_efa_share(hw, 144, 16), 0.5)
            self.assertEqual(mm.ep_efa_share(mm.rack_hw(hw, 128), 128, 16), 0.0)
            self.assertFalse(mm.ep_fits_domains(hw, 144, 1, 1, 16, 256))
            self.assertTrue(mm.ep_fits_domains(hw, 128, 1, 1, 16, 256))

    def test_eps_that_fit_the_domain_are_tried_not_only_powers_of_two(self):
        """Rule C: on x72 EP 16, 32, 64 and 72; on x36 16, 32 and 36; HGX keeps the powers of two."""
        self.assertTrue({16, 32, 72, 144}.issubset(mm.ep_candidates(288, 72, 4)))
        self.assertNotIn(64, mm.ep_candidates(288, 72, 4))  # 64 does not divide 288
        self.assertTrue({16, 32, 64, 128, 256}.issubset(mm.ep_candidates(256, 72, 4)))
        self.assertNotIn(72, mm.ep_candidates(256, 72, 4))
        self.assertTrue({16, 32, 36}.issubset(mm.ep_candidates(288, 36, 4)))
        self.assertEqual(mm.ep_candidates(288), [1, 2, 4, 8, 16, 32])  # HGX: powers of two, as before
        self.assertEqual(mm.ep_candidates(160), [1, 2, 4, 8, 16, 32])

    def test_auto_plan_keeps_ep_inside_the_domain(self):
        """Rules C and D: with EP free, a DeepSeek-V3-sized model with 288 experts on one GB300 x72 picks an EP that is not a power
        of two (36: 8 experts per GPU, two groups in the UltraServer) and keeps its all-to-all on NVLink."""
        arch = {**vr.ARCH["DeepSeek-V3"], "n_routed_experts": 288, "n_group": 1, "topk_group": 1}
        status, body = call(model_name="moe", architecture=arch, model_params_b=760, params_total_b=760, hardware="u-p6e-gb300x72",
                            num_gpus=72, total_tokens=1e12)
        self.assertEqual(status, 200, body)
        p, a2a = body["parallelism"], body["alltoall_communication"]
        self.assertTrue(body["fits_in_memory"])
        self.assertEqual((p["total_gpus"], p["expert_parallel"]), (72, 36))
        self.assertEqual(a2a["efa_share"], 0.0, a2a)

    def test_hgx_power_of_two_layouts_are_unchanged(self):
        """On 8-GPU nodes every power-of-two group either fits a node or spans whole nodes: the same share as mfu-model-5's
        1 - (GPUs per node / ETP) / EP."""
        hw = mm.HARDWARE["p5en.48xlarge"]
        for ep in (2, 4, 8, 16, 32, 64):
            for etp in (1, 2, 4, 8):
                old = max(0.0, 1 - max(1, 8 // etp) / ep)
                self.assertAlmostEqual(mm.ep_efa_share(hw, 512, ep, etp), old, msg=(ep, etp))


class EpOverEfaCalibration(unittest.TestCase):
    """mfu-model-6, rule E: the EP all-to-all over EFA, calibrated on our own measurements and checked on others.
    Kernel: DeepEP-class dispatch / combine micro-benchmarks, fit on the 4-node cells, hold-out 2 and 8 nodes and p5en.
    Training: our DeepEP-over-EFA p6-b300 pre-training runs, fit on Kimi-K2 16 nodes, hold-out the two scored 32-node rows of RUNS
    (+26.3% and +29.9% in mfu-model-5) and the Kimi-K2 32-node UCCL run. Fails on the MR 42 head (no calibration, rows at 26-30%)."""

    @classmethod
    def setUpClass(cls):
        cls.kernel = {r["id"]: r for r in vr.evaluate_ep_kernel()}
        cls.train = {r["id"]: r for r in vr.evaluate(vr.EP_EFA_RUNS)}
        with vr.mfu_model_5_ep():
            cls.train_before = {r["id"]: r for r in vr.evaluate(vr.EP_EFA_RUNS)}
            cls.runs_before = {r["id"]: r for r in vr.evaluate()}
        cls.runs_after = {r["id"]: r for r in vr.evaluate()}

    def test_constants_are_the_fits(self):
        self.assertEqual(vr.fit_ep_efa_kernel(), mm.EP_EFA_KERNEL)
        self.assertEqual(vr.fit_ep_efa_training(), mm.EP_EFA_TRAINING)
        self.assertEqual((mm.EP_EFA_KERNEL, mm.EP_EFA_TRAINING), (0.705, 0.31))
        self.assertEqual(mm.LINK_EFFICIENCY["ep_efa"], round(0.705 * 0.31, 3))

    def test_kernel_cells_fit_and_hold_out(self):
        fit = [r for r in self.kernel.values() if r["fit"]]
        self.assertEqual(sorted(r["id"] for r in fit), ["b300-4n", "p5-4n"])
        for r in self.kernel.values():  # every cell within 25% of the fitted kernel efficiency (worst: B300 2 nodes, NVSHMEM)
            self.assertLessEqual(abs(r["error_pct"]), 25.0, r)
            self.assertLess(r["efficiency"], 1.0, r)  # a kernel never beats the line rate once its local copies are removed
        self.assertEqual({k: r["error_pct"] for k, r in self.kernel.items()},
                         {"p5-4n": 4.8, "b300-4n": -4.6, "p5en-2n": 4.9, "p5-8n": 2.3, "b300-2n": 21.9, "b300-8n": -12.6})

    def test_training_fit_and_hold_out(self):
        self.assertEqual([k for k, r in self.train.items() if r["fit"]], ["repo-kimi-k2-b300-16n"])
        self.assertLessEqual(abs(self.train["repo-kimi-k2-b300-16n"]["error_pct"]), 0.5)
        for r in vr.ep_efa_scored(list(self.train.values())):  # rule E: within 15%
            self.assertLessEqual(abs(r["error_pct"]), 15.0, r)
        self.assertEqual({k: r["error_pct"] for k, r in self.train.items()},
                         {"repo-kimi-k2-b300-16n": 0.1, "repo-kimi-k2-b300-32n": 9.3, "repo-dsv3-b300-32n": 10.5,
                          "repo-kimi-k2-b300-32n-uccl": 12.0})

    def test_the_two_rows_were_26_and_30_percent_high(self):
        self.assertEqual((self.train_before["repo-kimi-k2-b300-32n"]["error_pct"], self.train_before["repo-dsv3-b300-32n"]["error_pct"]),
                         (26.3, 29.9))

    def test_the_training_share_is_below_every_kernel_cell(self):
        """A training step reaches about a third of the kernel rate: smaller dispatches (2,048 tokens per rank against 4,096),
        SMs shared with the GEMMs, the host proxy and the slowest of 32 ranks. It is not a property of one kernel cell."""
        self.assertLess(mm.LINK_EFFICIENCY["ep_efa"], min(r["efficiency"] for r in self.kernel.values()) / 2)

    def test_only_our_two_rows_move_and_the_38_run_summary_holds(self):
        moved = {k: (self.runs_before[k]["error_pct"], r["error_pct"]) for k, r in self.runs_after.items()
                 if r["error_pct"] != self.runs_before[k]["error_pct"]}
        self.assertEqual(moved, {"repo-kimi-k2-b300-32n": (26.3, 9.3), "repo-dsv3-b300-32n": (29.9, 10.5)})
        s = vr.summary(list(self.runs_after.values()))
        self.assertEqual((s["all"]["runs"], s["all"]["median_abs_error_pct"], s["all"]["max_abs_error_pct"]), (38, 6.0, 38.2))

    def test_efa_term_only_on_aws_efa_systems(self):
        efa = sorted(k for k, hw in mm.HARDWARE.items() if hw.get("fabric") == "efa")
        self.assertEqual(efa, sorted(["p5.48xlarge", "p5e.48xlarge", "p5en.48xlarge", "p6-b200.48xlarge", "p6-b300.48xlarge",
                                      "p4d.24xlarge", "p4de.24xlarge", "u-p6e-gb200x72", "u-p6e-gb200x36", "u-p6e-gb300x72",
                                      "u-p6e-gb300x36"]))
        shape = mm.model_shape(vr.ARCH["DeepSeek-V3"])
        for dgx in ("dgx-h100", "dgx-b200", "dgx-b300"):  # InfiniBand rows: the mfu-model-5 term, unchanged
            a = mm.estimate(shape, mm.HARDWARE[dgx], DSV3_PLAN, "fp8")["time_s"]["ep_total"]
            with vr.mfu_model_5_ep():
                b = mm.estimate(shape, mm.HARDWARE[dgx], DSV3_PLAN, "fp8")["time_s"]["ep_total"]
            self.assertEqual(a, b, dgx)

    def test_nvlink_only_plans_do_not_move(self):
        """An EP group inside one domain has no EFA part: mfu-model-5 and mfu-model-6 agree exactly."""
        shape = mm.model_shape(vr.ARCH["DeepSeek-V3"])
        for hw, plan in (("p5en.48xlarge", dict(DSV3_PLAN, ep=8)), ("u-p6e-gb300x72", dict(DSV3_PLAN, num_gpus=128, pp=2, ep=64))):
            a = mm.estimate(shape, mm.HARDWARE[hw], plan, "bf16")
            with vr.mfu_model_5_ep():
                b = mm.estimate(shape, mm.HARDWARE[hw], plan, "bf16")
            self.assertEqual((a["ep_efa_share"], a["mfu"]), (0.0, b["mfu"]), hw)

    def test_pinned_summary_is_current(self):
        v = mm.VALIDATION_SUMMARY
        hold = vr.ep_efa_scored(list(self.train.values()))
        self.assertEqual((v["ep_efa_kernel_cells"], v["ep_efa_kernel_max_abs_error_pct"]),
                         (len(self.kernel), max(abs(r["error_pct"]) for r in self.kernel.values())))
        self.assertEqual((v["ep_efa_training_hold_out_runs"], v["ep_efa_training_hold_out_max_abs_error_pct"]),
                         (len(hold), max(abs(r["error_pct"]) for r in hold)))

    def test_remote_copies_of_a_deduplicating_dispatcher(self):
        self.assertEqual(mm.efa_remote_copies(0.0, 8), 0.0)
        self.assertAlmostEqual(mm.efa_remote_copies(0.5, 8), 2 * (1 - 0.5 ** 8) * 0.5)      # 2 domains: about 1 copy
        self.assertAlmostEqual(mm.efa_remote_copies(0.75, 8), 4 * (1 - 0.75 ** 8) * 0.75)   # 4 domains: 2.7 copies, not 6
        self.assertLess(mm.efa_remote_copies(0.75, 4), mm.efa_remote_copies(0.75, 8))      # group-limited routing sends fewer


if __name__ == "__main__":
    unittest.main()
