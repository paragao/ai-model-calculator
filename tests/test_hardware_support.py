"""Hardware support: GB200 and GB300 (docs/CALCULATOR-HARDWARE.md has the sources and the plan).

    python3 -m pytest tests/test_hardware_support.py

Part 1 pins today's results for every hardware entry that existed before GB200 was added, so no
default (GPUs per node, price, bandwidth, search) can leak into p5, p5en, B200, B300 or A100.
The numbers below were produced by main 0ac7828, before any of these changes.
"""
from __future__ import annotations

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


PIN_MM = {  # key: hardware|model|precision -> [mfu, tokens/s/GPU]
    "p5.48xlarge|Llama-3-70B|bf16": [0.5421, 1113.7],
    "p5.48xlarge|Llama-3-70B|fp8": [0.4213, 1731.9],
    "p5.48xlarge|DeepSeek-V3|bf16": [0.2775, 943.7],
    "p5.48xlarge|DeepSeek-V3|fp8": [0.1997, 1358.9],
    "p5e.48xlarge|Llama-3-70B|bf16": [0.5421, 1113.7],
    "p5e.48xlarge|Llama-3-70B|fp8": [0.4213, 1731.9],
    "p5e.48xlarge|DeepSeek-V3|bf16": [0.2775, 943.7],
    "p5e.48xlarge|DeepSeek-V3|fp8": [0.1997, 1358.9],
    "p5en.48xlarge|Llama-3-70B|bf16": [0.5421, 1113.7],
    "p5en.48xlarge|Llama-3-70B|fp8": [0.4213, 1731.9],
    "p5en.48xlarge|DeepSeek-V3|bf16": [0.2775, 943.7],
    "p5en.48xlarge|DeepSeek-V3|fp8": [0.1997, 1358.9],
    "p6-b200.48xlarge|Llama-3-70B|bf16": [0.4196, 1960.9],
    "p6-b200.48xlarge|Llama-3-70B|fp8": [0.389, 3636.1],
    "p6-b200.48xlarge|DeepSeek-V3|bf16": [0.1941, 1501.7],
    "p6-b200.48xlarge|DeepSeek-V3|fp8": [0.1492, 2309.1],
    "p6-b300.48xlarge|Llama-3-70B|bf16": [0.4201, 1963.2],
    "p6-b300.48xlarge|Llama-3-70B|fp8": [0.3898, 3643.9],
    "p6-b300.48xlarge|DeepSeek-V3|bf16": [0.2246, 1737.5],
    "p6-b300.48xlarge|DeepSeek-V3|fp8": [0.1871, 2895.3],
    "p4d.24xlarge|Llama-3-70B|bf16": [0.4497, 291.4],
    "p4d.24xlarge|Llama-3-70B|fp8": [0.4497, 291.4],
    "p4d.24xlarge|DeepSeek-V3|bf16": [0.1891, 202.9],
    "p4d.24xlarge|DeepSeek-V3|fp8": [0.1891, 202.9],
    "p4de.24xlarge|Llama-3-70B|bf16": [0.4497, 291.4],
    "p4de.24xlarge|Llama-3-70B|fp8": [0.4497, 291.4],
    "p4de.24xlarge|DeepSeek-V3|bf16": [0.1891, 202.9],
    "p4de.24xlarge|DeepSeek-V3|fp8": [0.1891, 202.9],
    "dgx-h100|Llama-3-70B|bf16": [0.5421, 1113.7],
    "dgx-h100|Llama-3-70B|fp8": [0.4213, 1731.9],
    "dgx-h100|DeepSeek-V3|bf16": [0.2775, 943.7],
    "dgx-h100|DeepSeek-V3|fp8": [0.1997, 1358.9],
    "dgx-b200|Llama-3-70B|bf16": [0.4196, 1960.9],
    "dgx-b200|Llama-3-70B|fp8": [0.389, 3636.1],
    "dgx-b200|DeepSeek-V3|bf16": [0.1941, 1501.7],
    "dgx-b200|DeepSeek-V3|fp8": [0.1492, 2309.1],
    "dgx-b300|Llama-3-70B|bf16": [0.4201, 1963.2],
    "dgx-b300|Llama-3-70B|fp8": [0.3898, 3643.9],
    "dgx-b300|DeepSeek-V3|bf16": [0.2246, 1737.5],
    "dgx-b300|DeepSeek-V3|fp8": [0.1871, 2895.3],
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
    "llama70b-b200-64": [64, 1, 4, 1, 1, 1, 8, 0.4378, 140247, 183.4, 6338023.0, True],
    "llama70b-b300-128": [128, 1, 4, 1, 1, 1, 16, 0.4328, 277281, 92.8, 8904805.0, True],
    "qwen30b-p5en-64": [64, 1, 2, 1, 1, 1, 8, 0.3201, 726242, 35.4, 951966.0, True],
    "dsv3-b200-256": [256, 2, 16, 1, 1, 1, 32, 0.2902, 707848, 36.3, 5023045.0, True],
    "param2-p5en-64": [64, 1, 1, 1, 1, 1, 8, 0.3866, 1402960, 18.3, 492785.0, True],
    "glm-b200-64": [64, 1, 16, 1, 1, 1, 8, 0.3075, 377660, 68.1, 2353671.0, True],
    "glm-b300-auto30": [256, 2, 8, 1, 1, 1, 32, 0.2993, 1470710, 17.5, 3357745.0, True],
    "qwen8b-b300-auto": [64, 1, 1, 1, 1, 1, 8, 0.4428, 1232157, 20.9, 1001956.0, True],
    "llama8b-b300-12": [12, 1, 1, 1, 1, 8, 2, 0.443, 232371, 110.7, 1328228.0, True],
}

PIN_TRIED = {"glm-b300-auto30": [[8, True, 585.0], [16, True, 281.4], [32, True, 135.2], [64, True, 68.1], [128, True, 34.5], [256, True, 17.5]], "qwen8b-b300-auto": [[8, True, 166.9], [16, True, 83.5], [32, True, 41.7], [64, True, 20.9]]}

LLAMA70B_PLAN = dict(num_gpus=64, tp=2, pp=4, cp=1, vp=5, ep=1, mbs=1, seq=8192, gbs=128)
DSV3_PLAN = dict(num_gpus=256, tp=1, pp=8, cp=1, vp=2, ep=32, etp=1, mbs=1, seq=4096, gbs=2048,
                 dispatcher="deepep", a2a_overlap=True, dedup_by_group=True)


class PinnedExistingHardware(unittest.TestCase):
    """p5, p5e, p5en, B200, B300, A100 and the validation systems give the numbers they gave on main."""

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
        for given in ("trn2", "trn2.48xlarge", "trainium", "tpu-v5", "b800"):
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
        self.assertEqual(gb["alltoall_communication"]["comm_type"], "intra-node (NVLink)")
        hgx = run("DeepSeek-V3", hardware="b200", total_tokens_b=2000, num_gpus=256, expert_parallel=64)
        self.assertEqual(hgx["alltoall_communication"]["comm_type"], "inter-node (EFA)")

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
        "nemo-gb200-deepseek-v3": 85.0,   # NeMo 25.07 software; NeMo 26.08.01 runs this model on the same GPUs 2.1x faster
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

    def test_the_existing_38_run_summary_is_untouched(self):
        s = vr.summary(vr.evaluate())
        self.assertEqual((s["all"]["runs"], s["all"]["median_abs_error_pct"], s["all"]["max_abs_error_pct"]), (38, 6.1, 58.9))
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


if __name__ == "__main__":
    unittest.main()
