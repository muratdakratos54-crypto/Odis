"""Deterministik modda GPU: loop-loop ve grouped-loop bit bit mi? Hangi op'lar deterministik değil?"""
import os
import sys
import warnings

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch  # noqa: E402

import os  # noqa: E402
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(ROOT)
from scripts.verify_dispatch import compare, run_once  # noqa: E402
from src.checkpoint import load_model  # noqa: E402
from src.config import PRESETS  # noqa: E402
from src.data import TokenDataset  # noqa: E402
from src.model import GPT  # noqa: E402

torch.use_deterministic_algorithms(True, warn_only=True)
ds = TokenDataset("data/sp8k/val.bin", 512, "cuda")
x, y = ds.get_batch(32, torch.Generator().manual_seed(12345))
models = {"moe v1": load_model("checkpoints/moe/best.pt", "cuda")[0], "noaux v1": load_model("checkpoints/moe-noaux/best.pt", "cuda")[0]}
torch.manual_seed(1337)
models["moe init"] = GPT(PRESETS["moe"][0]).cuda()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    for name, model in models.items():
        model.train()
        for precision in ("bf16", "fp32"):
            l1 = run_once(model, x, y, precision, "loop")
            l2 = run_once(model, x, y, precision, "loop")
            g1 = run_once(model, x, y, precision, "grouped")
            g2 = run_once(model, x, y, precision, "grouped")
            for label, (a, b) in (("loop-loop", (l1, l2)), ("grouped-grouped", (g1, g2)), ("grouped-loop", (l1, g1))):
                c = compare(a, b)
                print(f"{name:9s} {precision} {label:16s} routing {c['routing_identical_all_layers']!s:5s} logits bit {c['logits_bitexact']!s:5s} "
                      f"grads bit {c['grads_bitexact']!s:5s} (rel {c['grads']:.1e}) params bit {c['params_bitexact']!s:5s} loss+1 {c['loss_after_step']:.1e}")
msgs = sorted({str(w.message)[:160] for w in caught if "deterministic" in str(w.message).lower()})
print("\nDeterministik olmayan op uyarıları:")
for m in msgs:
    print(" -", m)
