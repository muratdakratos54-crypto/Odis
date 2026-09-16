"""Deterministik mod, moe v1, bf16: grouped vs loop gradyan farkı hangi tensörlerde? fork32 bit bit mi?"""
import os
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch  # noqa: E402

import os  # noqa: E402
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(ROOT)
from scripts.verify_dispatch import compare, run_once  # noqa: E402
from src.checkpoint import load_model  # noqa: E402
from src.data import TokenDataset  # noqa: E402

torch.use_deterministic_algorithms(True, warn_only=True)
ds = TokenDataset("data/sp8k/val.bin", 512, "cuda")
x, y = ds.get_batch(32, torch.Generator().manual_seed(12345))
model = load_model("checkpoints/moe/best.pt", "cuda")[0].train()
L = run_once(model, x, y, "bf16", "loop")
G = run_once(model, x, y, "bf16", "grouped")
Fk = run_once(model, x, y, "bf16", "fork32")
for label, R in (("grouped", G), ("fork32", Fk)):
    c = compare(L, R)
    diff = [n for n in L["grads"] if not torch.equal(L["grads"][n], R["grads"][n])]
    print(f"{label}: grads bit {c['grads_bitexact']} rel {c['grads']:.2e}; farklı tensör {len(diff)}/{len(L['grads'])}")
    # en derin (son) farklı katman: backward sondan başa aktığı için farkın doğduğu yer
    layers = sorted({int(n.split('.')[1]) for n in diff if n.startswith('blocks.')})
    print("   farklı katmanlar:", layers)
    deepest = [n for n in diff if n.startswith(f"blocks.{layers[-1]}.")] if layers else diff
    print("   en derin katmanda farklı tensörler:", deepest[:12], "..." if len(deepest) > 12 else "")
    other = [n for n in diff if not n.startswith("blocks.")]
    print("   blok dışı:", other)
