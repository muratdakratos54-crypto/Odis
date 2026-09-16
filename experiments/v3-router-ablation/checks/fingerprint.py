import hashlib, json, sys, torch
sys.path.insert(0, ".")
from src.config import ModelConfig
from src.model import GPT
torch.set_num_threads(1)
cfg = ModelConfig(vocab_size=97, max_seq_len=16, d_model=32, n_layer=3, n_head=4, ffn_hidden=88,
                  n_experts=4, top_k=2, expert_hidden=256, n_shared_experts=1, first_dense_layers=1)
torch.manual_seed(0)
x, y = torch.randint(0, 97, (4, 16)), torch.randint(0, 97, (4, 16))
torch.manual_seed(0)
model = GPT(cfg)
opt = model.configure_optimizer(1e-2, 0.1, (0.9, 0.95), "cpu")
losses, aux = [], []
for _ in range(3):
    _, loss, info = model(x, y)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item()); aux.append(info["aux_loss"].item())
h = hashlib.sha256()
for n, p in sorted(model.named_parameters()):
    h.update(n.encode()); h.update(p.detach().numpy().tobytes())
print(json.dumps({"losses": losses, "aux": aux, "params_sha256": h.hexdigest()[:24]}))
