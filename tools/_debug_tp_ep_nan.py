"""Isolate which sharding plan produces NaN gradients on bwd: plain TP
(incl. TP-over-experts via base_model_tp_plan / tp_plan="auto") vs the
hybrid TP+EP plan (ep_router/grouped_gemm). No LoRA — just check whether
the router's score tensor gets a NaN gradient on the very first backward."""
import os
import sys

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

MODEL_DIR = "/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
VOCAB = 151936
MODE = sys.argv[1] if len(sys.argv) > 1 else "auto"

local_rank = int(os.environ["LOCAL_RANK"])
dist_rank = int(os.environ["RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


def log(m):
    if dist_rank == 0:
        print(m, flush=True)


torch.manual_seed(0)

log(f"mode={MODE}  loading...")
kwargs = dict(dtype=torch.bfloat16, tp_plan="auto")
if MODE == "ep_only":
    from transformers.distributed import DistributedConfig
    kwargs["distributed_config"] = DistributedConfig(enable_expert_parallel=True)

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, **kwargs)
model.train()
torch.cuda.synchronize()
log(f"loaded — {torch.cuda.memory_allocated(local_rank)/1e9:.1f} GB on rank {dist_rank}")

tp_group = model._device_mesh.get_group()

T = 128
batch = torch.randint(0, VOCAB, (1, T + 1), device=device)
dist.broadcast(batch, src=0, group=tp_group)
input_ids, labels = batch[:, :-1], batch[:, 1:]

# Hook the router (gate) of layer 0 to inspect its output grad for NaN.
hits = {}
def mk_hook(name):
    def hook(module, grad_input, grad_output):
        for j, g in enumerate(grad_output):
            if g is not None:
                hits[f"{name}.out[{j}]"] = (bool(torch.isnan(g).any().item()), bool(torch.isinf(g).any().item()), g.float().norm().item())
        for j, g in enumerate(grad_input):
            if g is not None:
                hits[f"{name}.in[{j}]"] = (bool(torch.isnan(g).any().item()), bool(torch.isinf(g).any().item()), g.float().norm().item())
    return hook

layer0 = model.model.layers[0]
h1 = layer0.mlp.gate.register_full_backward_hook(mk_hook("layer0.mlp.gate"))
h2 = layer0.mlp.experts.register_full_backward_hook(mk_hook("layer0.mlp.experts"))
h3 = layer0.mlp.register_full_backward_hook(mk_hook("layer0.mlp"))
h4 = layer0.self_attn.o_proj.register_full_backward_hook(mk_hook("layer0.self_attn.o_proj"))

out = model(input_ids=input_ids, labels=labels)
log(f"loss={out.loss.item():.4f}")
out.loss.backward()
torch.cuda.synchronize()

for k, v in hits.items():
    print(f"  [r{dist_rank}] {k:35s} nan={v[0]} inf={v[1]} norm={v[2]:.4e}", flush=True)

dist.destroy_process_group()
