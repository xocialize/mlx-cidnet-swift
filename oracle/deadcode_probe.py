"""Verify by ablation what code-reading suggested: I_LCA5's output is discarded.

    i_dec2 = self.I_LCA5(i_dec3, hv_3)        # (A)
    hv_2   = self.HV_LCA5(hv_3, i_dec3)       # (B)
    hv_2   = self.HVD_block2(hv_2, hv_jump1)  # consumes (B)  -> LIVE
    i_dec2 = self.ID_block2(i_dec3, v_jump1)  # overwrites (A) -> (A) DEAD

If that reading is right, corrupting I_LCA5's weights cannot change the output, while corrupting
HV_LCA5's must.
"""
import sys, torch
sys.path.insert(0, "upstream")
from net.CIDNet import CIDNet

torch.set_grad_enabled(False)
x = torch.from_numpy(__import__("numpy").random.default_rng(9001)
                     .random((1, 3, 128, 128), dtype="float32"))

def out_with_corrupted(module_name):
    m = CIDNet.from_pretrained("Fediory/HVI-CIDNet-Generalization").eval()
    if module_name:
        mod = getattr(m, module_name)
        for p in mod.parameters():
            p.mul_(0).add_(7.0)          # obliterate it
    return m(x)

base = out_with_corrupted(None)
for name in ["I_LCA5", "HV_LCA5", "I_LCA2", "I_LCA6"]:
    got = out_with_corrupted(name)
    d = (got - base).abs().max().item()
    verdict = "DEAD (output unchanged)" if d == 0.0 else f"live (max delta {d:.6f})"
    print(f"  corrupt {name:8s} -> {verdict}")
