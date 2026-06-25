"""Deterministic backward (split+reduce) test for HK GQA causal bwd.
Feeds O/L from a torch reference (HK fwd kernel doesn't build on ROCm 7.2), so it
only needs the bwd + prep modules. Checks (1) dQ bitwise-identical across two runs,
(2) dQ/dK/dV vs torch fp32. Square causal, D=128.

  make HK_DETERMINISTIC=1 ATTN_B=4 ATTN_N=4096 ATTN_H=64 ATTN_H_KV=8 PY_EXT_SUFFIX=.cpython-311-x86_64-linux-gnu.so
  python test_python_det.py 4 4096 64 8 1
"""
import sys, math, torch
import tk_kernel_bkwd, tk_kernel_bkwd_prep

torch.manual_seed(0)
dev = "cuda"
B   = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N   = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
H   = int(sys.argv[3]) if len(sys.argv) > 3 else 64
HKV = int(sys.argv[4]) if len(sys.argv) > 4 else 8
causal = bool(int(sys.argv[5])) if len(sys.argv) > 5 else True
D = 128
nsplit = N // 256
group = H // HKV
scale = D ** -0.5
print(f"B={B} N={N} H={H} HKV={HKV} D={D} causal={causal} nsplit={nsplit}")

def gen(s): return torch.randn(s, dtype=torch.bfloat16, device=dev)
Q_bhnd, K_bhnd, V_bhnd, dO_bhnd = gen((B,H,N,D)), gen((B,HKV,N,D)), gen((B,HKV,N,D)), gen((B,H,N,D))

# memory-light O/L (per-head forward, no autograd) — avoids the 17GB fp32 S on a busy node
tmask = torch.triu(torch.ones(N,N,device=dev,dtype=torch.bool),1) if causal else None
O_bhnd = torch.empty((B,H,N,D), device=dev, dtype=torch.float32)
lse = torch.empty((B,H,N), device=dev, dtype=torch.float32)
with torch.no_grad():
    for h in range(H):
        s = (Q_bhnd[:,h].float() @ K_bhnd[:,h//group].float().transpose(-1,-2)) * scale
        if causal: s = s.masked_fill(tmask, float("-inf"))
        lse[:,h] = torch.logsumexp(s, -1)
        O_bhnd[:,h] = torch.softmax(s,-1) @ V_bhnd[:,h//group].float()
        del s
Q_tk  = Q_bhnd.transpose(1,2).contiguous()
K_tk  = K_bhnd.transpose(1,2).contiguous()
V_tk  = V_bhnd.transpose(1,2).contiguous()
dO_tk = dO_bhnd.transpose(1,2).contiguous()
O_tk  = O_bhnd.transpose(1,2).bfloat16().contiguous()              # [B,N,H,D]
L_tk  = lse.reshape(B,H,1,N).float().contiguous()                  # [B,H,1,N] (natural lse)

# reference grads, per-head (memory-light: peak ~one [B,N,N] head, not the full 17GB S)
class _G:
    def __init__(s): s.grad=None
Qr, Kr, Vr = _G(), _G(), _G()
try:
    Qg = torch.zeros((B,H,N,D), device=dev)
    Kg = torch.zeros((B,HKV,N,D), device=dev)
    Vg = torch.zeros((B,HKV,N,D), device=dev)
    for h in range(H):
        qh = Q_bhnd[:,h].float().detach().requires_grad_(True)
        kh = K_bhnd[:,h//group].float().detach().requires_grad_(True)
        vh = V_bhnd[:,h//group].float().detach().requires_grad_(True)
        s = (qh @ kh.transpose(-1,-2)) * scale
        if causal: s = s.masked_fill(tmask, float("-inf"))
        (torch.softmax(s,-1) @ vh).backward(dO_bhnd[:,h].float())
        Qg[:,h] = qh.grad; Kg[:,h//group] += kh.grad; Vg[:,h//group] += vh.grad
        del s
    Qr.grad, Kr.grad, Vr.grad = Qg, Kg, Vg
except Exception as ex:
    print("[correctness] reference skipped:", type(ex).__name__)

def run_bwd_det():
    dQ_acc = torch.zeros((nsplit*B, H, N, D), dtype=torch.float32, device=dev).contiguous()
    dQ_in  = torch.zeros((B,H,N,D), dtype=torch.bfloat16, device=dev).contiguous()
    dQ     = torch.zeros((B,N,H,D), dtype=torch.bfloat16, device=dev).contiguous()
    dK     = torch.zeros((B,N,HKV,D), dtype=torch.bfloat16, device=dev).contiguous()
    dV     = torch.zeros((B,N,HKV,D), dtype=torch.bfloat16, device=dev).contiguous()
    delta  = torch.zeros((B,H,N,1), device=dev).float().transpose(-1,-2).contiguous()
    tk_kernel_bkwd_prep.dispatch_prep(O_tk, dO_tk, delta)
    tk_kernel_bkwd.dispatch_bwd_combined(Q_tk,K_tk,V_tk,dO_tk,dQ_in,dK,dV,L_tk,delta,dQ_acc)
    tk_kernel_bkwd_prep.dispatch_dq_reduce(dQ_acc, dQ_in)
    tk_kernel_bkwd_prep.dispatch_dq_shuffle(dQ_in, dQ)
    torch.cuda.synchronize()
    return dQ, dK, dV

dQ1, dK1, dV1 = run_bwd_det()
dQ2, dK2, dV2 = run_bwd_det()
print(f"[determinism] dQ equal across runs: {torch.equal(dQ1,dQ2)}  dK: {torch.equal(dK1,dK2)}  dV: {torch.equal(dV1,dV2)}")

def report(name, ref, hk):
    err = (ref.float()-hk.float()).abs()
    print(f"  {name}: max_abs={err.max().item():.4e}  mean_abs={err.mean().item():.4e}  ref_absmax={ref.abs().max().item():.3e}")
if Qr.grad is not None:
    print("[correctness vs torch fp32]")
    report("dQ", Qr.grad, dQ1.transpose(1,2))
    report("dK", Kr.grad, dK1.transpose(1,2))
    report("dV", Vr.grad, dV1.transpose(1,2))
