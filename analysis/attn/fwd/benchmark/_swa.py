import torch, sys, itertools, tk_kernel
# Naive causal-SWA (bottom-right) forward validation vs fp32 windowed reference.
# usage: _swa.py <seqlens_csv> <window_left> [seed] [H] [H_KV]  (self-attn; delta=0)
seqlens = [int(x) for x in sys.argv[1].split(',')]
W = int(sys.argv[2])                       # window_size_left; attend [i+delta-W, i+delta]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
H = int(sys.argv[4]) if len(sys.argv) > 4 else 64
H_KV = int(sys.argv[5]) if len(sys.argv) > 5 else 8
D = 64
dt = torch.bfloat16
torch.manual_seed(seed)
B = len(seqlens)
cu = torch.tensor([0] + list(itertools.accumulate(seqlens)), dtype=torch.int32, device='cuda')
total = int(cu[-1]); max_sq = max(seqlens)
q = torch.randn(total, H, D, dtype=dt, device='cuda')
k = torch.randn(total, H_KV, D, dtype=dt, device='cuda')
v = torch.randn(total, H_KV, D, dtype=dt, device='cuda')
out = torch.zeros(1, total, H, D, dtype=dt, device='cuda')
lse = torch.zeros(1, H, 1, total, dtype=torch.float32, device='cuda')
tk_kernel.dispatch_micro(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), out, lse, cu, cu, max_sq, B, W, 0)
torch.cuda.synchronize()

of = out[0].float()
G = H // H_KV; scale = 1.0 / (D ** 0.5)
worst = 0.0; badtot = 0; badbig = 0; worst_loc = None
for i in range(B):
    s, e = int(cu[i]), int(cu[i + 1]); L = e - s
    delta = 0                              # self-attn: seqlen_k == seqlen_q
    qi = q[s:e].float(); ki = k[s:e].float(); vi = v[s:e].float()
    ii = torch.arange(L, device='cuda')
    rel = (ii[:, None] + delta) - ii[None, :]       # q_pos - k_pos
    mask = (rel < 0) | (rel > W)                     # outside causal window band
    for h in range(H):
        hk = h // G
        S = (qi[:, h, :] @ ki[:, hk, :].t()) * scale
        S = S.masked_fill(mask, float('-inf'))
        O = torch.softmax(S, dim=-1) @ vi[:, hk, :]
        eh = (of[s:e, h, :] - O).abs()
        m = eh.max().item()
        if m > worst: worst = m; worst_loc = (i, h, L)
        badtot += (eh > 0.05).sum().item(); badbig += (eh > 0.5).sum().item()
print(f"[SWA seqlens={seqlens} W={W} B={B} total={total} H={H}/{H_KV} seed={seed}] "
      f"O worst={worst:.4f} O>0.5={badbig} O>0.05={badtot} worst_loc={worst_loc}")
