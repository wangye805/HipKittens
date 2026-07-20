import torch, sys, itertools, tk_kernel
# CROSS-attention SWA (bottom-right causal window). delta_i = seqlen_k_i - seqlen_q_i != 0.
# argv: "sq0,sq1,..." "sk0,sk1,..." window_left seed [H H_KV]
sqs = [int(x) for x in sys.argv[1].split(',')]
sks = [int(x) for x in sys.argv[2].split(',')]
W = int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
H = int(sys.argv[5]) if len(sys.argv) > 5 else 64
H_KV = int(sys.argv[6]) if len(sys.argv) > 6 else 8
D = 64; dt = torch.bfloat16; torch.manual_seed(seed)
B = len(sqs); assert len(sks) == B
cuq = torch.tensor([0] + list(itertools.accumulate(sqs)), dtype=torch.int32, device='cuda')
cuk = torch.tensor([0] + list(itertools.accumulate(sks)), dtype=torch.int32, device='cuda')
totq = int(cuq[-1]); totk = int(cuk[-1]); max_sq = max(sqs); PAD = 256
q = torch.randn(totq + PAD, H, D, dtype=dt, device='cuda')
k = torch.randn(totk + PAD, H_KV, D, dtype=dt, device='cuda')
v = torch.randn(totk + PAD, H_KV, D, dtype=dt, device='cuda')
out = torch.zeros(1, totq, H, D, dtype=dt, device='cuda')
lse = torch.zeros(1, H, 1, totq, dtype=torch.float32, device='cuda')
tk_kernel.dispatch_micro(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), out, lse,
                         cuq, cuk, max_sq, B, W, 0)
torch.cuda.synchronize()
of = out[0].float(); G = H // H_KV; scale = 1.0 / (D ** 0.5)
worst = 0.0; badbig = 0; wl = None
for i in range(B):
    qs, qe = int(cuq[i]), int(cuq[i+1]); ks, ke = int(cuk[i]), int(cuk[i+1])
    Lq = qe - qs; Lk = ke - ks; delta = Lk - Lq
    qi = q[qs:qe].float(); ki = k[ks:ke].float(); vi = v[ks:ke].float()
    rq = torch.arange(Lq, device='cuda'); rk = torch.arange(Lk, device='cuda')
    rel = (rq[:, None] + delta) - rk[None, :]           # q_pos - k_pos (bottom-right)
    mask = (rel < 0) | (rel > W)                         # outside causal window band
    for h in range(H):
        hk = h // G
        S = (qi[:, h, :] @ ki[:, hk, :].t()) * scale
        S = S.masked_fill(mask, float('-inf'))
        O = torch.softmax(S, dim=-1) @ vi[:, hk, :]
        eh = (of[qs:qe, h, :] - O).abs(); m = eh.max().item()
        if m > worst: worst = m; wl = (i, h, Lq, Lk)
        badbig += (eh > 0.5).sum().item()
print(f"[XSWA sqs={sqs} sks={sks} W={W} H={H}/{H_KV} seed={seed}] O worst={worst:.4f} O>0.5={badbig} worst_loc(seq,h,Lq,Lk)={wl}")
