"""Test the FA-compatible wrapper: call it exactly like flash_attn_varlen_func, check vs fp32."""
import sys, itertools, torch
import hk_flash_attn

# args: "sq0,sq1,..." "sk0,sk1,..." seed [H H_KV]   (H/H_KV must match the compiled kernel)
sqs = [int(x) for x in sys.argv[1].split(',')]
sks = [int(x) for x in sys.argv[2].split(',')]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
H = int(sys.argv[4]) if len(sys.argv) > 4 else 64
H_KV = int(sys.argv[5]) if len(sys.argv) > 5 else 8
D = 64; dt = torch.bfloat16; torch.manual_seed(seed); B = len(sqs)

cuq = torch.tensor([0] + list(itertools.accumulate(sqs)), dtype=torch.int32, device='cuda')
cuk = torch.tensor([0] + list(itertools.accumulate(sks)), dtype=torch.int32, device='cuda')
totq, totk = int(cuq[-1]), int(cuk[-1])
q = torch.randn(totq, H, D, dtype=dt, device='cuda')   # NOTE: 3-D packed, exactly like FA (no pad!)
k = torch.randn(totk, H_KV, D, dtype=dt, device='cuda')
v = torch.randn(totk, H_KV, D, dtype=dt, device='cuda')

out, lse = hk_flash_attn.flash_attn_varlen_func(
    q, k, v, cuq, cuk, max(sqs), max(sks), causal=True, return_lse=True)
torch.cuda.synchronize()

of = out.float(); G = H // H_KV; scale = 1.0 / (D ** 0.5)
worst = 0.0; badbig = 0; lworst = 0.0
for i in range(B):
    qs, qe = int(cuq[i]), int(cuq[i+1]); ks, ke = int(cuk[i]), int(cuk[i+1])
    Lq, Lk = qe - qs, ke - ks; delta = Lk - Lq
    qi, ki, vi = q[qs:qe].float(), k[ks:ke].float(), v[ks:ke].float()
    rq = torch.arange(Lq, device='cuda'); rk = torch.arange(Lk, device='cuda')
    cmask = rk[None, :] > (rq[:, None] + delta)
    for h in range(H):
        hk = h // G
        S = (qi[:, h, :] @ ki[:, hk, :].t()) * scale
        S = S.masked_fill(cmask, float('-inf'))
        O = torch.softmax(S, dim=-1) @ vi[:, hk, :]
        eh = (of[qs:qe, h, :] - O).abs()
        worst = max(worst, eh.max().item()); badbig += (eh > 0.5).sum().item()
        lworst = max(lworst, (lse[h, qs:qe] - torch.logsumexp(S, dim=-1)).abs().max().item())
print(f"[wrapper sqs={sqs} sks={sks} H={H}/{H_KV} seed={seed}] out.shape={tuple(out.shape)} "
      f"lse.shape={tuple(lse.shape)} O worst={worst:.4f} O>0.5={badbig} LSE worst={lworst:.4f}")
