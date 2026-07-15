import torch, math, sys, tk_kernel

torch.manual_seed(0)

# ---- config (must match the Makefile: NTILES = TOPK/32, D=512, NW=8, QB=16) ----
S     = int(sys.argv[1]) if len(sys.argv) > 1 else 4096   # query tokens (grid = one CTA/token)
TOPK  = int(sys.argv[2]) if len(sys.argv) > 2 else 1152
HAS_SINK = int(sys.argv[3]) if len(sys.argv) > 3 else 1
D, NW, QB, TILE_K, T_KV = 512, 8, 16, 32, 4096
H = NW * QB                      # 128 heads per token
NTILES = TOPK // TILE_K
NKK = NTILES * TILE_K
scale = 1.0 / math.sqrt(D)
dev, dt = 'cuda', torch.bfloat16

# ---- inputs (4-D layouts the kernel/pybind expect) ----
q    = torch.randn(S, NW, QB, D, device=dev, dtype=dt) * 0.1
kv   = (torch.randn(T_KV, D, device=dev, dtype=dt) * 0.1).view(1, 1, T_KV, D).contiguous()
topk = torch.randint(0, T_KV, (S, 1, NTILES, TILE_K), device=dev, dtype=torch.int32)
sink = (torch.randn(NW, QB, device=dev, dtype=torch.float32) * 0.5).view(1, 1, NW, QB).contiguous()
o    = torch.zeros(S, NW, QB, D, device=dev, dtype=dt)
lse  = torch.zeros(S, 1, NW, QB, device=dev, dtype=torch.float32)

def flops(s, h, topk, d):
    return 4 * s * h * topk * d          # QK + PV, apples-to-apples w/ flydsl

def efficiency(fl, ms):
    return (fl / 1e12) / (ms / 1e3)

# ---- torch reference (dense K==V, per-token top-k gather, softmax+sink, LSE) ----
def reference(q, kv, topk, sink):
    qh   = q.reshape(S, H, D).float()                      # [S,H,D]
    kvf  = kv.reshape(T_KV, D).float()
    idx  = topk.reshape(S, NKK).long()                     # [S,NKK]
    valid = idx >= 0
    gnb  = kvf[idx.clamp_min(0)]                           # [S,NKK,D]
    s_   = torch.einsum('shd,skd->shk', qh, gnb) * scale   # [S,H,NKK]
    s_   = s_.masked_fill(~valid.unsqueeze(1), float('-inf'))
    m    = s_.amax(-1, keepdim=True)                        # [S,H,1]
    p    = torch.exp(s_ - m)                                # invalid -> 0
    l    = p.sum(-1, keepdim=True)                          # [S,H,1]
    denom = l
    if HAS_SINK:
        sk = sink.reshape(H).float().view(1, H, 1)
        denom = l + torch.exp(sk - m)
    o_ref  = torch.einsum('shk,skd->shd', p, gnb) / denom   # [S,H,D]
    lse_ref = (m + torch.log(denom)).squeeze(-1)            # [S,H]
    return o_ref, lse_ref

# ---- run + time (mirrors kernels/attn/gqa/test_python.py) ----
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
for _ in range(50):
    tk_kernel.dispatch(q, kv, topk, sink, o, lse)
torch.cuda.synchronize()
timings = []
for _ in range(100):
    start.record(); tk_kernel.dispatch(q, kv, topk, sink, o, lse); end.record()
    torch.cuda.synchronize(); timings.append(start.elapsed_time(end))
ms = sorted(timings)[len(timings)//2]
print(f"DSA-V4 fwd  S={S} H={H} D={D} topk={TOPK} sink={HAS_SINK}")
print(f"  median {ms:.4f} ms   {efficiency(flops(S,H,TOPK,D), ms):.1f} TFLOPS")

# ---- correctness ----
o_ref, lse_ref = reference(q, kv, topk, sink)
o_hk   = o.reshape(S, H, D).float()
lse_hk = lse.reshape(S, H).float()
o_err  = (o_hk - o_ref).abs()
l_err  = (lse_hk - lse_ref).abs()
o_ok = (o_err <= 0.001 + 0.05 * o_ref.abs().clamp_min(1e-6)).float().mean().item()
print(f"  O   max_abs={o_err.max().item():.5f}  within-tol={100*o_ok:.3f}%  "
      f"cos={torch.nn.functional.cosine_similarity(o_hk.flatten(), o_ref.flatten(), dim=0).item():.6f}")
print(f"  LSE max_abs={l_err.max().item():.6f}")
print("  PASS" if (o_err.max().item() < 0.05 and l_err.max().item() < 0.01) else "  FAIL")
