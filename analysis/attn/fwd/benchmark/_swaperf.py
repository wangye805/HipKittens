import torch, sys, itertools, tk_kernel
# Timed SWA throughput vs full-causal at same (L,B). Naive SWA visits all tiles up to the causal
# frontier => wall-time is expected ~= causal regardless of window; useful (windowed) FLOPs are far
# fewer, so effective-TFLOPS exposes the wasted work that exp1 loop-prune would remove.
# argv: L B window_left [seed]
L = int(sys.argv[1]); B = int(sys.argv[2]); W = int(sys.argv[3]); seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
H, H_KV, D = 64, 8, 64
dt = torch.bfloat16; torch.manual_seed(seed)
seqlens = [L] * B
cu = torch.tensor([0] + list(itertools.accumulate(seqlens)), dtype=torch.int32, device='cuda')
total = int(cu[-1]); max_sq = L
q = torch.randn(total, H, D, dtype=dt, device='cuda')
k = torch.randn(total, H_KV, D, dtype=dt, device='cuda')
v = torch.randn(total, H_KV, D, dtype=dt, device='cuda')
out = torch.zeros(1, total, H, D, dtype=dt, device='cuda')
lse = torch.zeros(1, H, 1, total, dtype=torch.float32, device='cuda')
qp, kp, vp = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)

def bench(wl):
    for _ in range(50):
        tk_kernel.dispatch_micro(qp, kp, vp, out, lse, cu, cu, max_sq, B, wl, 0)
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(100):
        tk_kernel.dispatch_micro(qp, kp, vp, out, lse, cu, cu, max_sq, B, wl, 0)
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / 100

# attended (q,k) pairs for a bottom-right causal window over one seq of length L (self-attn, delta=0)
pairs = sum(min(i + 1, W + 1) for i in range(L))
useful_flop = B * 4 * pairs * H * D          # 2 matmuls (QK^T, PV), 2 mac -> 4 * pairs * H * D
causal_flop = B * (4 * L * L * H * D // 2)

ms_c = bench(-1)                              # full causal
ms_s = bench(W)                              # SWA window W
tf_useful = useful_flop / 1e12 / (ms_s / 1e3)
tf_work   = causal_flop / 1e12 / (ms_s / 1e3)   # if you (wrongly) counted causal work
print(f"[SWA perf] L={L} B={B} W={W}  causal={ms_c:.4f}ms  swa={ms_s:.4f}ms  "
      f"swa/causal={ms_s/ms_c:.3f}x  useful={tf_useful:.1f}TF/s  (as-causal-work={tf_work:.1f}TF/s)")
